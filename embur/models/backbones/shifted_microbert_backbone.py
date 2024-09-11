
# File: embur/models/backbones/shifted_microbert_backbone.py

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.backbones.backbone import Backbone
import torch.cuda.graphs as torch_graphs

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class ShiftedCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Shifted attention specific
        self.attn_proj = nn.Linear(config.n_embd, 7 * config.n_head)
        self.layer_attn_proj = nn.Linear(config.n_embd, config.n_head)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, prev_attn=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Shifted attention mechanisms
        shifted_atts = [att]
        for i in range(1, 4):
            shifted_atts.append(F.pad(att[:, :, :, i:], (0, i), value=0))
            shifted_atts.append(F.pad(att[:, :, :, :-i], (i, 0), value=0))
        
        attn_proj = self.attn_proj(x).view(B, T, 7, self.n_head).permute(0, 2, 3, 1)
        beta = F.softmax(attn_proj, dim=1).unsqueeze(-1)
        
        interpolated_att = sum(beta[:, i] * shifted_atts[i] for i in range(7))
        
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, -1, -1)
        interpolated_att = interpolated_att.masked_fill(causal_mask, 0)
        
        if prev_attn is not None:
            layer_attn_proj = self.layer_attn_proj(x).transpose(1, 2)
            gamma = torch.sigmoid(layer_attn_proj).unsqueeze(-1)
            final_att = gamma * interpolated_att + (1 - gamma) * prev_attn
        else:
            final_att = interpolated_att
        
        y = final_att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, final_att

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = ShiftedCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, prev_attn=None):
        attn_output, new_attn = self.attn(self.ln_1(x), prev_attn)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_attn

@Backbone.register("shifted_microbert")
class ShiftedMicroBERTBackbone(Backbone):
    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_attention_heads: int,
        tokenizer_path: str,
        position_embedding_type: str = "absolute",
        activation: str = "gelu",
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self._vocab = vocab
        self._namespace = "tokens"
        
        config = type('Config', (), {
            'n_embd': embedding_dim,
            'n_head': num_attention_heads,
            'n_layer': num_layers,
            'block_size': 1024,  # You might want to make this configurable
            'bias': True,
            'dropout': hidden_dropout,
        })()
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab.get_vocab_size(self._namespace), config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, vocab.get_vocab_size(self._namespace), bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, text):
        device = text["tokens"]["token_ids"].device
        with torch.cuda.device(device):
            with torch.random.fork_rng(devices=[device]):
                torch.random.manual_seed(0)  # Use a fixed seed for reproducibility
                b, t = text["tokens"]["token_ids"].size()
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
                pos = torch.arange(0, t, dtype=torch.long, device=device)

                tok_emb = self.transformer.wte(text["tokens"]["token_ids"])
                pos_emb = self.transformer.wpe(pos)
                
                with torch_graphs.DisableGraphCapture():
                    x = self.transformer.drop(tok_emb + pos_emb)
                
                prev_attn = None
                for block in self.transformer.h:
                    x, prev_attn = block(x, prev_attn)
                
                x = self.transformer.ln_f(x)
                
                outputs = {
                    "encoded_text": x,
                    "encoded_text_mask": text["tokens"]["mask"],
                    "token_ids": text["tokens"]["token_ids"],
                }
                
                return outputs
            
    def make_output_human_readable(self, output_dict):
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self._vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        del output_dict["token_ids"]
        return output_dict