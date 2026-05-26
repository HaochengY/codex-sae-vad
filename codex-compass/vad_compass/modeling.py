import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_in, num_latents, k):
        super().__init__()
        self.d_in = int(d_in)
        self.num_latents = int(num_latents)
        self.k = int(k)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, num_latents)
        self.decoder = nn.Linear(num_latents, d_in, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def remove_decoder_parallel_grad(self):
        if self.decoder.weight.grad is None:
            return
        grad = self.decoder.weight.grad
        weight = self.decoder.weight.data
        parallel = (grad * weight).sum(dim=0, keepdim=True) * weight
        grad.sub_(parallel)

    def encode(self, x):
        acts = F.relu(self.encoder(x - self.b_dec))
        values, indices = torch.topk(acts, k=min(self.k, acts.shape[-1]), dim=-1, sorted=False)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, indices, values)
        return sparse, values, indices

    def forward(self, x):
        sparse, values, indices = self.encode(x)
        x_hat = self.decoder(sparse) + self.b_dec
        return x_hat, sparse, values, indices


class QueryBookSlots(nn.Module):
    def __init__(
        self,
        k_slots=4,
        sae_dim=65536,
        slot_dim=512,
        nhead=8,
        num_layers=2,
        ff_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.k_slots = int(k_slots)
        self.sae_dim = int(sae_dim)
        self.slot_dim = int(slot_dim)
        self.codebook = nn.Linear(self.sae_dim, self.slot_dim, bias=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.slot_dim,
            nhead=nhead,
            dim_feedforward=ff_mult * self.slot_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(self.slot_dim)
        self.slot_queries = nn.Parameter(torch.zeros(1, self.k_slots, self.slot_dim))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)

    def forward(self, sparse_tokens, attention_mask=None):
        # sparse_tokens: [B, L, d_sae]
        bsz = sparse_tokens.size(0)
        x = self.codebook(sparse_tokens)
        slots = self.slot_queries.expand(bsz, -1, -1)
        x = torch.cat([x, slots], dim=1)
        key_padding_mask = None
        if attention_mask is not None:
            valid = attention_mask.to(torch.bool)
            pad_slots = torch.zeros(bsz, self.k_slots, dtype=torch.bool, device=valid.device)
            key_padding_mask = torch.cat([~valid, pad_slots], dim=1)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.final_norm(x[:, -self.k_slots :])


class VadCompassHead(nn.Module):
    """SAE + query-codebook slots + video-level anomaly classifier.

    This is the VAD analogue of SegCompass' sparse-concept path. Since this
    task is video-level classification, the slot confidences are aggregated
    into one abnormal probability instead of decoding pixel masks.
    """

    def __init__(
        self,
        d_in,
        num_latents,
        topk,
        k_slots=4,
        slot_dim=512,
        nhead=8,
        num_layers=2,
        freeze_sae=False,
    ):
        super().__init__()
        self.sae = TopKSparseAutoencoder(d_in=d_in, num_latents=num_latents, k=topk)
        self.query_book = QueryBookSlots(
            k_slots=k_slots,
            sae_dim=num_latents,
            slot_dim=slot_dim,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.pos_proj = nn.Linear(d_in, slot_dim, bias=False)
        self.query_fuse = nn.Sequential(
            nn.LayerNorm(2 * slot_dim),
            nn.Linear(2 * slot_dim, slot_dim),
            nn.SiLU(inplace=True),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(slot_dim // 2, 1),
        )
        if freeze_sae:
            for p in self.sae.parameters():
                p.requires_grad_(False)

    def forward(self, hidden_tokens, pos_hidden):
        # hidden_tokens: [L, D] or [B, L, D]; pos_hidden: [K, D] or [B, K, D]
        single = hidden_tokens.dim() == 2
        if single:
            hidden_tokens = hidden_tokens.unsqueeze(0)
        if pos_hidden.dim() == 2:
            pos_hidden = pos_hidden.unsqueeze(0)
        bsz, length, d = hidden_tokens.shape
        flat = hidden_tokens.reshape(bsz * length, d)
        x_hat, sparse, values, indices = self.sae(flat)
        sparse_tokens = sparse.reshape(bsz, length, -1)
        token_mask = torch.ones(bsz, length, dtype=torch.bool, device=hidden_tokens.device)
        concept_slots = self.query_book(sparse_tokens, token_mask)
        pos_slots = self.pos_proj(pos_hidden.to(dtype=concept_slots.dtype, device=concept_slots.device))
        fused_slots = self.query_fuse(torch.cat([pos_slots, concept_slots], dim=-1))
        slot_logits = self.conf_head(fused_slots).squeeze(-1)
        slot_probs = torch.sigmoid(slot_logits)
        video_prob = 1.0 - torch.prod(1.0 - slot_probs.clamp(1e-6, 1.0 - 1e-6), dim=-1)
        recon_loss = F.mse_loss(x_hat, flat)
        out = {
            "slot_logits": slot_logits,
            "slot_probs": slot_probs,
            "video_prob": video_prob,
            "recon_loss": recon_loss,
            "concept_slots": concept_slots,
            "fused_slots": fused_slots,
            "active_latents": torch.tensor(float(values.shape[-1]), device=hidden_tokens.device),
        }
        return out
