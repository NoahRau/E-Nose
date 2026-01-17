import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ModelTraining.model import CrossAttentionBlock, PatchEmbed

logger = logging.getLogger(__name__)

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (Three-Matrix Implementation).
    Entspricht der LLaMA/PaLM Architektur: (Swish(xW) * xV) * W2
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Kombinierte Projektion für Gate (W) und Value (V) -> Effizienter als 2 Layer
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        # Output Projektion (W2)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # 1. Projektion: Erzeugt Gate- und Value-Logits gleichzeitig
        x12 = self.w12(x)
        # 2. Split: Trennt Gate (x1) und Value (x2)
        x1, x2 = x12.chunk(2, dim=-1)
        # 3. Gating: Swish Activation auf Gate * Lineare Value
        hidden = F.silu(x1) * x2
        # 4. Output Projektion
        return self.w3(hidden)

class MoldClassifierHead(nn.Module):
    """
    Klassifizierungs-Kopf für Phase 2 (Fine-Tuning).
    """
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            # SwiGLU für bessere nicht-lineare Trennung der Klassen
            SwiGLU(embed_dim, hidden_features=embed_dim*2, out_features=embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class FridgeMoCA_V3(nn.Module):
    def __init__(
            self,
            seq_len=512,
            patch_size=16,
            gas_chans=1,   # <--- ANPASSUNG: Standard auf 1 (Dein bme_gas)
            env_chans=6,   # <--- ANPASSUNG: Standard auf 6 (Deine 6 Env-Werte)
            embed_dim=384,
            depth=6,
            num_heads=6,
            out_dim=4096,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # --- Encoders ---
        # 1. Chemie-Encoder (Gas)
        self.patch_embed_gas = PatchEmbed(patch_size, gas_chans, embed_dim)
        self.pos_embed_gas = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 2. Physik-Encoder (Env)
        self.patch_embed_env = PatchEmbed(patch_size, env_chans, embed_dim)
        self.pos_embed_env = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # --- Transformer Backbone ---
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- DINOv3 Heads ---
        self.dino_head = nn.Sequential(
            SwiGLU(embed_dim, hidden_features=2048, out_features=out_dim),
            weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        )

        self.ibot_head = nn.Sequential(
            SwiGLU(embed_dim, hidden_features=2048, out_features=out_dim),
            weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        )

        # Weight Norm Init
        with torch.no_grad():
            self.dino_head[1].parametrizations.weight.original0.fill_(1.0)
            self.ibot_head[1].parametrizations.weight.original0.fill_(1.0)

    def forward_features(self, x_gas, x_env):
        """Encoder Logic (für Inference/Classification)"""
        emb_gas = self.patch_embed_gas(x_gas) + self.pos_embed_gas
        emb_env = self.patch_embed_env(x_env) + self.pos_embed_env

        B = x_gas.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concat: [CLS, Gas, Env]
        x = torch.cat([cls_tokens, emb_gas, emb_env], dim=1)

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)

    def forward(self, x_gas, x_env):
        """Pre-Training Forward Pass"""
        x = self.forward_features(x_gas, x_env)

        cls_feat = x[:, 0]
        patch_feat = x[:, 1:]

        dino_out = self.dino_head(F.normalize(cls_feat, dim=-1))

        patch_out_flat = self.ibot_head(F.normalize(patch_feat.reshape(-1, patch_feat.shape[-1]), dim=-1))
        patch_out = patch_out_flat.reshape(x.shape[0], -1, patch_out_flat.shape[-1])

        return dino_out, patch_out, cls_feat, patch_feat