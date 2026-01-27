import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

# Make sure this import matches your file structure
from ModelTraining.model import CrossAttentionBlock, PatchEmbed

logger = logging.getLogger(__name__)

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (Three-Matrix Implementation).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

class MoldClassifierHead(nn.Module):
    """
    Classifier Head for Phase 2.
    """
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
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
            gas_chans=1,
            env_chans=6,
            embed_dim=384,
            depth=6,
            num_heads=6,
            out_dim=4096,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        # --- Encoders ---
        self.patch_embed_gas = PatchEmbed(patch_size, gas_chans, embed_dim)
        self.pos_embed_gas = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking.
        x: [Batch, Patches, Dim]
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Generiere Rauschen für Sortierung
        noise = torch.rand(N, L, device=x.device)

        # Sortieren: kleine Werte behalten, große verwerfen
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep indices
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather: Wir behalten nur die ausgewählten Patches
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Maske generieren (0 = keep, 1 = remove) für Loss
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x_gas, x_env, mask_ratio=0.0):
        """
        Encoder Logic.
        mask_ratio: 0.0 for Teacher/Inference, >0.0 for Student Training
        """
        # 1. Embed & Positional Info
        emb_gas = self.patch_embed_gas(x_gas) + self.pos_embed_gas
        emb_env = self.patch_embed_env(x_env) + self.pos_embed_env

        # 2. Concatenate Gas and Env BEFORE masking
        # (Allows model to learn relations between Gas and Env)
        x = torch.cat([emb_gas, emb_env], dim=1)

        # 3. Apply Masking (only if requested)
        if mask_ratio > 0.0:
            x, mask, _ = self.random_masking(x, mask_ratio)
        else:
            mask = None

        # 4. Append CLS Token (never masked)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 5. Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x_gas, x_env, mask_ratio=0.0):
        """Pre-Training Forward Pass"""
        # Pass mask_ratio down to features
        x = self.forward_features(x_gas, x_env, mask_ratio=mask_ratio)

        cls_feat = x[:, 0]    # [Batch, Dim]
        patch_feat = x[:, 1:] # [Batch, N_Visible_Patches, Dim]

        # Heads
        dino_out = self.dino_head(F.normalize(cls_feat, dim=-1))

        # Flatten patches for iBOT head
        patch_out_flat = self.ibot_head(F.normalize(patch_feat.reshape(-1, patch_feat.shape[-1]), dim=-1))

        # Reshape back to [Batch, N, Dim]
        patch_out = patch_out_flat.reshape(x.shape[0], -1, patch_out_flat.shape[-1])

        return dino_out, patch_out, cls_feat, patch_feat