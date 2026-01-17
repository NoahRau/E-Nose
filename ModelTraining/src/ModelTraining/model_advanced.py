import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ModelTraining.model import CrossAttentionBlock, PatchEmbed

logger = logging.getLogger(__name__)

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Leistungsfähigere Alternative zu Standard MLPs (GeLU/ReLU).
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
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class MoldClassifierHead(nn.Module):
    """
    Dieser Head wird erst in Phase 2 benutzt!
    Er setzt auf das vortrainierte Foundation Modell auf.
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
        # x ist das CLS Token aus dem Backbone
        return self.head(x)

class FridgeMoCA_V3(nn.Module):
    def __init__(
            self,
            seq_len=512,
            patch_size=16,
            gas_chans=10,
            env_chans=3,
            embed_dim=384,   # Erhöht für V3 (Capacity Boost)
            depth=6,
            num_heads=6,
            out_dim=4096,    # DINO Output Dimension
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

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

        # --- DINOv3 Heads (mit SwiGLU) ---
        # DINO Head (für CLS Token)
        self.dino_head = nn.Sequential(
            SwiGLU(embed_dim, hidden_features=2048, out_features=out_dim),
            weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        )

        # iBOT Head (für Patches)
        self.ibot_head = nn.Sequential(
            SwiGLU(embed_dim, hidden_features=2048, out_features=out_dim),
            weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        )

        # Init weight norm reference magnitude
        with torch.no_grad():
            self.dino_head[1].parametrizations.weight.original0.fill_(1.0)
            self.ibot_