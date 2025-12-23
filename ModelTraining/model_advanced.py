import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PatchEmbed, CrossAttentionBlock # Wir erben vom alten Modell oder importieren Teile

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # DINOv2 Style MLP: Linear -> GELU -> Linear -> GELU -> Linear (Bottleneck) -> WeightNorm -> Linear (Prototypes)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # L2 Norm vor dem letzten Layer (KoLeo mag das)
        x = self.last_layer(x)
        return x

class FridgeMoCA_Pro(nn.Module):
    def __init__(self, seq_len=512, patch_size=16, gas_chans=10, env_chans=3,
                 embed_dim=192, depth=6, num_heads=6, out_dim=4096): # out_dim = Anzahl "Prototypen"
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # Encoders
        self.patch_embed_gas = PatchEmbed(patch_size, gas_chans, embed_dim)
        self.pos_embed_gas = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.patch_embed_env = PatchEmbed(patch_size, env_chans, embed_dim)
        self.pos_embed_env = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer
        self.blocks = nn.ModuleList([CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # --- DINOv2 Spezialitäten ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # DINO Head (für das [CLS] Token)
        self.dino_head = DINOHead(embed_dim, out_dim)

        # iBOT Head (für die Patches - "Online Tokenizer")
        # Oft teilt man sich Gewichte, aber separat ist flexibler
        self.ibot_head = DINOHead(embed_dim, out_dim)

    def forward_features(self, x_gas, x_env, mask_gas=None, mask_env=None):
        """Shared Encoder Logic"""
        # Embed
        emb_gas = self.patch_embed_gas(x_gas) + self.pos_embed_gas
        emb_env = self.patch_embed_env(x_env) + self.pos_embed_env

        # CLS Token vorbereiten
        B = x_gas.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Masking anwenden (falls Masken da sind - Student Mode)
        if mask_gas is not None:
            # Wir nehmen an, mask_gas ist Bool [B, L]
            # Hier vereinfacht: Wir Nullen die maskierten Stellen (oder ersetzen durch MASK Token)
            # DINOv2 ersetzt oft durch Learnable Mask Token
            pass # (Implementation detail: Mask Token einfügen)

        # Concat: [CLS, Gas, Env]
        x = torch.cat([cls_tokens, emb_gas, emb_env], dim=1)

        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x # [Batch, 1 + L_Gas + L_Env, Dim]

    def forward(self, x_gas, x_env, mask_info=None):
        # x_gas: [B, 10, L], x_env: [B, 3, L]

        # 1. Features extrahieren
        # (Im Student-Pass würden wir hier maskieren)
        x = self.forward_features(x_gas, x_env)

        # 2. Heads anwenden
        cls_feat = x[:, 0]
        patch_feat = x[:, 1:]

        # DINO Output (CLS)
        dino_out = self.dino_head(cls_feat)

        # iBOT Output (Patches)
        # Wir flatten die Patches für den Head: [B * L, Dim]
        patch_out_flat = self.ibot_head(patch_feat.reshape(-1, patch_feat.shape[-1]))
        patch_out = patch_out_flat.reshape(x.shape[0], -1, patch_out_flat.shape[-1])

        return dino_out, patch_out, cls_feat # cls_feat (vor Head) für KoLeo Loss