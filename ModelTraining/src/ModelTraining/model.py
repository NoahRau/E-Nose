import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --- 1. Patch Embedding (Entry point into Transformer) ---
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [Batch, Channels, Time] -> [Batch, Embed_Dim, Patches]
        x = self.proj(x)
        # Transpose für Transformer: [Batch, Patches, Embed_Dim]
        return x.transpose(1, 2)


# --- 2. Der "Fridge-MoCA" Block ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # Self-Attention + Residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP + Residual
        x = x + self.mlp(self.norm2(x))
        return x


# --- 3. Das Hauptmodell ---
class FridgeMoCA(nn.Module):
    def __init__(
        self,
        seq_len=512,  # Fenstergröße (z.B. ca. 17 Minuten bei 2s Takt)
        patch_size=16,  # Wir fassen 16 Zeitschritte (32s) zusammen
        gas_chans=10,  # BME688 Kanäle
        env_chans=3,  # CO2, Temp, Hum
        embed_dim=128,  # Größe des "Gedankens" pro Patch
        depth=4,  # Wie tief denkt das Modell?
        num_heads=4,
        mask_ratio=0.5,
    ):  # SSL: 50% der Daten verstecken
        super().__init__()

        self.mask_ratio = mask_ratio
        self.num_patches = seq_len // patch_size

        # --- A. Encoders (Getrennte Eingänge für MoCA) ---
        # 1. Chemie-Encoder (Gas)
        self.patch_embed_gas = PatchEmbed(patch_size, gas_chans, embed_dim)
        self.pos_embed_gas = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 2. Physik-Encoder (Env)
        self.patch_embed_env = PatchEmbed(patch_size, env_chans, embed_dim)
        self.pos_embed_env = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 3. Transformer Blöcke (Shared oder Separate? Hier Shared für Fusion)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # --- B. Decoder (Rekonstruktion) ---
        # Wir wollen die Originalwerte vorhersagen
        self.decoder_pred_gas = nn.Linear(embed_dim, patch_size * gas_chans)
        self.decoder_pred_env = nn.Linear(embed_dim, patch_size * env_chans)

    def random_masking(self, x):
        """
        Der SSL-Trick: Wir verstecken zufällige Patches.
        x: [Batch, Patches, Dim]
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))

        # Rauschen erzeugen um Indizes zu sortieren
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # Zufällige Reihenfolge
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # Um später zurück zu sortieren

        # Wir behalten nur die ersten 'len_keep' Patches
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather (Auswählen der sichtbaren Patches)
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Maske generieren (0 = keep, 1 = remove) für Loss-Berechnung
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x_gas, x_env):
        # 1. Embedden
        # x_gas Input: [Batch, 10, SeqLen]
        emb_gas = self.patch_embed_gas(x_gas) + self.pos_embed_gas
        emb_env = self.patch_embed_env(x_env) + self.pos_embed_env

        # 2. MoCA Masking Strategie:
        # Wir maskieren Gas und Env UNABHÄNGIG voneinander.
        # Das zwingt das Modell, fehlendes Gas durch vorhandenes CO2 zu raten.
        gas_vis, mask_gas, ids_restore_gas = self.random_masking(emb_gas)
        env_vis, mask_env, ids_restore_env = self.random_masking(emb_env)

        # 3. Fusion (Concatenation der sichtbaren Patches)
        # Für den Encoder mischen wir alles, was sichtbar ist.
        # (Hinweis: Einfache Version. Für echtes Cross-Attn müsste man komplexer routen,
        # aber Concat + Self-Attention ist mathematisch äquivalent zu Global Attention)
        x = torch.cat([gas_vis, env_vis], dim=1)

        # 4. Transformer Durchlauf
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 5. Split back (Wir müssen wissen, was Gas und was Env war)
        # Da wir unterschiedliche Längen haben können (durch Randomness),
        # ist dies im einfachen Code trickreich.
        # TRICK: Wir verarbeiten hier nur die Latents weiter.

        return x, mask_gas, mask_env, ids_restore_gas, ids_restore_env

    def forward(self, x_gas, x_env):
        """
        Der komplette Trainingsschritt (Forward + Loss)
        Inputs: [Batch, Channels, SeqLen]
        """
        # --- 1. Encoding ---
        # Wir lassen das Masking und Encoding laufen
        # (Vereinfachung: Hier implementiere ich eine symmetrische Maskierung für den Einstieg)

        # Embedden
        emb_gas = self.patch_embed_gas(x_gas) + self.pos_embed_gas
        emb_env = self.patch_embed_env(x_env) + self.pos_embed_env

        # Masking (Hier: Synchrones Masking oder Random? Random ist besser!)
        # Wir maskieren ~60% der Gas-Daten, aber nur ~20% der Env-Daten (Asymmetrie!)
        # Für diesen Code-Snippet nutzen wir die interne Methode mit 50%
        x_gas_masked, mask_gas, ids_restore_gas = self.random_masking(emb_gas)
        x_env_masked, mask_env, ids_restore_env = self.random_masking(emb_env)

        # Fusion für Attention: Wir sagen dem Modell "Hier sind die Gas-Infos und hier die Env-Infos"
        # Wir addieren ein "Modality Token" (nicht im Code, aber implizit durch Position)
        x_combined = torch.cat([x_gas_masked, x_env_masked], dim=1)

        # Transformer
        for blk in self.blocks:
            x_combined = blk(x_combined)
        x_combined = self.norm(x_combined)

        # --- 2. Decoding (Rekonstruktion) ---
        # Wir splitten den Output wieder auf
        len_gas_keep = x_gas_masked.shape[1]
        out_gas_vis = x_combined[:, :len_gas_keep, :]
        out_env_vis = x_combined[:, len_gas_keep:, :]

        # HIER DER MAE TRICK:
        # Wir müssen die "leeren" (maskierten) Patches mit einem learnable [MASK] Token füllen
        # und dann alles in die richtige Reihenfolge bringen.
        # (Aus Platzgründen im Chat: Wir rekonstruieren direkt aus den Latents.
        #  State-of-the-Art MAE rekonstruiert ALLES, aber hier reicht Loss auf Visible für Code-Simplicity oft nicht.
        #  Wir vereinfachen: Wir nutzen den Encoder-Output, um die Original-Patches vorherzusagen.)

        # Einfacher Decoder (Projektion):
        # Note: pred_gas for visible patches not used - MAE reconstructs masked patches instead
        _ = self.decoder_pred_gas(out_gas_vis)  # Vorhersage für die SICHTBAREN (zum Lernen der Repräsentation)
        # STOP! Echtes MAE muss die *unsichtbaren* vorhersagen.
        # Dazu fügt man Mask-Tokens ein.

        # VOLLSTÄNDIGER DECODER-PART:
        B, L_gas, D = emb_gas.shape
        # Mask Tokens erstellen
        mask_token = torch.zeros(
            B, 1, D, device=x_gas.device
        )  # Learnable Parameter wäre besser

        # Gas Rekonstruktion
        # Wir bauen eine Sequenz: [Sichtbar, Mask_Token, Mask_Token...]
        # Dann sortieren wir sie zurück mit ids_restore
        mask_tokens_gas = mask_token.repeat(1, L_gas - len_gas_keep, 1)
        x_gas_full = torch.cat([out_gas_vis, mask_tokens_gas], dim=1)
        x_gas_full = torch.gather(
            x_gas_full, 1, ids_restore_gas.unsqueeze(-1).repeat(1, 1, D)
        )
        pred_gas_full = self.decoder_pred_gas(x_gas_full)

        # Env Rekonstruktion
        B, L_env, D = emb_env.shape
        mask_tokens_env = mask_token.repeat(1, L_env - x_env_masked.shape[1], 1)
        x_env_full = torch.cat([out_env_vis, mask_tokens_env], dim=1)
        x_env_full = torch.gather(
            x_env_full, 1, ids_restore_env.unsqueeze(-1).repeat(1, 1, D)
        )
        pred_env_full = self.decoder_pred_env(x_env_full)

        # --- 3. Loss Berechnung ---
        # Wir vergleichen Vorhersage mit Original (gepatcht)

        # Original patchen für Vergleich
        target_gas = self.patchify(x_gas, self.patch_size)
        target_env = self.patchify(x_env, self.patch_size)

        # MSE Loss, aber NUR auf den maskierten Bereichen!
        loss_gas = (pred_gas_full - target_gas) ** 2
        loss_gas = (loss_gas.mean(dim=-1) * mask_gas).sum() / mask_gas.sum()

        loss_env = (pred_env_full - target_env) ** 2
        loss_env = (loss_env.mean(dim=-1) * mask_env).sum() / mask_env.sum()

        total_loss = loss_gas + loss_env
        return total_loss, pred_gas_full, pred_env_full

    def patchify(self, imgs, p):
        """Hilfsfunktion: Macht aus Zeitreihe Patches"""
        # imgs: (N, C, L)
        # x: (N, L, patch_size * C) -> Das ist Standard Vision Transformer Logic
        # Wir brauchen: (N, Patches, patch_size * C)
        assert imgs.shape[2] % p == 0
        h = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], h, p)
        x = torch.einsum("nchp->nhpc", x)
        x = x.reshape(imgs.shape[0], h, -1)
        return x


# --- Test Script ---
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Batch Size 2, 10 Gas channels, 512 timesteps (~17 min)
    gas_data = torch.randn(2, 10, 512)
    env_data = torch.randn(2, 3, 512)

    model = FridgeMoCA(seq_len=512, patch_size=16)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", num_params)

    loss, _, _ = model(gas_data, env_data)
    logger.info("MoCA Loss: %.4f", loss.item())
    logger.info("Model test passed!")
