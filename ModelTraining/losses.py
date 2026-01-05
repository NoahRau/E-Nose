import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko Differential Entropy Regularizer.
    Sorgt dafür, dass die Features den Raum gleichmäßig nutzen (kein "Clumping").
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_output, eps=1e-8):
        # student_output: [Batch, Dim] - muss L2-normalisiert sein!
        # Wir berechnen die Distanz zu allen anderen im Batch
        pdist = torch.pdist(student_output, p=2).pow(2)

        # Wir wollen den nächsten Nachbarn für jedes Sample finden (außer sich selbst)
        # pdist gibt kondensierte Distanzmatrix zurück, wir brauchen die volle
        # (Vereinfachte Implementierung für kleine Batches)
        n = student_output.shape[0]
        dists = torch.cdist(student_output, student_output)

        # Diagonale auf unendlich setzen (damit man sich nicht selbst als Nachbar findet)
        dists.fill_diagonal_(float("inf"))

        # Kleinste Distanz für jedes Sample finden
        min_dist, _ = torch.min(dists, dim=1)

        # KoLeo Loss: Wir wollen diese Distanz maximieren (Log-Summe minimieren)
        loss = -torch.log(min_dist + eps).mean()
        return loss


class DINOLoss(nn.Module):
    """
    Kombiniert DINO Loss (auf [CLS] Token) und iBOT Loss (auf Masked Patches).
    Beinhaltet Sinkhorn-Knopp Zentrierung für den Teacher.
    """

    def __init__(
        self,
        out_dim,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=5,
        nepochs=100,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Teacher Temperatur Scheduler (Startet kalt, wird wärmer)
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    @torch.no_grad()
    def sinkhorn_knopp_normalization(self, logits, temperature=0.1, n_iters=3):
        """
        Der Trick von SwAV/DINO: Sorgt für gleichmäßige Verteilung der Klassen (Cluster).
        """
        Q = torch.exp(logits / temperature).t()  # [Dim, Batch]
        B = Q.shape[1]  # Batch size
        K = Q.shape[0]  # Output dim

        # Normalisierung
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(n_iters):
            # Zeilen normalisieren (auf 1/K)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # Spalten normalisieren (auf 1/B)
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B  # Skalieren, damit Summe = B ist
        return Q.t()  # [Batch, Dim]

    def forward(self, student_output, teacher_output, epoch, is_ibot=False):
        """
        student_output: Logits vom Student [Batch, Dim]
        teacher_output: Logits vom Teacher [Batch, Dim]
        is_ibot: Wenn True, wenden wir das auf Patches an (iBOT), sonst auf CLS (DINO)
        """
        # 1. Teacher vorbereiten (Zentrieren & Sharpening)
        # Sinkhorn oder einfaches Softmax mit Temp? DINOv2 nutzt Softmax + Centering.
        # Sinkhorn ist für SwAV, DINO nutzt Centering. Wir nutzen hier die DINO-Variante mit Centering.

        # Teacher output zentrieren (Moving Average)
        teacher_out = teacher_output - self.center

        # Temperatur holen
        temp = self.teacher_temp_schedule[epoch]

        # Target Distribution (vom Teacher)
        teacher_probs = F.softmax(teacher_out / temp, dim=-1)

        # 2. Student Logits (Log-Softmax)
        student_log_probs = F.log_softmax(student_output / self.student_temp, dim=-1)

        # 3. Cross-Entropy Loss: -Sum(Teacher * log(Student))
        loss = torch.sum(-teacher_probs * student_log_probs, dim=-1).mean()

        # 4. Center Update (nur im DINO Pass, nicht bei jedem Patch-Pass um Rauschen zu meiden)
        if not is_ibot:
            self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # EMA Update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
