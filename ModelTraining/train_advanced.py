import copy

import torch
from torch.utils.data import DataLoader

from ModelTraining.dataset import FridgeDataset
from ModelTraining.losses import DINOLoss, KoLeoLoss
from ModelTraining.model_advanced import FridgeMoCA_Pro

# Config
EPOCHS = 50
LR = 0.0005
MOMENTUM_TEACHER = 0.996  # Teacher lernt langsam vom Student
LAMBDA_DINO = 1.0
LAMBDA_IBOT = 1.0
LAMBDA_KOLEO = 0.1


def update_teacher(student, teacher, momentum):
    """EMA Update: Teacher = momentum * Teacher + (1-momentum) * Student"""
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Daten
    dataset = FridgeDataset("data/deine_daten.csv", seq_len=512, mode="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # 2. Modelle (Student & Teacher)
    student = FridgeMoCA_Pro().to(device)
    teacher = FridgeMoCA_Pro().to(device)

    # Teacher braucht keine Gradients (lernt nur via EMA)
    teacher.load_state_dict(student.state_dict())  # Starten identisch
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=4096, nepochs=EPOCHS).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

    # 4. Loop
    print("Starte DINOv2 + iBOT Training...")

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0

        for gas, env in dataloader:
            gas, env = gas.to(device), env.to(device)

            # --- MASKEN ERSTELLEN ---
            # Wir brauchen Masken für iBOT (Student sieht maskiert, Teacher sieht alles)
            # Einfache Random Maske für Patches (hier vereinfacht)
            B, C, L = gas.shape
            # ... (Code für Maskengenerierung hier einfügen, ca 50% patches) ...

            # --- FORWARD PASS ---

            # 1. Teacher (Global View, Unmasked)
            with torch.no_grad():
                t_dino, t_ibot, _ = teacher(gas, env)

            # 2. Student (Local/Masked View)
            # Hier würden wir dem Student eigentlich maskierte Inputs geben
            # student_gas = gas * mask ...
            s_dino, s_ibot, s_cls_feat = student(gas, env)

            # --- LOSS BERECHNUNG ---

            # A) DINO Loss (CLS Token Vergleich)
            # Vergleiche Student CLS mit Teacher CLS
            l_dino = dino_loss_fn(s_dino, t_dino, epoch, is_ibot=False)

            # B) iBOT Loss (Patch Token Vergleich - Online Tokenizer)
            # Wir vergleichen nur die maskierten Patches!
            # (Hier vereinfacht als Mean über alle, in echt nur 'masked_indices')
            l_ibot = dino_loss_fn(
                s_ibot.reshape(-1, 4096), t_ibot.reshape(-1, 4096), epoch, is_ibot=True
            )

            # C) KoLeo Loss (Feature Spreading)
            l_koleo = koleo_loss_fn(s_cls_feat)

            # Summe
            loss = (
                (LAMBDA_DINO * l_dino)
                + (LAMBDA_IBOT * l_ibot)
                + (LAMBDA_KOLEO * l_koleo)
            )

            # --- BACKPROP ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- TEACHER UPDATE ---
            update_teacher(student, teacher, MOMENTUM_TEACHER)

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss/len(dataloader):.4f}")

        # DINO Loss Schedule Update (für Temperature)
        # (Passiert automatisch via epoch index im forward)


if __name__ == "__main__":
    main()
