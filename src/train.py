# src/train.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model import TwoTowerRecSys
from src.data_loader import generate_user_data, GENRES

def train_two_tower_model(epochs: int = 15, batch_size: int = 16, lr: float = 0.003):
    catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    users, interactions = generate_user_data(num_users=50, catalog_path=catalog_path)
    item_genres = [GENRES.index(item["genre"]) if item["genre"] in GENRES else 0 for item in catalog]

    num_users = len(users)
    num_items = len(catalog)
    num_genres = len(GENRES)

    print("=" * 60)
    print("🔥 TRAINING PYTORCH TWO-TOWER NEURAL RECOMMENDATION SYSTEM")
    print("=" * 60)
    print(f"Users: {num_users} | Items: {num_items} | Total Interactions: {len(interactions)}")

    model = TwoTowerRecSys(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        embed_dim=32,
        temperature=0.07
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_history = []

    # Prepare training tensors
    u_ids = []
    u_genres = []
    i_ids = []
    i_genres = []

    for u_idx, item_idx in interactions:
        u_ids.append(u_idx)
        u_genres.append(users[u_idx]["preferred_genre_idx"])
        i_ids.append(item_idx)
        i_genres.append(item_genres[item_idx])

    u_ids = torch.tensor(u_ids, dtype=torch.long)
    u_genres = torch.tensor(u_genres, dtype=torch.long)
    i_ids = torch.tensor(i_ids, dtype=torch.long)
    i_genres = torch.tensor(i_genres, dtype=torch.long)

    dataset_size = len(interactions)
    indices = np.arange(dataset_size)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []

        for start_idx in range(0, dataset_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            if len(batch_idx) < 4:
                continue

            b_u = u_ids[batch_idx]
            b_ug = u_genres[batch_idx]
            b_i = i_ids[batch_idx]
            b_ig = i_genres[batch_idx]

            optimizer.zero_grad()
            sim_matrix = model(b_u, b_ug, b_i, b_ig)
            loss = model.compute_loss(sim_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] — In-Batch Contrastive Loss: {avg_loss:.4f}")

    # Save Model Weights
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "two_tower_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Model weights saved to {weights_path}")

    # Plot Loss Curve
    _plot_loss_curve(loss_history)
    print("=" * 60)
    return model, loss_history

def _plot_loss_curve(loss_history):
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, "training_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#05070e")
    ax.set_facecolor("#0f172a")

    ax.plot(range(1, len(loss_history) + 1), loss_history, color="#f43f5e", linewidth=2.5, marker="o", markersize=4, label="In-Batch Contrastive Loss (τ=0.07)")
    ax.set_title("Two-Tower Neural Network — Training Loss Curve", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#cbd5e1")
    ax.grid(color="#334155", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#1e293b", edgecolor="none", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"✅ Training loss plot saved to {out_path}")

if __name__ == "__main__":
    train_two_tower_model(epochs=15, batch_size=16, lr=0.003)
