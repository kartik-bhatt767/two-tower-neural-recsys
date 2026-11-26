# src/train.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.data_loader import (
    GENRES,
    build_catalog_tag_matrix,
    build_tag_vocabulary,
    encode_tags_multihot,
    generate_user_data,
    load_catalog,
)
from src.model import TwoTowerRecSys


def train_two_tower_model(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.004,
    num_users: int = 150,
    embed_dim: int = 32,
    temperature: float = 0.07,
):
    torch.manual_seed(42)
    np.random.seed(42)

    catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
    catalog = load_catalog(catalog_path)
    tag_vocab = build_tag_vocabulary(catalog)
    item_tag_matrix = build_catalog_tag_matrix(catalog, tag_vocab)

    users, train_interactions = generate_user_data(
        num_users=num_users,
        catalog_path=catalog_path,
        holdout_ratio=0.25,
        seed=42,
    )

    item_genres = [GENRES.index(item["genre"]) if item["genre"] in GENRES else 0 for item in catalog]
    user_tag_matrix = np.vstack(
        [encode_tags_multihot(u.get("preferred_tags", []), tag_vocab) for u in users]
    )

    num_items = len(catalog)
    num_genres = len(GENRES)
    num_tags = len(tag_vocab)

    print("=" * 64)
    print("TRAINING PYTORCH TWO-TOWER NEURAL RECOMMENDATION SYSTEM")
    print("=" * 64)
    print(
        f"Users: {num_users} | Catalog Items: {num_items} | "
        f"Tag Vocab: {num_tags} | Train Interactions: {len(train_interactions)}"
    )

    model = TwoTowerRecSys(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        num_tags=num_tags,
        embed_dim=embed_dim,
        temperature=temperature,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    u_ids = torch.tensor([u_idx for u_idx, _ in train_interactions], dtype=torch.long)
    u_genres = torch.tensor(
        [users[u_idx]["preferred_genre_idx"] for u_idx, _ in train_interactions], dtype=torch.long
    )
    u_tags = torch.tensor(
        np.vstack([user_tag_matrix[u_idx] for u_idx, _ in train_interactions]), dtype=torch.float32
    )

    i_ids = torch.tensor([i_idx for _, i_idx in train_interactions], dtype=torch.long)
    i_genres = torch.tensor([item_genres[i_idx] for _, i_idx in train_interactions], dtype=torch.long)
    i_tags = torch.tensor(
        np.vstack([item_tag_matrix[i_idx] for _, i_idx in train_interactions]), dtype=torch.float32
    )

    dataset_size = len(train_interactions)
    indices = np.arange(dataset_size)

    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []

        for start_idx in range(0, dataset_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            if len(batch_idx) < 4:
                continue

            optimizer.zero_grad()
            sim_matrix = model(
                u_ids[batch_idx],
                u_genres[batch_idx],
                i_ids[batch_idx],
                i_genres[batch_idx],
                u_tags[batch_idx],
                i_tags[batch_idx],
            )
            loss = model.compute_loss(sim_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] — Contrastive Loss: {avg_loss:.4f}")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "two_tower_weights.pt")
    checkpoint = {
        "state_dict": model.state_dict(),
        "num_users": num_users,
        "num_items": num_items,
        "num_genres": num_genres,
        "num_tags": num_tags,
        "embed_dim": embed_dim,
        "temperature": temperature,
        "tag_vocab": tag_vocab,
        "genres": GENRES,
    }
    torch.save(checkpoint, weights_path)
    print(f"Saved model checkpoint to {weights_path}")

    _plot_loss_curve(loss_history)
    print("=" * 64)
    return model, loss_history


def _plot_loss_curve(loss_history):
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, "training_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#05070e")
    ax.set_facecolor("#0f172a")

    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="#f43f5e",
        linewidth=2.5,
        marker="o",
        markersize=3.5,
        label="Symmetric InfoNCE Loss (tau=0.07)",
    )
    ax.set_title(
        "Two-Tower Neural Network — Training Loss Curve",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#cbd5e1")
    ax.grid(color="#334155", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#1e293b", edgecolor="none", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Training loss plot saved to {out_path}")


if __name__ == "__main__":
    train_two_tower_model()
