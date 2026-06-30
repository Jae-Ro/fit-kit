"""
Train OutfitTransformer on Polyvore with FashionSigLIP

Two-stage training:
    1. CP  (Compatibility Prediction): category-preserving hard negatives
    2. CIR (Complementary Item Retrieval): InfoNCE contrastive loss

Data:
    - owj0421/polyvore (HuggingFace): item metadata + images + categories
    - owj0421/polyvore-outfits (HuggingFace): outfit compositions + tasks
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fit_kit.models.outfit_transformer import OutfitTransformer, OutfitTransformerConfig
from fit_kit.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy(y_prob, y_true, reduction="none")
        p_t = y_prob * y_true + (1 - y_prob) * (1 - y_true)
        loss = ce * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        return (alpha_t * loss).mean()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for CIR.

    More stable than triplet loss — uses all in-batch negatives with
    temperature-scaled cosine similarity. No margin collapse issues.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb: torch.Tensor, answer_emb: torch.Tensor) -> torch.Tensor:
        # both should be L2-normalized already
        # cosine similarity matrix: (batch, batch)
        logits = query_emb @ answer_emb.T / self.temperature
        # positive pairs are on the diagonal
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)


# Step 1: Precompute FashionSigLIP embeddings


def encode_polyvore(device: str, output_dir: str, batch_size: int = 64):
    """Encode all Polyvore items with FashionSigLIP (image + text embeddings)."""
    from io import BytesIO

    import open_clip
    from datasets import load_dataset
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb_path = out / "polyvore_embeddings.pt"
    if emb_path.exists():
        logger.info(f"embeddings already exist at {emb_path}, skipping")
        return

    # load Polyvore items
    logger.info("loading Polyvore items from HuggingFace...")
    items = load_dataset("owj0421/polyvore", split="data")
    logger.info(f"{len(items)} items loaded")

    # load FashionSigLIP
    logger.info(f"loading FashionSigLIP on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()

    # encode all items
    all_image_embs = []
    all_text_embs = []
    all_item_ids = []
    failed = 0

    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        batch = items[start:end]
        batch_size_actual = end - start

        # encode text (url_name as description)
        texts = [name.replace("-", " ") for name in batch["url_name"]]
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # encode images
        images = []
        valid_mask = []
        for img in batch["image"]:
            try:
                if img is None:
                    valid_mask.append(False)
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.open(BytesIO(img)).convert("RGB")
                else:
                    img = img.convert("RGB")
                images.append(preprocess(img))
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
                failed += 1

        if images:
            img_tensor = torch.stack(images).to(device)
            with torch.no_grad():
                img_feats = model.encode_image(img_tensor)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # rebuild full batch with zeros for failed images
            img_idx = 0
            batch_img_embs = []
            for valid in valid_mask:
                if valid:
                    batch_img_embs.append(img_feats[img_idx].cpu())
                    img_idx += 1
                else:
                    batch_img_embs.append(torch.zeros(text_feats.shape[1]))
            batch_img_embs = torch.stack(batch_img_embs)
        else:
            batch_img_embs = torch.zeros(batch_size_actual, text_feats.shape[1])

        all_image_embs.append(batch_img_embs)
        all_text_embs.append(text_feats.cpu())
        all_item_ids.extend(batch["item_id"])

        if end % (batch_size * 20) == 0 or end == len(items):
            logger.info(f"encoded {end}/{len(items)} items (failed: {failed})")

    all_image_embs = torch.cat(all_image_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)

    # save
    torch.save(
        {
            "image_embeddings": all_image_embs,
            "text_embeddings": all_text_embs,
            "item_ids": all_item_ids,
        },
        emb_path,
    )

    logger.info(
        f"saved embeddings: {emb_path} (image={tuple(all_image_embs.shape)}, "
        f"text={tuple(all_text_embs.shape)}, "
        f"{len(all_item_ids)} items, "
        f"{failed} failed)"
    )


# Datasets


class EmbeddingStore:
    """Precomputed FashionSigLIP embeddings for Polyvore items."""

    def __init__(self, embeddings_path: str):
        data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.image_embs = data["image_embeddings"]  # (N, 768)
        self.text_embs = data["text_embeddings"]  # (N, 768)
        item_ids = data["item_ids"]
        self.id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        logger.info(f"loaded {len(item_ids)} item embeddings from {embeddings_path}")

    def get_item_embedding(self, item_id: str) -> torch.Tensor:
        """Return concat(image, text) = 1536-dim embedding for an item."""
        idx = self.id_to_idx[item_id]
        return torch.cat([self.image_embs[idx], self.text_embs[idx]], dim=-1)

    def get_all_cir_embeddings(self) -> torch.Tensor:
        """Return text embeddings for all items (for CIR retrieval evaluation)."""
        return self.text_embs


class PolyvoreCompatibilityDataset(Dataset):
    """Polyvore outfit compatibility dataset with category-preserving hard negatives.

    Negatives are generated by replacing items with other items from the SAME
    Polyvore category but different outfits. This forces the model to learn
    fine-grained style compatibility rather than just category co-occurrence.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        split: str = "train",
        dataset_type: str = "nondisjoint",
        category_index: dict | None = None,
    ):
        from collections import defaultdict

        from datasets import load_dataset

        # load outfit compositions
        task_data = load_dataset(
            "owj0421/polyvore-outfits",
            f"{dataset_type}_compatibility",
            split=split,
        )

        # load item ID mappings from ALL splits (compatibility shuffles items across splits)
        self.id_converter = {}
        for s in ["train", "validation", "test"]:
            set_data = load_dataset(
                "owj0421/polyvore-outfits",
                f"{dataset_type}_default",
                split=s,
            )
            for outfit in set_data:
                for item in outfit["items"]:
                    self.id_converter[f"{outfit['set_id']}_{item['index']}"] = item["item_id"]

        # load or reuse Polyvore item categories
        if category_index is not None:
            self.item_to_category = category_index["item_to_category"]
            self.category_to_items = category_index["category_to_items"]
        else:
            logger.info("loading Polyvore item categories...")
            metadata = load_dataset("owj0421/polyvore", split="data")
            self.item_to_category = {}
            self.category_to_items = defaultdict(list)
            for row in metadata:
                iid = row["item_id"]
                cat = row.get("category", "unknown")
                if iid in store.id_to_idx:
                    self.item_to_category[iid] = cat
                    self.category_to_items[cat].append(iid)

        logger.info(f"{len(self.category_to_items)} categories, {len(self.item_to_category)} items with categories")  # fmt: skip

        # Only load positive (compatible) outfits — we generate negatives on the fly.
        self.positives = []
        skipped = 0
        label_counts = {}
        for row in task_data:
            label = row["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            if not label:  # label=0 or empty → negative, skip
                continue
            item_ids = []
            for sid in row["items"]:
                if not sid:  # empty string
                    break
                iid = self.id_converter.get(sid)
                if iid is None or iid not in store.id_to_idx:
                    break
                item_ids.append(iid)
            else:
                if item_ids:
                    self.positives.append(item_ids)
                    continue
            skipped += 1

        logger.info(f"  label distribution: {label_counts}")

        # track which items belong to which outfit (for excluding same-outfit swaps)
        self.item_to_outfits = defaultdict(set)
        for oidx, outfit in enumerate(self.positives):
            for iid in outfit:
                self.item_to_outfits[iid].add(oidx)

        self.store = store
        logger.info(
            f"CP {split}: {len(self.positives)} positive outfits ({skipped} skipped), "
            f"{len(self.category_to_items)} categories for hard negatives"
        )

    def __len__(self):
        return len(self.positives) * 2  # 1 positive + 1 negative per outfit

    def _sample_same_category(self, item_id: str, outfit_idx: int) -> str:
        """Sample a replacement item from the same category but different outfit."""
        cat = self.item_to_category.get(item_id, "unknown")
        candidates = self.category_to_items.get(cat, [])

        # try to find an item not in the same outfit (up to 10 attempts)
        outfit_items = set(self.positives[outfit_idx])
        for _ in range(10):
            replacement = random.choice(candidates) if candidates else item_id
            if replacement not in outfit_items:
                return replacement

        # fallback: any item from same category
        return random.choice(candidates) if candidates else item_id

    def __getitem__(self, idx):
        is_positive = idx % 2 == 0
        outfit_idx = idx // 2

        item_ids = self.positives[outfit_idx]

        if is_positive:
            embeddings = torch.stack([self.store.get_item_embedding(iid) for iid in item_ids])
            return embeddings, 1.0
        else:
            # hard negative: replace 1-2 items with SAME CATEGORY items from other outfits
            neg_ids = list(item_ids)
            n_replace = random.randint(1, max(1, len(neg_ids) // 2))
            replace_indices = random.sample(range(len(neg_ids)), n_replace)
            for i in replace_indices:
                neg_ids[i] = self._sample_same_category(neg_ids[i], outfit_idx)
            embeddings = torch.stack([self.store.get_item_embedding(iid) for iid in neg_ids])
            return embeddings, 0.0


class PolyvoreTripletDataset(Dataset):
    """Polyvore triplet dataset for CIR training.

    For each outfit, randomly remove one item as the answer.
    The partial outfit is the query, the removed item is the positive.
    """

    def __init__(
        self, store: EmbeddingStore, split: str = "train", dataset_type: str = "nondisjoint"
    ):
        from datasets import load_dataset

        set_data = load_dataset(
            "owj0421/polyvore-outfits",
            f"{dataset_type}_default",
            split=split,
        )

        self.samples = []
        skipped = 0
        for outfit in set_data:
            item_ids = [item["item_id"] for item in outfit["items"]]
            if len(item_ids) < 2:
                skipped += 1
                continue
            if any(iid not in store.id_to_idx for iid in item_ids):
                skipped += 1
                continue
            self.samples.append(item_ids)

        self.store = store
        logger.info(f"CIR {split}: {len(self.samples)} outfits ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_ids = self.samples[idx]
        # randomly pick one item as the answer
        answer_idx = random.randint(0, len(item_ids) - 1)
        answer_id = item_ids[answer_idx]
        outfit_ids = [iid for i, iid in enumerate(item_ids) if i != answer_idx]

        outfit_emb = torch.stack([self.store.get_item_embedding(iid) for iid in outfit_ids])
        answer_emb = self.store.get_item_embedding(answer_id)
        return outfit_emb, answer_emb


def _load_category_index(store: EmbeddingStore) -> dict:
    """Load Polyvore item categories once for reuse across datasets."""
    from collections import defaultdict

    from datasets import load_dataset

    logger.info("loading Polyvore item categories...")
    metadata = load_dataset("owj0421/polyvore", split="data")
    item_to_category = {}
    category_to_items = defaultdict(list)
    for row in metadata:
        iid = row["item_id"]
        cat = row.get("category", "unknown")
        if iid in store.id_to_idx:
            item_to_category[iid] = cat
            category_to_items[cat].append(iid)

    logger.info(f"{len(category_to_items)} categories, {len(item_to_category)} items with categories")  # fmt: skip
    return {"item_to_category": item_to_category, "category_to_items": dict(category_to_items)}


# Collate functions


def cp_collate_fn(batch):
    """Collate variable-length outfits for CP."""
    embeddings = [item[0] for item in batch]  # list of (n_items, 1536)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return embeddings, labels


def cir_collate_fn(batch):
    """Collate variable-length outfits + answers for CIR."""
    outfits = [item[0] for item in batch]  # list of (n_items, 1536)
    answers = torch.stack([item[1] for item in batch])  # (batch, 1536)
    return outfits, answers


# Training loops


def train_cp(
    model: OutfitTransformer,
    store: EmbeddingStore,
    device: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 2e-5,
    checkpoint_dir: str = "checkpoints",
    dataset_type: str = "nondisjoint",
    patience: int = 10,
):
    """Train CP (Compatibility Prediction) task with category-preserving hard negatives."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    category_index = _load_category_index(store)
    train_ds = PolyvoreCompatibilityDataset(store, "train", dataset_type, category_index)
    valid_ds = PolyvoreCompatibilityDataset(store, "validation", dataset_type, category_index)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=cp_collate_fn,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=cp_collate_fn,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for embeddings, labels in train_dl:
            embeddings = [e.to(device) for e in embeddings]
            labels = labels.to(device)

            scores = model.predict_compatibility(embeddings)
            loss = loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # validate
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, labels in valid_dl:
                embeddings = [e.to(device) for e in embeddings]
                labels = labels.to(device)

                scores = model.predict_compatibility(embeddings)
                loss = loss_fn(scores, labels)

                val_loss += loss.item()
                all_preds.append(scores.cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(valid_dl)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # metrics
        auc = _compute_auc(all_preds, all_labels)
        acc = ((all_preds > 0.5) == all_labels.bool()).float().mean().item()

        logger.info(
            f"CP epoch {epoch + 1}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_auc={auc:.4f}  "
            f"val_acc={acc:.4f}"
        )

        # save best (by val_loss — better generalization for cross-domain transfer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            path = ckpt_dir / "cp_best.pt"
            torch.save(
                {
                    "config": model.config.__dict__,
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "auc": auc,
                    "val_loss": val_loss,
                },
                path,
            )
            logger.info(f"saved best CP checkpoint (val_loss={val_loss:.4f}, auc={auc:.4f}) → {path}")  # fmt: skip
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")  # fmt: skip
                break

    return best_val_loss


def train_cir(
    model: OutfitTransformer,
    store: EmbeddingStore,
    device: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 2e-5,
    temperature: float = 0.07,
    checkpoint_dir: str = "checkpoints",
    dataset_type: str = "nondisjoint",
    patience: int = 10,
):
    """Train CIR (Complementary Item Retrieval) task with InfoNCE loss."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    train_ds = PolyvoreTripletDataset(store, "train", dataset_type)
    valid_ds = PolyvoreTripletDataset(store, "validation", dataset_type)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=cir_collate_fn,
        pin_memory=True,
        drop_last=True,  # InfoNCE needs consistent batch sizes
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=cir_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    loss_fn = InfoNCELoss(temperature=temperature)

    best_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for outfits, answers in train_dl:
            outfits = [o.to(device) for o in outfits]
            answers = answers.to(device)

            # embed partial outfits → query embeddings (L2-normalized)
            query_embs = model.embed_outfit(outfits)
            # embed answer items → target embeddings (L2-normalized)
            answer_embs = model.embed_items(answers)

            loss = loss_fn(query_embs, answer_embs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for outfits, answers in valid_dl:
                outfits = [o.to(device) for o in outfits]
                answers = answers.to(device)

                query_embs = model.embed_outfit(outfits)
                answer_embs = model.embed_items(answers)
                loss = loss_fn(query_embs, answer_embs)
                val_loss += loss.item()

        val_loss /= len(valid_dl)

        logger.info(f"CIR epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")  # fmt: skip

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            path = ckpt_dir / "cir_best.pt"
            torch.save(
                {
                    "config": model.config.__dict__,
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                path,
            )
            logger.info(f"saved best CIR checkpoint (loss={val_loss:.4f}) → {path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")  # fmt: skip
                break

    return best_loss


# Metrics


def _compute_auc(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUC-ROC."""
    try:
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(labels.numpy(), preds.numpy())
    except Exception:
        # fallback: approximate AUC
        pos = preds[labels == 1]
        neg = preds[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        return (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()


# CLI


def main():
    ap = argparse.ArgumentParser(description="Train OutfitTransformer on Polyvore")
    sub = ap.add_subparsers(dest="command")

    # encode
    enc = sub.add_parser("encode", help="Precompute FashionSigLIP embeddings")
    enc.add_argument("--device", default="cuda:0")
    enc.add_argument("--output-dir", default="data/polyvore")
    enc.add_argument("--batch-size", type=int, default=64)

    # train-cp
    cp = sub.add_parser("train-cp", help="Train compatibility prediction")
    cp.add_argument("--device", default="cuda:0")
    cp.add_argument("--embeddings", default="data/polyvore/polyvore_embeddings.pt")
    cp.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    cp.add_argument("--checkpoint-dir", default="checkpoints")
    cp.add_argument("--epochs", type=int, default=50)
    cp.add_argument("--batch-size", type=int, default=256)
    cp.add_argument("--lr", type=float, default=2e-5)
    cp.add_argument("--dataset-type", default="nondisjoint")

    # train-cir
    cir = sub.add_parser("train-cir", help="Train complementary item retrieval")
    cir.add_argument("--device", default="cuda:0")
    cir.add_argument("--embeddings", default="data/polyvore/polyvore_embeddings.pt")
    cir.add_argument("--checkpoint", default=None, help="Resume from CP checkpoint")
    cir.add_argument("--checkpoint-dir", default="checkpoints")
    cir.add_argument("--epochs", type=int, default=50)
    cir.add_argument("--batch-size", type=int, default=256)
    cir.add_argument("--lr", type=float, default=2e-5)
    cir.add_argument("--temperature", type=float, default=0.07)
    cir.add_argument("--patience", type=int, default=10)
    cir.add_argument("--dataset-type", default="nondisjoint")

    # all
    allcmd = sub.add_parser("all", help="Run full pipeline: encode → train-cp → train-cir")
    allcmd.add_argument("--device", default="cuda:0")
    allcmd.add_argument("--output-dir", default="data/polyvore")
    allcmd.add_argument("--checkpoint-dir", default="checkpoints")
    allcmd.add_argument("--epochs", type=int, default=50)
    allcmd.add_argument("--dataset-type", default="nondisjoint")

    args = ap.parse_args()

    if args.command == "encode":
        encode_polyvore(args.device, args.output_dir, args.batch_size)

    elif args.command == "train-cp":
        store = EmbeddingStore(args.embeddings)
        config = OutfitTransformerConfig()
        model = OutfitTransformer(config)

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            logger.info(f"loaded checkpoint: {args.checkpoint}")

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"model: {n_params:,} params")

        train_cp(
            model,
            store,
            args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            dataset_type=args.dataset_type,
        )

    elif args.command == "train-cir":
        store = EmbeddingStore(args.embeddings)
        config = OutfitTransformerConfig()
        model = OutfitTransformer(config)

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            logger.info(f"loaded CP checkpoint: {args.checkpoint}")

        train_cir(
            model,
            store,
            args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            checkpoint_dir=args.checkpoint_dir,
            dataset_type=args.dataset_type,
            patience=args.patience,
        )

    elif args.command == "all":
        emb_path = f"{args.output_dir}/polyvore_embeddings.pt"

        # step 1: encode
        encode_polyvore(args.device, args.output_dir)

        # step 2: train CP
        store = EmbeddingStore(emb_path)
        config = OutfitTransformerConfig()
        model = OutfitTransformer(config)
        logger.info(f"model: {sum(p.numel() for p in model.parameters()):,} params")

        train_cp(
            model,
            store,
            args.device,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            dataset_type=args.dataset_type,
        )

        # step 3: train CIR from best CP checkpoint
        cp_ckpt = torch.load(
            f"{args.checkpoint_dir}/cp_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(cp_ckpt["model"])
        logger.info("loaded best CP checkpoint for CIR training")

        train_cir(
            model,
            store,
            args.device,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            dataset_type=args.dataset_type,
        )

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
