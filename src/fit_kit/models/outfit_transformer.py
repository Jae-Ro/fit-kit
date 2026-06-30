from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fit_kit.utils.log_utils import get_custom_logger

logger = get_custom_logger()


@dataclass
class OutfitTransformerConfig:
    """Configuration for OutfitTransformer."""

    # item embedding (FashionSigLIP image 768 + text 768 concatenated)
    d_item: int = 1536

    # transformer encoder
    n_heads: int = 16  # 1536 / 16 = 96 dim per head
    d_ffn: int = 2048
    n_layers: int = 6
    dropout: float = 0.3
    max_items: int = 16  # max items per outfit

    # CIR output embedding
    d_cir: int = 768  # matches FashionSigLIP dim for retrieval

    # modality weighting
    image_weight: float = 1.0  # scale image half vs text half (>1 = heavier image)


class OutfitTransformer(nn.Module):
    """Transformer for outfit compatibility prediction and complementary item retrieval.
    Adapted from bigohofone/outfit-transformer (CVPR 2023) with FashionSigLIP backbone.

    Architecture:
    * Per item:   FashionSigLIP image(768) + text(768) → 1536-dim item embedding
    * Per outfit: TransformerEncoder(d_model=1536, 6 layers, 16 heads)

    * CP mode:  [cp_task, item1, item2, ...] → transformer → cp_head → compatibility score
    * CIR mode: [cir_task, item1, item2, ...] → transformer → cir_head → query embedding

    Training:
    * CP:  Binary classification on compatible vs incompatible outfits (Polyvore)
    * CIR: Triplet loss — embed(partial_outfit) should be close to the missing item's
        embedding and far from negative items

    Key Points:
    * Operates on precomputed FashionSigLIP embeddings — no image/text encoder needed at inference time
    * Item embeddings are the concatenation of centered image and text embeddings: [image_emb(768) || text_emb(768)] = 1536-dim
    """

    def __init__(self, config: OutfitTransformerConfig | None = None):
        super().__init__()
        self.config = config or OutfitTransformerConfig()
        d = self.config.d_item
        d_half = d // 2

        # learned special tokens
        # task tokens prepended to the sequence to indicate CP vs CIR mode.
        # split into two halves to match the image||text concat structure.
        self.task_emb = nn.Parameter(torch.randn(d_half) * 0.02)
        self.cp_emb = nn.Parameter(torch.randn(d_half) * 0.02)
        self.cir_emb = nn.Parameter(torch.randn(d_half) * 0.02)
        self.pad_emb = nn.Parameter(torch.randn(d) * 0.02)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ffn,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation=F.mish,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            enable_nested_tensor=False,
        )

        # task sepecific heads
        self.cp_head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )
        self.cir_head = nn.Linear(d, self.config.d_cir, bias=False)

    def _pad_and_mask(
        self,
        item_embeddings: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal method to pad variable-length outfit sequences and create attention mask

        :param item_embeddings: list of (n_items, d_item) tensors, one per outfit
        :return: tuple of (padded, mask) where:
            padded = (batch, max_len, d_item)
            mask = (batch, max_len) bool — True for padding positions
        """
        batch_size = len(item_embeddings)
        max_len = min(
            max(e.shape[0] for e in item_embeddings),
            self.config.max_items,
        )
        d = self.config.d_item
        device = item_embeddings[0].device

        padded = (
            self.pad_emb.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                max_len,
                d,
            )
            .clone()
        )
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

        for i, emb in enumerate(item_embeddings):
            n = min(emb.shape[0], max_len)
            padded[i, :n] = emb[:n]
            mask[i, :n] = False

        return padded, mask

    def _forward(
        self,
        item_embeddings: list[torch.Tensor],
        task_token: torch.Tensor,
    ) -> torch.Tensor:
        """Internal method to run model with a task token prepended

        :param item_embeddings: list of (n_items, d_item) tensors
        :param task_token: (d_item,) task embedding
        :return: (batch, d_item) — transformer output at the task token position
        """
        padded, mask = self._pad_and_mask(item_embeddings)
        batch_size = padded.shape[0]

        # prepend task token
        task = task_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        x = torch.cat([task, padded], dim=1)
        mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )

        # normalize each modality half independently before transformer
        d_half = self.config.d_item // 2
        img_half = F.normalize(x[..., :d_half], p=2, dim=-1) * self.config.image_weight
        txt_half = F.normalize(x[..., d_half:], p=2, dim=-1)
        x = torch.cat([img_half, txt_half], dim=-1)

        out = self.transformer(x, src_key_padding_mask=mask)
        return out[:, 0, :]  # task token output

    def predict_compatibility(
        self,
        item_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Method to score CP (outfit compatibility)

        :param item_embeddings: list of (n_items, d_item) tensors, one per outfit
        :return: scores of shape (batch,) compatibility scores in [0, 1]
        """
        task_token = torch.cat([self.task_emb, self.cp_emb])
        out = self._forward(item_embeddings, task_token)
        return self.cp_head(out).squeeze(-1)

    def embed_outfit(
        self,
        item_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Method to produce CIR query embedding from a partial outfit.
        The resulting embedding can be compared against item embeddings
        (from embed_items) to find complementary items.

        :param item_embeddings: list of (n_items, d_item) tensors (partial outfits)
        :return: query embeddings of shape (batch, d_cir) L2-normalized
        """
        task_token = torch.cat([self.task_emb, self.cir_emb])
        out = self._forward(item_embeddings, task_token)
        return F.normalize(self.cir_head(out), p=2, dim=-1)

    def embed_items(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Method to produce CIR item embeddings for candidate items

        :param item_embeddings: (n_items, d_item) — precomputed concat embeddings
        :return: embeddings of shape (n_items, d_cir) L2-normalized
        """
        # wrap each item as a single-item "outfit"
        items_as_outfits = [emb.unsqueeze(0) for emb in item_embeddings]
        return self.embed_outfit(items_as_outfits)

    def forward(
        self,
        item_embeddings: list[torch.Tensor],
        mode: str = "cp",
    ) -> torch.Tensor:
        """Top-level method to run forward pass for either CP or CIR tasks.

        :param item_embeddings:  list of (n_items, d_item) tensors
        :param mode: "cp" for compatibility, "cir" for outfit embedding, defaults to "cp"
        :raises ValueError: mode is not one of "cp" or "cir"
        :return: for cp mode: (batch,) compatibility scores.
            For cir mode: (batch, d_cir) query embeddings
        """
        if mode == "cp":
            return self.predict_compatibility(item_embeddings)
        elif mode == "cir":
            return self.embed_outfit(item_embeddings)
        else:
            raise ValueError(f"unknown mode: {mode}")


def build_item_embedding(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
) -> torch.Tensor:
    """Helper function to concatenate image and text embeddings into an item embedding.
    Uses RAW (uncentered) embeddings — the OutfitTransformer learns its own
    representation space. Centering is only for the retrieval module.

    :param image_emb: (batch, 768) FashionSigLIP image embedding
    :param text_emb: (batch, 768) FashionSigLIP text embedding
    :return: (batch, 1536) concatenated embedding
    """
    return torch.cat([image_emb, text_emb], dim=-1)
