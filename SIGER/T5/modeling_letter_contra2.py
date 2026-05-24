import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5LayerNorm


class LETTER(T5ForConditionalGeneration):

    def __init__(
            self,
            config: T5Config,
            model_temperature=1.0,
            use_cls_contra=False,
            cls_contra_weight=0.1,
            cls_contra_type="infonce",
            cls_temperature=0.1,
            item_id_base=0,
            emb_file="",
            cls_num_negatives=256,
            *args,
            **kwargs,
    ):
        super().__init__(config)

        self.model_temperature = model_temperature
        self.use_cls_contra = use_cls_contra
        self.cls_contra_weight = cls_contra_weight
        self.cls_contra_type = cls_contra_type
        self.cls_temperature = cls_temperature
        self.cls_item_id_base = item_id_base
        self.emb_file = emb_file
        self.cls_num_negatives = cls_num_negatives

        self.item_num = 0
        self.cls_token_embedding = None
        self.cls_mlp_projection = None
        self.register_buffer("item_embeddings", None)
        self.latest_loss_breakdown = {}

        if self.use_cls_contra:
            self._ensure_cls_token()
            self._load_item_embeddings(emb_file)


    def _ensure_cls_token(self):
        if self.cls_token_embedding is None:
            self.cls_token_embedding = nn.Parameter(
                torch.randn(1, self.config.d_model) * 0.02
            )

    def _load_item_embeddings(self, emb_file):
        if not emb_file:
            raise ValueError("`emb_file` must be provided when `use_cls_contra` is enabled.")

        item_emb = np.load(emb_file)
        if item_emb.ndim != 2:
            raise ValueError(f"`emb_file` must contain a 2D array, but got shape {item_emb.shape}.")

        item_emb_tensor = torch.tensor(item_emb, dtype=torch.float32)
        self.item_num = item_emb_tensor.size(0)
        self.emb_file = emb_file
        self._buffers["item_embeddings"] = item_emb_tensor

        item_emb_dim = item_emb_tensor.size(1)
        if self.cls_mlp_projection is None or self.cls_mlp_projection[0].in_features != self.config.d_model:
            self.cls_mlp_projection = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.GELU(),
                nn.Linear(self.config.d_model, item_emb_dim),
            )

    def _prepare_decoder_inputs(
            self,
            decoder_input_ids,
            decoder_attention_mask=None,
            append_cls_token=False,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).long()
        else:
            decoder_attention_mask = decoder_attention_mask.long()

        decoder_attention_mask = decoder_attention_mask.clone()
        decoder_attention_mask[:, 0] = 1
        decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)
        cls_positions = None

        if append_cls_token:
            batch_size, seq_len, hidden_dim = decoder_inputs_embeds.shape
            cls_positions = decoder_attention_mask.sum(dim=1).long()
            batch_indices = torch.arange(batch_size, device=decoder_inputs_embeds.device)
            pad_embed = self.get_input_embeddings().weight[self.config.pad_token_id].view(1, 1, hidden_dim)
            packed_embeds = pad_embed.expand(batch_size, seq_len + 1, hidden_dim).clone()
            packed_embeds[:, :seq_len, :] = decoder_inputs_embeds
            packed_embeds[batch_indices, cls_positions] = self.cls_token_embedding.expand(batch_size, -1)

            position_ids = torch.arange(seq_len + 1, device=decoder_attention_mask.device).unsqueeze(0)
            packed_mask = (position_ids <= cls_positions.unsqueeze(1)).long()

            decoder_inputs_embeds = packed_embeds
            decoder_attention_mask = packed_mask

        return decoder_inputs_embeds, decoder_attention_mask, cls_positions

    def set_hyper(
            self,
            model_temperature=None,
            use_cls_contra=None,
            cls_contra_weight=None,
            cls_contra_type=None,
            cls_temperature=None,
            item_id_base=None,
            emb_file=None,
            cls_num_negatives=None,
    ):
        if model_temperature is not None:
            self.model_temperature = model_temperature
        if use_cls_contra is not None:
            self.use_cls_contra = use_cls_contra
        if cls_contra_weight is not None:
            self.cls_contra_weight = cls_contra_weight
        if cls_contra_type is not None:
            self.cls_contra_type = cls_contra_type
        if cls_temperature is not None:
            self.cls_temperature = cls_temperature
        if item_id_base is not None:
            self.cls_item_id_base = item_id_base
        if cls_num_negatives is not None:
            self.cls_num_negatives = cls_num_negatives
        if self.use_cls_contra:
            self._ensure_cls_token()
            emb_path = emb_file if emb_file is not None else self.emb_file
            self._load_item_embeddings(emb_path)
        elif emb_file is not None:
            self.emb_file = emb_file

    def ranking_loss(self, lm_logits, labels):
        t_logits = lm_logits / self.model_temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(lm_logits.device)
        return loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))

    def cosine_loss(self, cls_output: torch.Tensor, E_t: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss. Minimizes 1 - cos(cls[i], E_t[i]) per positive pair,
        no in-batch negatives.
        """
        cls_norm = F.normalize(cls_output, p=2, dim=-1)
        E_t_norm = F.normalize(E_t, p=2, dim=-1)
        cos_sim = (cls_norm * E_t_norm).sum(dim=-1)  # [B]
        return (1.0 - cos_sim).mean()

    def infonce_loss_inbatch(self, cls_output: torch.Tensor, E_t: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss with in-batch negatives.
        Positive for sample i is E_t[i]; negatives are E_t[j] for j != i.
        """
        B = cls_output.size(0)
        cls_norm = F.normalize(cls_output, p=2, dim=-1)
        E_t_norm = F.normalize(E_t, p=2, dim=-1)
        logits = (cls_norm @ E_t_norm.T) / self.cls_temperature  # [B, B]
        labels = torch.arange(B, dtype=torch.long, device=cls_output.device)
        return F.cross_entropy(logits, labels)

    def infonce_loss_random_neg(self, cls_output: torch.Tensor, E_t: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss with randomly sampled negatives from the full item pool.

        For each sample i:
          - positive  : E_t[i]
          - negatives : cls_num_negatives items randomly drawn from item_embeddings

        Logit layout per row: [pos_score | neg_score_0 | ... | neg_score_{K-1}]
        The label is always 0 (positive is placed at index 0).

        Args:
            cls_output : [B, item_emb_dim]  projected CLS representations
            E_t        : [B, item_emb_dim]  positive item embeddings (already fetched)
        Returns:
            scalar loss
        """
        B, device = cls_output.size(0), cls_output.device
        K = self.cls_num_negatives

        # --- sample K random negative indices for every sample in the batch ---
        neg_idx = torch.randint(0, self.item_num, (B, K), device=device)  # [B, K]

        item_emb = self.item_embeddings.to(device)          # [N, item_emb_dim]
        neg_embs = item_emb[neg_idx]                        # [B, K, item_emb_dim]

        # --- normalize all vectors ---
        cls_norm = F.normalize(cls_output, p=2, dim=-1)     # [B, d]
        pos_norm = F.normalize(E_t, p=2, dim=-1)            # [B, d]
        neg_norm = F.normalize(neg_embs, p=2, dim=-1)       # [B, K, d]

        # --- positive score: [B, 1] ---
        pos_score = (cls_norm * pos_norm).sum(dim=-1, keepdim=True) / self.cls_temperature

        # --- negative scores: [B, K]  (bmm: [B, K, d] x [B, d, 1] -> [B, K, 1]) ---
        neg_score = torch.bmm(neg_norm, cls_norm.unsqueeze(-1)).squeeze(-1) / self.cls_temperature

        # --- concat and compute CE loss (label = 0, i.e. positive is always first) ---
        logits = torch.cat([pos_score, neg_score], dim=1)   # [B, K+1]
        labels = torch.zeros(B, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def get_projected_item_embeddings(self, target_item_ids: torch.Tensor) -> torch.Tensor:
        if self.item_embeddings is None:
            raise ValueError("Item embeddings are not initialized. Please provide a valid `emb_file`.")
        target_idx = target_item_ids.long() - int(self.cls_item_id_base)
        target_idx = torch.clamp(target_idx, 0, self.item_num - 1)
        item_embeddings = self.item_embeddings.to(target_item_ids.device)
        return item_embeddings[target_idx]  # [B, item_emb_dim], no projection

    def total_loss(self, lm_logits, labels, cls_output=None, target_item_ids=None):
        loss_ranking = self.ranking_loss(lm_logits, labels)
        if not self.use_cls_contra:
            self.latest_loss_breakdown = {
                "ranking_loss": loss_ranking.detach(),
                "cls_loss": loss_ranking.detach().new_zeros(()),
                "total_loss": loss_ranking.detach(),
            }
            return loss_ranking

        if cls_output is None or target_item_ids is None:
            raise ValueError("`cls_output` and `target_item_ids` are required when `use_cls_contra` is enabled.")

        E_t = self.get_projected_item_embeddings(target_item_ids.to(cls_output.device))
        cls_projected = self.cls_mlp_projection(cls_output)  # [B, item_emb_dim]

        if self.cls_contra_type == "cosine":
            loss_contra = self.cosine_loss(cls_projected, E_t)
        elif self.cls_contra_type == "infonce":
            loss_contra = self.infonce_loss_inbatch(cls_projected, E_t)
        elif self.cls_contra_type == "random_neg":
            loss_contra = self.infonce_loss_random_neg(cls_projected, E_t)
        else:
            raise ValueError(
                f"Unknown cls_contra_type: {self.cls_contra_type!r}. "
                "Choose 'infonce', 'cosine', or 'random_neg'."
            )

        total = loss_ranking + self.cls_contra_weight * loss_contra
        self.latest_loss_breakdown = {
            "ranking_loss": loss_ranking.detach(),
            "cls_loss": loss_contra.detach(),
            "total_loss": total.detach(),
        }
        return total
        
    def forward(
            self,
            input_ids=None,
            whole_word_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            decoder_head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            reduce_loss=False,
            return_hidden_state=False,
            target_item_ids=None,
            history_item_ids=None,
            **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if (
                labels is not None
                and decoder_inputs_embeds is None
                and decoder_input_ids is not None
                and past_key_values is None
                and self.use_cls_contra
        ):
            decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)
            if decoder_attention_mask is None:
                decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).long()
                decoder_attention_mask[:, 0] = 1
            cls_emb = self.cls_token_embedding.expand(decoder_input_ids.size(0), 1, -1)
            decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, cls_emb], dim=1)
            cls_mask = torch.ones(
                decoder_attention_mask.size(0),
                1,
                dtype=decoder_attention_mask.dtype,
                device=decoder_attention_mask.device,
            )
            decoder_attention_mask = torch.cat([decoder_attention_mask, cls_mask], dim=1)
            decoder_input_ids = None

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        cls_hidden = None
        lm_sequence_output = sequence_output
        if self.use_cls_contra and labels is not None:
            cls_hidden = sequence_output[:, -1, :]
            lm_sequence_output = sequence_output[:, :labels.size(1), :]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            lm_sequence_output = lm_sequence_output.to(self.lm_head.weight.device)
            if cls_hidden is not None:
                cls_hidden = cls_hidden.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            lm_sequence_output = lm_sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(lm_sequence_output)

        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, cls_hidden, target_item_ids)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# =============================================================================
# Hierarchical Encoder Extension
# =============================================================================

class ItemLevelEncoder(nn.Module):
    """Single-layer Transformer that compresses an item's 4 code tokens into
    one item-level representation (Stage 1 of the hierarchical encoder).

    Architecture (pre-norm, T5-style RMS LayerNorm):
        normed  = LayerNorm(x)                          # [N, 4, D]
        x       = x + MultiHeadSelfAttention(normed)    # residual
        normed  = LayerNorm(x)
        x       = x + FFN(normed)                       # residual
        output  = mean_pool(x)                          # [N, D]

    Initialization
    --------------
    Output projections (o_proj, ff2) are zero-initialized so that at the
    start of training the item representation equals the mean of the 4 input
    token embeddings — a smooth starting point close to the LETTER baseline.
    All other projections use N(0, 0.02).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        d_ff = d_model * ffn_ratio
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()

        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.ff1):
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.ff2.weight)

    def _mhsa(self, x: torch.Tensor) -> torch.Tensor:
        N, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(N, S, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(N, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(N, S, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        attn = self.drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, S, D)
        return self.o_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 4 code tokens → single item representation [N, D]."""
        x = x + self.drop(self._mhsa(self.norm1(x)))
        x = x + self.drop(self.ff2(self.act(self.ff1(self.norm2(x)))))
        return x.mean(dim=1)


class LETTER_HiEncoder(LETTER):
    """LETTER (contra2) 的两阶段层次化编码器扩展。

    Stage 1 — ItemLevelEncoder
        每个 item 的 4 个 RQ-VAE code token 经小型 Transformer 压缩为单一
        item 表示，将 T5 Encoder 见到的序列长度缩减 4 倍。

    Stage 2 — T5 Encoder（继承）
        以 item 粒度序列 [item_0, item_1, …, item_{N-1}, EOS] 为输入，
        输出供 Decoder 交叉注意力使用的 item 级上下文。

    所有其他 LETTER 特性（sideinfo 对比损失、CLS token、多投影头）完全保留。
    Encoder hidden states 和 attention mask 统一工作在 item 级别。

    新增构造参数
    -----------
    item_encoder_heads   : int   — ItemLevelEncoder 注意力头数（默认 4）
    item_encoder_dropout : float — ItemLevelEncoder dropout 概率（默认 0.1）
    """

    def __init__(
        self,
        config: T5Config,
        item_encoder_heads: int = 4,
        item_encoder_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.item_encoder = ItemLevelEncoder(
            d_model=config.d_model,
            n_heads=item_encoder_heads,
            dropout=item_encoder_dropout,
        )

    # ------------------------------------------------------------------
    # Helpers: token-level → item-level conversion
    # ------------------------------------------------------------------

    def _build_item_level_inputs(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ):
        """Convert token-level encoder inputs to item-level inputs.

        Assumes add_prefix=False: each item occupies exactly 4 consecutive
        token positions, followed by a single EOS at position 4·N.

        Returns
        -------
        item_inputs_embeds : Tensor [B, max_n+1, D]
        item_attn_mask     : LongTensor [B, max_n+1]
        n_items            : LongTensor [B]
        """
        tok_embs = self.get_input_embeddings()(input_ids)  # [B, L, D]
        B, L, D = tok_embs.shape
        device = input_ids.device

        seq_len = attention_mask.sum(dim=-1).long()         # [B]
        n_items = ((seq_len - 1) // 4).clamp(min=1)        # [B]
        max_n = int(n_items.max().item())

        item_idx = torch.arange(max_n, device=device)      # [max_n]
        code_idx = torch.arange(4, device=device)           # [4]
        tok_idx = (
            (item_idx.unsqueeze(1) * 4 + code_idx.unsqueeze(0))
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clamp(0, L - 1)
        )                                                    # [B, max_n, 4]
        b_idx = (
            torch.arange(B, device=device)
            .view(B, 1, 1)
            .expand(B, max_n, 4)
        )
        item_tok_embs = tok_embs[b_idx, tok_idx]           # [B, max_n, 4, D]

        item_reprs = self.item_encoder(
            item_tok_embs.view(B * max_n, 4, D)
        ).view(B, max_n, D)                                  # [B, max_n, D]

        eos_pos = (seq_len - 1).clamp(0, L - 1)            # [B]
        eos_embs = tok_embs[
            torch.arange(B, device=device), eos_pos
        ].unsqueeze(1)                                       # [B, 1, D]

        item_inputs_embeds = torch.cat(
            [item_reprs, eos_embs], dim=1
        )                                                    # [B, max_n+1, D]

        item_positions = torch.arange(
            max_n + 1, device=device
        ).unsqueeze(0)                                       # [1, max_n+1]
        item_attn_mask = (
            item_positions <= n_items.unsqueeze(1)
        ).long()                                             # [B, max_n+1]

        return item_inputs_embeds, item_attn_mask, n_items

    @staticmethod
    def _item_attn_mask_from_token_mask(
        attention_mask: torch.Tensor,  # [B, L]  token-level
    ):
        """Recompute item-level attention mask from the original token mask."""
        device = attention_mask.device
        seq_len = attention_mask.sum(dim=-1).long()
        n_items = ((seq_len - 1) // 4).clamp(min=1)
        max_n = int(n_items.max().item())
        item_positions = torch.arange(
            max_n + 1, device=device
        ).unsqueeze(0)
        item_attn_mask = (
            item_positions <= n_items.unsqueeze(1)
        ).long()
        return item_attn_mask, n_items

    # ------------------------------------------------------------------
    # Override: forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,
        return_hidden_state=False,
        target_item_ids=None,
        history_item_ids=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # ---- Stage 1 + Stage 2: Hierarchical Encoding ----
        item_attn_mask = None

        if encoder_outputs is None:
            if (
                input_ids is not None
                and inputs_embeds is None
                and attention_mask is not None
            ):
                item_inputs_embeds, item_attn_mask, _ = self._build_item_level_inputs(
                    input_ids, attention_mask
                )
                encoder_outputs = self.encoder(
                    input_ids=None,
                    attention_mask=item_attn_mask,
                    inputs_embeds=item_inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        # 交叉注意力 mask：层次化路径使用 item 级，否则沿用 token 级
        cross_attn_mask = item_attn_mask if item_attn_mask is not None else attention_mask

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # ---- Decoder 输入准备（与基类 LETTER 完全相同）----
        cls_positions = None
        if (
            decoder_inputs_embeds is None
            and decoder_input_ids is not None
            and past_key_values is None
            and self.use_cls_contra
            and labels is not None
        ):
            decoder_inputs_embeds, decoder_attention_mask, cls_positions = (
                self._prepare_decoder_inputs(
                    decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    append_cls_token=True,
                )
            )
            decoder_input_ids = None

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds.to(self.decoder.first_device)
            if cross_attn_mask is not None:
                cross_attn_mask = cross_attn_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            if cls_positions is not None:
                cls_positions = cls_positions.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=cross_attn_mask,  # item 级 mask
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        cls_hidden = None
        if cls_positions is not None:
            batch_indices = torch.arange(
                sequence_output.size(0), device=sequence_output.device
            )
            cls_hidden = sequence_output[batch_indices, cls_positions]

        if labels is not None and cls_positions is not None:
            batch_size, target_seq_len = labels.size(0), labels.size(1)
            gather_base = (
                torch.arange(target_seq_len, device=sequence_output.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            gather_shift = (gather_base >= cls_positions.unsqueeze(1)).long()
            gather_indices = gather_base + gather_shift
            lm_sequence_output = sequence_output.gather(
                1,
                gather_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
            )
        else:
            lm_sequence_output = (
                sequence_output[:, : labels.size(1), :]
                if labels is not None
                else sequence_output
            )

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            lm_sequence_output = lm_sequence_output.to(self.lm_head.weight.device)
            if cls_hidden is not None:
                cls_hidden = cls_hidden.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            lm_sequence_output = lm_sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(lm_sequence_output)

        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, cls_hidden, target_item_ids)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Override: _prepare_encoder_decoder_kwargs_for_generation
    # ------------------------------------------------------------------

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor,
        model_kwargs,
        model_input_name,
        generation_config,
    ):
        """在 generate() beam-search 循环前执行层次化编码。

        HuggingFace 默认实现直接调用 self.encoder（原始 T5Stack），完全绕过
        ItemLevelEncoder。此覆写在此执行完整两阶段编码，并将 model_kwargs 中的
        token 级 attention_mask 替换为 item 级 mask，保证后续所有 forward() 调用
        的 encoder outputs 与 attention mask 维度一致。
        """
        attention_mask = model_kwargs.get("attention_mask", None)

        if attention_mask is not None:
            item_inputs_embeds, item_attn_mask, _ = self._build_item_level_inputs(
                inputs_tensor, attention_mask
            )
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=item_attn_mask,
                inputs_embeds=item_inputs_embeds,
                return_dict=True,
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["attention_mask"] = item_attn_mask
        else:
            model_kwargs = super(
                LETTER, self
            )._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        return model_kwargs

    # ------------------------------------------------------------------
    # Override: prepare_inputs_for_generation
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 跳过 LETTER 的覆写，直接调用 T5ForConditionalGeneration 的实现
        inputs = super(LETTER, self).prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        # attention_mask 已由 _prepare_encoder_decoder_kwargs_for_generation
        # 替换为 item 级。若不一致（边界情况），重新计算一次。
        if encoder_outputs is not None and attention_mask is not None:
            enc_len = encoder_outputs[0].shape[1]
            if attention_mask.shape[1] != enc_len:
                item_attn_mask, _ = self._item_attn_mask_from_token_mask(
                    attention_mask
                )
                inputs["attention_mask"] = item_attn_mask

        return inputs
