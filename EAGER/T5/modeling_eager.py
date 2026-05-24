import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5ForConditionalGeneration,
    T5LayerNorm,
    T5Stack,
)


def _build_decoder_config(config: T5Config, num_layers: int) -> T5Config:
    decoder_config = config.to_dict()
    decoder_config["is_decoder"] = True
    decoder_config["use_cache"] = config.use_cache
    decoder_config["is_encoder_decoder"] = False
    decoder_config["num_layers"] = int(num_layers)
    return T5Config.from_dict(decoder_config)


def _build_encoder_config(config: T5Config, num_layers: int) -> T5Config:
    encoder_config = config.to_dict()
    encoder_config["is_decoder"] = False
    encoder_config["use_cache"] = False
    encoder_config["is_encoder_decoder"] = False
    encoder_config["num_layers"] = int(num_layers)
    return T5Config.from_dict(encoder_config)


def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1)


def _normalize_if_needed(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def _info_nce(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    reduction: str = "mean",
    negative_mode: str = "unpaired",
    norm: bool = True,
) -> torch.Tensor:
    if norm:
        query, positive_key, negative_keys = _normalize_if_needed(
            query, positive_key, negative_keys
        )

    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == "unpaired":
            negative_logits = query @ _transpose_last_two(negative_keys)
        elif negative_mode == "paired":
            negative_logits = (query.unsqueeze(1) @ _transpose_last_two(negative_keys)).squeeze(1)
        else:
            raise ValueError(f"Unsupported negative_mode: {negative_mode}")
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ _transpose_last_two(positive_key)
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def _load_teacher_embedding(path: str) -> np.ndarray:
    if path.endswith(".pt") or path.endswith(".pth"):
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            tensor = None
            for key in ["embeddings", "embedding", "weight", "tensor"]:
                if key in loaded:
                    tensor = loaded[key]
                    break
            if tensor is None and len(loaded) == 1:
                tensor = next(iter(loaded.values()))
            if tensor is None:
                raise TypeError(f"Unsupported teacher embedding checkpoint format: {path}")
            loaded = tensor
        loaded = torch.as_tensor(loaded).detach().cpu().float().numpy()
    else:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "embeddings" in loaded:
                loaded = loaded["embeddings"]
            else:
                first_key = loaded.files[0]
                loaded = loaded[first_key]
    return np.asarray(loaded, dtype=np.float32)


def _truncate_semantic_teacher(teacher_table: np.ndarray, target_dim: int = 256) -> np.ndarray:
    if teacher_table.ndim != 2:
        raise ValueError("Semantic teacher embedding must be a 2D table.")
    if teacher_table.shape[1] < target_dim:
        raise ValueError(
            f"Semantic teacher embedding dim {teacher_table.shape[1]} is smaller than required {target_dim}."
        )
    return teacher_table[:, :target_dim]


class EAGERAuxiliaryEncoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(idx == 0)) for idx in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.size(0),
                hidden_states.size(1),
                device=hidden_states.device,
                dtype=torch.long,
            )
        extended_attention_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(hidden_states.dtype).min

        for block in self.block:
            layer_outputs = block(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                use_cache=False,
                output_attentions=False,
                return_dict=False,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


@dataclass
class EAGERModelOutput(Seq2SeqLMOutput):
    cf_logits: Optional[torch.Tensor] = None
    sem_logits: Optional[torch.Tensor] = None


class EAGER(T5ForConditionalGeneration):
    def __init__(
        self,
        config: T5Config,
        encoder_input_mode: str = "code",
        cf_teacher_emb_path: str = "",
        sem_teacher_emb_path: str = "",
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 4,
        aux_num_layers: int = 1,
        item_id_base: int = 1,
        model_temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(config)
        self.model_temperature = model_temperature
        self.latest_loss_breakdown = {}
        self.encoder_input_mode = encoder_input_mode
        self.model_dim = config.d_model
        self.item_id_base = int(item_id_base)

        if encoder_input_mode not in {"code", "item_id"}:
            raise ValueError(f"Unsupported encoder_input_mode: {encoder_input_mode}")

        encoder_config = _build_encoder_config(config, encoder_num_layers)
        decoder_config = _build_decoder_config(config, decoder_num_layers)
        aux_config = _build_encoder_config(config, aux_num_layers)

        shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared = shared
        self.encoder = T5Stack(encoder_config, shared)
        self.decoder_cf = T5Stack(decoder_config, shared)
        self.decoder_sem = T5Stack(decoder_config, shared)
        self.decoder = self.decoder_cf
        self.auxiliary_encoder = EAGERAuxiliaryEncoder(aux_config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.cf_teacher_proj = None
        self.sem_teacher_proj = None
        self.guide_proj = nn.Linear(config.d_model, config.d_model)
        self.estimate_head = nn.Linear(config.d_model, 1)
        self.start_vec = nn.Parameter(torch.zeros(config.d_model))
        self.mask_vec = nn.Parameter(torch.zeros(config.d_model))

        self.cf_item_embedding = None
        self.sem_item_embedding = None
        self.teacher_pad_index = 0
        self.register_buffer(
            "cf_first_token_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "cf_second_token_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.cf_first_to_second_token_ids = {}

        if encoder_input_mode == "item_id":
            cf_teacher_table = _load_teacher_embedding(cf_teacher_emb_path)
            sem_teacher_table = _truncate_semantic_teacher(_load_teacher_embedding(sem_teacher_emb_path))
            if cf_teacher_table.shape[0] != sem_teacher_table.shape[0]:
                raise ValueError("Teacher embedding tables must have the same number of items.")
            self.teacher_pad_index = cf_teacher_table.shape[0]
            self.cf_item_embedding = nn.Embedding(
                self.teacher_pad_index + 1,
                config.d_model,
                padding_idx=self.teacher_pad_index,
            )
            self.sem_item_embedding = nn.Embedding(
                self.teacher_pad_index + 1,
                config.d_model,
                padding_idx=self.teacher_pad_index,
            )

        cf_teacher_table = _load_teacher_embedding(cf_teacher_emb_path)
        sem_teacher_table = _truncate_semantic_teacher(_load_teacher_embedding(sem_teacher_emb_path))
        self.cf_teacher_embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(cf_teacher_table),
            freeze=True,
        )
        self.sem_teacher_embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(sem_teacher_table),
            freeze=True,
        )
        self.cf_teacher_proj = nn.Linear(config.d_model, self.cf_teacher_embeddings.embedding_dim)
        self.sem_teacher_proj = nn.Linear(config.d_model, self.sem_teacher_embeddings.embedding_dim)

        self.config.encoder_input_mode = encoder_input_mode
        self.config.cf_teacher_emb_path = cf_teacher_emb_path
        self.config.sem_teacher_emb_path = sem_teacher_emb_path
        self.config.num_layers = encoder_num_layers
        self.config.num_decoder_layers = decoder_num_layers
        self.config.aux_num_layers = aux_num_layers
        self.config.item_id_base = self.item_id_base
        self.config.use_cache = False

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.get("config")
        if config is not None:
            kwargs.setdefault("encoder_input_mode", getattr(config, "encoder_input_mode", "code"))
            kwargs.setdefault("cf_teacher_emb_path", getattr(config, "cf_teacher_emb_path", ""))
            kwargs.setdefault("sem_teacher_emb_path", getattr(config, "sem_teacher_emb_path", ""))
            kwargs.setdefault("encoder_num_layers", getattr(config, "num_layers", 1))
            kwargs.setdefault("decoder_num_layers", getattr(config, "num_decoder_layers", 4))
            kwargs.setdefault("aux_num_layers", getattr(config, "aux_num_layers", 1))
            kwargs.setdefault("item_id_base", getattr(config, "item_id_base", 1))
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _project_hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            hidden_states = hidden_states * (self.model_dim ** -0.5)
        return self.lm_head(hidden_states)

    def ranking_loss(self, lm_logits, labels):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(lm_logits.device)
        t_logits = lm_logits / self.model_temperature
        return loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))

    def _prepare_item_history_inputs(
        self,
        history_item_ids: torch.Tensor,
        branch: str,
    ):
        if history_item_ids is None:
            raise ValueError("`history_item_ids` is required when encoder_input_mode='item_id'.")
        if branch == "cf":
            embedding_table = self.cf_item_embedding
        else:
            embedding_table = self.sem_item_embedding
        history_ids = history_item_ids.to(next(embedding_table.parameters()).device)
        attn_mask = history_ids.ne(-1).long()
        history_ids = history_ids.masked_fill(history_ids.lt(0), self.teacher_pad_index)
        history_ids = torch.where(history_ids.eq(self.teacher_pad_index), history_ids, history_ids - self.item_id_base)
        if torch.any(history_ids.lt(0)):
            raise ValueError("History item ids fall below the configured `item_id_base`.")
        inputs_embeds = embedding_table(history_ids)
        return inputs_embeds, attn_mask

    def _encode_branch(
        self,
        branch: str,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        history_item_ids: Optional[torch.Tensor],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: bool,
    ):
        if self.encoder_input_mode == "code":
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            branch_attention_mask = attention_mask
        else:
            inputs_embeds, branch_attention_mask = self._prepare_item_history_inputs(history_item_ids, branch)
            encoder_outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=branch_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return encoder_outputs, branch_attention_mask

    def _decode_branch(
        self,
        decoder: T5Stack,
        encoder_outputs: BaseModelOutput,
        encoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        decoder_input_ids: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        past_key_values,
        use_cache: bool,
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: bool,
    ):
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        logits = self._project_hidden_to_logits(sequence_output)
        return decoder_outputs, logits

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder_cf

    def set_cf_codebook(
        self,
        first_token_ids: torch.Tensor,
        first_to_second_token_ids,
    ):
        self.cf_first_token_ids = first_token_ids.detach().clone()
        self.cf_first_to_second_token_ids = {
            int(first_token_id): token_ids.detach().clone()
            for first_token_id, token_ids in first_to_second_token_ids.items()
        }
        all_second_token_ids = sorted(
            {
                int(token_id)
                for token_ids in self.cf_first_to_second_token_ids.values()
                for token_id in token_ids.tolist()
            }
        )
        self.cf_second_token_ids = torch.tensor(all_second_token_ids, dtype=torch.long)

    def encode_branches(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        history_item_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ):
        cf_encoder_outputs, cf_attention_mask = self._encode_branch(
            branch="cf",
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_item_ids=history_item_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sem_encoder_outputs, sem_attention_mask = self._encode_branch(
            branch="sem",
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_item_ids=history_item_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {
            "cf": (cf_encoder_outputs, cf_attention_mask),
            "sem": (sem_encoder_outputs, sem_attention_mask),
        }

    def _compute_auxiliary_losses(
        self,
        guide_summary: torch.Tensor,
        cf_labels: torch.Tensor,
    ):
        if (
            self.cf_first_token_ids.numel() == 0
            or self.cf_second_token_ids.numel() == 0
            or not self.cf_first_to_second_token_ids
        ):
            zero = guide_summary.new_zeros(())
            return zero, zero
        if cf_labels.size(1) < 2:
            zero = guide_summary.new_zeros(())
            return zero, zero

        code_labels = cf_labels[:, :2]
        valid_code_mask = code_labels.ne(-100)
        if not valid_code_mask.all():
            zero = guide_summary.new_zeros(())
            return zero, zero

        # Original EAGER uses a single hierarchical mask decision for the whole batch:
        # either mask the first code level or mask the second level conditioned on the first.
        mask_first_level = bool(torch.rand((), device=cf_labels.device).item() <= 0.5)
        masked_pos_idx = 0 if mask_first_level else 1
        label_embeddings = self.shared(code_labels.clamp_min(0))
        aux_inputs = label_embeddings.clone()
        aux_inputs[:, masked_pos_idx] = self.mask_vec.view(1, -1)
        aux_inputs = torch.cat(
            [self.start_vec.view(1, 1, -1).expand(aux_inputs.size(0), -1, -1), aux_inputs],
            dim=1,
        )
        aux_attention_mask = torch.ones(
            code_labels.size(0),
            code_labels.size(1) + 1,
            device=cf_labels.device,
            dtype=torch.long,
        )
        guide_bias = self.guide_proj(guide_summary.detach()).unsqueeze(1)
        aux_inputs = aux_inputs + guide_bias

        aux_hidden = self.auxiliary_encoder(
            inputs_embeds=aux_inputs,
            attention_mask=aux_attention_mask,
        )
        masked_hidden = aux_hidden[:, masked_pos_idx + 1]
        target_token_ids = code_labels[:, masked_pos_idx]
        positive_emb = self.shared(target_token_ids).detach()
        if mask_first_level:
            negative_token_ids = self.cf_first_token_ids.to(cf_labels.device)
            negative_emb = self.shared(negative_token_ids).detach().unsqueeze(0).expand(
                code_labels.size(0), -1, -1
            )
        else:
            negative_token_ids = self.cf_second_token_ids.to(cf_labels.device)
            negative_token_ids = negative_token_ids.unsqueeze(0).expand(code_labels.size(0), -1)
            negative_emb = self.shared(negative_token_ids).detach()
        recon_loss = _info_nce(
            masked_hidden,
            positive_emb,
            negative_emb,
            temperature=1.0,
            negative_mode="paired",
            norm=False,
        )

        corrupted_inputs = torch.cat(
            [self.start_vec.view(1, 1, -1).expand(label_embeddings.size(0), -1, -1), label_embeddings.clone()],
            dim=1,
        )
        if mask_first_level:
            sampled_neg_indices = torch.randint(
                low=0,
                high=self.cf_first_token_ids.numel(),
                size=(code_labels.size(0),),
                device=cf_labels.device,
            )
            sampled_neg_ids = self.cf_first_token_ids.to(cf_labels.device).index_select(
                0, sampled_neg_indices
            )
        else:
            sampled_neg_indices = torch.randint(
                low=0,
                high=self.cf_second_token_ids.numel(),
                size=(code_labels.size(0),),
                device=cf_labels.device,
            )
            sampled_neg_ids = self.cf_second_token_ids.to(cf_labels.device).index_select(
                0, sampled_neg_indices
            )
        corrupted_inputs[:, masked_pos_idx + 1] = self.shared(sampled_neg_ids)
        positive_inputs = torch.cat(
            [self.start_vec.view(1, 1, -1).expand(label_embeddings.size(0), -1, -1), label_embeddings],
            dim=1,
        ) + guide_bias
        corrupted_inputs = corrupted_inputs + guide_bias
        pos_hidden = self.auxiliary_encoder(
            inputs_embeds=positive_inputs,
            attention_mask=aux_attention_mask,
        )[:, 0]
        neg_hidden = self.auxiliary_encoder(
            inputs_embeds=corrupted_inputs,
            attention_mask=aux_attention_mask,
        )[:, 0]
        pos_logits = self.estimate_head(pos_hidden)
        neg_logits = self.estimate_head(neg_hidden)
        estimate_logits = torch.cat([pos_logits, neg_logits], dim=0)
        estimate_labels = torch.cat(
            [
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits),
            ],
            dim=0,
        )
        estimate_loss = F.binary_cross_entropy_with_logits(estimate_logits, estimate_labels)
        return recon_loss, estimate_loss

    def prepare_inputs_for_generation(self, *args, **kwargs):
        history_item_ids = kwargs.get("history_item_ids")
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if history_item_ids is not None:
            model_inputs["history_item_ids"] = history_item_ids
        return model_inputs

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        model_kwargs = None
        if len(args) >= 2 and isinstance(args[1], dict):
            model_kwargs = args[1]
        elif isinstance(kwargs.get("model_kwargs"), dict):
            model_kwargs = kwargs["model_kwargs"]

        history_item_ids = None
        if model_kwargs is not None and "history_item_ids" in model_kwargs:
            history_item_ids = model_kwargs.pop("history_item_ids")
        input_ids = None
        if len(args) >= 1 and torch.is_tensor(args[0]):
            input_ids = args[0]
        elif torch.is_tensor(kwargs.get("input_ids")):
            input_ids = kwargs["input_ids"]

        prepared_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)
        if history_item_ids is not None:
            prepared_kwargs["history_item_ids"] = history_item_ids
        return prepared_kwargs

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
        cf_labels=None,
        sem_labels=None,
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
        del whole_word_ids, reduce_loss, return_hidden_state, decoder_inputs_embeds, inputs_embeds
        del head_mask, decoder_head_mask, cross_attn_head_mask

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if labels is None:
            labels = cf_labels if cf_labels is not None else sem_labels
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        branch_past_key_values = {"cf": past_key_values, "sem": past_key_values}
        if (
            isinstance(past_key_values, tuple)
            and len(past_key_values) == 2
            and all(isinstance(value, tuple) for value in past_key_values)
        ):
            branch_past_key_values = {
                "cf": past_key_values[0],
                "sem": past_key_values[1],
            }

        if labels is None:
            branch_encodings = self.encode_branches(
                input_ids=input_ids,
                attention_mask=attention_mask,
                history_item_ids=history_item_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            cf_encoder_outputs, cf_attention_mask = branch_encodings["cf"]
            sem_encoder_outputs, sem_attention_mask = branch_encodings["sem"]
            sem_decoder_outputs, sem_logits = self._decode_branch(
                self.decoder_sem,
                sem_encoder_outputs,
                sem_attention_mask,
                sem_labels,
                decoder_input_ids,
                decoder_attention_mask,
                branch_past_key_values["sem"],
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
            cf_decoder_outputs, cf_logits = self._decode_branch(
                self.decoder_cf,
                cf_encoder_outputs,
                cf_attention_mask,
                cf_labels,
                decoder_input_ids,
                decoder_attention_mask,
                branch_past_key_values["cf"],
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
            logits = cf_logits + sem_logits

            return EAGERModelOutput(
                loss=None,
                logits=logits,
                past_key_values=(
                    cf_decoder_outputs.past_key_values,
                    sem_decoder_outputs.past_key_values,
                ),
                decoder_hidden_states=cf_decoder_outputs.hidden_states,
                decoder_attentions=cf_decoder_outputs.attentions,
                cross_attentions=cf_decoder_outputs.cross_attentions,
                encoder_last_hidden_state=cf_encoder_outputs.last_hidden_state,
                encoder_hidden_states=cf_encoder_outputs.hidden_states,
                encoder_attentions=cf_encoder_outputs.attentions,
                cf_logits=cf_logits,
                sem_logits=sem_logits,
            )

        branch_encodings = self.encode_branches(
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_item_ids=history_item_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        cf_encoder_outputs, cf_attention_mask = branch_encodings["cf"]
        sem_encoder_outputs, sem_attention_mask = branch_encodings["sem"]

        sem_decoder_outputs, sem_logits = self._decode_branch(
            self.decoder_sem,
            sem_encoder_outputs,
            sem_attention_mask,
            sem_labels,
            decoder_input_ids,
            decoder_attention_mask,
            branch_past_key_values["sem"],
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        cf_decoder_outputs, cf_logits = self._decode_branch(
            self.decoder_cf,
            cf_encoder_outputs,
            cf_attention_mask,
            cf_labels,
            decoder_input_ids,
            decoder_attention_mask,
            branch_past_key_values["cf"],
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        cf_loss = None
        sem_loss = None
        eos_loss = None
        recon_loss = None
        estimate_loss = None
        total_loss = None

        if labels is not None:
            cf_loss = self.ranking_loss(cf_logits, cf_labels if cf_labels is not None else labels)
            sem_loss = self.ranking_loss(sem_logits, sem_labels if sem_labels is not None else labels)
            if target_item_ids is None:
                raise ValueError("`target_item_ids` is required for EAGER training.")

            target_teacher_ids = target_item_ids - self.item_id_base
            if torch.any(target_teacher_ids.lt(0)):
                raise ValueError("Target item ids fall below the configured `item_id_base`.")
            cf_teacher = self.cf_teacher_embeddings(target_teacher_ids)
            sem_teacher = self.sem_teacher_embeddings(target_teacher_ids)

            cf_summary = cf_decoder_outputs[0][:, -1]
            sem_summary = sem_decoder_outputs[0][:, -1]
            cf_projected = self.cf_teacher_proj(cf_summary)
            sem_projected = self.sem_teacher_proj(sem_summary)
            eos_loss = F.smooth_l1_loss(cf_projected, cf_teacher) + F.smooth_l1_loss(sem_projected, sem_teacher)

            if cf_labels is not None:
                recon_loss, estimate_loss = self._compute_auxiliary_losses(
                    guide_summary=sem_summary,
                    cf_labels=cf_labels,
                )
            else:
                zero = cf_summary.new_zeros(())
                recon_loss, estimate_loss = zero, zero
            total_loss = cf_loss + sem_loss + eos_loss + recon_loss + estimate_loss
            self.latest_loss_breakdown = {
                "cf_ranking_loss": cf_loss.detach(),
                "sem_ranking_loss": sem_loss.detach(),
                "eos_loss": eos_loss.detach(),
                "recon_loss": recon_loss.detach(),
                "estimate_loss": estimate_loss.detach(),
                "total_loss": total_loss.detach(),
            }

        if not return_dict:
            output = (cf_logits,) + cf_decoder_outputs[1:] + cf_encoder_outputs
            return ((total_loss,) + output) if total_loss is not None else output

        return EAGERModelOutput(
            loss=total_loss,
            logits=cf_logits,
            past_key_values=(
                cf_decoder_outputs.past_key_values,
                sem_decoder_outputs.past_key_values,
            ),
            decoder_hidden_states=cf_decoder_outputs.hidden_states,
            decoder_attentions=cf_decoder_outputs.attentions,
            cross_attentions=cf_decoder_outputs.cross_attentions,
            encoder_last_hidden_state=cf_encoder_outputs.last_hidden_state,
            encoder_hidden_states=cf_encoder_outputs.hidden_states,
            encoder_attentions=cf_encoder_outputs.attentions,
            cf_logits=cf_logits,
            sem_logits=sem_logits,
        )
