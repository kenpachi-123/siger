import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.scale = input_dim ** -0.5
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.attn_net(x).squeeze(-1) * self.scale
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        return (weights * x).sum(dim=1)


class PromptFormer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        codebook_weight, stage_sizes = self._build_codebook(config)
        self.codebook = nn.Embedding.from_pretrained(codebook_weight, freeze=True)
        self.stage_sizes = stage_sizes
        self.latent_size = codebook_weight.shape[1]
        self.alpha = config["alpha"]

        self.collab_embedding = nn.Parameter(torch.zeros_like(self.codebook.weight))
        self.register_buffer("fused_embedding", torch.zeros_like(self.codebook.weight))

        self.layernorm = nn.LayerNorm(self.latent_size, eps=1e-6)
        self.collab_layernorm = nn.LayerNorm(self.latent_size, eps=1e-6)
        self.collab_projection = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
        )
        self.projection = nn.Linear(2 * self.latent_size, config["t5_dim"])

        self.attention_pooling = AttentionPooling(self.latent_size, hidden_dim=128)
        self.q_ctx = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.k_candidates = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.bos_queries = nn.Parameter(torch.zeros(config["bos_queries"], self.latent_size))

        if len(set(self.stage_sizes)) != 1:
            raise ValueError("DECOR currently expects equal-sized codebooks across all stages.")
        self.stage_width = self.stage_sizes[0]
        self.register_buffer(
            "fixed_bins",
            torch.stack(
                [
                    torch.arange(1 + idx * self.stage_width, 1 + (idx + 1) * self.stage_width)
                    for idx in range(len(self.stage_sizes))
                ]
            ),
        )
        stage_id_map = torch.full((self.codebook.num_embeddings,), -1, dtype=torch.long)
        for idx in range(len(self.stage_sizes)):
            start = 1 + idx * self.stage_width
            end = start + self.stage_width
            stage_id_map[start:end] = idx
        self.register_buffer("stage_id_map", stage_id_map)

    def _build_codebook(self, config: dict):
        codebook_path = config.get("codebook_path") or ""
        collision_size = config["collision_size"]
        if codebook_path and os.path.exists(codebook_path):
            rq_codebooks = self._load_rq_codebooks(codebook_path)
            latent_size = rq_codebooks[0].shape[1]
            collision_codebook = torch.zeros(collision_size, latent_size, dtype=torch.float32)
            stage_codebooks = rq_codebooks + [collision_codebook]
            stage_sizes = [codebook.shape[0] for codebook in stage_codebooks]
            padding = torch.zeros(1, latent_size, dtype=torch.float32)
            eos = torch.zeros(1, latent_size, dtype=torch.float32)
            weight = torch.cat([padding] + stage_codebooks + [eos], dim=0)
            return weight, stage_sizes

        latent_size = config["latent_size"]
        rq_n_codebooks = config["rq_n_codebooks"]
        rq_codebook_size = config["rq_codebook_size"]
        stage_sizes = [rq_codebook_size] * rq_n_codebooks + [collision_size]
        vocab_size = 1 + sum(stage_sizes) + 1
        return torch.zeros(vocab_size, latent_size, dtype=torch.float32), stage_sizes

    @staticmethod
    def _load_rq_codebooks(codebook_path: str):
        if codebook_path.endswith(".npz"):
            saved = np.load(codebook_path, allow_pickle=True)
            codebooks = [
                torch.as_tensor(np.asarray(codebook, dtype=np.float32))
                for codebook in saved["codebooks"].tolist()
            ]
            if len(codebooks) < 3:
                raise ValueError(f"Expected at least 3 codebooks in npz: {codebook_path}")
            return codebooks[:3]

        saved = torch.load(codebook_path, map_location="cpu", weights_only=False)
        lookup_sources = [saved]
        if isinstance(saved, dict) and isinstance(saved.get("state_dict"), dict):
            lookup_sources.insert(0, saved["state_dict"])

        for source in lookup_sources:
            stage_keys = sorted(
                key for key in source.keys()
                if key.startswith("rq.vq_layers.") and key.endswith(".embedding.weight")
            )
            if len(stage_keys) >= 3:
                return [source[key].detach().float() for key in stage_keys[:3]]

            stage_keys = sorted(
                key for key in source.keys()
                if key.endswith(".embed") and "quantization_layer.quantization_layers." in key
            )
            if len(stage_keys) >= 3:
                return [source[key].detach().float().T for key in stage_keys[:3]]

        raise ValueError(f"Unsupported codebook checkpoint format: {codebook_path}")

    @property
    def eos_id(self) -> int:
        return self.codebook.num_embeddings - 1

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        txt_embeds = self.layernorm(self.codebook(input_ids))
        collab_embeds = self.collab_embedding[input_ids]
        collab_embeds = self.collab_layernorm(collab_embeds)
        collab_embeds = self.collab_projection(collab_embeds)
        fused_embeds = self.projection(torch.cat([txt_embeds, collab_embeds], dim=-1))
        self.fused_embedding[input_ids] = fused_embeds.detach().to(self.fused_embedding.dtype)
        return fused_embeds

    def get_decoder_embedding(
        self,
        input_ids: torch.Tensor,
        fused_embeds: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
    ):
        batch_size, seq_len = input_ids.shape
        e_ctx = self.attention_pooling(fused_embeds, attention_mask=encoder_attention_mask)
        scores = torch.matmul(e_ctx, self.bos_queries.T)
        probs = F.softmax(scores, dim=-1)
        bos_vec = torch.matmul(probs, self.bos_queries)
        e_fused = self.forward(input_ids)

        if seq_len == 1:
            return self.alpha * bos_vec.unsqueeze(1) + (1 - self.alpha) * e_fused, None

        if encoder_outputs is None:
            return self.fused_embedding[input_ids].to(e_fused.dtype), None

        decoder_ids = input_ids[:, 1:]
        stage_ids = self.stage_id_map[decoder_ids.reshape(-1)]
        if (stage_ids < 0).any():
            raise ValueError("Decoder inputs contain ids that do not map to any DECOR codebook stage.")

        candidate_bins = self.fixed_bins[stage_ids]
        candidates = self.fused_embedding[candidate_bins].to(e_ctx.dtype)
        ctx = self.q_ctx(e_ctx).unsqueeze(1).expand(-1, seq_len - 1, -1).reshape(-1, e_ctx.shape[-1])
        attn_scores = torch.matmul(
            ctx.unsqueeze(1),
            self.k_candidates(candidates).transpose(-2, -1),
        ).squeeze(1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
        e_soft = (attn_weights.unsqueeze(-1) * candidates).sum(dim=1).reshape(batch_size, seq_len - 1, -1)
        decoder_embeddings = torch.cat([bos_vec.unsqueeze(1), e_soft], dim=1)
        e_final = self.alpha * decoder_embeddings + (1 - self.alpha) * e_fused
        return e_final, entropy


class DECOR(T5ForConditionalGeneration):
    def __init__(
        self,
        config: T5Config,
        tokenizer=None,
        model_temperature=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(config)
        self.model_temperature = model_temperature
        self.latest_loss_breakdown = {}
        self.config.num_decoder_layers = len(self.decoder.block)

        promptformer_config = {
            "t5_dim": config.d_model,
            "codebook_path": getattr(config, "codebook_path", ""),
            "alpha": getattr(config, "alpha", 0.5),
            "bos_queries": getattr(config, "bos_queries", 16),
            "latent_size": getattr(config, "decor_latent_size", config.d_model),
            "rq_n_codebooks": getattr(config, "decor_rq_n_codebooks", 3),
            "rq_codebook_size": getattr(config, "decor_rq_codebook_size", 256),
            "collision_size": getattr(config, "decor_collision_size", 256),
        }
        self.fusion_module = PromptFormer(promptformer_config)

        self._tied_weights_keys = [
            key for key in self._tied_weights_keys if key != "lm_head.weight"
        ]
        self.shared.requires_grad_(False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.register_buffer(
            "tokenizer_codebook_map",
            torch.full((config.vocab_size,), -1, dtype=torch.long),
        )
        if tokenizer is not None:
            self._init_tokenizer_codebook_map(tokenizer)

    @staticmethod
    def _parse_rq_token(token: str):
        if not (token.startswith("<") and token.endswith(">")):
            return None
        inner = token[1:-1]
        if "_" not in inner:
            return None
        stage_tag, code_tag = inner.rsplit("_", 1)
        if not code_tag.isdigit():
            return None
        stage_tag = stage_tag.lower()
        if len(stage_tag) == 1 and "a" <= stage_tag <= "z":
            stage_idx = ord(stage_tag) - ord("a")
        elif stage_tag.isdigit():
            stage_idx = int(stage_tag)
        else:
            return None
        return stage_idx, int(code_tag)

    def _init_tokenizer_codebook_map(self, tokenizer):
        mapping = torch.full_like(self.tokenizer_codebook_map, -1)
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id < mapping.numel():
            mapping[tokenizer.pad_token_id] = 0
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id < mapping.numel():
            mapping[tokenizer.eos_token_id] = self.fusion_module.eos_id

        offsets = [0]
        for stage_size in self.fusion_module.stage_sizes[:-1]:
            offsets.append(offsets[-1] + stage_size)

        for token, token_id in tokenizer.get_vocab().items():
            parsed = self._parse_rq_token(token)
            if parsed is None:
                continue
            stage_idx, code_idx = parsed
            if stage_idx >= len(self.fusion_module.stage_sizes):
                continue
            if code_idx >= self.fusion_module.stage_sizes[stage_idx]:
                continue
            mapping[token_id] = 1 + offsets[stage_idx] + code_idx

        self.tokenizer_codebook_map.copy_(mapping)

    def _map_tokenizer_ids_to_codebook(self, input_ids: torch.Tensor) -> torch.Tensor:
        codebook_ids = self.tokenizer_codebook_map[input_ids]
        if (codebook_ids < 0).any():
            raise ValueError(
                "DECOR encountered tokenizer ids without a codebook mapping. "
                "Use SID-only sequences or ensure all custom tokens are registered."
            )
        return codebook_ids

    def _project_hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            hidden_states = hidden_states * (self.model_dim ** -0.5)
        return self.lm_head(hidden_states)

    def ranking_loss(self, lm_logits, labels):
        logits = lm_logits / self.model_temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(lm_logits.device)
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def _forward_ar(
        self,
        input_ids=None,
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
        decor_fused_embeds=None,
    ):
        codebook_input_ids = None
        if encoder_outputs is None:
            if inputs_embeds is None:
                codebook_input_ids = self._map_tokenizer_ids_to_codebook(input_ids)
                decor_fused_embeds = self.fusion_module(codebook_input_ids)
            else:
                decor_fused_embeds = inputs_embeds
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=decor_fused_embeds,
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

        if decor_fused_embeds is None:
            if codebook_input_ids is None:
                codebook_input_ids = self._map_tokenizer_ids_to_codebook(input_ids)
            decor_fused_embeds = self.fusion_module(codebook_input_ids)

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            codebook_decoder_ids = self._map_tokenizer_ids_to_codebook(decoder_input_ids)
            decoder_inputs_embeds, _ = self.fusion_module.get_decoder_embedding(
                input_ids=codebook_decoder_ids,
                fused_embeds=decor_fused_embeds,
                encoder_attention_mask=attention_mask,
                encoder_outputs=hidden_states,
            )
            decoder_input_ids = None

        decoder_outputs = self.decoder(
            input_ids=None,
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
        lm_logits = self._project_hidden_to_logits(sequence_output)
        loss = None
        if labels is not None:
            loss = self.ranking_loss(lm_logits, labels)
            self.latest_loss_breakdown = {
                "ranking_loss": loss.detach(),
                "total_loss": loss.detach(),
            }

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": None,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": False,
            "decor_fused_embeds": kwargs.get("decor_fused_embeds"),
        }

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        model_kwargs = None
        if len(args) >= 2 and isinstance(args[1], dict):
            model_kwargs = args[1]
        elif isinstance(kwargs.get("model_kwargs"), dict):
            model_kwargs = kwargs["model_kwargs"]
        if model_kwargs is None:
            return super()._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

        input_ids = args[0] if args and torch.is_tensor(args[0]) else model_kwargs.get("input_ids")
        attention_mask = model_kwargs.get("attention_mask")
        codebook_input_ids = self._map_tokenizer_ids_to_codebook(input_ids)
        fused_embeds = self.fusion_module(codebook_input_ids)
        encoder_outputs = self.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=fused_embeds,
            return_dict=True,
        )
        prepared_kwargs = dict(model_kwargs)
        prepared_kwargs["encoder_outputs"] = encoder_outputs
        prepared_kwargs["decor_fused_embeds"] = fused_embeds
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
        maskgit_target_ids=None,
        decor_fused_embeds=None,
        **kwargs,
    ):
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self._forward_ar(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decor_fused_embeds=decor_fused_embeds,
        )
