import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


class LETTER(T5ForConditionalGeneration):
    def __init__(
            self,
            config: T5Config,
            model_temperature=1.0,
            *args,
            **kwargs,
    ):
        super().__init__(config)
        self.model_temperature = model_temperature
        self.use_maskgit = False
        self.latest_loss_breakdown = {}
        self.config.num_decoder_layers = len(self.decoder.block)

    def _project_hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            hidden_states = hidden_states * (self.model_dim ** -0.5)
        return self.lm_head(hidden_states)

    def ranking_loss(self, lm_logits, labels):
        t_logits = lm_logits / self.model_temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(lm_logits.device)
        return loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))

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
            history_item_ids=None,
    ):
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

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)
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
            **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
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
            history_item_ids=history_item_ids,
        )
