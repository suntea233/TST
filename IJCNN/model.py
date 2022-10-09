from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizerFast,BertModel,BartModel
from transformers.modeling_outputs import Seq2SeqModelOutput

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 768
units = (2048,d_model)
vocab_size = 50265
numofblock = 2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = BartModel.from_pretrained(r"facebook/bart-base").to(DEVICE)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            )

        sequence_output = outputs.encoder_last_hidden_state[ : , 0 , : ]
        return outputs.encoder_last_hidden_state,sequence_output


class FC(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.layer1 = nn.Linear(input_shape,units[0])
        self.layer2 = nn.Linear(units[0],units[1])
        self.relu = nn.ReLU()
        self.LayerNormalization = nn.LayerNorm(d_model)


    def forward(self,x):
        outputs = self.layer1(x)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs += x
        outputs = self.LayerNormalization(outputs)
        return outputs.to(DEVICE)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.bart = BartModel.from_pretrained(r"facebook/bart-base")
        self.decoder = self.bart.decoder.to(DEVICE)
        self.logits = nn.Linear(self.bart.config.d_model,self.bart.shared.num_embeddings).to(DEVICE)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        encoder_hidden_states = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        outputs = outputs.last_hidden_state

        logits = self.logits(outputs)

        logits = logits.view(-1, self.bart.config.vocab_size)

        return logits