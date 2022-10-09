import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 768


class Diffusion(nn.Module):
    def __init__(self, encoder, decoder,pad_id):
        super(Diffusion, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.decoder = decoder
        self.T = 1000
        self.T_cumsum = torch.tensor([i for i in range(self.T)])
        self.z_extract = nn.Sequential(nn.Linear(d_model, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, d_model),
                                       nn.ReLU()).to(DEVICE)
        self.a = nn.Parameter(torch.tensor([0.5]),requires_grad=True)
        self.pad_id = pad_id

    def forward(self, x0=None, xT=None, x0_attention_mask=None, xT_attention_mask=None, x0_output=None, xT_output=None):
        "loss = crossentropy + ?(reconstruction_loss)? + mse_loss(tilde_z,delta_z) + sentiment_classification_loss"

        """train"""

        batch_size, seq_len = x0.shape
        if xT != None:
            a = self.a.to(DEVICE)
            t = torch.randint(0, self.T, (batch_size,), device=self.DEVICE)
            first_hidden_state, first_sentiment_hidden_state = self.encoder(x0, x0_attention_mask)
            last_hidden_state, last_sentiment_hidden_state = self.encoder(xT, xT_attention_mask)

            h0_prime = first_sentiment_hidden_state
            # first_sentiment_hidden_state = first_sentiment_hidden_state.unsqueeze(1)
            # first_sentiment_hidden_state = first_sentiment_hidden_state.repeat(1, seq_len, 1)

            hT_prime = last_sentiment_hidden_state
            # last_sentiment_hidden_state = last_sentiment_hidden_state.unsqueeze(1)
            # last_sentiment_hidden_state = last_sentiment_hidden_state.repeat(1, seq_len, 1)

            first_encoder_hidden_states = (a) * F.softmax(first_sentiment_hidden_state.unsqueeze(1),dim=-1) +  (1-a) * F.softmax(first_hidden_state,dim=-1)
            last_encoder_hidden_states = (a) * F.softmax(last_sentiment_hidden_state.unsqueeze(1),dim=-1) + (1-a) * F.softmax(last_hidden_state,dim=-1)

            first_decode_logits = self.decoder(decoder_input_ids=x0, decoder_attention_mask=x0_attention_mask,
                                               encoder_hidden_states=first_encoder_hidden_states)
            last_decode_logits = self.decoder(decoder_input_ids=xT, decoder_attention_mask=xT_attention_mask,
                                              encoder_hidden_states=last_encoder_hidden_states)

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            x0_lm_loss = loss_fct(first_decode_logits, x0_output.contiguous().view(-1))
            xT_lm_loss = loss_fct(last_decode_logits, xT_output.contiguous().view(-1))

            lm_loss = x0_lm_loss + xT_lm_loss

            delta_z = ((hT_prime - h0_prime) / self.T)
            t_cumsum = self.extract(self.T_cumsum, t)
            t_cumsum = t_cumsum.reshape(batch_size, 1).float()
            t_hidden_state = ((delta_z * t_cumsum) + h0_prime)
            tilde_z = self.z_extract(t_hidden_state)

            diffusion_loss = F.mse_loss(tilde_z, delta_z)

            return lm_loss, diffusion_loss


    def sample(self, inputs, x0, inputs_attention_mask=None, x0_attention_mask=None):
        """sample"""
        a = self.a.to(DEVICE)
        first_hidden_state, first_sentiment_hidden_state = self.encoder(x0, x0_attention_mask)

        inputs_hidden_state, inputs_sentiment_hidden_state = self.encoder(inputs)
        tilde_z = self.z_extract(first_sentiment_hidden_state)

        last_sentiment_hidden_state = first_sentiment_hidden_state + (tilde_z * self.T)



        last_encoder_hidden_states = a * F.softmax(last_sentiment_hidden_state.unsqueeze(1),dim=-1) + (1-a) * F.softmax(inputs_hidden_state,dim=-1)
        #         print(last_sentiment_hidden_state)

        last_hidden_logits = self.decoder(decoder_input_ids=inputs, decoder_attention_mask=inputs_attention_mask,
                                          encoder_hidden_states=last_encoder_hidden_states)

        return last_hidden_logits

    def extract(self, a, t):
        a = a.to(self.DEVICE)
        return a.gather(-1, t)


