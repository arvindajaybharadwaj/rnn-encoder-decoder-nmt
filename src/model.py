import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        # the encoder block has only 2 layers: embedding and a gru/lstm cell
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)

        # gru layer
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, enc_in, enc_lengths):
        embedded = self.embedding(enc_in)

        # telling model to ignore the <pad> token
        packed = pack_padded_sequence(embedded, enc_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)

        # context vector is the last layer of the hidden states
        context = hidden[-1]

        return context

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, dec_in, hidden):
        embedded = self.embedding(dec_in)
        outputs, hidden = self.gru(embedded, hidden)
        logits = self.fc_out(outputs)

        return logits, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_in, enc_lengths, dec_input):
        # encode the sentence into a context vector
        context = self.encoder(enc_in, enc_lengths)

        # initial hidden state of decoder is context vector
        hidden_0 = context.unsqueeze(0)

        # getting target sentence
        # dec_input is passed for teacher forcing
        logits, _ = self.decoder(dec_input, hidden_0)

        return logits