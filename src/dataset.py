import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomEuroParlDataset(Dataset):
    def __init__(self, enc_in, dec_in, dec_tgt):
        self.enc_inputs = enc_in
        self.dec_inputs = dec_in
        self.dec_targets = dec_tgt

    def __len__(self):
        return len(self.enc_inputs)
    
    def __getitem__(self, index):
        enc_input = torch.tensor(self.enc_inputs[index], dtype=torch.int64)
        dec_input = torch.tensor(self.dec_inputs[index], dtype=torch.int64)
        dec_target = torch.tensor(self.dec_targets[index], dtype=torch.int64)
        return (enc_input, dec_input, dec_target)

def collate_fn(batch):
    enc_seqs = [item[0] for item in batch]
    dec_in_seqs = [item[1] for item in batch]
    dec_tgt_seqs = [item[2] for item in batch]

    enc_lens = torch.tensor([len(seq) for seq in enc_seqs], dtype=torch.int64)
    dec_lens = torch.tensor([len(seq) for seq in dec_in_seqs], dtype=torch.int64)

    padded_enc = pad_sequence(enc_seqs, batch_first=True, padding_value=0)
    padded_dec_in = pad_sequence(dec_in_seqs, batch_first=True, padding_value=0)
    padded_dec_tgt = pad_sequence(dec_tgt_seqs, batch_first=True, padding_value=0)

    return padded_enc, padded_dec_in, padded_dec_tgt, enc_lens, dec_lens
