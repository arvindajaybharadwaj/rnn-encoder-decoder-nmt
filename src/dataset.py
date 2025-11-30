import torch
from torch.utils.data import Dataset

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
    
