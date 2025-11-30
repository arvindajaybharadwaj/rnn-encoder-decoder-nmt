import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# loading the processed data
enc_in = torch.load("../data/processed/enc_in.pt")
dec_in = torch.load("../data/processed/dec_in.pt")
dec_tgt = torch.load("../data/processed/dec_tgt.pt")

# split the data into train, val, and test sets
train_enc_in, test_enc_main, train_dec_in, test_dec_in_main, train_dec_tgt, test_dec_tgt_main = train_test_split(enc_in, dec_in, dec_tgt, test_size=0.1, random_state=42)
val_enc_in, test_enc_in, val_dec_in,  test_dec_in, val_dec_tgt, test_dec_tgt = train_test_split(test_enc_main, test_dec_in_main, test_dec_tgt_main, test_size=0.5, random_state=42)
