import torch 
from sklearn.model_selection import train_test_split

def load_processed_data():
    enc_in = torch.load("../data/processed/enc_in.pt")
    dec_in = torch.load("../data/processed/dec_in.pt")
    dec_tgt = torch.load("../data/processed/dec_tgt.pt")

    return (enc_in, dec_in, dec_tgt)

def split_data(enc_in, dec_in, dec_tgt, train_ratio=0.9, val_ratio=0.05, random_state=42):
    test_main_ratio = 1 - train_ratio
    val_test_rel_ratio = val_ratio / test_main_ratio

    train_enc_in, test_enc_main, train_dec_in, test_dec_in_main, train_dec_tgt, test_dec_tgt_main = train_test_split(enc_in, dec_in, dec_tgt, test_size=test_main_ratio, random_state=random_state)
    val_enc_in, test_enc_in, val_dec_in,  test_dec_in, val_dec_tgt, test_dec_tgt = train_test_split(test_enc_main, test_dec_in_main, test_dec_tgt_main, test_size=val_test_rel_ratio, random_state=random_state)

    return (train_enc_in, val_enc_in, test_enc_in, train_dec_in, val_dec_in, test_dec_in, train_dec_tgt, val_dec_tgt, test_dec_tgt)


