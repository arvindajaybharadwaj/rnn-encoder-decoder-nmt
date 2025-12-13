import torch 
from sklearn.model_selection import train_test_split
from src.dataset import CustomEuroParlDataset
from src.dataset import collate_fn
from torch.utils.data import DataLoader

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

def get_dataloaders(batch_size, train_ratio=0.9, val_ratio=0.05, random_state=42):
    # loading the dataset
    enc_in, dec_in, dec_tgt = load_processed_data()

    #splitting data
    split_dataset = split_data(enc_in, dec_in, dec_tgt, train_ratio=train_ratio, val_ratio=val_ratio, random_state=random_state)

    train_enc_in, val_enc_in, test_enc_in = split_dataset[0], split_dataset[1], split_dataset[2]
    train_dec_in, val_dec_in, test_dec_in = split_dataset[3], split_dataset[4], split_dataset[5]
    train_dec_tgt, val_dec_tgt, test_dec_tgt = split_dataset[6], split_dataset[7], split_dataset[8]

    # creating the datasets
    train_dataset = CustomEuroParlDataset(train_enc_in, train_dec_in, train_dec_tgt)
    val_dataset = CustomEuroParlDataset(val_enc_in, val_dec_in, val_dec_tgt)
    test_dataset = CustomEuroParlDataset(test_enc_in, test_dec_in, test_dec_tgt)

    # creating the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # returning the loaders
    return (train_loader, val_loader, test_loader)