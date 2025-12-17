import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.model import Encoder, Decoder, Seq2Seq
from src.data_utils import get_dataloaders

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_CHECKPOINT = None

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# defining all the hyperparameters
SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PADDING_IDX = 0

encoder = Encoder(SRC_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, PADDING_IDX)
decoder = Decoder(TGT_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, PADDING_IDX)

model = Seq2Seq(encoder, decoder).to(device)

# loss function is cross entropy loss (it should be padding aware)
# ignores index 0 which represents <pad>
criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)

optimizer = optim.Adam(model.parameters(), lr=LR)

train_loader, val_loader, _ = get_dataloaders(BATCH_SIZE)

start_epoch = 0

if RESUME_CHECKPOINT is not None:
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resumed from epoch {start_epoch}")

# training loop
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0

    for enc_in, dec_in, dec_tgt, enc_lens, _ in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"
    ):
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)
        enc_lens = enc_lens.to(device)

        optimizer.zero_grad()

        logits = model(enc_in, enc_lens, dec_in)

        logits = logits.view(-1, logits.size(-1))
        dec_tgt = dec_tgt.view(-1)

        loss = criterion(logits, dec_tgt)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for enc_in, dec_in, dec_tgt, enc_lens, _ in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"
        ):
            enc_in = enc_in.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)
            enc_lens = enc_lens.to(device)

            logits = model(enc_in, enc_lens, dec_in)
            logits = logits.view(-1, logits.size(-1))
            dec_tgt = dec_tgt.view(-1)

            loss = criterion(logits, dec_tgt)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # saving the model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        checkpoint_path
    )

    print(f"Saved checkpoint: {checkpoint_path}")
