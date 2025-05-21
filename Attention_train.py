import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import wandb
from model import Encoder, Decoder, Seq2Seq, TransliterationDataset  # Assumes model.py contains your model classes
from utils import load_vocabularies, load_data  # Custom utils for loading vocab and data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for enc_in, dec_in, dec_out in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        output = model(enc_in, dec_in)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        targets = dec_out[:, 1:].argmax(dim=2).reshape(-1)

        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        preds = output.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return epoch_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for enc_in, dec_in, dec_out in tqdm(dataloader, desc="Evaluating"):
            output = model(enc_in, dec_in, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            targets = dec_out[:, 1:].argmax(dim=2).reshape(-1)

            loss = criterion(output, targets)
            epoch_loss += loss.item()
            preds = output.argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return epoch_loss / len(dataloader), correct / total


def main(args):
    wandb.init(project="transliteration-attention", config=args)
    config = wandb.config

    input_vocab, output_vocab = load_vocabularies(args.vocab_path)
    train_data, val_data = load_data(
        args.train_path, args.val_path, input_vocab, output_vocab, args.max_len
    )

    train_dataset = TransliterationDataset(*train_data)
    val_dataset = TransliterationDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    encoder = Encoder(
        input_size=len(input_vocab),
        embedding_size=config.embedding_dim,
        hidden_size=config.hidden_dim,
        num_layers=config.enc_layers,
        dropout=config.dropout,
        cell_type=config.rnn_type,
        bidirectional=config.bidirectional
    ).to(device)

    decoder = Decoder(
        output_size=len(output_vocab),
        embedding_size=config.embedding_dim,
        hidden_size=config.hidden_dim,
        num_layers=config.dec_layers,
        dropout=config.dropout,
        cell_type=config.rnn_type,
        enc_hid_dim=config.hidden_dim
    ).to(device)

    model = Seq2Seq(encoder, decoder, config.enc_layers, config.dec_layers, config.rnn_type).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Acc = {train_acc:.4f} | Val Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--rnn_type", type=str, choices=["RNN", "GRU", "LSTM"], default="LSTM")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")

    args = parser.parse_args()
    main(args)