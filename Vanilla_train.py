# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom Dataset
class TransliterationDataset(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_targets):
        self.enc_inputs = torch.LongTensor(enc_inputs).to(device)
        self.dec_inputs = torch.LongTensor(dec_inputs).to(device)
        self.dec_targets = torch.FloatTensor(dec_targets).to(device)

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_targets[idx]


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, cell_type, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.cell_type = cell_type

        rnn_cls = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN
        }[cell_type]

        self.rnn = rnn_cls(embedding_size, hidden_size, num_layers, dropout=dropout, 
                           bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


# Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        rnn_cls = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN
        }[cell_type]
        self.rnn = rnn_cls(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        prediction = self.fc(output)
        return prediction, hidden


# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, n_enc_layers, n_dec_layers, cell_type):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        _, hidden = self.encoder(src)

        # Match encoder/decoder layers
        if self.n_enc_layers != self.n_dec_layers:
            if self.cell_type == "LSTM":
                hidden = (hidden[0][:self.n_dec_layers], hidden[1][:self.n_dec_layers])
            else:
                hidden = hidden[:self.n_dec_layers]

        input = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output.squeeze(1)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs


# Accuracy
def calculate_accuracy(preds, targets):
    preds = preds.argmax(dim=2)
    correct = (preds == targets).float()
    mask = (targets != 0).float()
    return (correct * mask).sum() / mask.sum()


# Training
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0
    for enc_in, dec_in, dec_out in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        output = model(enc_in, dec_in)
        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]),
                         dec_out[:, 1:].reshape(-1, dec_out.shape[-1]))
        loss.backward()
        optimizer.step()
        acc = calculate_accuracy(output[:, 1:], dec_in[:, 1:])
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(dataloader), total_acc / (2 * len(dataloader))


# Evaluation
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for enc_in, dec_in, dec_out in tqdm(dataloader, desc="Evaluating"):
            output = model(enc_in, dec_in, teacher_forcing_ratio=0)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]),
                             dec_out[:, 1:].reshape(-1, dec_out.shape[-1]))
            acc = calculate_accuracy(output[:, 1:], dec_in[:, 1:])
            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss / len(dataloader), total_acc / (2 * len(dataloader))


# Main
def main(args):
    # Dummy dataset placeholders (replace with real loading logic)
    train_enc_in = np.random.randint(1, args.input_vocab_size, (1000, args.seq_len))
    train_dec_in = np.random.randint(1, args.output_vocab_size, (1000, args.seq_len))
    train_dec_out = np.eye(args.output_vocab_size)[train_dec_in]

    val_enc_in = train_enc_in.copy()
    val_dec_in = train_dec_in.copy()
    val_dec_out = train_dec_out.copy()

    train_dataset = TransliterationDataset(train_enc_in, train_dec_in, train_dec_out)
    val_dataset = TransliterationDataset(val_enc_in, val_dec_in, val_dec_out)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    encoder = Encoder(args.input_vocab_size, args.embed_size, args.hidden_size, args.n_enc_layers,
                      args.dropout, args.rnn_type, args.bidirectional).to(device)
    decoder = Decoder(args.output_vocab_size, args.embed_size, args.hidden_size, args.n_dec_layers,
                      args.dropout, args.rnn_type).to(device)
    model = Seq2Seq(encoder, decoder, args.n_enc_layers, args.n_dec_layers, args.rnn_type).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_vocab_size", type=int, default=100)
    parser.add_argument("--output_vocab_size", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_enc_layers", type=int, default=2)
    parser.add_argument("--n_dec_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rnn_type", type=str, choices=["RNN", "GRU", "LSTM"], default="RNN")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
