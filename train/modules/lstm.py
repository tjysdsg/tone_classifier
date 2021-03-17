import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMNet(nn.Module):
    def __init__(self, num_classes: int, hidden_size=512, embedding_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.5
        )
        self.fc = nn.Linear(2 * num_layers * hidden_size, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)

        out = torch.flatten(h, 1)
        out = self.fc(out)
        out = F.softmax(out, -1)
        return out
