import torch
import torch.nn as nn
import torch.nn.functional as F


class End2End(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            vocab_size: int,
            blank=0,
    ):
        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank = blank
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, stride=1, kernel_size=11)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=11)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=11)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.dropout = nn.Dropout(0.5)

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(2 * hidden_size, vocab_size)

        self.ctc = nn.CTCLoss(self.blank)

    def lift(self, speech: torch.Tensor):
        """
        :param speech (batch, sequence, nfft)
        """
        ret = self.conv1(speech)
        ret = self.pool1(ret)
        ret = F.relu(ret)
        ret = self.conv2(ret)
        ret = self.pool2(ret)
        ret = F.relu(ret)
        ret = self.conv3(ret)
        ret = self.pool3(ret)
        ret = F.relu(ret)

        ret = self.dropout(ret)  # (batch, channels, time convolved, nfft convolved)
        ret = ret.permute(0, 2, 1, 3)  # (batch, time convolved, channels, nfft convolved)
        ret = torch.flatten(ret, 2)  # (batch, time convolved, channels * nfft convolved)

        return ret

    def forward(
            self,
            speech: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        """
        :param speech: (batch, sequence, 512)
        :param text: (batch, sequence)
        :param text_lengths: (batch, )
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # check that batch_size is unified
        assert (speech.shape[0] == text.shape[0] == text_lengths.shape[0]), \
            (speech.shape, text.shape, text_lengths.shape)

        max_target_len = speech.shape[1]
        batch_size = speech.shape[0]

        lifters = self.lift(speech)
        lengths = torch.full((batch_size, max_target_len), lifters.shape[1])

        out, _ = self.gru(lifters)  # out: (batch, time convolved, 2 * hidden_size)
        pred = self.linear(out)  # (batch, time convolved, vocab_size)
        return pred, lengths

    def loss(self, pred: torch.Tensor, y: torch.Tensor, pred_lengths: torch.Tensor, y_lengths: torch.Tensor):
        """
        :param pred (batch, time convolved, vocab_size)
        :param y (batch, sequence)
        :param pred_lengths (batch, )
        :param y_lengths (batch, )
        """
        assert torch.max(y) < self.vocab_size
        pred = F.log_softmax(pred, dim=-1)
        pred = pred.permute(1, 0, 2)  # (time convolved, batch, vocab_size)
        return self.ctc(pred, pred_lengths, y, y_lengths)
