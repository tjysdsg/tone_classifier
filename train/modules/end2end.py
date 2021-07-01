import torch
import torch.nn as nn
import torch.nn.functional as F


class End2End(torch.nn.Module):
    def __init__(
            self,
            hidden_size: int,
            vocab_size: int,
            blank=0,
    ):
        super().__init__()
        self.blank = blank
        self.vocab_size = vocab_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, stride=1, kernel_size=11)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=11)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=11)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.dropout = nn.Dropout(0.5)

        # TODO: auto calc input size
        self.gru = nn.GRU(848, hidden_size, bidirectional=True, batch_first=True)

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

    def forward(self, speech: torch.Tensor):
        """
        :param speech: (batch, sequence, 512)
        """
        batch_size = speech.shape[0]

        speech = torch.unsqueeze(speech, 1)  # (batch, channel, sequence, 512)

        lifters = self.lift(speech)
        lengths = torch.full((batch_size,), lifters.shape[1])

        out, _ = self.gru(lifters)  # out: (batch, time convolved, 2 * hidden_size)
        pred = self.linear(out)  # (batch, time convolved, vocab_size)
        return pred, lengths

    def loss(self, pred: torch.Tensor, y: torch.Tensor, pred_lengths: torch.Tensor, y_lengths: torch.Tensor):
        """
        :param pred (batch, time convolved, vocab_size)
        :param y (batch, sequence, vocab_size)
        :param pred_lengths (batch, )
        :param y_lengths (batch, )
        """
        assert torch.max(y) < self.vocab_size
        pred = F.log_softmax(pred, dim=-1)
        pred = pred.permute(1, 0, 2)  # (time convolved, batch, vocab_size)
        return self.ctc(pred, y, pred_lengths, y_lengths)

    def predict(self, speech: torch.Tensor):
        from ctcdecode import CTCBeamDecoder

        pred, _ = self(speech)
        pred = F.log_softmax(pred, dim=-1)

        decoder = CTCBeamDecoder(
            '_012345',
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=40,
            num_processes=10,
            blank_id=0,
            log_probs_input=True
        )
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(pred)
        beam_results = beam_results.detach().cpu().numpy().tolist()
        batch_size = speech.shape[0]
        res = [beam_results[i][0][:out_lens[i][0]] for i in range(batch_size)]
        return res
