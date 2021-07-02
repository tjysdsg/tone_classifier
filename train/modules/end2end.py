import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple, List
from train.utils import log_add


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

    def ctc_prefix_beam_search(
            self,
            speech: torch.Tensor,
            beam_size: int = None,
    ):
        """CTC prefix beam search inner implementation

        :param speech (batch, max_len, feat_dim)
        :param beam_size beam size for beam search

        :return List[(best_path, score)] nbest results
        """
        beam_size = beam_size or self.vocab_size
        assert beam_size <= self.vocab_size, "Beam size must be less than or equal to the vocab size"
        batch_size = speech.shape[0]
        # for CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1

        # let B = batch_size = 1 and N = beam_size
        ctc_probs, _ = self(speech)
        ctc_probs = F.log_softmax(ctc_probs, dim=-1)  # (B, maxlen, vocab_size)
        maxlen = ctc_probs.size(1)
        ctc_probs = ctc_probs.squeeze(0)

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]

        # CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # first beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        # update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add(list(x[1])),
                reverse=True
            )
            cur_hyps = next_hyps[:beam_size]
        hyps = [[y[0], log_add([y[1][0], y[1][1]])] for y in cur_hyps]
        return hyps
