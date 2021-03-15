import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import scipy


def length_norm(mat):
    return mat / np.sqrt(np.sum(mat * mat, axis=1))[:, None]


def compute_eer(fnr, fpr):
    """Computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """Computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided
    to equalize the counts over score partitions (if there is such
    partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """Computes normalized minimum detection cost function (DCF) given
       the costs for false accepts and false rejects as well as a priori
       probability for target speakers
    """

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, c_det_ind = min(dcf), np.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det / c_def, c_det_ind


def compute_equalized_min_cost(labels, scores, ptar=[0.01, 0.001]):
    sorted_ndx = np.argsort(scores)
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_c = 0.
    for pt in ptar:
        tmp, idx = compute_c_norm(fnr, fpr, pt)
        min_c += tmp
    return eer * 100, min_c / len(ptar)


def score_norm(scores, enrol_scores, num_cohort):
    enrol_scores.sort()
    enrol_scores = enrol_scores[:, ::-1][:, :num_cohort]
    return (scores - enrol_scores.mean(axis=1)[:, None]) / enrol_scores.std(axis=1)[:, None] + \
           (scores - enrol_scores.mean(axis=1)[None, :]) / enrol_scores.std(axis=1)[None, :]


labels = [line.split()[0] for line in open('data/vox_test/wav.scp')]
trials_mask = np.zeros((len(labels), len(labels)))
for line in open('data/vox_test/trials'):
    ei = labels.index(line.split()[1])
    ti = labels.index(line.split()[2])
    trials_mask[ei, ti] = 1
trials_mask = trials_mask.reshape(-1).astype(bool)
labels = [line.split()[0][0:7] for line in open('data/vox_test/wav.scp')]
labels = (np.array([[labels]]).transpose() == np.array([[labels]])).reshape(-1)[trials_mask]

embd = length_norm(np.load('exp/vox12_ResNet34StatsPool-32-128_AMsoftmax-32-0.1/vox_test_wav.npy'))
scores = embd.dot(embd.T)
eer = compute_equalized_min_cost(labels, scores.reshape(-1)[trials_mask])
print("minDCF is %.3f & EER is %.2f%%" % (eer[1], eer[0]))
