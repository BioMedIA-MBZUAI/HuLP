import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix

from pycox.models import loss as pycox_loss
from pycox.models.data import pair_rank_mat
from pycox.models.utils import pad_col, cumsum_reverse

# https://docs.monai.io/en/0.3.0/transforms.html
import monai
from monai.transforms import (
    MapTransform,
)

def create_multiclass_dict(data_train, one_hot_cols):
    i = 0
    multiclass_dict = {}
    for col in one_hot_cols:
        # num_unique_vals = data_train[col].nunique()
        num_unique_vals = len([i for i in data_train.columns if col in i])
        
        # TODO: make more dynamic to account for unordered columns
        multiclass_dict.update({col: slice(i, i + num_unique_vals)})
        i += num_unique_vals

    return multiclass_dict

########################### DEEPHIT ###########################
# modified from https://github.com/havakv/pycox/blob/master/pycox/models/deephit.py

def deephit_encode_survival(time, event, bins) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """

    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0]), dtype=torch.float, device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if bin_idx >= bins.shape[0]:
            bin_idx = bins.shape[0] - 1
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def predict_surv_df(input, duration_index, loss_type="deephit"):
    # TODO: duration_index is currently not being used. Need to interpolate duration_index and return as DataFrame index
    """Predict the survival function for `input`, i.e., survive all of the event types,
    and return as a pandas DataFrame.
    See `prediction_surv_df` to return a DataFrame instead.

    Arguments:
        input {tuple, np.ndarra, or torch.tensor} -- Input to net.
    
    Keyword Arguments:
        batch_size {int} -- Batch size (default: {8224})
        eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
        num_workers {int} -- Number of workes in created dataloader (default: {0})
    
    Returns:
        pd.DataFrame -- Predictions
    """
    if "deephit" in loss_type:
        loss_type = "deephit"
    elif "mtlr" in loss_type:
        loss_type = "mtlr"
    surv = predict_surv(input, False, loss_type)
    surv = pd.DataFrame(surv).T
    return surv

def predict_surv(input, numpy=None, loss_type="deephit"):
    # SURVIVAL FUNCTION
    """Predict the survival function for `input`, i.e., survive all of the event types.
    See `prediction_surv_df` to return a DataFrame instead.

    Arguments:
        input {tuple, np.ndarra, or torch.tensor} -- Input to net.

    Keyword Arguments:
        batch_size {int} -- Batch size (default: {8224})
        numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
            (default: {None})
        eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
        to_cpu {bool} -- For larger data sets we need to move the results to cpu
            (default: {False})
        num_workers {int} -- Number of workes in created dataloader (default: {0})

    Returns:
        [TupleTree, np.ndarray or tensor] -- Predictions
    """
    cif = predict_cif(input, False, loss_type)
    surv = torch.exp(-cif)
    
    if type(surv) == monai.data.meta_tensor.MetaTensor:
        surv = surv.as_tensor().cpu().numpy()
    else:
        surv = surv.cpu().numpy()

    return surv

def predict_cif(input, numpy=None, loss_type="deephit"):
    # CUMULATIVE HAZARD FUNCTION
    """Predict the cumulative incidence function (cif) for `input`.

    Arguments:
        input {tuple, np.ndarray, or torch.tensor} -- Input to net.

    Keyword Arguments:
        batch_size {int} -- Batch size (default: {8224})
        numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
            (default: {None})
        eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
        to_cpu {bool} -- For larger data sets we need to move the results to cpu
            (default: {False})
        num_workers {int} -- Number of workers in created dataloader (default: {0})

    Returns:
        [np.ndarray or tensor] -- Predictions
    """
    pmf = interpolate_predict_pmf(input, 10, 'const_pdf', None, loss_type)
    cif = pmf.cumsum(1)
    return cif  

def interpolate_predict_pmf(input, sub=10, scheme='const_pdf', duration_index=None, loss_type="deephit"):
    if not scheme in ['const_pdf', 'lin_surv']:
        raise NotImplementedError
    pmf = predict_pmf(input, False, loss_type)
    n, m = pmf.shape
    pmf_cdi = pmf[:, 1:].contiguous().view(-1, 1).repeat(1, sub).div(sub).view(n, -1)
    pmf_cdi = pad_col(pmf_cdi, where='start')
    pmf_cdi[:, 0] = pmf[:, 0]
    return pmf_cdi

def predict_pmf(preds, numpy=None, loss_type="deephit"):
    # HAZARD FUNCTION
    """Predict the probability mass fuction (PMF) for `input`.

    Arguments:
        input {tuple, np.ndarray, or torch.tensor} -- Input to net.

    Keyword Arguments:
        batch_size {int} -- Batch size (default: {8224})
        numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
            (default: {None})
        eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
        grads {bool} -- If gradients should be computed (default: {False})
        to_cpu {bool} -- For larger data sets we need to move the results to cpu
            (default: {False})
        num_workers {int} -- Number of workers in created dataloader (default: {0})

    Returns:
        [np.ndarray or tensor] -- Predictions
    """
    if loss_type == "deephit":
        pass
    elif loss_type == "mtlr":
        preds = cumsum_reverse(preds, dim=1)
    pmf = pad_col(preds).softmax(1)[:, :-1]
    return pmf 


########################### DEEPHIT ###########################


########################### CORN PROG LOSS ###########################
def prognosis_ranking_loss(logits, y_train, y_events, num_classes, device):
    """
    rename num_classes to num_bins
    rename y_train to y_time
    """
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.

    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the bin labels.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    # 1. Separate censored vs. uncensored
    censored = y_events == 0
    y_censored = y_train[censored]
    y_uncensored = y_train[~censored]
    logits_censored = logits.as_tensor()[censored]
    logits_uncensored = logits.as_tensor()[~censored]

    # 2. Expand censored
    y_censored_expanded = torch.tensor([], dtype=torch.int64, device=device)
    logits_censored_expanded = torch.tensor([], dtype=torch.float32, device=device)
    for idx, bin_num in enumerate(y_censored):
        bin_num = int(bin_num.cpu().numpy())
        for expanded_class in range(bin_num, num_classes):
            y_censored_expanded = torch.cat(
                (
                    y_censored_expanded,
                    torch.tensor([expanded_class], dtype=torch.int64, device=device),
                ),
                dim=0,
            )
            logits_censored_expanded = torch.cat(
                (logits_censored_expanded, logits_censored[idx][None, ...]), dim=0
            )

    # 3. Combine censored and uncensored
    y_train = torch.cat((y_censored_expanded, y_uncensored), dim=0)
    logits = torch.cat((logits_censored_expanded, logits_uncensored), dim=0)

    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > i - 1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.0
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(
            F.logsigmoid(pred) * train_labels
            + (F.logsigmoid(pred) - pred) * (1 - train_labels)
        )
        losses += loss

    return losses / num_examples


########################### CORN PROG LOSS ###########################


def compute_metric_at_times(metric, time_true, prob_pred, event_observed, score_times):
    """Helper function to evaluate a metric at given timepoints."""
    scores = []
    for time, pred in zip(score_times, prob_pred.T):
        target = time_true > time
        uncensored = target | event_observed.astype(bool)
        try:
            scores.append(metric(target[uncensored], pred[uncensored]))
        except:
            scores.append(-1)
    return scores


def brier_score_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(
        brier_score_loss, time_true, prob_pred, event_observed, score_times
    )
    return scores


def roc_auc_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(
        roc_auc_score, time_true, prob_pred, event_observed, score_times
    )
    return scores


# Assuming `labels` and `outputs` are numpy arrays or PyTorch tensors
# You may need to convert them to numpy arrays using `.cpu().numpy()` if they are PyTorch tensors
def compute_metrics(outputs, labels):
    # AUC
    auc = roc_auc_score(
        labels.argmax(axis=1), outputs.softmax(dim=1)[:, 1].cpu().numpy()
    )

    # Confusion Matrix
    cm = confusion_matrix(labels.argmax(axis=1), outputs.argmax(dim=1).cpu().numpy())

    # Sensitivity (True Positive Rate or Recall)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # Specificity (True Negative Rate)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    # Balanced Accuracy
    balanced_accuracy = (sensitivity + specificity) / 2

    weighted_score = (
        0.4 * auc + 0.2 * sensitivity + 0.2 * specificity + 0.2 * balanced_accuracy
    )

    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Weighted Score: {weighted_score:.4f}")

    return auc, sensitivity, specificity, balanced_accuracy, weighted_score


def deephit_loss(scores, labels, censors):
    rank_mat = pair_rank_mat(labels.cpu().numpy(), censors.cpu().numpy())
    rank_mat = torch.from_numpy(rank_mat)
    rank_mat = rank_mat.to("cuda")
    loss_single = pycox_loss.DeepHitSingleLoss(0.2, 0.1)
    loss = loss_single(scores, labels, censors.to(torch.int64), rank_mat)
    return loss

def deepmtlr_loss(phi, idx_durations, events, reduction: str = 'mean',
         epsilon: float = 1e-7):
    """phi: scores/preds
    idx_durations: time/labels
    events: censor"""
    phi = cumsum_reverse(phi, dim=1)

    return nll_pmf(phi, idx_durations, events, reduction, epsilon)
             
def nll_pmf(phi, idx_durations, events, reduction: str = 'mean',
            epsilon: float = 1e-7):
    """Negative log-likelihood for the PMF parametrized model [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                        f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                        f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = pad_col(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return _reduction(loss, reduction)

def _reduction(loss, reduction: str = 'mean'):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")
