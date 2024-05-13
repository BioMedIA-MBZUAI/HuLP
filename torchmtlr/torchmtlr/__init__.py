from typing import Union

import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn


class MTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1 # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")


def masked_logsumexp(x: torch.Tensor,
                     mask: torch.Tensor,
                     dim: int = -1) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask,
                  dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits: torch.Tensor,
                            target: torch.Tensor,
                            model: torch.nn.Module,
                            C1: float,
                            average: bool = False) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    return nll_total


def mtlr_survival(logits: torch.Tensor) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    G = torch.tril(torch.ones(logits.size(1),
                              logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)


def mtlr_survival_at_times(logits: torch.Tensor,
                           train_times: Union[torch.Tensor, np.ndarray],
                           pred_times: np.ndarray) -> np.ndarray:
    """Generates predicted survival curves at arbitrary timepoints using linear
    interpolation.

    Notes
    -----
    This function uses scipy.interpolate internally and returns a Numpy array,
    in contrast with `mtlr_survival`.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    train_times
        Time bins used for model training. Must have the same length as the
        first dimension of `pred`.
    pred_times
        Array of times used to compute the survival curve.

    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values
        are linearly interpolated at timepoints not used for training.
    """
    train_times = np.pad(train_times, (1, 0))
    surv = mtlr_survival(logits).detach().cpu().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(
        logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits: torch.Tensor) -> torch.Tensor:
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)


# MR ADDED
def prog_rank_survival(logits: torch.Tensor) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    logits = torch.sigmoid(logits)
    # print('sigmoid', probas)
    logits = torch.cumprod(logits, dim=1)

    G = torch.tril(torch.ones(logits.size(1),
                              logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)

def prog_rank_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(
        logits, dim=1)[:, :-1] / (prog_rank_survival(logits) + 1e-15)[:, 1:]

def prog_rank_risk(logits: torch.Tensor) -> torch.Tensor:
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = prog_rank_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)

def mtlr_survival_bin_neg_log_likelihood(logits: torch.Tensor,
                            target: torch.Tensor,
                            model: torch.nn.Module,
                            C1: float,
                            average: bool = False,
                            alpha=1.,
                            bin_penalty_type='abs',
                            penalize_censored=False) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    
    idx_logits = torch.argmax(logits, 1, keepdim=True)
    idx_target = torch.argmax(target, 1, keepdim=True)

    if bin_penalty_type == 'abs':
        bin_penalty = (idx_target-idx_logits)/(target.shape[1] - 1)
        # print('bin_penalty', bin_penalty.shape, bin_penalty)
        
        # if penalize_censored:
        #     bin_penalty_censored = bin_penalty[censored]
        #     # print('bin_penalty[censored]', bin_penalty_censored.shape, bin_penalty_censored)
        #     bin_penalty_censored = torch.where(bin_penalty_censored > 0, bin_penalty_censored, torch.Tensor([np.float32(1)]).to(logits.device))
        #     # print('bin_penalty_censored torch.where(bin_penalty_censored > 0', bin_penalty_censored.shape, bin_penalty_censored)
        #     nll_censored = masked_logsumexp(logits[censored], target[censored])
        #     # print('nll_censored', nll_censored.shape, nll_censored)
        #     nll_censored = ((1-alpha)*bin_penalty_censored+alpha*nll_censored).sum() if censored.any() else 0
        #     # print(f'after alpha {alpha} nll_censored', nll_censored.shape, nll_censored)
            
        # else:
        #     nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0

        # bin_penalty = torch.abs(bin_penalty)
        # # print('abg bin_penalty', bin_penalty.shape, bin_penalty)
        # nll_uncensored = (logits[~censored] * target[~censored]) if (~censored).any() else 0
        # # print('nll_uncensored = (logits[~censored] * target[~censored]) if (~censored).any() else 0', nll_uncensored)
        # nll_uncensored = (alpha*nll_uncensored + (1-alpha)*bin_penalty[~censored]).sum()
        # # print(f'after alpha {alpha} nll_uncensored = (alpha*nll_uncensored + (1-alpha)*bin_penalty[~censored]).sum()', nll_uncensored)
        # # print("")
        

        # MR REDO 12/17/23
        if penalize_censored:
            bin_penalty_censored = bin_penalty[censored]
            # print('bin_penalty[censored]', bin_penalty_censored.shape, bin_penalty_censored)
            bin_penalty_censored = torch.where(bin_penalty_censored > 0, bin_penalty_censored, torch.Tensor([np.float32(1)]).to(logits.device))
            bin_penalty_censored = bin_penalty_censored.sum() if censored.any() else 0
            # print('bin_penalty_censored torch.where(bin_penalty_censored > 0', bin_penalty_censored.shape, bin_penalty_censored)
            # print('nll_censored', nll_censored.shape, nll_censored)



            # nll_censored = ((1-alpha)*bin_penalty_censored+alpha*nll_censored).sum() if censored.any() else 0
            # print(f'after alpha {alpha} nll_censored', nll_censored.shape, nll_censored)
            
        # else:
            # nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0

        bin_penalty = torch.abs(bin_penalty)
        bin_penalty_uncensored = bin_penalty[~censored].sum() if (~censored).any() else 0

        total_bin_penalty = bin_penalty_censored + bin_penalty_uncensored
        print('bin_penalty_censored', bin_penalty_censored)
        print('bin_penalty_uncensored', bin_penalty_uncensored)
        print('total_bin_penalty', total_bin_penalty)

        # print('abg bin_penalty', bin_penalty.shape, bin_penalty)
        # nll_uncensored = (logits[~censored] * target[~censored]) if (~censored).any() else 0
        # print('nll_uncensored = (logits[~censored] * target[~censored]) if (~censored).any() else 0', nll_uncensored)
        # nll_uncensored = (alpha*nll_uncensored + (1-alpha)*bin_penalty[~censored]).sum()
        # print(f'after alpha {alpha} nll_uncensored = (alpha*nll_uncensored + (1-alpha)*bin_penalty[~censored]).sum()', nll_uncensored)
        # print("")
         
    elif bin_penalty_type == 'squared':
        bin_penalty = (idx_logits-idx_target)**2/(target.shape[1] - 1)**2
        nll_uncensored = (logits[~censored] * target[~censored]) if (~censored).any() else 0
        # print('nll_uncensored', nll_uncensored.sum())
        nll_uncensored = (alpha*nll_uncensored).sum() + (1-alpha)*torch.sqrt(bin_penalty[~censored].sum())
            

    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0


    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()
    # print('logits', logits.shape, logits)
    # print('norm', norm)

    nll_total = -(nll_censored + nll_uncensored - norm)
    # print('nll_total', nll_total)
    # print('nll_censored', nll_censored)
    # print('nll_uncensored', nll_uncensored)
    # print('norm', norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    # print('nll_total_FINAL', nll_total)
    # print("")
    # print('after alpha nll_total', nll_total, 'nll_censored', nll_censored, 'nll_uncensored', nll_uncensored)
    # return 0.75*nll_total + 0.25*total_bin_penalty #, nll_censored, nll_uncensored
    return 1*nll_total + 0*total_bin_penalty #, nll_censored, nll_uncensored
    # return nll_total + total_bin_penalty #, nll_censored, nll_uncensored
    # return nll_total  #, nll_censored, nll_uncensored

# def mtlr_survival_bin_neg_log_likelihood(logits: torch.Tensor,
#                             target: torch.Tensor,
#                             model: torch.nn.Module,
#                             C1: float,
#                             average: bool = False,
#                             alpha=1.) -> torch.Tensor:
#     """Computes the negative log-likelihood of a batch of model predictions.

#     Parameters
#     ----------
#     logits : torch.Tensor, shape (num_samples, num_time_bins)
#         Tensor with the time-logits (as returned by the MTLR module) for one
#         instance in each row.
#     target : torch.Tensor, shape (num_samples, num_time_bins)
#         Tensor with the encoded ground truth survival.
#     model
#         PyTorch Module with at least `MTLR` layer.
#     C1
#         The L2 regularization strength.
#     average
#         Whether to compute the average log likelihood instead of sum
#         (useful for minibatch training).

#     Returns
#     -------
#     torch.Tensor
#         The negative log likelihood.
#     """
#     censored = target.sum(dim=1) > 1
#     # print('logits', logits.shape)
    
#     idx_logits = torch.argmax(logits, 1, keepdim=True)
#     # print('idx_logits', idx_logits.shape)
#     idx_target = torch.argmax(target, 1, keepdim=True)

#     bin_penalty = torch.abs(idx_logits-idx_target)/(target.shape[1] - 1)
    
#     # nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
#     # nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0
    
#     nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
#     # nll_censored = (nll_censored * bin_penalty[censored]).sum()
#     # nll_censored = (alpha*nll_censored + (1-alpha)*bin_penalty[censored]).sum()
    
#     # alpha = 0.3
#     nll_uncensored = (logits[~censored] * target[~censored]) if (~censored).any() else 0
#     nll_uncensored = (alpha*nll_uncensored + (1-alpha)*bin_penalty[~censored]).sum()
#     # nll_uncensored = torch.logsumexp(alpha*nll_uncensored + (1-alpha)*bin_penalty[~censored], dim=1).sum()
#     # nll_uncensored = (nll_uncensored * bin_penalty[~censored]).sum()
#     # nll_uncensored = (nll_uncensored * (1+bin_penalty[~censored])).sum() # error!

#     # the normalising constant
#     norm = torch.logsumexp(logits, dim=1).sum()

#     nll_total = -(nll_censored + nll_uncensored - norm)
#     if average:
#         nll_total = nll_total / target.size(0)

#     # L2 regularization
#     for k, v in model.named_parameters():
#         if "mtlr_weight" in k:
#             nll_total += C1/2 * torch.sum(v**2)

#     return nll_total