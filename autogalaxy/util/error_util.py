import numpy as np
from autofit.non_linear.samples import quantile


def quantile_1d_profile(profile_1d_list, q, weights=None):
    """
    This function is adapted from from corner.py

    It compute sample quantiles specifically for 1D profiles of light and mass profiles. This is so that their
    errors can be plotted on 1D plots, via random draws from a PDF.

    The input to the function is a list of 1D profiles, where each entry in this list contains a list or numpy array
    of the 1D profile values. The quantile method is used on each list of radial coordinates to compute the 1D profile
    at a given input confidence (e.g. `q=0.5` gives the median 1D profile).

    Note
    ----
    When ``weight_list`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weight_list``.
    """

    radial_quantile = np.zeros(shape=profile_1d_list[0].shape[0])

    for radial_index in range(profile_1d_list[0].shape[0]):

        radial_list = [profile_1d[radial_index] for profile_1d in profile_1d_list]

        radial_quantile[radial_index] = quantile(x=radial_list, q=q, weights=weights)[0]

    return radial_quantile
