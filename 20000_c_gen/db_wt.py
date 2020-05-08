import numpy as np
import scipy.signal

def db_filters(k):
    """Returns Daubechies filter coefficients
    Args:
        k: integer between [1, 5] corresponding to which db transform to use
    Returns:
        [h0, h1, g0, g1]: lpa, hpa, lps, hps filters
    """
    c = {
            1 : np.array([
                [0.707106781186548,   0.707106781186548,   0.707106781186548,  -0.707106781186548],
                [0.707106781186548,  -0.707106781186548,   0.707106781186548,   0.707106781186548]
                ]),
            2 : np.array([
                [0.482962913144534,  -0.129409522551260,  -0.129409522551260,  -0.482962913144534],
                [0.836516303737808,  -0.224143868042013,   0.224143868042013,   0.836516303737808],
                [0.224143868042013,   0.836516303737808,   0.836516303737808,  -0.224143868042013],
                [-0.129409522551260,  -0.482962913144534,   0.482962913144534,  -0.129409522551260]
                ]),
            3 : np.array([
                [0.332670552950083,   0.035226291885710,   0.035226291885710,  -0.332670552950083],
                [0.806891509311093,  0.085441273882027,  -0.085441273882027,   0.806891509311093],
                [0.459877502118491,  -0.135011020010255,  -0.135011020010255,  -0.459877502118491],
                [-0.135011020010255,  -0.459877502118491,   0.459877502118491,  -0.135011020010255],
                [-0.085441273882027,   0.806891509311093,  0.806891509311093,   0.085441273882027],
                [0.035226291885710,  -0.332670552950083,   0.332670552950083,   0.035226291885710]
                ]),
            4 : np.array([
                [0.230377813308896,  -0.010597401785069,  -0.010597401785069,  -0.230377813308896],
                [0.714846570552915,  -0.032883011666885,   0.032883011666885,   0.714846570552915],
                [0.630880767929859,   0.030841381835560,   0.030841381835560,  -0.630880767929859],
                [-0.027983769416859,   0.187034811719093,  -0.187034811719093,  -0.027983769416859],
                [-0.187034811719093,  -0.027983769416859,  -0.027983769416859,   0.187034811719093],
                [0.030841381835560,  -0.630880767929859,   0.630880767929859,   0.030841381835560],
                [0.032883011666885,   0.714846570552915,   0.714846570552915,  -0.032883011666885],
                [-0.010597401785069,  -0.230377813308896,   0.230377813308896,  -0.010597401785069]
                ]),
            5 : np.array([
                [0.160102397974193,   0.003335725285474,   0.003335725285474,  -0.160102397974193],
                [0.603829269797189,   0.012580751999082,  -0.012580751999082,   0.603829269797189],
                [0.724308528437772,  -0.006241490212798,  -0.006241490212798,  -0.724308528437772],
                [0.138428145901320,  -0.077571493840046,   0.077571493840046,   0.138428145901320],
                [-0.242294887066382,  -0.032244869584638,  -0.032244869584638,   0.242294887066382],
                [-0.032244869584638,   0.242294887066382,  -0.242294887066382,  -0.032244869584638],
                [0.077571493840046,   0.138428145901320,   0.138428145901320,  -0.077571493840046],
                [-0.006241490212798,  -0.724308528437772,   0.724308528437772,  -0.006241490212798],
                [-0.012580751999082,   0.603829269797189,   0.603829269797189,   0.012580751999082],
                [0.003335725285474,  -0.160102397974193,   0.160102397974193,   0.003335725285474]
                ])
    }[k]
    
    h0 = c[:, 0]    #% low-pass analysis filter
    h1 = c[:, 1]    #% high-pass analysis filter
    g0 = c[:, 2]    #% low-pass synthesis filter
    g1 = c[:, 3]    #% high-pass synthesis filter

    return [h0, h1, g0, g1]


def analysis_fb(h0, h1, x):
    """Analysis filter bank
    Args:
        h0: low pass analysis filter
        h1: high pass analysis filter
        x: 1D data signal to be transformed
    Returns:
        [c, d]: coarse and detail coefficients
    """
    # low pass filter and downsample
    c = scipy.signal.upfirdn(h0, x, down=2)

    # high pass filter and downsample
    d = scipy.signal.upfirdn(h1, x, down=2)

    return [c, d]


def synthesis_fb(g0, g1, c, d):
    """Synthesis filter bank
    Args:
        g0: low pass synthesis filter
        g1: high pass synthesis filter
        c: coarse coefficients
        d: list of detail coefficients
    Returns:
        x_: reconstructed signal
    """
    if ((len(d) % 2 != 0) and (len(d) != len(c))):
        c = np.hstack((c, np.array([0])))

    # upsample and filter
    y_0 = scipy.signal.upfirdn(g0, c, up=2)

    # upsample and filter
    y_1 = scipy.signal.upfirdn(g1, d, up=2)

    # reconstruct signal
    y = y_0 + y_1

    # convolution adds length of the filter - 1 to the signal
    zrs = len(g0) - 1

    # reconstruct original signal
    x_ = y[zrs:-zrs]

    return x_


def forward(x, m, k):
    """Forward wavelet transform of a 1D signal
    Args:
        x: data signal to be transformed
        m: number of levels in transform
        k: which daubechies transform to use
    Returns:
        [c, d]: coarse and detail coefficients
    """

    # find analysis and synthesis coefficients... we'll only use analysis
    [h0, h1, g0, g1] = db_filters(k)

    # c = x at the first level
    c = x
    d = []

    # analysis (forward transform)
    for i in range(m):
        [c, d_n] = analysis_fb(h0, h1, c)
        d.append(d_n)

    return [c, d]


def inverse(c, d, k):
    """Inverse wavelet transform of a 1D signal
    Args:
        c: coarse coefficients
        d: detail coefficients
        k: which daubechies transform to use
    Returns:
        c: reconstructed signal
    """

    m = 1

    # find analysis and synthesis coefficients... we'll only use analysis
    [h0, h1, g0, g1] = db_filters(k)

    # synthesis (inverse transform)
    for i in range(m-1, -1, -1):
        c = synthesis_fb(g0, g1, c, d)

    return c
