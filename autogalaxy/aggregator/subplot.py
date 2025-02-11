from enum import Enum

class SubplotDataset(Enum):
    """
    The subplots that can be extracted from the subplot_fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    Data = (0, 0)
    DataLog10 = (1, 0)
    NoiseMap = (2, 0)
    PSF = (0, 1)
    PSFLog10 = (1, 1)
    SignalToNoiseMap = (2, 1)
    OverSampleSizeLp = (0, 2)
    OverSampleSizePixelization = (1, 2)

class SubplotFit(Enum):
    """
    The subplots that can be extracted from the subplot_fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    Data = (0, 0)
    SignalToNoiseMap = (1, 0)
    ModelImage = (2, 0)
    NormalizedResidualMap = (0, 1)
    NormalizedResidualMapOneSigma = (1, 1)
    ChiSquaredMap = (2, 1)