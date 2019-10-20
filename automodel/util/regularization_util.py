import numpy as np
from autoarray import decorator_util


@decorator_util.jit()
def constant_regularization_matrix_from_pixel_neighbors(
    coefficient, pixel_neighbors, pixel_neighbors_size
):
    """From the pixel-neighbors, setup the regularization matrix using the constant regularization scheme.

    Parameters
    ----------
    coefficients : tuple
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    pixel_neighbors : ndarray
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : ndarrayy
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    pixels = len(pixel_neighbors)

    regularization_matrix = np.zeros(shape=(pixels, pixels))

    regularization_coefficient = coefficient ** 2.0

    for i in range(pixels):
        regularization_matrix[i, i] += 1e-8
        for j in range(pixel_neighbors_size[i]):
            neighbor_index = pixel_neighbors[i, j]
            regularization_matrix[i, i] += regularization_coefficient
            regularization_matrix[i, neighbor_index] -= regularization_coefficient

    return regularization_matrix


def adaptive_regularization_weights_from_pixel_signals(
    inner_coefficient, outer_coefficient, pixel_signals
):
    """Compute the regularization weights, which are the effective regularization coefficient of every \
    pixel. They are computed using the (hyper) pixel-signal of each pixel.

    Two regularization coefficients are used, corresponding to the:

    1) (pixel_signals) - pixels with a high pixel-signal (i.e. where the signal is located in the pixelization).
    2) (1.0 - pixel_signals) - pixels with a low pixel-signal (i.e. where the signal is not located in the \
     pixelization).

    Parameters
    ----------
    coefficients : (float, float)
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    pixel_signals : ndarray
        The estimated signal in every pixelization pixel, used to change the regularization weighting of high signal \
        and low signal pixelizations.
    """
    return (
        inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)
    ) ** 2.0


@decorator_util.jit()
def weighted_regularization_matrix_from_pixel_neighbors(
    regularization_weights, pixel_neighbors, pixel_neighbors_size
):
    """From the pixel-neighbors, setup the regularization matrix using the weighted regularization scheme.

    Parameters
    ----------
    regularization_weights : ndarray
        The regularization_ weight of each pixel, which governs how much smoothing is applied to that individual pixel.
    pixel_neighbors : ndarray
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : ndarrayy
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    pixels = len(regularization_weights)

    regularization_matrix = np.zeros(shape=(pixels, pixels))

    regularization_weight = regularization_weights ** 2.0

    for i in range(pixels):
        regularization_matrix[i, i] += 1e-8
        for j in range(pixel_neighbors_size[i]):
            neighbor_index = pixel_neighbors[i, j]
            regularization_matrix[i, i] += regularization_weight[neighbor_index]
            regularization_matrix[
                neighbor_index, neighbor_index
            ] += regularization_weight[neighbor_index]
            regularization_matrix[i, neighbor_index] -= regularization_weight[
                neighbor_index
            ]
            regularization_matrix[neighbor_index, i] -= regularization_weight[
                neighbor_index
            ]

    return regularization_matrix
