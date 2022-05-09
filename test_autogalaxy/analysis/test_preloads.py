import numpy as np
from os import path

import autofit as af

import autogalaxy as ag


def test__set_blurred_image():

    # Blurred image is all zeros so preloads as zeros

    fit_0 = ag.m.MockFitImaging(blurred_image=np.zeros(2))
    fit_1 = ag.m.MockFitImaging(blurred_image=np.zeros(2))

    preloads = ag.Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_image == np.zeros(2)).all()

    # Blurred image are different, indicating the model parameters change the grid, so no preloading.

    fit_0 = ag.m.MockFitImaging(blurred_image=np.array([1.0]))
    fit_1 = ag.m.MockFitImaging(blurred_image=np.array([2.0]))

    preloads = ag.Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_image is None

    # Blurred images are the same meaning they are fixed in the model, so do preload.

    fit_0 = ag.m.MockFitImaging(blurred_image=np.array([1.0]))
    fit_1 = ag.m.MockFitImaging(blurred_image=np.array([1.0]))

    preloads = ag.Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_image == np.array([1.0])).all()


def test__info():

    file_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")

    file_preloads = path.join(file_path, "preloads.summary")

    preloads = ag.Preloads(
        blurred_image=np.zeros(3),
        w_tilde=None,
        use_w_tilde=False,
        sparse_image_plane_grid_pg_list=None,
        relocated_grid=None,
        mapper_list=None,
        operated_mapping_matrix=None,
        curvature_matrix_preload=None,
    )

    af.formatter.output_list_of_strings_to_file(
        file=file_preloads, list_of_strings=preloads.info
    )

    results = open(file_preloads)
    lines = results.readlines()

    i = 0

    assert lines[i] == f"W Tilde = False\n"
    i += 1
    assert lines[i] == f"Use W Tilde = False\n"
    i += 1
    assert lines[i] == f"\n"
    i += 1
    assert lines[i] == f"Blurred Image = False\n"
    i += 1
    assert lines[i] == f"Mapper = False\n"
    i += 1
    assert lines[i] == f"Blurred Mapping Matrix = False\n"
    i += 1
    assert lines[i] == f"Curvature Matrix Sparse = False\n"
    i += 1
    assert lines[i] == f"Curvature Matrix = False\n"
    i += 1
    assert lines[i] == f"Regularization Matrix = False\n"
    i += 1
    assert lines[i] == f"Log Det Regularization Matrix Term = False\n"
    i += 1

    preloads = ag.Preloads(
        blurred_image=1,
        w_tilde=1,
        use_w_tilde=True,
        sparse_image_plane_grid_pg_list=1,
        mapper_list=1,
        operated_mapping_matrix=1,
        curvature_matrix_preload=1,
        curvature_matrix=1,
        regularization_matrix=1,
        log_det_regularization_matrix_term=1,
    )

    af.formatter.output_list_of_strings_to_file(
        file=file_preloads, list_of_strings=preloads.info
    )

    results = open(file_preloads)
    lines = results.readlines()

    i = 0

    assert lines[i] == f"W Tilde = True\n"
    i += 1
    assert lines[i] == f"Use W Tilde = True\n"
    i += 1
    assert lines[i] == f"\n"
    i += 1
    assert lines[i] == f"Blurred Image = True\n"
    i += 1
    assert lines[i] == f"Mapper = True\n"
    i += 1
    assert lines[i] == f"Blurred Mapping Matrix = True\n"
    i += 1
    assert lines[i] == f"Curvature Matrix Sparse = True\n"
    i += 1
    assert lines[i] == f"Curvature Matrix = True\n"
    i += 1
    assert lines[i] == f"Regularization Matrix = True\n"
    i += 1
    assert lines[i] == f"Log Det Regularization Matrix Term = True\n"
    i += 1
