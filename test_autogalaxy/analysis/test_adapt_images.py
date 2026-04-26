import pytest
import numpy as np

import autofit as af
import autogalaxy as ag


def test__instance_with_associated_adapt_images_from(masked_imaging_7x7):
    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=1.0)

    galaxy_image_dict = {
        g0: ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        g1: ag.Array2D.full(fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0),
    }

    adapt_images = ag.AdaptImages(
        galaxy_image_dict=galaxy_image_dict,
    )

    assert adapt_images.model_image.native == pytest.approx(
        3.0 * np.ones((3, 3)), 1.0e-4
    )


def test__image_for_galaxy__resolves_after_galaxy_identity_changes():
    """
    Simulates the post-``jax.jit`` unflatten boundary: ``adapt_images.galaxy_image_dict`` is keyed by the
    trace-time ``Galaxy`` instances, but the lookup at ``GalaxiesToInversion.mapper_galaxy_dict`` is performed
    against fresh ``Galaxy`` objects whose ``__hash__`` differs. The path-tuple lookup via
    ``galaxy_name_image_dict`` must still resolve to the right adapt image.
    """
    galaxies = af.ModelInstance()
    galaxies.lens = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_name_image_dict = {
        str(("galaxies", "lens")): ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        str(("galaxies", "source")): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    trace_galaxies = [galaxies.lens, galaxies.source]

    adapt_images = ag.AdaptImages(
        galaxy_name_image_dict=galaxy_name_image_dict,
    ).updated_via_instance_from(instance=instance, galaxies=trace_galaxies)

    assert adapt_images.galaxy_path_list == [
        str(("galaxies", "lens")),
        str(("galaxies", "source")),
    ]

    # Fast path: by-instance lookup still works for the trace-time galaxies.
    assert adapt_images.image_for_galaxy(
        trace_galaxies[0], trace_galaxies
    ).native == pytest.approx(np.ones((3, 3)), 1.0e-4)

    # Simulate post-unflatten: fresh ``Galaxy`` objects with new ``.id`` values
    # placed at the same positions as the trace-time list. ``galaxy_image_dict``
    # cannot resolve them (hash mismatch) so the helper must fall back to
    # ``galaxy_name_image_dict`` via ``galaxy_path_list``.
    fresh_galaxies = [ag.Galaxy(redshift=0.5), ag.Galaxy(redshift=1.0)]

    assert adapt_images.galaxy_image_dict.get(fresh_galaxies[0]) is None
    assert adapt_images.image_for_galaxy(
        fresh_galaxies[0], fresh_galaxies
    ).native == pytest.approx(np.ones((3, 3)), 1.0e-4)
    assert adapt_images.image_for_galaxy(
        fresh_galaxies[1], fresh_galaxies
    ).native == pytest.approx(2.0 * np.ones((3, 3)), 1.0e-4)


def test__image_plane_mesh_grid_for_galaxy__resolves_after_galaxy_identity_changes():
    """
    Companion to :func:`test__image_for_galaxy__resolves_after_galaxy_identity_changes` for the mesh-grid
    lookup path used by ``GalaxiesToInversion.image_plane_mesh_grid_list``.
    """
    galaxies = af.ModelInstance()
    galaxies.lens = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_name_image_plane_mesh_grid_dict = {
        str(("galaxies", "lens")): ag.Grid2DIrregular(values=[(3.0, 3.0), (3.0, 3.0)]),
        str(("galaxies", "source")): ag.Grid2DIrregular(values=[(4.0, 4.0), (4.0, 4.0)]),
    }

    trace_galaxies = [galaxies.lens, galaxies.source]

    adapt_images = ag.AdaptImages(
        galaxy_name_image_plane_mesh_grid_dict=galaxy_name_image_plane_mesh_grid_dict,
    ).updated_via_instance_from(instance=instance, galaxies=trace_galaxies)

    fresh_galaxies = [ag.Galaxy(redshift=0.5), ag.Galaxy(redshift=1.0)]

    assert adapt_images.image_plane_mesh_grid_for_galaxy(
        fresh_galaxies[0], fresh_galaxies
    ) == pytest.approx(3.0 * np.ones((2, 2)), 1.0e-4)
    assert adapt_images.image_plane_mesh_grid_for_galaxy(
        fresh_galaxies[1], fresh_galaxies
    ) == pytest.approx(4.0 * np.ones((2, 2)), 1.0e-4)
