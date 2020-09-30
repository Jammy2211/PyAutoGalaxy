from pathlib import Path

import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag
from autoarray import FitInterferometer
from autoconf import conf
from autogalaxy.pipeline.phase.dataset import PhaseDataset
from autogalaxy.plot.lensing_plotters import Include
from test_autogalaxy import mock


############
# AutoGalaxy #
############

# Lens Datasets #


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return aa.Kernel.ones(shape_2d=(3, 3), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7(
        interferometer_7, visibilities_mask_7x2, mask_7x7, sub_grid_7x7, transformer_7x7_7
):
    return aa.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        settings=aa.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=aa.TransformerDFT
        ),
    )


@pytest.fixture(name="rectangular_pixelization_grid_3x3")
def make_rectangular_pixelization_grid_3x3(grid_7x7):
    return aa.GridRectangular.overlay_grid(grid=grid_7x7, shape_2d=(3, 3))


@pytest.fixture(name="rectangular_inversion_7x7_3x3")
def make_rectangular_inversion_7x7_3x3(masked_imaging_7x7, rectangular_mapper_7x7_3x3):
    regularization = aa.reg.Constant(coefficient=1.0)

    return aa.Inversion(
        masked_dataset=masked_imaging_7x7,
        mapper=rectangular_mapper_7x7_3x3,
        regularization=regularization,
    )


@pytest.fixture(name="voronoi_inversion_9_3x3")
def make_voronoi_inversion_9_3x3(masked_imaging_7x7, voronoi_mapper_9_3x3):
    regularization = aa.reg.Constant(coefficient=1.0)
    return aa.Inversion(
        masked_dataset=masked_imaging_7x7,
        mapper=voronoi_mapper_9_3x3,
        regularization=regularization,
    )


@pytest.fixture(name="rectangular_mapper_7x7_3x3")
def make_rectangular_mapper_7x7_3x3(grid_7x7, rectangular_pixelization_grid_3x3):
    return aa.Mapper(grid=grid_7x7, pixelization_grid=rectangular_pixelization_grid_3x3)


@pytest.fixture(name="fit_interferometer_7")
def make_masked_interferometer_fit_x1_plane_7(masked_interferometer_7):
    fit_interferometer = FitInterferometer(
        masked_interferometer=masked_interferometer_7,
        model_visibilities=5.0 * masked_interferometer_7.visibilities,
    )
    fit_interferometer.masked_dataset = masked_interferometer_7
    return fit_interferometer


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return aa.Array.full(fill_value=2.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="sub_mask_7x7")
def make_sub_mask_7x7():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D.manual(mask=mask, sub_size=2, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7(image_7x7, psf_3x3, noise_map_7x7):
    return aa.Imaging(
        image=image_7x7, psf=psf_3x3, noise_map=noise_map_7x7, name="mock_imaging_7x7"
    )


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return aa.Array.ones(shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return ag.MaskedImaging(
        imaging=imaging_7x7,
        mask=sub_mask_7x7,
        settings=ag.SettingsMaskedImaging(sub_size=1),
    )


@pytest.fixture(name="interferometer_7")
def make_interferometer_7(visibilities_7x2, noise_map_7x2, uv_wavelengths_7x2):
    return aa.Interferometer(
        visibilities=visibilities_7x2,
        noise_map=noise_map_7x2,
        uv_wavelengths=uv_wavelengths_7x2,
    )


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)


@pytest.fixture(name="convolver_7x7")
def make_convolver_7x7(mask_7x7, blurring_mask_7x7, psf_3x3):
    return aa.Convolver(mask=mask_7x7, kernel=psf_3x3)


@pytest.fixture(name="mask_7x7_1_pix")
def make_mask_7x7_1_pix():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="grid_iterate_7x7")
def make_grid_iterate_7x7(mask_7x7):
    return aa.GridIterate.from_mask(
        mask=mask_7x7, fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    )


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7(uv_wavelengths_7x2, mask_7x7):
    return aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2, real_space_mask=mask_7x7
    )


@pytest.fixture(name="positions_7x7")
def make_positions_7x7():
    return aa.GridCoordinates(coordinates=[[(0.1, 0.1), (0.2, 0.2)], [(0.3, 0.3)]])


@pytest.fixture(name="visibilities_mask_7x2")
def make_visibilities_mask_7x2():
    return np.full(fill_value=False, shape=(7, 2))


@pytest.fixture(name="grid_7x7")
def make_grid_7x7(mask_7x7):
    return aa.Grid.from_mask(mask=mask_7x7)


@pytest.fixture(name="sub_grid_7x7")
def make_sub_grid_7x7(sub_mask_7x7):
    return aa.Grid.from_mask(mask=sub_mask_7x7)


@pytest.fixture(name="sub_grid_7x7_simple")
def make_sub_grid_7x7_simple(mask_7x7, sub_grid_7x7):
    sub_grid_7x7[0] = np.array([1.0, 1.0])
    sub_grid_7x7[1] = np.array([1.0, 0.0])
    sub_grid_7x7[2] = np.array([1.0, 1.0])
    sub_grid_7x7[3] = np.array([1.0, 0.0])
    return sub_grid_7x7


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7(
        interferometer_7, mask_7x7, visibilities_mask_7x2, sub_grid_7x7
):
    return ag.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        settings=ag.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=ag.TransformerDFT
        ),
    )


@pytest.fixture(name="masked_interferometer_7_lop")
def make_masked_interferometer_7_lop(
        interferometer_7, mask_7x7, visibilities_mask_7x2, sub_grid_7x7
):
    return ag.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        settings=ag.SettingsMaskedInterferometer(transformer_class=ag.TransformerNUFFT),
    )


#
# MODEL #
#

# PROFILES #


@pytest.fixture(name="lp_0")
def make_lp_0():
    # noinspection PyTypeChecker
    return ag.lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)


@pytest.fixture(name="lp_1")
def make_lp_1():
    # noinspection PyTypeChecker
    return ag.lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)


@pytest.fixture(name="mp_0")
def make_mp_0():
    # noinspection PyTypeChecker
    return ag.mp.SphericalIsothermal(einstein_radius=1.0)


@pytest.fixture(name="mp_1")
def make_mp_1():
    # noinspection PyTypeChecker
    return ag.mp.SphericalIsothermal(einstein_radius=2.0)


@pytest.fixture(name="lmp_0")
def make_lmp_0():
    return ag.lmp.EllipticalSersicRadialGradient()


@pytest.fixture(name="dmp_0")
def make_dmp_0():
    # noinspection PyTypeChecker
    return ag.mp.SphericalNFW(kappa_s=1.0)


@pytest.fixture(name="dmp_1")
def make_dmp_1():
    # noinspection PyTypeChecker
    return ag.mp.SphericalNFW(kappa_s=2.0)


@pytest.fixture(name="smp_0")
def make_smp_0():
    # noinspection PyTypeChecker
    return ag.lmp.EllipticalSersic(intensity=1.0, mass_to_light_ratio=1.0)


@pytest.fixture(name="smp_1")
def make_smp_1():
    # noinspection PyTypeChecker
    return ag.lmp.EllipticalSersic(intensity=2.0, mass_to_light_ratio=2.0)


# GALAXY #


@pytest.fixture(name="gal_x1_lp")
def make_gal_x1_lp(lp_0):
    return ag.Galaxy(redshift=0.5, light_profile_0=lp_0)


@pytest.fixture(name="gal_x2_lp")
def make_gal_x2_lp(lp_0, lp_1):
    return ag.Galaxy(redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1)


@pytest.fixture(name="gal_x1_mp")
def make_gal_x1_mp(mp_0):
    return ag.Galaxy(redshift=0.5, mass_profile_0=mp_0)


@pytest.fixture(name="gal_x2_mp")
def make_gal_x2_mp(mp_0, mp_1):
    return ag.Galaxy(redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1)


@pytest.fixture(name="gal_x1_lp_x1_mp")
def make_gal_x1_lp_x1_mp(lp_0, mp_0):
    return ag.Galaxy(redshift=0.5, light_profile_0=lp_0, mass_profile_0=mp_0)


@pytest.fixture(name="hyper_galaxy")
def make_hyper_galaxy():
    return ag.HyperGalaxy(noise_factor=1.0, noise_power=1.0, contribution_factor=1.0)


# Plane #


@pytest.fixture(name="plane_7x7")
def make_plane_7x7(gal_x1_lp_x1_mp):
    return ag.Plane(galaxies=[gal_x1_lp_x1_mp])


@pytest.fixture(name="plane_x2_galaxy_inversion_7x7")
def make_plane_x2_galaxy_inversion_7x7(gal_x1_lp, gal_x1_mp):
    source_gal_inversion = ag.Galaxy(
        redshift=1.0,
        pixelization=ag.pix.Rectangular(),
        regularization=ag.reg.Constant(),
    )

    return ag.Plane(galaxies=[gal_x1_lp, source_gal_inversion])


# GALAXY DATA #


@pytest.fixture(name="gal_data_7x7")
def make_gal_data_7x7(image_7x7, noise_map_7x7):
    return ag.GalaxyData(
        image=image_7x7, noise_map=noise_map_7x7, pixel_scales=image_7x7.pixel_scales
    )


@pytest.fixture(name="gal_fit_data_7x7_image")
def make_gal_fit_data_7x7_image(gal_data_7x7, sub_mask_7x7):
    return ag.MaskedGalaxyDataset(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
    )


@pytest.fixture(name="gal_fit_data_7x7_convergence")
def make_gal_fit_data_7x7_convergence(gal_data_7x7, sub_mask_7x7):
    return ag.MaskedGalaxyDataset(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_convergence=True
    )


@pytest.fixture(name="gal_fit_data_7x7_potential")
def make_gal_fit_data_7x7_potential(gal_data_7x7, sub_mask_7x7):
    return ag.MaskedGalaxyDataset(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_potential=True
    )


@pytest.fixture(name="gal_fit_data_7x7_deflections_y")
def make_gal_fit_data_7x7_deflections_y(gal_data_7x7, sub_mask_7x7):
    return ag.MaskedGalaxyDataset(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_y=True
    )


@pytest.fixture(name="gal_fit_data_7x7_deflections_x")
def make_gal_fit_data_7x7_deflections_x(gal_data_7x7, sub_mask_7x7):
    return ag.MaskedGalaxyDataset(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_x=True
    )


# GALAXY FIT #


@pytest.fixture(name="gal_fit_7x7_image")
def make_gal_fit_7x7_image(gal_fit_data_7x7_image, gal_x1_lp):
    return ag.FitGalaxy(
        masked_galaxy_dataset=gal_fit_data_7x7_image, model_galaxies=[gal_x1_lp]
    )


@pytest.fixture(name="gal_fit_7x7_convergence")
def make_gal_fit_7x7_convergence(gal_fit_data_7x7_convergence, gal_x1_mp):
    return ag.FitGalaxy(
        masked_galaxy_dataset=gal_fit_data_7x7_convergence, model_galaxies=[gal_x1_mp]
    )


@pytest.fixture(name="gal_fit_7x7_potential")
def make_gal_fit_7x7_potential(gal_fit_data_7x7_potential, gal_x1_mp):
    return ag.FitGalaxy(
        masked_galaxy_dataset=gal_fit_data_7x7_potential, model_galaxies=[gal_x1_mp]
    )


@pytest.fixture(name="gal_fit_7x7_deflections_y")
def make_gal_fit_7x7_deflections_y(gal_fit_data_7x7_deflections_y, gal_x1_mp):
    return ag.FitGalaxy(
        masked_galaxy_dataset=gal_fit_data_7x7_deflections_y, model_galaxies=[gal_x1_mp]
    )


@pytest.fixture(name="gal_fit_7x7_deflections_x")
def make_gal_fit_7x7_deflections_x(gal_fit_data_7x7_deflections_x, gal_x1_mp):
    return ag.FitGalaxy(
        masked_galaxy_dataset=gal_fit_data_7x7_deflections_x, model_galaxies=[gal_x1_mp]
    )


### HYPER GALAXIES ###


@pytest.fixture(name="hyper_model_image_7x7")
def make_hyper_model_image_7x7(mask_7x7):
    return ag.Array.manual_mask(
        np.full(fill_value=5.0, shape=mask_7x7.pixels_in_mask), mask=mask_7x7
    )


@pytest.fixture(name="hyper_galaxy_image_0_7x7")
def make_hyper_galaxy_image_0_7x7(mask_7x7):
    return ag.Array.manual_mask(
        np.full(fill_value=2.0, shape=mask_7x7.pixels_in_mask), mask=mask_7x7
    )


@pytest.fixture(name="hyper_galaxy_image_1_7x7")
def make_hyper_galaxy_image_1_7x7(mask_7x7):
    return ag.Array.manual_mask(
        np.full(fill_value=3.0, shape=mask_7x7.pixels_in_mask), mask=mask_7x7
    )


@pytest.fixture(name="hyper_galaxy_image_path_dict_7x7")
def make_hyper_galaxy_image_path_dict_7x7(
        hyper_galaxy_image_0_7x7, hyper_galaxy_image_1_7x7
):
    hyper_galaxy_image_path_dict = {}

    hyper_galaxy_image_path_dict[("g0",)] = hyper_galaxy_image_0_7x7
    hyper_galaxy_image_path_dict[("g1",)] = hyper_galaxy_image_1_7x7

    return hyper_galaxy_image_path_dict


@pytest.fixture(name="contribution_map_7x7")
def make_contribution_map_7x7(
        hyper_model_image_7x7, hyper_galaxy_image_0_7x7, hyper_galaxy
):
    return hyper_galaxy.contribution_map_from_hyper_images(
        hyper_model_image=hyper_model_image_7x7,
        hyper_galaxy_image=hyper_galaxy_image_0_7x7,
    )


@pytest.fixture(name="hyper_noise_map_7x7")
def make_hyper_noise_map_7x7(
        masked_imaging_fit_x2_plane_7x7, contribution_map_7x7, hyper_galaxy
):
    hyper_noise = hyper_galaxy.hyper_noise_map_from_contribution_map(
        noise_map=masked_imaging_fit_x2_plane_7x7.noise_map,
        contribution_map=contribution_map_7x7,
    )
    return masked_imaging_fit_x2_plane_7x7.noise_map + hyper_noise


### FITS ###


@pytest.fixture(name="masked_imaging_fit_7x7")
def make_masked_imaging_fit_7x7(masked_imaging_7x7, plane_7x7):
    return ag.FitImaging(masked_imaging=masked_imaging_7x7, plane=plane_7x7)


@pytest.fixture(name="masked_imaging_fit_x2_galaxy_7x7")
def make_masked_imaging_fit_x2_galaxy_7x7(masked_imaging_7x7, gal_x1_lp):
    plane = ag.Plane(galaxies=[gal_x1_lp, gal_x1_lp])

    return ag.FitImaging(masked_imaging=masked_imaging_7x7, plane=plane)


@pytest.fixture(name="masked_imaging_fit_x2_galaxy_inversion_7x7")
def make_masked_imaging_fit_x2_galaxy_inversion_7x7(
        masked_imaging_7x7, plane_x2_galaxy_inversion_7x7
):
    return ag.FitImaging(
        masked_imaging=masked_imaging_7x7, plane=plane_x2_galaxy_inversion_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_7x7")
def make_masked_interferometer_fit_7x7(masked_interferometer_7, plane_7x7):
    return ag.FitInterferometer(
        masked_interferometer=masked_interferometer_7, plane=plane_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x2_galaxy_inversion_7x7")
def make_masked_interferometer_fit_x2_galaxy_inversion_7x7(
        masked_interferometer_7, plane_x2_galaxy_inversion_7x7
):
    return ag.FitInterferometer(
        masked_interferometer=masked_interferometer_7,
        plane=plane_x2_galaxy_inversion_7x7,
    )


@pytest.fixture(name="samples_with_result")
def make_samples_with_result():
    galaxies = [
        ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)),
        ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic(intensity=2.0)),
    ]

    plane = ag.Plane(galaxies=galaxies)

    return mock.MockSamples(max_log_likelihood_instance=plane)


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data(mask_7x7):
    return PhaseDataset(
        settings=ag.SettingsPhaseImaging(),
        search=mock.MockSearch(
            phase_name="test_phase",
        ),
    )


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7():
    return ag.PhaseImaging(
        search=mock.MockSearch(
            phase_name="test_phase",
        )
    )


@pytest.fixture(name="phase_interferometer_7")
def make_phase_interferometer_7(mask_7x7):
    return ag.PhaseInterferometer(
        search=mock.MockSearch(
            phase_name="test_phase",
        ),
        real_space_mask=mask_7x7
    )


@pytest.fixture(name="voronoi_pixelization_grid_9")
def make_voronoi_pixelization_grid_9(grid_7x7):
    grid_9 = aa.Grid.manual_1d(
        grid=[
            [0.6, -0.3],
            [0.5, -0.8],
            [0.2, 0.1],
            [0.0, 0.5],
            [-0.3, -0.8],
            [-0.6, -0.5],
            [-0.4, -1.1],
            [-1.2, 0.8],
            [-1.5, 0.9],
        ],
        shape_2d=(3, 3),
        pixel_scales=1.0,
    )
    return aa.GridVoronoi(
        grid=grid_9,
        nearest_pixelization_1d_index_for_mask_1d_index=np.zeros(
            shape=grid_7x7.shape_1d, dtype="int"
        ),
    )


@pytest.fixture(name="voronoi_mapper_9_3x3")
def make_voronoi_mapper_9_3x3(grid_7x7, voronoi_pixelization_grid_9):
    return aa.Mapper(grid=grid_7x7, pixelization_grid=voronoi_pixelization_grid_9)


# PLOTTING #


@pytest.fixture(name="include_all")
def make_include_all():
    conf.instance = conf.Config.for_directory(
        Path(__file__).parent.parent
    )
    return Include(
        origin=True,
        mask=True,
        grid=True,
        border=True,
        positions=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        critical_curves=True,
        caustics=True,
        multiple_images=True,
        inversion_pixelization_grid=True,
        inversion_grid=True,
        inversion_border=True,
        inversion_image_pixelization_grid=True,
        preloaded_critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
        preload_caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
    )
