import autogalaxy as ag
from autoarray.mock.fixtures import *
from autofit.mock.mock_search import MockSamples, MockSearch
from autogalaxy.plot.mat_wrap.lensing_include import Include1D, Include2D


def make_masked_imaging_7x7():

    imaging_7x7 = make_imaging_7x7()

    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=make_sub_mask_2d_7x7())

    return masked_imaging_7x7.apply_settings(settings=ag.SettingsImaging(sub_size=1))


#
# MODEL #
#

# PROFILES #


def make_ps_0():
    # noinspection PyTypeChecker
    return ag.ps.PointFlux(flux=0.0)


def make_ps_1():
    # noinspection PyTypeChecker
    return ag.ps.PointFlux(flux=1.0)


def make_lp_0():
    # noinspection PyTypeChecker
    return ag.lp.SphSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)


def make_lp_1():
    # noinspection PyTypeChecker
    return ag.lp.SphSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)


def make_mp_0():
    # noinspection PyTypeChecker
    return ag.mp.SphIsothermal(einstein_radius=1.0)


def make_mp_1():
    # noinspection PyTypeChecker
    return ag.mp.SphIsothermal(einstein_radius=2.0)


def make_lmp_0():
    return ag.lmp.EllSersicRadialGradient()


def make_dmp_0():
    # noinspection PyTypeChecker
    return ag.mp.SphNFW(kappa_s=1.0)


def make_dmp_1():
    # noinspection PyTypeChecker
    return ag.mp.SphNFW(kappa_s=2.0)


def make_smp_0():
    # noinspection PyTypeChecker
    return ag.lmp.EllSersic(intensity=1.0, mass_to_light_ratio=1.0)


def make_smp_1():
    # noinspection PyTypeChecker
    return ag.lmp.EllSersic(intensity=2.0, mass_to_light_ratio=2.0)


# GALAXY #


def make_gal_x1_lp():
    return ag.Galaxy(redshift=0.5, light_profile_0=make_lp_0())


def make_gal_x2_lp():
    return ag.Galaxy(
        redshift=0.5, light_profile_0=make_lp_0(), light_profile_1=make_lp_1()
    )


def make_gal_x1_mp():
    return ag.Galaxy(redshift=0.5, mass_profile_0=make_mp_0())


def make_gal_x2_mp():
    return ag.Galaxy(
        redshift=0.5, mass_profile_0=make_mp_0(), mass_profile_1=make_mp_1()
    )


def make_gal_x1_lp_x1_mp():
    return ag.Galaxy(
        redshift=0.5, light_profile_0=make_lp_0(), mass_profile_0=make_mp_0()
    )


def make_hyper_galaxy():
    return ag.HyperGalaxy(noise_factor=1.0, noise_power=1.0, contribution_factor=1.0)


# Plane #


def make_plane_7x7():
    return ag.Plane(galaxies=[make_gal_x1_lp_x1_mp()])


def make_plane_x2_galaxy_inversion_7x7():
    source_gal_inversion = ag.Galaxy(
        redshift=1.0,
        pixelization=ag.pix.Rectangular(),
        regularization=ag.reg.Constant(),
    )

    return ag.Plane(galaxies=[make_gal_x1_lp(), source_gal_inversion])


# GALAXY DATA #


def make_gal_data_7x7():
    return ag.GalaxyData(
        image=make_image_7x7(),
        noise_map=make_noise_map_7x7(),
        pixel_scales=make_image_7x7().pixel_scales,
    )


def make_gal_fit_data_7x7_image():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(), mask=make_sub_mask_2d_7x7(), use_image=True
    )


def make_gal_fit_data_7x7_convergence():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(),
        mask=make_sub_mask_2d_7x7(),
        use_convergence=True,
    )


def make_gal_fit_data_7x7_potential():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(), mask=make_sub_mask_2d_7x7(), use_potential=True
    )


def make_gal_fit_data_7x7_deflections_y():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(),
        mask=make_sub_mask_2d_7x7(),
        use_deflections_y=True,
    )


def make_gal_fit_data_7x7_deflections_x():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(),
        mask=make_sub_mask_2d_7x7(),
        use_deflections_x=True,
    )


# GALAXY FIT #


def make_gal_fit_7x7_image():
    return ag.FitGalaxy(
        masked_galaxy_dataset=make_gal_fit_data_7x7_image(),
        model_galaxies=[make_gal_x1_lp()],
    )


def make_gal_fit_7x7_convergence():
    return ag.FitGalaxy(
        masked_galaxy_dataset=make_gal_fit_data_7x7_convergence(),
        model_galaxies=[make_gal_x1_mp()],
    )


def make_gal_fit_7x7_potential():
    return ag.FitGalaxy(
        masked_galaxy_dataset=make_gal_fit_data_7x7_potential(),
        model_galaxies=[make_gal_x1_mp()],
    )


def make_gal_fit_7x7_deflections_y():
    return ag.FitGalaxy(
        masked_galaxy_dataset=make_gal_fit_data_7x7_deflections_y(),
        model_galaxies=[make_gal_x1_mp()],
    )


def make_gal_fit_7x7_deflections_x():
    return ag.FitGalaxy(
        masked_galaxy_dataset=make_gal_fit_data_7x7_deflections_x(),
        model_galaxies=[make_gal_x1_mp()],
    )


# HYPER GALAXIES #


def make_hyper_model_image_7x7():
    return ag.Array2D.manual_mask(
        np.full(fill_value=5.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )


def make_hyper_galaxy_image_0_7x7():
    return ag.Array2D.manual_mask(
        np.full(fill_value=2.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )


def make_hyper_galaxy_image_path_dict_7x7():
    hyper_galaxy_image_path_dict = {
        ("g0",): make_hyper_galaxy_image_0_7x7(),
        ("g1",): make_hyper_galaxy_image_1_7x7(),
    }

    return hyper_galaxy_image_path_dict


def make_hyper_galaxy_image_1_7x7():
    return ag.Array2D.manual_mask(
        np.full(fill_value=3.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )


def make_fit_imaging_7x7():
    return ag.FitImaging(imaging=make_masked_imaging_7x7(), plane=make_plane_7x7())


def make_fit_imaging_x2_galaxy_7x7():
    plane = ag.Plane(galaxies=[make_gal_x1_lp(), make_gal_x1_lp()])

    return ag.FitImaging(imaging=make_masked_imaging_7x7(), plane=plane)


def make_fit_imaging_x2_galaxy_inversion_7x7():
    return ag.FitImaging(
        imaging=make_masked_imaging_7x7(), plane=make_plane_x2_galaxy_inversion_7x7()
    )


def make_fit_interferometer_7x7():
    return ag.FitInterferometer(
        interferometer=make_interferometer_7(), plane=make_plane_7x7()
    )


def make_fit_interferometer_x2_galaxy_inversion_7x7():
    return ag.FitInterferometer(
        interferometer=make_interferometer_7(),
        plane=make_plane_x2_galaxy_inversion_7x7(),
    )


def make_samples_with_result():
    galaxies = [
        ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=1.0)),
        ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic(intensity=2.0)),
    ]

    plane = ag.Plane(galaxies=galaxies)

    return MockSamples(max_log_likelihood_instance=plane)


def make_analysis_imaging_7x7():
    return ag.AnalysisImaging(dataset=make_masked_imaging_7x7())


def make_analysis_interferometer_7():
    return ag.AnalysisInterferometer(dataset=make_interferometer_7())


def make_include_1d_all():
    return Include1D(half_light_radius=True)


def make_include_2d_all():
    return Include2D(
        origin=True,
        mask=True,
        border=True,
        positions=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        critical_curves=False,
        caustics=False,
        multiple_images=False,
        mapper_source_pixelization_grid=True,
        mapper_data_pixelization_grid=True,
    )
