from autoarray.fixtures import *

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt


def make_masked_imaging_7x7():
    imaging_7x7 = make_imaging_7x7()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_7x7_sub_2():
    imaging_7x7 = make_imaging_7x7_sub_2()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_covariance_7x7():
    imaging_7x7 = make_imaging_covariance_7x7()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


# PROFILES #


def make_ps_0():
    # noinspection PyTypeChecker
    return ag.ps.PointFlux(flux=0.0)


def make_ps_1():
    # noinspection PyTypeChecker
    return ag.ps.PointFlux(flux=1.0)


def make_lp_0():
    # noinspection PyTypeChecker
    return ag.lp.SersicSph(intensity=1.0, effective_radius=2.0, sersic_index=2.0)


def make_lp_1():
    # noinspection PyTypeChecker
    return ag.lp.SersicSph(intensity=2.0, effective_radius=2.0, sersic_index=2.0)


def make_lp_linear_0():
    # noinspection PyTypeChecker
    return ag.lp_linear.Gaussian()


def make_lp_operated_0():
    # noinspection PyTypeChecker
    return ag.lp_operated.Gaussian(intensity=1.0)


def make_mp_0():
    # noinspection PyTypeChecker
    return ag.mp.IsothermalSph(einstein_radius=1.0)


def make_mp_1():
    # noinspection PyTypeChecker
    return ag.mp.IsothermalSph(einstein_radius=2.0)


def make_lmp_0():
    return ag.lmp.SersicRadialGradient()


def make_dmp_0():
    # noinspection PyTypeChecker
    return ag.mp.NFWSph(kappa_s=1.0)


def make_dmp_1():
    # noinspection PyTypeChecker
    return ag.mp.NFWSph(kappa_s=2.0)


def make_smp_0():
    # noinspection PyTypeChecker
    return ag.lmp.Sersic(intensity=1.0, mass_to_light_ratio=1.0)


def make_smp_1():
    # noinspection PyTypeChecker
    return ag.lmp.Sersic(intensity=2.0, mass_to_light_ratio=2.0)


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


# Galaxies #


def make_galaxies_7x7():
    return ag.Galaxies(galaxies=[make_gal_x1_lp_x1_mp()])


def make_galaxies_x2_7x7():
    return ag.Galaxies(galaxies=[make_gal_x1_lp_x1_mp(), make_gal_x1_lp_x1_mp()])


def make_galaxies_x2_inversion_7x7():
    source_gal_inversion = ag.Galaxy(
        redshift=1.0,
        pixelization=ag.mesh.Rectangular(),
        regularization=ag.reg.Constant(),
    )

    return [make_gal_x1_lp(), source_gal_inversion]


# COSMOLOGY #


def make_Planck15():
    return ag.cosmo.Planck15()


# QUANTITY DATASET AND FIT #


def make_dataset_quantity_7x7_array_2d():
    return ag.DatasetQuantity(
        data=aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        noise_map=aa.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )


def make_dataset_quantity_7x7_vector_yx_2d():
    return ag.DatasetQuantity(
        data=aa.VectorYX2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        noise_map=aa.VectorYX2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )


def make_fit_quantity_7x7_array_2d():
    return ag.FitQuantity(
        dataset=make_dataset_quantity_7x7_array_2d(),
        light_mass_obj=make_galaxies_7x7(),
        func_str="convergence_2d_from",
    )


def make_fit_quantity_7x7_vector_yx_2d():
    return ag.FitQuantity(
        dataset=make_dataset_quantity_7x7_vector_yx_2d(),
        light_mass_obj=make_galaxies_7x7(),
        func_str="deflections_yx_2d_from",
    )


# galaxies #


def make_adapt_galaxy_name_image_dict_7x7():
    image_0 = ag.Array2D(
        np.full(fill_value=2.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )

    image_1 = ag.Array2D(
        np.full(fill_value=3.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )

    adapt_galaxy_name_image_dict = {
        str(("galaxies", "g0")): image_0,
        str(("galaxies", "g1")): image_1,
    }

    return adapt_galaxy_name_image_dict


def make_adapt_images_7x7():
    return ag.AdaptImages(
        galaxy_name_image_dict=make_adapt_galaxy_name_image_dict_7x7(),
    )


def make_fit_imaging_x2_galaxy_7x7():
    return ag.FitImaging(
        dataset=make_masked_imaging_7x7(),
        galaxies=[make_gal_x1_lp(), make_gal_x1_lp(), make_gal_x1_mp()],
    )


def make_fit_imaging_x2_galaxy_inversion_7x7():
    return ag.FitImaging(
        dataset=make_masked_imaging_7x7(), galaxies=make_galaxies_x2_inversion_7x7()
    )


def make_fit_interferometer_7x7():
    return ag.FitInterferometer(
        dataset=make_interferometer_7(),
        galaxies=make_galaxies_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )


def make_fit_interferometer_x2_galaxy_inversion_7x7():
    return ag.FitInterferometer(
        dataset=make_interferometer_7(),
        galaxies=make_galaxies_x2_inversion_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )


def make_samples_summary_with_result():
    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=af.Model(ag.lp.Sersic))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    instance = model.instance_from_prior_medians()

    return ag.m.MockSamplesSummary(max_log_likelihood_instance=instance)


def make_analysis_imaging_7x7():
    analysis = ag.AnalysisImaging(
        dataset=make_masked_imaging_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )
    analysis._adapt_images = make_adapt_images_7x7()
    return analysis


def make_analysis_interferometer_7():
    analysis = ag.AnalysisInterferometer(
        dataset=make_interferometer_7(),
    )
    analysis._adapt_images = make_adapt_images_7x7()
    return analysis


def make_include_1d_all():
    return aplt.Include1D(half_light_radius=True)


def make_include_2d_all():
    return aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        positions=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        tangential_critical_curves=False,
        radial_critical_curves=False,
        tangential_caustics=False,
        radial_caustics=False,
        multiple_images=False,
        mapper_source_plane_mesh_grid=True,
        mapper_image_plane_mesh_grid=True,
    )
