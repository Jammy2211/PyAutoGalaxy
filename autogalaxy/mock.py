import math

from astropy import constants

import autofit as af
import autogalaxy as ag
from autoarray.mock import *
from autoarray.structures import grids
from autofit.non_linear.mock.mock_search import MockSamples
from autogalaxy.pipeline.phase.dataset import PhaseDataset
from autogalaxy.plot.lensing_plotters import Include
from autofit.mock import MockSearch


# MockProfiles #


class MockLightProfile(ag.lp.LightProfile):
    def __init__(self, value, size=1):
        self.value = value
        self.size = size

    def image_from_grid(self, grid):
        return np.array(self.size * [self.value])


class MockMassProfile:
    def __init__(self, value):
        self.value = value

    def surface_density_from_grid(self, grid):
        return np.array([self.value])

    def potential_from_grid(self, grid):
        return np.array([self.value])

    def deflections_from_grid(self, grid):
        return np.array([self.value, self.value])


# Mock Galaxy #


class MockGalaxy:
    def __init__(self, value, shape=1):
        self.value = value
        self.shape = shape

    @grids.grid_like_to_structure
    def image_from_grid(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    @grids.grid_like_to_structure
    def convergence_from_grid(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    @grids.grid_like_to_structure
    def deflections_from_grid(self, grid):
        return np.full(shape=(self.shape, 2), fill_value=self.value)


class MockHyperGalaxy:
    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

    def contributions_from_model_image_and_galaxy_image(
        self, model_image, galaxy_image, minimum_value
    ):
        contributions = galaxy_image / (model_image + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def hyper_noise_from_contributions(self, noise_map, contributions):
        return self.noise_factor * (noise_map * contributions) ** self.noise_power


# Mock Cosmology #


class Value:
    def __init__(self, value):
        self.value = value

    def to(self, *args, **kwargs):
        return Value(value=self.value)


class MockCosmology:
    def __init__(
        self,
        arcsec_per_kpc=0.5,
        kpc_per_arcsec=2.0,
        critical_surface_density=2.0,
        cosmic_average_density=2.0,
    ):
        self.arcsec_per_kpc = arcsec_per_kpc
        self.kpc_per_arcsec = kpc_per_arcsec
        self.critical_surface_density = critical_surface_density
        self.cosmic_average_density = cosmic_average_density

    def arcsec_per_kpc_proper(self, z):
        return Value(value=self.arcsec_per_kpc)

    def kpc_per_arcsec_proper(self, z):
        return Value(value=self.kpc_per_arcsec)

    def angular_diameter_distance(self, z):
        return Value(value=1.0)

    def angular_diameter_distance_z1z2(self, z1, z2):
        const = constants.c.to("kpc / s") ** 2.0 / (
            4 * math.pi * constants.G.to("kpc3 / (solMass s2)")
        )
        return Value(value=self.critical_surface_density * const.value)

    def critical_density(self, z):
        return Value(value=self.cosmic_average_density)


# Mock Pipeline / Phase #


class MockResult(af.MockResult):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        search=None,
        mask=None,
        model_image=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        use_as_hyper_dataset=False,
    ):
        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.mask = mask
        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_visibilities_path_dict = hyper_galaxy_visibilities_path_dict
        self.hyper_model_visibilities = hyper_model_visibilities
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.pixelization = pixelization
        self.use_as_hyper_dataset = use_as_hyper_dataset

        self.max_log_likelihood_plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)])

    @property
    def last(self):
        return self


class MockResults(af.ResultsCollection):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        search=None,
        mask=None,
        model_image=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        use_as_hyper_dataset=False,
    ):
        """
        A collection of results from previous phases. Results can be obtained using an index or the name of the phase
        from whence they came.
        """

        super().__init__()

        result = MockResult(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
            mask=mask,
            model_image=model_image,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=hyper_model_visibilities,
            pixelization=pixelization,
            use_as_hyper_dataset=use_as_hyper_dataset,
        )

        self.__result_list = [result]

    @property
    def last(self):
        """
        The result of the last phase
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    def __getitem__(self, item):
        """
        Get the result of a previous phase by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous phase
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)


def make_masked_imaging_7x7():
    return ag.MaskedImaging(
        imaging=make_imaging_7x7(),
        mask=make_sub_mask_7x7(),
        settings=ag.SettingsMaskedImaging(sub_size=1),
    )


def make_masked_interferometer_7():
    return ag.MaskedInterferometer(
        interferometer=make_interferometer_7(),
        visibilities_mask=make_visibilities_mask_7x2(),
        real_space_mask=make_mask_7x7(),
        settings=ag.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=ag.TransformerDFT
        ),
    )


def make_masked_interferometer_7_lop():
    return ag.MaskedInterferometer(
        interferometer=make_interferometer_7(),
        visibilities_mask=make_visibilities_mask_7x2(),
        real_space_mask=make_mask_7x7(),
        settings=ag.SettingsMaskedInterferometer(transformer_class=ag.TransformerNUFFT),
    )


#
# MODEL #
#

# PROFILES #


def make_lp_0():
    # noinspection PyTypeChecker
    return ag.lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)


def make_lp_1():
    # noinspection PyTypeChecker
    return ag.lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)


def make_mp_0():
    # noinspection PyTypeChecker
    return ag.mp.SphericalIsothermal(einstein_radius=1.0)


def make_mp_1():
    # noinspection PyTypeChecker
    return ag.mp.SphericalIsothermal(einstein_radius=2.0)


def make_lmp_0():
    return ag.lmp.EllipticalSersicRadialGradient()


def make_dmp_0():
    # noinspection PyTypeChecker
    return ag.mp.SphericalNFW(kappa_s=1.0)


def make_dmp_1():
    # noinspection PyTypeChecker
    return ag.mp.SphericalNFW(kappa_s=2.0)


def make_smp_0():
    # noinspection PyTypeChecker
    return ag.lmp.EllipticalSersic(intensity=1.0, mass_to_light_ratio=1.0)


def make_smp_1():
    # noinspection PyTypeChecker
    return ag.lmp.EllipticalSersic(intensity=2.0, mass_to_light_ratio=2.0)


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
        galaxy_data=make_gal_data_7x7(), mask=make_sub_mask_7x7(), use_image=True
    )


def make_gal_fit_data_7x7_convergence():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(), mask=make_sub_mask_7x7(), use_convergence=True
    )


def make_gal_fit_data_7x7_potential():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(), mask=make_sub_mask_7x7(), use_potential=True
    )


def make_gal_fit_data_7x7_deflections_y():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(),
        mask=make_sub_mask_7x7(),
        use_deflections_y=True,
    )


def make_gal_fit_data_7x7_deflections_x():
    return ag.MaskedGalaxyDataset(
        galaxy_data=make_gal_data_7x7(),
        mask=make_sub_mask_7x7(),
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
    return ag.Array.manual_mask(
        np.full(fill_value=5.0, shape=make_mask_7x7().pixels_in_mask),
        mask=make_mask_7x7(),
    )


def make_hyper_galaxy_image_0_7x7():
    return ag.Array.manual_mask(
        np.full(fill_value=2.0, shape=make_mask_7x7().pixels_in_mask),
        mask=make_mask_7x7(),
    )


def make_hyper_galaxy_image_path_dict_7x7():
    hyper_galaxy_image_path_dict = {
        ("g0",): make_hyper_galaxy_image_0_7x7(),
        ("g1",): make_hyper_galaxy_image_1_7x7(),
    }

    return hyper_galaxy_image_path_dict


def make_hyper_galaxy_image_1_7x7():
    return ag.Array.manual_mask(
        np.full(fill_value=3.0, shape=make_mask_7x7().pixels_in_mask),
        mask=make_mask_7x7(),
    )


def make_masked_imaging_fit_7x7():
    return ag.FitImaging(
        masked_imaging=make_masked_imaging_7x7(), plane=make_plane_7x7()
    )


def make_masked_imaging_fit_x2_galaxy_7x7():
    plane = ag.Plane(galaxies=[make_gal_x1_lp(), make_gal_x1_lp()])

    return ag.FitImaging(masked_imaging=make_masked_imaging_7x7(), plane=plane)


def make_masked_imaging_fit_x2_galaxy_inversion_7x7():
    return ag.FitImaging(
        masked_imaging=make_masked_imaging_7x7(),
        plane=make_plane_x2_galaxy_inversion_7x7(),
    )


def make_masked_interferometer_fit_7x7():
    return ag.FitInterferometer(
        masked_interferometer=make_masked_interferometer_7(), plane=make_plane_7x7()
    )


def make_masked_interferometer_fit_x2_galaxy_inversion_7x7():
    return ag.FitInterferometer(
        masked_interferometer=make_masked_interferometer_7(),
        plane=make_plane_x2_galaxy_inversion_7x7(),
    )


def make_samples_with_result():
    galaxies = [
        ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)),
        ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic(intensity=2.0)),
    ]

    plane = ag.Plane(galaxies=galaxies)

    return MockSamples(max_log_likelihood_instance=plane)


def make_phase_data():
    return PhaseDataset(
        settings=ag.SettingsPhaseImaging(), search=MockSearch(name="test_phase")
    )


def make_phase_imaging_7x7():
    return ag.PhaseImaging(search=MockSearch(name="test_phase"))


def make_phase_interferometer_7():
    return ag.PhaseInterferometer(
        search=MockSearch(name="test_phase"), real_space_mask=make_mask_7x7()
    )


def make_include_all():
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
