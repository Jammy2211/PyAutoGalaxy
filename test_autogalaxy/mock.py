import numpy as np
import math

from astropy import constants
from autoarray.structures import grids
import autogalaxy as ag

### MockProfiles ###

# noinspection PyUnusedLocal
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


### Mock Galaxy ###


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


### Mock Cosmology ###


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


### Mock Pipeline / Phase ###

import autofit as af
from test_autofit.mock import MockSearch, MockSamples


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
