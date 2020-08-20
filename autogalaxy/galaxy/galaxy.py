from itertools import count

import numpy as np
from astropy import cosmology as cosmo
from autoarray.inversion import pixelizations as pix
from autoarray.structures import arrays, grids
from autofit.mapper.model_object import ModelObject
from autofit.text import formatter
from autogalaxy import dimensions as dim
from autogalaxy import exc
from autogalaxy import lensing
from autogalaxy.profiles import light_profiles as lp
from autogalaxy.profiles import mass_profiles as mp
from autogalaxy.profiles.mass_profiles import (
    dark_mass_profiles as dmp,
    stellar_mass_profiles as smp,
)
from autogalaxy.util import cosmology_util


def is_light_profile(obj):
    return isinstance(obj, lp.LightProfile)


def is_mass_profile(obj):
    return isinstance(obj, mp.MassProfile)


class Galaxy(ModelObject, lensing.LensingObject):
    """
    @DynamicAttrs
    """

    def __init__(
        self,
        redshift,
        pixelization=None,
        regularization=None,
        hyper_galaxy=None,
        **kwargs,
    ):
        """Class representing a galaxy, which is composed of attributes used for fitting hyper_galaxies (e.g. light profiles, \
        mass profiles, pixelizations, etc.).
        
        All *has_* methods retun *True* if galaxy has that attribute, *False* if not.

        Parameters
        ----------
        redshift: float
            The redshift of the galaxy.
        light_profiles: [lp.LightProfile]
            A list of the galaxy's light profiles.
        mass_profiles: [mp.MassProfile]
            A list of the galaxy's mass profiles.
        hyper_galaxy : HyperGalaxy
            The hyper_galaxies-parameters of the hyper_galaxies-galaxy, which is used for performing a hyper_galaxies-analysis on the noise-map.
            
        Attributes
        ----------
        pixelization : inversion.Pixelization
            The pixelization of the galaxy used to reconstruct an observed image using an inversion.
        regularization : inversion.Regularization
            The regularization of the pixel-grid used to reconstruct an observed using an inversion.
        """
        super().__init__()
        self.redshift = redshift

        self.hyper_model_image = None
        self.hyper_galaxy_image = None

        for name, val in kwargs.items():
            setattr(self, name, val)

        self.pixelization = pixelization
        self.regularization = regularization

        if pixelization is not None and regularization is None:
            raise exc.GalaxyException(
                "If the galaxy has a pixelization, it must also have a regularization."
            )
        if pixelization is None and regularization is not None:
            raise exc.GalaxyException(
                "If the galaxy has a regularization, it must also have a pixelization."
            )

        self.hyper_galaxy = hyper_galaxy

    def __hash__(self):
        return self.id

    @property
    def light_profiles(self):
        return [value for value in self.__dict__.values() if is_light_profile(value)]

    @property
    def light_profile_keys(self):
        return [key for key, value in self.__dict__.items() if is_light_profile(value)]

    @property
    def mass_profiles(self):
        return [value for value in self.__dict__.values() if is_mass_profile(value)]

    @property
    def mass_profile_keys(self):
        return [key for key, value in self.__dict__.items() if is_mass_profile(value)]

    @property
    def has_redshift(self):
        return self.redshift is not None

    @property
    def has_pixelization(self):
        return self.pixelization is not None

    @property
    def has_regularization(self):
        return self.regularization is not None

    @property
    def has_hyper_galaxy(self):
        return self.hyper_galaxy is not None

    @property
    def has_light_profile(self):
        return len(self.light_profiles) > 0

    @property
    def has_mass_profile(self):
        return len(self.mass_profiles) > 0

    @property
    def has_only_mass_sheets(self):

        if not self.has_mass_profile:
            return False

        mass_sheet_bools = [
            mass_profile.is_mass_sheet for mass_profile in self.mass_profiles
        ]
        total_mass_sheets = sum(mass_sheet_bools)

        return len(self.mass_profiles) == total_mass_sheets

    @property
    def has_profile(self):
        return len(self.mass_profiles) + len(self.light_profiles) > 0

    @property
    def light_profile_centres(self):
        """Returns the light profile centres of the galaxy as a *GridCoordinates* object, which structures the centres
        in lists according to which light profile they come from. 
        
        Fo example, if a galaxy has two light profiles, the first with one centre and second with two centres this 
        returns:
        
        [[(y0, x0)], [(y0, x0), (y1, x1)]]

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.

        NOTE: Currently, no light profiles can have more than one centre (it unlikely one ever will). The structure of 
        the output follows this convention to follow other methods in the *Galaxy* class that return profile 
        attributes."""

        centres = [[light_profile.centre] for light_profile in self.light_profiles]

        if len(centres) == 0:
            return []

        centres_dict = {}

        for key, centre in zip(self.light_profile_keys, centres):
            centres_dict[key] = centre

        return grids.GridCoordinates(coordinates=centres_dict)

    @property
    def mass_profile_centres(self):
        """Returns the mass profile centres of the galaxy as a *GridCoordinates* object, which structures the centres
        in lists according to which mass profile they come from. 

        Fo example, if a galaxy has two mass profiles, the first with one centre and second with two centres this 
        returns:

        [[(y0, x0)], [(y0, x0), (y1, x1)]]

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.

        NOTE: Currently, no mass profiles can have more than one centre (it unlikely one ever will). The structure of 
        the output follows this convention to follow other methods in the *Galaxy* class that return profile 
        attributes.

        The centres of mass-sheets are omitted, as their centres are not relevant to lensing calculations."""
        centres = [
            [mass_profile.centre]
            for mass_profile in self.mass_profiles
            if not mass_profile.is_mass_sheet
        ]

        if len(centres) == 0:
            return []

        centres_dict = {}

        for key, centre in zip(self.mass_profile_keys, centres):
            if centre is not None:
                centres_dict[key] = centre

        return grids.GridCoordinates(coordinates=centres_dict)

    @property
    def mass_profile_axis_ratios(self):
        """Returns the mass profile axis-ratios of the galaxy as a *Values* object, which structures the axis-ratios
        in lists according to which mass profile they come from. 

        Fo example, if a galaxy has two mass profiles, the first with one axis-ratio and second with two axis-ratios
        this returns:

        [[axis_ratio_0], [axis_ratio_0, axis_ratio_1]]

        This is used for visualization, for example plotting the axis-ratios of all mass profiles colored by their
        profile.

        """

        axis_ratios = [[mass_profile.axis_ratio] for mass_profile in self.mass_profiles]

        if len(axis_ratios) == 0:
            return []

        axis_ratios_dict = {}

        for key, axis_ratio in zip(self.mass_profile_keys, axis_ratios):
            axis_ratios_dict[key] = axis_ratio

        return arrays.Values(values=axis_ratios_dict)

    @property
    def mass_profile_phis(self):
        """Returns the mass profile phis of the galaxy as a *Values* object, which structures the phis in lists
        according to which mass profile they come from.

        Fo example, if a galaxy has two mass profiles, the first with one phi and second with two phis this returns:

        [[phi_0], [phi_0, phi_1]]

        This is used for visualization, for example plotting the phis of all mass profiles colored by their profile.

        """
        phis = [[mass_profile.phi] for mass_profile in self.mass_profiles]

        if len(phis) == 0:
            return []

        phis_dict = {}

        for key, phi in zip(self.mass_profile_keys, phis):
            phis_dict[key] = phi

        return arrays.Values(values=phis_dict)

    @property
    def uses_cluster_inversion(self):
        return type(self.pixelization) is pix.VoronoiBrightnessImage

    @property
    def has_stellar_profile(self):
        return len(self.stellar_profiles) > 0

    @property
    def has_dark_profile(self):
        return len(self.dark_profiles) > 0

    @property
    def stellar_profiles(self):
        return [
            profile
            for profile in self.mass_profiles
            if isinstance(profile, smp.StellarProfile)
        ]

    @property
    def dark_profiles(self):
        return [
            profile
            for profile in self.mass_profiles
            if isinstance(profile, dmp.DarkProfile)
        ]

    def stellar_mass_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_mass="angular",
        redshift_source=None,
        cosmology=cosmo.Planck15,
    ):
        if self.has_stellar_profile:
            return sum(
                [
                    profile.mass_within_circle_in_units(
                        radius=radius,
                        unit_mass=unit_mass,
                        redshift_object=self.redshift,
                        redshift_source=redshift_source,
                        cosmology=cosmology,
                    )
                    for profile in self.stellar_profiles
                ]
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a stellar mass-based calculation on a galaxy which does not have a stellar mass-profile"
            )

    def dark_mass_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_mass="angular",
        redshift_source=None,
        cosmology=cosmo.Planck15,
    ):
        if self.has_dark_profile:
            return sum(
                [
                    profile.mass_within_circle_in_units(
                        radius=radius,
                        unit_mass=unit_mass,
                        redshift_object=self.redshift,
                        redshift_source=redshift_source,
                        cosmology=cosmology,
                    )
                    for profile in self.dark_profiles
                ]
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a dark mass-based calculation on a galaxy which does not have a dark mass-profile"
            )

    def stellar_fraction_at_radius(self, radius):
        return 1.0 - self.dark_fraction_at_radius(radius=radius)

    def dark_fraction_at_radius(self, radius):

        stellar_mass = self.stellar_mass_within_circle_in_units(radius=radius)
        dark_mass = self.dark_mass_within_circle_in_units(radius=radius)

        return dark_mass / (stellar_mass + dark_mass)

    @property
    def cosmology(self):
        return cosmo.Planck15

    @property
    def arcsec_per_kpc(self):
        return cosmology_util.arcsec_per_kpc_from(
            redshift=self.redshift, cosmology=self.cosmology
        )

    @property
    def kpc_per_arcsec(self):
        return 1.0 / self.arcsec_per_kpc

    @property
    def unit_length(self):
        if self.has_light_profile:
            return self.light_profiles[0].unit_length
        elif self.has_mass_profile:
            return self.mass_profiles[0].unit_length
        else:
            return None

    @property
    def unit_luminosity(self):
        if self.has_light_profile:
            return self.light_profiles[0].unit_luminosity
        elif self.has_mass_profile:
            return self.mass_profiles[0].unit_luminosity
        else:
            return None

    @property
    def unit_mass(self):
        if self.has_mass_profile:
            return self.mass_profiles[0].unit_mass
        else:
            return None

    def __repr__(self):
        string = "Redshift: {}".format(self.redshift)
        if self.pixelization:
            string += "\nPixelization:\n{}".format(str(self.pixelization))
        if self.regularization:
            string += "\nRegularization:\n{}".format(str(self.regularization))
        if self.hyper_galaxy:
            string += "\nHyper Galaxy:\n{}".format(str(self.hyper_galaxy))
        if self.light_profiles:
            string += "\nLight Profiles:\n{}".format(
                "\n".join(map(str, self.light_profiles))
            )
        if self.mass_profiles:
            string += "\nMass Profiles:\n{}".format(
                "\n".join(map(str, self.mass_profiles))
            )
        return string

    def __eq__(self, other):
        return all(
            (
                isinstance(other, Galaxy),
                self.pixelization == other.pixelization,
                self.redshift == other.redshift,
                self.hyper_galaxy == other.hyper_galaxy,
                self.light_profiles == other.light_profiles,
                self.mass_profiles == other.mass_profiles,
            )
        )

    def new_object_with_units_converted(
        self,
        unit_length=None,
        unit_luminosity=None,
        unit_mass=None,
        kpc_per_arcsec=None,
        exposure_time=None,
        critical_surface_density=None,
    ):

        new_dict = {
            key: value.new_object_with_units_converted(
                unit_length=unit_length,
                unit_luminosity=unit_luminosity,
                unit_mass=unit_mass,
                kpc_per_arcsec=kpc_per_arcsec,
                exposure_time=exposure_time,
                critical_surface_density=critical_surface_density,
            )
            if is_light_profile(value) or is_mass_profile(value)
            else value
            for key, value in self.__dict__.items()
        }

        return self.__class__(**new_dict)

    @grids.grid_like_to_structure
    def image_from_grid(self, grid):
        """Calculate the summed image of all of the galaxy's light profiles using a grid of Cartesian (y,x) \
        coordinates.
        
        If the galaxy has no light profiles, a grid of zeros is returned.
        
        See *profiles.light_profiles* for a description of how light profile image are computed.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_light_profile:
            return sum(map(lambda p: p.image_from_grid(grid=grid), self.light_profiles))
        return np.zeros((grid.shape[0],))

    def blurred_image_from_grid_and_psf(self, grid, psf, blurring_grid=None):

        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return psf.convolved_array_from_array_2d_and_mask(
            array_2d=image.in_2d_binned + blurring_image.in_2d_binned, mask=grid.mask
        )

    def blurred_image_from_grid_and_convolver(self, grid, convolver, blurring_grid):

        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return convolver.convolved_image_from_image_and_blurring_image(
            image=image.in_1d_binned, blurring_image=blurring_image.in_1d_binned
        )

    def profile_visibilities_from_grid_and_transformer(self, grid, transformer):

        image = self.image_from_grid(grid=grid)

        return transformer.visibilities_from_image(image=image.in_1d_binned)

    def luminosity_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_luminosity="eps",
        exposure_time=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):
        """Compute the total luminosity of the galaxy's light profiles within a circle of specified radius.

        See *light_profiles.luminosity_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        unit_luminosity : str
            The unit_label the luminosity is returned in {esp, counts}.
        exposure_time : float
            The exposure time of the observation, which converts luminosity from electrons per second unit_label to counts.
        """
        if self.has_light_profile:
            return sum(
                map(
                    lambda p: p.luminosity_within_circle_in_units(
                        radius=radius,
                        unit_luminosity=unit_luminosity,
                        redshift_object=self.redshift,
                        exposure_time=exposure_time,
                        cosmology=cosmology,
                        kwargs=kwargs,
                    ),
                    self.light_profiles,
                )
            )
        return None

    @grids.grid_like_to_structure
    def convergence_from_grid(self, grid):
        """Compute the summed convergence of the galaxy's mass profiles using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.
        
        See *profiles.mass_profiles* module for details of how this is performed.

        The *grid_like_to_structure* decorator reshapes the NumPy arrays the convergence is outputted on. See \
        *aa.grid_like_to_structure* for a description of the output.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.convergence_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        """Compute the summed gravitational potential of the galaxy's mass profiles \
        using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See *profiles.mass_profiles* module for details of how this is performed.

        The *grid_like_to_structure* decorator reshapes the NumPy arrays the convergence is outputted on. See \
        *aa.grid_like_to_structure* for a description of the output.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.potential_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grids.grid_like_to_structure
    def deflections_from_grid(self, grid):
        """Compute the summed (y,x) deflection angles of the galaxy's mass profiles \
        using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, two grid of zeros are returned.

        See *profiles.mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.deflections_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0], 2))

    def mass_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_mass="angular",
        redshift_source=None,
        cosmology=cosmo.Planck15,
    ):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following unit_label for mass can be specified and output:

        - Dimensionless angular unit_label (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The unit_label the mass is returned in {angular, angular}.
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            unit_label to phsical unit_label (e.g. solar masses).
        """
        if self.has_mass_profile:
            return sum(
                map(
                    lambda p: p.mass_within_circle_in_units(
                        radius=radius,
                        unit_mass=unit_mass,
                        redshift_object=self.redshift,
                        redshift_source=redshift_source,
                        cosmology=cosmology,
                    ),
                    self.mass_profiles,
                )
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a mass-based calculation on a galaxy which does not have a mass-profile"
            )

    @property
    def contribution_map(self):
        """Compute the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        -----------
        hyper_model_image : ndarray
            The best-fit model image to the observed image from a previous analysis
            phase. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image : ndarray
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis phase.
        """
        return self.hyper_galaxy.contribution_map_from_hyper_images(
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image=self.hyper_galaxy_image,
        )

    def summarize_in_units(
        self,
        radii,
        whitespace=80,
        unit_length="arcsec",
        unit_luminosity="eps",
        unit_mass="solMass",
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):

        if hasattr(self, "name"):
            summary = ["Galaxy = {}\n".format(self.name)]
            prefix_galaxy = self.name + "_"
        else:
            summary = ["Galaxy\n"]
            prefix_galaxy = ""

        summary += [
            formatter.label_and_value_string(
                label=prefix_galaxy + "redshift",
                value=self.redshift,
                whitespace=whitespace,
            )
        ]

        if self.has_light_profile:
            summary += self.summarize_light_profiles_in_units(
                whitespace=whitespace,
                prefix=prefix_galaxy,
                radii=radii,
                unit_length=unit_length,
                unit_luminosity=unit_luminosity,
                redshift_source=redshift_source,
                cosmology=cosmology,
                kwargs=kwargs,
            )

        if self.has_mass_profile:
            summary += self.summarize_mass_profiles_in_units(
                whitespace=whitespace,
                prefix=prefix_galaxy,
                radii=radii,
                unit_length=unit_length,
                unit_mass=unit_mass,
                redshift_source=redshift_source,
                cosmology=cosmology,
                kwargs=kwargs,
            )

        return summary

    def summarize_light_profiles_in_units(
        self,
        radii,
        whitespace=80,
        prefix="",
        unit_length="arcsec",
        unit_luminosity="eps",
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):

        summary = ["\nGALAXY LIGHT\n\n"]

        for radius in radii:
            luminosity = self.luminosity_within_circle_in_units(
                unit_luminosity=unit_luminosity,
                radius=radius,
                redshift_source=redshift_source,
                cosmology=cosmology,
                kwargs=kwargs,
            )

            summary += [
                formatter.within_radius_label_value_and_unit_string(
                    prefix=prefix + "luminosity",
                    radius=radius,
                    unit_length=unit_length,
                    value=luminosity,
                    unit_value=unit_luminosity,
                    whitespace=whitespace,
                )
            ]

        summary.append("\nLIGHT PROFILES:\n\n")

        for light_profile in self.light_profiles:
            summary += light_profile.summarize_in_units(
                radii=radii,
                whitespace=whitespace,
                unit_length=unit_length,
                unit_luminosity=unit_luminosity,
                redshift_profile=self.redshift,
                redshift_source=redshift_source,
                cosmology=cosmology,
                kwargs=kwargs,
            )

            summary += "\n"

        return summary

    def summarize_mass_profiles_in_units(
        self,
        radii,
        whitespace=80,
        prefix="",
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):

        summary = ["\nGALAXY MASS\n\n"]

        einstein_radius = self.einstein_radius_in_units(
            unit_length=unit_length, cosmology=cosmology
        )

        summary += [
            formatter.label_value_and_unit_string(
                label=prefix + "einstein_radius",
                value=einstein_radius,
                unit=unit_length,
                whitespace=whitespace,
            )
        ]

        einstein_mass = self.einstein_mass_in_units(
            unit_mass=unit_mass, redshift_source=redshift_source, cosmology=cosmology
        )

        summary += [
            formatter.label_value_and_unit_string(
                label=prefix + "einstein_mass",
                value=einstein_mass,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        for radius in radii:
            mass = self.mass_within_circle_in_units(
                unit_mass=unit_mass,
                radius=radius,
                redshift_source=redshift_source,
                cosmology=cosmology,
            )

            summary += [
                formatter.within_radius_label_value_and_unit_string(
                    prefix=prefix + "mass",
                    radius=radius,
                    unit_length=unit_length,
                    value=mass,
                    unit_value=unit_mass,
                    whitespace=whitespace,
                )
            ]

        summary += ["\nMASS PROFILES:\n\n"]

        for mass_profile in self.mass_profiles:
            summary += mass_profile.summarize_in_units(
                radii=radii,
                whitespace=whitespace,
                unit_length=unit_length,
                unit_mass=unit_mass,
                redshift_profile=self.redshift,
                redshift_source=redshift_source,
                cosmology=cosmology,
            )

            summary += "\n"

        return summary


class HyperGalaxy:
    _ids = count()

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        """ If a *Galaxy* is given a *HyperGalaxy* as an attribute, the noise-map in \
        the regions of the image that the galaxy is located will be hyper, to prevent \
        over-fitting of the galaxy.
        
        This is performed by first computing the hyper_galaxies-galaxy's 'contribution-map', \
        which determines the fraction of flux in every pixel of the image that can be \
        associated with this particular hyper_galaxies-galaxy. This is computed using \
        hyper_galaxies-hyper_galaxies set (e.g. fitting.fit_data.FitDataHyper), which includes  best-fit \
        unblurred_image_1d of the galaxy's light from a previous analysis phase.
         
        The *HyperGalaxy* class contains the hyper_galaxies-parameters which are associated \
        with this galaxy for scaling the noise-map.
        
        Parameters
        -----------
        contribution_factor : float
            Factor that adjusts how much of the galaxy's light is attributed to the
            contribution map.
        noise_factor : float
            Factor by which the noise-map is increased in the regions of the galaxy's
            contribution map.
        noise_power : float
            The power to which the contribution map is raised when scaling the
            noise-map.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

        self.component_number = next(self._ids)

    def contribution_map_from_hyper_images(self, hyper_model_image, hyper_galaxy_image):
        """Compute the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        -----------
        hyper_model_image : ndarray
            The best-fit model image to the observed image from a previous analysis
            phase. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image : ndarray
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis phase.
        """
        contribution_map = np.divide(
            hyper_galaxy_image, np.add(hyper_model_image, self.contribution_factor)
        )
        return np.divide(contribution_map, np.max(contribution_map))

    def hyper_noise_map_from_hyper_images_and_noise_map(
        self, hyper_model_image, hyper_galaxy_image, noise_map
    ):
        contribution_map = self.contribution_map_from_hyper_images(
            hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
        )
        return self.hyper_noise_map_from_contribution_map(
            noise_map=noise_map, contribution_map=contribution_map
        )

    def hyper_noise_map_from_contribution_map(self, noise_map, contribution_map):
        """Compute a hyper galaxy hyper_galaxies noise-map from a baseline noise-map.

        This uses the galaxy contribution map and the *noise_factor* and *noise_power*
        hyper_galaxies-parameters.

        Parameters
        -----------
        noise_map : ndarray
            The observed noise-map (before scaling).
        contribution_map : ndarray
            The galaxy contribution map.
        """
        return self.noise_factor * (noise_map * contribution_map) ** self.noise_power

    def __eq__(self, other):
        if isinstance(other, HyperGalaxy):
            return (
                self.contribution_factor == other.contribution_factor
                and self.noise_factor == other.noise_factor
                and self.noise_power == other.noise_power
            )
        return False

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])


class Redshift(float):
    def __new__(cls, redshift):
        # noinspection PyArgumentList
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)
