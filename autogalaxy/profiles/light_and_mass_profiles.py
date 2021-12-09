from typing import Tuple

from autogalaxy.profiles import light_profiles as lp
from autogalaxy.profiles import mass_profiles as mp

"""
Mass and light profiles describe both mass distributions and light distributions with a single set of parameters. This
means that the light and mass of these profiles are tied together. Galaxy instances interpret these
objects as being both mass and light profiles. 
"""


class LightMassProfile:

    pass


class EllGaussian(lp.EllGaussian, mp.EllGaussian, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.EllGaussian.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            sigma=sigma,
        )
        mp.EllGaussian.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            sigma=sigma,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllSersic(lp.EllSersic, mp.EllSersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.EllSersic.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        mp.EllSersic.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphSersic(EllSersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SphSersic mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        EllSersic.__init__(
            self,
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllExponential(EllSersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The EllExponential mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
        angle
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        EllSersic.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphExponential(EllExponential, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SphExponential mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        EllExponential.__init__(
            self,
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllDevVaucouleurs(EllSersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The EllDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
        angle
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphDevVaucouleurs(EllDevVaucouleurs, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SphDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        EllDevVaucouleurs.__init__(
            self,
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllSersicRadialGradient(
    lp.EllSersic, mp.EllSersicRadialGradient, LightMassProfile
):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The origin of the profiles
        axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
        angle
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        sersic_index
            The concentration of the light profiles
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        lp.EllSersic.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        mp.EllSersicRadialGradient.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class SphSersicRadialGradient(EllSersicRadialGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The origin of the profiles
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        sersic_index
            The concentration of the light profiles
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """

        EllSersicRadialGradient.__init__(
            self,
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class EllExponentialRadialGradient(EllSersicRadialGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup an Exponential mass and light profiles.

        Parameters
        ----------
        centre
            The origin of the profiles
        axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
        angle
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """

        EllSersicRadialGradient.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class SphExponentialRadialGradient(SphSersicRadialGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup an Exponential mass and light profiles.

        Parameters
        ----------
        centre
            The origin of the profiles
        axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
        angle
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """

        SphSersicRadialGradient.__init__(
            self,
            centre=centre,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class EllSersicCore(lp.EllSersicCore, mp.EllSersicCore, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.EllSersicCore.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity_break=intensity_break,
            gamma=gamma,
            alpha=alpha,
        )
        mp.EllSersicCore.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity_break=intensity_break,
            gamma=gamma,
            alpha=alpha,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphSersicCore(EllSersicCore, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SphSersic mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        EllSersicCore.__init__(
            self,
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity_break=intensity_break,
            gamma=gamma,
            alpha=alpha,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllChameleon(lp.EllChameleon, mp.EllChameleon, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.005,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.EllChameleon.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        mp.EllChameleon.__init__(
            self,
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphChameleon(EllChameleon, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.005,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SphChameleon mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of the origin of the profiles
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        EllChameleon.__init__(
            self,
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )
