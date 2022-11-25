from typing import Tuple

from autogalaxy.profiles.light import standard as lp
from autogalaxy.profiles import mass as mp

"""
Mass and light profiles describe both mass distributions and light distributions with a single set of parameters. This
means that the light and mass of these profiles are tied together. Galaxy instances interpret these
objects as being both mass and light profiles. 
"""


class LightMassProfile:

    pass


class Gaussian(lp.Gaussian, mp.Gaussian, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.Gaussian.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
        )
        mp.Gaussian.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class Sersic(lp.Sersic, mp.Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        mp.Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SersicSph(Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SersicSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
        Sersic.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class Exponential(Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
        Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class ExponentialSph(Exponential, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The ExponentialSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
        Exponential.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class DevVaucouleurs(Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class DevVaucouleursSph(DevVaucouleurs, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The DevVaucouleursSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
        DevVaucouleurs.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SersicRadialGradient(lp.Sersic, mp.SersicRadialGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
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
        lp.Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        mp.SersicRadialGradient.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class SphSersicRadialGradient(SersicRadialGradient, LightMassProfile):
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

        SersicRadialGradient.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class EllExponentialRadialGradient(SersicRadialGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
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

        SersicRadialGradient.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
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


class SersicCore(lp.SersicCore, mp.SersicCore, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.SersicCore.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
        )
        mp.SersicCore.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SersicCoreSph(SersicCore, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SersicSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
        SersicCore.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class Chameleon(lp.Chameleon, mp.Chameleon, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.005,
        mass_to_light_ratio: float = 1.0,
    ):

        lp.Chameleon.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        mp.Chameleon.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class ChameleonSph(Chameleon, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.005,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The ChameleonSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
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
        Chameleon.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )
