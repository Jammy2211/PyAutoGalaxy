import inspect
from typing import Dict, Tuple

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp


class SersicCore(lp.SersicCore, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            alpha=alpha,
            gamma=gamma,
        )

    @property
    def lp_cls(self):
        return lp.SersicCore

    def parameters_dict_from(self, intensity: float) -> Dict[str, float]:
        """
        Returns a dictionary of the parameters of the linear light profile with the `intensity` added.

        This `intenisty` will likely have come from the value inferred via the linear inversion.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        parameters_dict = vars(self)

        args = inspect.getfullargspec(self.lp_cls.__init__).args
        args.remove("self")

        parameters_dict = {key: parameters_dict[key] for key in args}
        parameters_dict["intensity"] = intensity

        return parameters_dict
