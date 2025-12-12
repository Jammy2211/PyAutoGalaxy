import numpy as np
from typing import Optional, Tuple

import autoarray as aa
import autolens as al


from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.standard.shapelets.abstract import AbstractShapelet

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

def genlaguerre_jax_recurrence(n, alpha, x):
    """
    Generalized (associated) Laguerre polynomial $L_n^{(\alpha)}(x)$ 
    calculated using the three-term recurrence relation in pure JAX.

    Optimized for JAX via `lax.fori_loop` for loop unrolling.

    .. math::
        (k+1) L_{k+1}^{(\alpha)}(x) = (2k + 1 + \alpha - x) L_k^{(\alpha)}(x) - (k + \alpha) L_{k-1}^{(\alpha)}(x)

    Parameters
    ----------
    n : int
        Degree of the polynomial. MUST be a non-negative static Python integer 
        for optimal JAX compilation.
    alpha : Union[float, JAXArray]
        Parameter $\alpha > -1$.
    x : JAXArray
        Input array (points at which to evaluate).

    Returns
    -------
    L : JAXArray
        Generalized Laguerre polynomial evaluated at x.
    """
    
    # --- 0. Input Validation (Requires static Python int n) ---
    if not isinstance(n, int) or n < 0:
         raise ValueError(f"Degree n must be a non-negative Python integer (static), got {n}.")

    # --- 1. Base Cases ---
    L0 = jnp.ones_like(x)
    if n == 0:
        return L0

    L1 = 1 + alpha - x
    if n == 1:
        return L1
    
    # --- 2. JAX Recurrence Calculation ---
    
    def body(k, state):
        # state = (L_{k-1}, L_k)
        L_nm1, L_n = state
        
        # Recurrence relation:
        # L_{k+1} = ((2k + 1 + alpha - x) * L_k - (k + alpha) * L_{k-1}) / (k + 1)
        L_np1 = ((2 * k + 1 + alpha - x) * L_n - (k + alpha) * L_nm1) / (k + 1)
        
        # Return new state: (L_k, L_{k+1})
        return (L_n, L_np1)

    # fori_loop(start, stop, body, init_state)
    _, res_Ln = lax.fori_loop(1, n, body, (L0, L1))
    return res_Ln

def genlaguerre_jax_summation(n, alpha, x):
    """
    Generalized (associated) Laguerre polynomial L_n^alpha(x) 
    calculated using the explicit summation formula, optimized for JAX vectorization.

    Parameters:
        n (int): Degree of the polynomial (static Python integer).
        alpha (Numeric): Parameter alpha > -1.
        x (Array): Input array (evaluation points).
    """
    # 0. Input Validation (Requires static Python int n)
    if not isinstance(n, int) or n < 0:
         # Use Python's math.isnan/isinf check if n is float, otherwise type error
         raise ValueError(f"Degree n must be a non-negative Python integer (static), got {n}.")
    
    # Base Case L0
    if n == 0:
        return jnp.ones_like(x)

    # 1. Generate k values for summation range [0, 1, 2, ..., n]
    k_values = jnp.arange(n + 1) # (n+1,)

    # 2. Reshape inputs for broadcasting (x: (M, 1), k: (1, n+1))
    x_expanded = jnp.expand_dims(x, axis=-1) 
    k_values_expanded = jnp.expand_dims(k_values, axis=0)

    # --- A. Binomial Factor (BF) Calculation ---
    # BF = exp( log( (n+alpha)! / ((n-k)! * (alpha+k)!) ) )
    
    log_N_plus_alpha_fact = gammaln(n + alpha + 1)
    
    log_BF_k = (
        log_N_plus_alpha_fact
        - gammaln(n - k_values + 1)      # log( (n-k)! )
        - gammaln(alpha + k_values + 1)  # log( (alpha+k)! )
    )
    
    BF_k = jnp.exp(log_BF_k) # Shape: (n+1,)

    # --- B. Term Factor (TF) Calculation ---
    # TF = (-x)^k / k!
    
    # Note: jnp.math.gamma(k_values + 1) is equivalent to k! in log-gamma space
    TF_k = jnp.power(-x_expanded, k_values_expanded) / jnp.exp(gammaln(k_values_expanded + 1)) 
    # TF_k Shape: (M, n+1)

    # --- C. Final Summation ---
    # Sum over the last axis (axis=1), which corresponds to k
    # BF_k broadcasts over the M dimension of TF_k
    return jnp.sum(BF_k * TF_k, axis=1)

class ShapeletPolar(AbstractShapelet):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        q: float = 1.0,
        phi: float = 0.0,
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n
            The n order of the shapelets basis function.
        m
            The m order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        q
            The axis-ratio of the elliptical coordinate system, where a perfect circle has q=1.0.
        phi
            The position angle (in degrees) of the elliptical coordinate system, measured counter-clockwise from the 
            positive x-axis.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.n = int(n)
        self.m = int(m)
        self.phi = float(phi)
        self.q = float(q)

        super().__init__(
            centre=centre, beta=beta, intensity=intensity
        )

    @property
    def coefficient_tag(self) -> str:
        return f"n_{self.n}_m_{self.m}"

    @aa.over_sample
    @aa.grid_dec.to_array
    @check_operated_only
    # @aa.grid_dec.transform
    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns the Polar Shapelet light profile's 2D image from a 2D grid of Polar (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Polar Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """
        # from scipy.special import genlaguerre
        from jax.scipy.special import factorial

        # laguerre = genlaguerre(n=(self.n - xp.abs(self.m)) / 2.0, alpha=xp.abs(self.m))
        grid = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid.array, centre=self.centre, angle=self.phi, xp=xp
        )
        const = (
            ((-1) ** ((self.n - xp.abs(self.m)) // 2))
            * xp.sqrt(
                factorial((self.n - xp.abs(self.m)) // 2)
                / factorial((self.n + xp.abs(self.m)) // 2)
            )
            / self.beta
            / xp.sqrt(xp.pi)
        )
        rsq = (grid[:, 0] ** 2 + (grid[:, 1]/self.q) ** 2) / self.beta**2
        theta = xp.arctan2(grid[:, 1], grid[:, 0])

        m_abs = abs(self.m)
        n_laguerre = (self.n - m_abs) // 2
        laguerre_vals = genlaguerre_jax_summation(n=n_laguerre, alpha=m_abs, x=rsq)

        radial = rsq ** (xp.abs(self.m) / 2.0) * xp.exp(-rsq / 2.0) * laguerre_vals

        if self.m == 0:
            azimuthal = 1
        elif self.m > 0:
            azimuthal = xp.sin((-1) * self.m * theta)
        else:
            azimuthal = xp.cos((-1) * self.m * theta)

        return self._intensity * const * radial * azimuthal


class ShapeletPolarSph(ShapeletPolar):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        phi: float = 0.0,
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n_y
            The order of the shapelets basis function in the y-direction.
        n_x
            The order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        phi
            The position angle (in degrees) of the elliptical coordinate system, measured counter-clockwise from the 
            positive x-axis.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        super().__init__(
            n=n,
            m=m,
            centre=centre,
            q=1.0,
            phi=phi,
            intensity=intensity,
            beta=beta,
        )
