from autofit.non_linear.plot.nest_plotters import NestPlotter
from autofit.non_linear.plot.mcmc_plotters import MCMCPlotter
from autofit.non_linear.plot.mle_plotters import MLEPlotter

from autoarray.plot.wrap.base import (
    Cmap,
    Colorbar,
    Output,
)
from autoarray.plot.wrap.two_d import DelaunayDrawer

from autoarray.structures.plot.structure_plotters import Array2DPlotter
from autoarray.structures.plot.structure_plotters import Grid2DPlotter
from autoarray.structures.plot.structure_plotters import YX1DPlotter
from autoarray.structures.plot.structure_plotters import YX1DPlotter as Array1DPlotter
from autoarray.inversion.plot.mapper_plotters import MapperPlotter
from autoarray.inversion.plot.inversion_plotters import InversionPlotter
from autoarray.dataset.plot.imaging_plotters import ImagingPlotter
from autoarray.dataset.plot.interferometer_plotters import InterferometerPlotter

from autogalaxy.plot.wrap import (
    HalfLightRadiusAXVLine,
    EinsteinRadiusAXVLine,
    ModelFluxesYXScatter,
    LightProfileCentresScatter,
    MassProfileCentresScatter,
    MultipleImagesScatter,
    TangentialCriticalCurvesPlot,
    RadialCriticalCurvesPlot,
    TangentialCausticsPlot,
    RadialCausticsPlot,
)

# Standalone plot functions — light profiles
from autogalaxy.profiles.plot.light_profile_plots import plot_image_2d as plot_light_profile_image_2d

# Standalone plot functions — mass profiles
from autogalaxy.profiles.plot.mass_profile_plots import (
    plot_convergence_2d as plot_mass_profile_convergence_2d,
    plot_potential_2d as plot_mass_profile_potential_2d,
    plot_deflections_y_2d as plot_mass_profile_deflections_y_2d,
    plot_deflections_x_2d as plot_mass_profile_deflections_x_2d,
    plot_magnification_2d as plot_mass_profile_magnification_2d,
)

# Standalone plot functions — basis
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image

# Standalone plot functions — galaxy
from autogalaxy.galaxy.plot.galaxy_plots import (
    plot_image_2d as plot_galaxy_image_2d,
    plot_convergence_2d as plot_galaxy_convergence_2d,
    plot_potential_2d as plot_galaxy_potential_2d,
    plot_deflections_y_2d as plot_galaxy_deflections_y_2d,
    plot_deflections_x_2d as plot_galaxy_deflections_x_2d,
    plot_magnification_2d as plot_galaxy_magnification_2d,
    subplot_of_light_profiles as subplot_galaxy_light_profiles,
    subplot_of_mass_profiles as subplot_galaxy_mass_profiles,
)

# Standalone plot functions — galaxies
from autogalaxy.galaxy.plot.galaxies_plots import (
    plot_image_2d as plot_galaxies_image_2d,
    plot_convergence_2d as plot_galaxies_convergence_2d,
    plot_potential_2d as plot_galaxies_potential_2d,
    plot_deflections_y_2d as plot_galaxies_deflections_y_2d,
    plot_deflections_x_2d as plot_galaxies_deflections_x_2d,
    plot_magnification_2d as plot_galaxies_magnification_2d,
    subplot_galaxies,
    subplot_galaxy_images,
)

# Standalone plot functions — adapt
from autogalaxy.galaxy.plot.adapt_plots import (
    plot_model_image as plot_adapt_model_image,
    plot_galaxy_image as plot_adapt_galaxy_image,
    subplot_adapt_images,
)

# Standalone plot functions — fit imaging
from autogalaxy.imaging.plot.fit_imaging_plots import (
    plot_data as plot_fit_imaging_data,
    plot_noise_map as plot_fit_imaging_noise_map,
    plot_signal_to_noise_map as plot_fit_imaging_signal_to_noise_map,
    plot_model_image as plot_fit_imaging_model_image,
    plot_residual_map as plot_fit_imaging_residual_map,
    plot_normalized_residual_map as plot_fit_imaging_normalized_residual_map,
    plot_chi_squared_map as plot_fit_imaging_chi_squared_map,
    subplot_fit as subplot_fit_imaging,
    plot_subtracted_image_of_galaxy,
    plot_model_image_of_galaxy,
    subplot_of_galaxy as subplot_fit_imaging_of_galaxy,
)

# Standalone plot functions — fit interferometer
from autogalaxy.interferometer.plot.fit_interferometer_plots import (
    subplot_fit as subplot_fit_interferometer,
    subplot_fit_dirty_images,
    subplot_fit_real_space,
)

# Standalone plot functions — fit quantity
from autogalaxy.quantity.plot.fit_quantity_plots import (
    plot_data as plot_fit_quantity_data,
    subplot_fit as subplot_fit_quantity,
)

# Standalone plot functions — fit ellipse
from autogalaxy.ellipse.plot.fit_ellipse_plots import (
    plot_data as plot_fit_ellipse_data,
    plot_ellipse_residuals,
    subplot_fit_ellipse,
    subplot_ellipse_errors,
)
