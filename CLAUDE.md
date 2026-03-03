# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
# All tests
python -m pytest test_autogalaxy/

# Single test file
python -m pytest test_autogalaxy/galaxy/test_galaxy.py

# Single test
python -m pytest test_autogalaxy/galaxy/test_galaxy.py::TestGalaxy::test_name

# With output
python -m pytest test_autogalaxy/imaging/test_fit_imaging.py -s
```

### Formatting
```bash
black autogalaxy/
```

## Architecture

**PyAutoGalaxy** is a Bayesian galaxy morphology fitting library. It depends on two sibling packages:
- **`autoarray`** – low-level data structures (grids, masks, arrays, imaging/interferometer datasets, inversions/pixelizations)
- **`autofit`** – non-linear search / model-fitting framework (defines `af.Analysis`, `af.ModelInstance`, `af.Result`)

### Core Class Hierarchy

```
GeometryProfile (geometry_profiles.py)
├── SphProfile
└── EllProfile
    ├── LightProfile (profiles/light/abstract.py) + OperateImage mixin
    │   ├── lp.Sersic, lp.Exponential, lp.Gaussian, etc. (profiles/light/standard/)
    │   ├── LightProfileLinear (profiles/light/linear/) - intensity solved via inversion
    │   ├── LightProfileOperated (profiles/light/operated/) - PSF already applied
    │   └── Basis (profiles/basis.py) - collection of profiles for MGE / shapelets
    └── MassProfile (profiles/mass/abstract/abstract.py) + OperateDeflections mixin
        ├── mp.Isothermal, mp.NFW, mp.PowerLaw, etc.
        └── stellar, dark, total subdirectories

Galaxy (galaxy/galaxy.py) - inherits af.ModelObject, OperateImageList, OperateDeflections
  └── holds arbitrary kwargs: light profiles, mass profiles, pixelizations, etc.

Galaxies (galaxy/galaxies.py) - a List[Galaxy] with group operations
```

### Fit Classes

Each dataset type has a `Fit*` class that orchestrates the full fitting pipeline:

- `FitImaging` (`imaging/fit_imaging.py`) – CCD imaging
- `FitInterferometer` (`interferometer/fit_interferometer.py`) – ALMA/interferometry
- `FitQuantity` (`quantity/fit_quantity.py`) – arbitrary quantity datasets
- `FitEllipse` (`ellipse/fit_ellipse.py`) – isophote/ellipse fitting

All inherit from `AbstractFitInversion` (`abstract_fit.py`), which handles the linear algebra inversion step when `LightProfileLinear` or pixelization-based profiles are present.

### Analysis Classes (autofit integration)

Each dataset type has an `Analysis*` class that implements `log_likelihood_function`:

- `AnalysisImaging` (`imaging/model/analysis.py`)
- `AnalysisInterferometer` (`interferometer/model/analysis.py`)
- `AnalysisQuantity` (`quantity/model/analysis.py`)
- `AnalysisEllipse` (`ellipse/model/analysis.py`)

These inherit from `AnalysisDataset` → `Analysis` (in `analysis/analysis/`), which inherits `af.Analysis`. The `log_likelihood_function` builds a `Fit*` object from the `af.ModelInstance` and returns its `figure_of_merit`.

### JAX Support

JAX is integrated via the `xp` parameter pattern throughout the codebase. Fit classes accept `xp=np` (NumPy, default) or `xp=jnp` (JAX). The `AbstractFitInversion.use_jax` property tracks which backend is active. The `AnalysisImaging.__init__` has `use_jax: bool = True`. The conftest.py forces JAX backend initialization before tests run.

### Linear Light Profiles & Inversions

`LightProfileLinear` subclasses do not take an `intensity` parameter—it is solved via a linear inversion (provided by `autoarray`). The `GalaxiesToInversion` class (`galaxy/to_inversion.py`) handles converting galaxies with linear profiles or pixelizations into the inversion objects needed by `autoarray`.

### Adapt Images (Multi-stage fitting)

`AdaptImages` (`analysis/adapt_images/`) stores per-galaxy model images from a previous search. These are passed to subsequent searches to drive adaptive mesh/regularization schemes. `galaxy_name_image_dict_via_result_from` extracts adapt images from a `Result`.

### Configuration

Default priors, visualization settings, and general config live in `autogalaxy/config/`. Tests push a local config directory via `conf.instance.push(...)` in `test_autogalaxy/conftest.py`.

### Operate Mixins

- `OperateImage` (`operate/image.py`) – provides `blurred_image_2d_from`, `visibilities_from`, etc. on light objects
- `OperateDeflections` (`operate/deflections.py`) – provides deflection-related operations on mass objects

Both are mixin classes inherited by `LightProfile`, `MassProfile`, `Galaxy`, and `Galaxies`.

### Workspace (Examples & Notebooks)

The `autogalaxy_workspace` at `/mnt/c/Users/Jammy/Code/PyAutoJAX/autogalaxy_workspace` contains runnable examples and tutorials. Key locations:

- `start_here.ipynb` / `start_here.py` – entry point overview of the API
- `scripts/imaging/` – end-to-end scripts: `simulator.py`, `fit.py`, `modeling.py`, `likelihood_function.py`, `features/`
- `scripts/interferometer/`, `scripts/ellipse/`, `scripts/multi/` – same structure for other dataset types
- `scripts/howtogalaxy/` – chapter-by-chapter tutorial scripts (chapters 1–4 + optional)
- `notebooks/` – Jupyter notebook equivalents of all `scripts/`
- `scripts/guides/` – topic guides (e.g. linear profiles, pixelizations, chaining)
- `config/` – workspace-level config that overrides package defaults when running workspace scripts

### Namespace Conventions

When importing `autogalaxy as ag`:
- `ag.lp.*` – standard light profiles
- `ag.lp_linear.*` – linear light profiles
- `ag.lp_operated.*` – operated light profiles
- `ag.lp_basis.*` / `ag.lp_snr.*` – basis and SNR profiles
- `ag.mp.*` – mass profiles
- `ag.lmp.*` – light+mass profiles
- `ag.ps.*` – point sources
- `ag.Galaxy`, `ag.Galaxies`
- `ag.FitImaging`, `ag.AnalysisImaging`, `ag.SimulatorImaging`
