# Copilot Coding Agent Instructions

You are working on **PyAutoGalaxy**, a Bayesian galaxy morphology fitting library.

## Key Rules

- Run tests after every change: `python -m pytest test_autogalaxy/`
- Format code with `black autogalaxy/`
- All files must use Unix line endings (LF, `\n`)
- Decorated functions (`@to_array`, `@to_grid`, `@to_vector_yx`) must return **raw arrays**, not autoarray wrappers
- The `xp` parameter controls NumPy (`xp=np`) vs JAX (`xp=jnp`) — never import JAX at module level
- Functions called inside `jax.jit` must guard autoarray wrapping with `if xp is np:`
- Use `grid.array[:, 0]` to access grid coordinates (not `grid[:, 0]`)
- If changing public API, clearly document what changed in your PR description — downstream packages depend on this

## Architecture

- `autogalaxy/profiles/` — Light profiles (`lp.*`), mass profiles (`mp.*`), linear profiles (`lp_linear.*`)
- `autogalaxy/galaxy/` — `Galaxy`, `Galaxies` classes
- `autogalaxy/imaging/`, `interferometer/`, `ellipse/` — Dataset-specific fit and analysis classes
- `autogalaxy/operate/` — `OperateImage`, `OperateDeflections`, `LensCalc` mixins
- `test_autogalaxy/` — Test suite

## Sandboxed runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autogalaxy/
```
