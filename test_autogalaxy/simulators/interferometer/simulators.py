import autogalaxy as ag

from test_autogalaxy.simulators.interferometer import instrument_util


def simulate__galaxy_x1__dev_vaucouleurs(instrument):

    data_name = "galaxy_x1__dev_vaucouleurs"

    # This galaxy-only system has a Dev Vaucouleurs spheroid / bulge.

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
            intensity=0.1,
            effective_radius=1.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[galaxy_galaxy, ag.Galaxy(redshift=1.0)],
    )


def simulate__galaxy_x1__bulge_disk(instrument):

    data_name = "galaxy_x1__bulge_disk"

    # This source-only system has a Dev Vaucouleurs spheroid / bulge and surrounding Exponential envelope

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
            intensity=0.1,
            effective_radius=1.0,
        ),
        envelope=ag.lp.EllipticalExponential(
            centre=(0.0, 0.0),
            elliptical_comps=(0.152828, -0.088235),
            intensity=1.0,
            effective_radius=2.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[galaxy_galaxy, ag.Galaxy(redshift=1.0)],
    )


def simulate__galaxy_x2__sersics(instrument):

    data_name = "galaxy_x2__sersics"

    # This source-only system has two Sersic bulges separated by 2.0"

    galaxy_galaxy_0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllipticalSersic(
            centre=(-1.0, -1.0),
            axis_ratio=0.8,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
        ),
    )

    galaxy_galaxy_1 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllipticalSersic(
            centre=(1.0, 1.0),
            axis_ratio=0.8,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[galaxy_galaxy_0, galaxy_galaxy_1, ag.Galaxy(redshift=1.0)],
    )


def simulate__galaxy_x1__dev_vaucouleurs__offset_centre(instrument):

    data_name = "galaxy_x1__dev_vaucouleurs__offset_centre"

    # This galaxy-only system has a Dev Vaucouleurs spheroid / bulge.

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllipticalDevVaucouleurs(
            centre=(2.0, 2.0),
            axis_ratio=0.9,
            phi=45.0,
            intensity=0.1,
            effective_radius=1.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[galaxy_galaxy, ag.Galaxy(redshift=1.0)],
    )
