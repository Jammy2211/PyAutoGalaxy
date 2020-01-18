import autoastro as aast

from test_autoarray.unit.conftest import *

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "test_files/config"), path.join(directory, "output")
    )


#
# MODEL #
#

# PROFILES #


@pytest.fixture(name="lp_0")
def make_lp_0():
    # noinspection PyTypeChecker
    return aast.lp.SphericalSersic(
        intensity=1.0, effective_radius=2.0, sersic_index=2.0
    )


@pytest.fixture(name="lp_1")
def make_lp_1():
    # noinspection PyTypeChecker
    return aast.lp.SphericalSersic(
        intensity=2.0, effective_radius=2.0, sersic_index=2.0
    )


@pytest.fixture(name="mp_0")
def make_mp_0():
    # noinspection PyTypeChecker
    return aast.mp.SphericalIsothermal(einstein_radius=1.0)


@pytest.fixture(name="mp_1")
def make_mp_1():
    # noinspection PyTypeChecker
    return aast.mp.SphericalIsothermal(einstein_radius=2.0)


@pytest.fixture(name="lmp_0")
def make_lmp_0():
    return aast.lmp.EllipticalSersicRadialGradient()


# GALAXY #


@pytest.fixture(name="gal_x1_lp")
def make_gal_x1_lp(lp_0):
    return aast.Galaxy(redshift=0.5, light_profile_0=lp_0)


@pytest.fixture(name="gal_x2_lp")
def make_gal_x2_lp(lp_0, lp_1):
    return aast.Galaxy(redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1)


@pytest.fixture(name="gal_x1_mp")
def make_gal_x1_mp(mp_0):
    return aast.Galaxy(redshift=0.5, mass_profile_0=mp_0)


@pytest.fixture(name="gal_x2_mp")
def make_gal_x2_mp(mp_0, mp_1):
    return aast.Galaxy(redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1)


@pytest.fixture(name="gal_x1_lp_x1_mp")
def make_gal_x1_lp_x1_mp(lp_0, mp_0):
    return aast.Galaxy(redshift=0.5, light_profile_0=lp_0, mass_profile_0=mp_0)


@pytest.fixture(name="hyper_galaxy")
def make_hyper_galaxy():
    return aast.HyperGalaxy(noise_factor=1.0, noise_power=1.0, contribution_factor=1.0)


# GALAXY DATA #


@pytest.fixture(name="gal_data_7x7")
def make_gal_data_7x7(image_7x7, noise_map_7x7):
    return aast.galaxy_data(
        image=image_7x7, noise_map=noise_map_7x7, pixel_scales=image_7x7.pixel_scales
    )


@pytest.fixture(name="gal_fit_data_7x7_image")
def make_gal_fit_data_7x7_image(gal_data_7x7, sub_mask_7x7):
    return aast.masked.galaxy_data(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
    )


@pytest.fixture(name="gal_fit_data_7x7_convergence")
def make_gal_fit_data_7x7_convergence(gal_data_7x7, sub_mask_7x7):
    return aast.masked.galaxy_data(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_convergence=True
    )


@pytest.fixture(name="gal_fit_data_7x7_potential")
def make_gal_fit_data_7x7_potential(gal_data_7x7, sub_mask_7x7):
    return aast.masked.galaxy_data(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_potential=True
    )


@pytest.fixture(name="gal_fit_data_7x7_deflections_y")
def make_gal_fit_data_7x7_deflections_y(gal_data_7x7, sub_mask_7x7):
    return aast.masked.galaxy_data(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_y=True
    )


@pytest.fixture(name="gal_fit_data_7x7_deflections_x")
def make_gal_fit_data_7x7_deflections_x(gal_data_7x7, sub_mask_7x7):
    return aast.masked.galaxy_data(
        galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_x=True
    )


# GALAXY FIT #


@pytest.fixture(name="gal_fit_7x7_image")
def make_gal_fit_7x7_image(gal_fit_data_7x7_image, gal_x1_lp):
    return aast.fit_galaxy(
        galaxy_data=gal_fit_data_7x7_image, model_galaxies=[gal_x1_lp]
    )


@pytest.fixture(name="gal_fit_7x7_convergence")
def make_gal_fit_7x7_convergence(gal_fit_data_7x7_convergence, gal_x1_mp):
    return aast.fit_galaxy(
        galaxy_data=gal_fit_data_7x7_convergence, model_galaxies=[gal_x1_mp]
    )


@pytest.fixture(name="gal_fit_7x7_potential")
def make_gal_fit_7x7_potential(gal_fit_data_7x7_potential, gal_x1_mp):
    return aast.fit_galaxy(
        galaxy_data=gal_fit_data_7x7_potential, model_galaxies=[gal_x1_mp]
    )


@pytest.fixture(name="gal_fit_7x7_deflections_y")
def make_gal_fit_7x7_deflections_y(gal_fit_data_7x7_deflections_y, gal_x1_mp):
    return aast.fit_galaxy(
        galaxy_data=gal_fit_data_7x7_deflections_y, model_galaxies=[gal_x1_mp]
    )


@pytest.fixture(name="gal_fit_7x7_deflections_x")
def make_gal_fit_7x7_deflections_x(gal_fit_data_7x7_deflections_x, gal_x1_mp):
    return aast.fit_galaxy(
        galaxy_data=gal_fit_data_7x7_deflections_x, model_galaxies=[gal_x1_mp]
    )


from autoastro.plot.lensing_plotters import Include

# PLOTTING #


@pytest.fixture(name="include_all")
def make_include_all():
    return Include(
        origin=True,
        mask=True,
        grid=True,
        border=True,
        positions=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        critical_curves=True,
        caustics=True,
        multiple_images=True,
        inversion_pixelization_grid=True,
        inversion_grid=True,
        inversion_border=True,
        inversion_image_pixelization_grid=True,
        preloaded_critical_curves=[(1.0, 1.0), (2.0, 2.0)],
        preload_caustics=[(1.0, 1.0), (2.0, 2.0)],
    )
