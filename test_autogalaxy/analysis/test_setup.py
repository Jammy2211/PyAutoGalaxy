import autofit as af
import autogalaxy as ag
from autogalaxy import exc

import pytest


class MockResult:
    def __init__(self, last):

        self.last = last


class TestSetupHyper:
    def test__hyper_search(self):

        setup = ag.SetupHyper(search=None)
        assert setup.search.n_live_points == 50
        assert setup.search.evidence_tolerance == pytest.approx(0.059, 1.0e-4)

        setup = ag.SetupHyper(
            search=af.DynestyStatic(n_live_points=51)
        )
        assert setup.search.n_live_points == 51

        setup = ag.SetupHyper(hyper_galaxies=True, evidence_tolerance=0.5)
        assert setup.search.evidence_tolerance == 0.5
        assert setup.search.evidence_tolerance == 0.5

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=3.0,
            )


class TestSetupLightParametric:
    def test__prior_models(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalExponential),
        )

        assert setup.bulge_prior_model.cls is ag.lp.EllipticalSersic
        assert setup.disk_prior_model.cls is ag.lp.EllipticalExponential
        assert setup.envelope_prior_model is None

    def test__set_bulge_disk_prior_model_assertions(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
        )

        setup.bulge_prior_model._assertions = []
        setup.disk_prior_model._assertions = []

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            assert_bulge_sersic_above_disk=True,
        )

        assert isinstance(
            setup.bulge_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
        )

        setup.bulge_prior_model._assertions = []
        setup.disk_prior_model._assertions = []

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            assert_bulge_sersic_above_disk=False,
        )

        assert setup.bulge_prior_model._assertions == []

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.mp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.mp.EllipticalExponential),
        )

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            assert_bulge_sersic_above_disk=True,
        )

        assert setup.bulge_prior_model._assertions == []

    def test__set_chameleon_prior_model_assertions(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalChameleon),
            disk_prior_model=af.PriorModel(ag.lp.SphericalChameleon),
            envelope_prior_model=af.PriorModel(ag.lmp.EllipticalChameleon),
        )

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            envelope_prior_model=setup.envelope_prior_model,
            assert_chameleon_core_radius_0_above_core_radius_1=True,
        )

        assert isinstance(
            setup.bulge_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.disk_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.envelope_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.mp.EllipticalChameleon),
            disk_prior_model=af.PriorModel(ag.mp.SphericalChameleon),
            envelope_prior_model=af.PriorModel(ag.lmp.SphericalChameleon),
        )

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            envelope_prior_model=setup.envelope_prior_model,
            assert_chameleon_core_radius_0_above_core_radius_1=True,
        )

        assert isinstance(
            setup.bulge_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.disk_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.envelope_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalChameleon),
            disk_prior_model=af.PriorModel(ag.lp.SphericalChameleon),
            envelope_prior_model=af.PriorModel(ag.lmp.EllipticalChameleon),
        )

        setup.bulge_prior_model._assertions = []
        setup.disk_prior_model._assertions = []
        setup.envelope_prior_model._assertions = []

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            envelope_prior_model=setup.envelope_prior_model,
            assert_bulge_sersic_above_disk=False,
            assert_chameleon_core_radius_0_above_core_radius_1=False,
        )

        assert setup.bulge_prior_model._assertions == []
        assert setup.disk_prior_model._assertions == []
        assert setup.envelope_prior_model._assertions == []

    def test__input_light_centre__centres_of_prior_models_are_aligned(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.SphericalSersic),
            disk_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            envelope_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            light_centre=(0.0, 0.0),
        )

        assert setup.bulge_prior_model.centre == (0.0, 0.0)
        assert setup.disk_prior_model.centre == (0.0, 0.0)
        assert setup.envelope_prior_model.centre == (0.0, 0.0)

    def test__input_align_centre_and_elliptical_comps__components_are_aligned(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalExponential),
            align_bulge_disk_centre=True,
            align_bulge_disk_elliptical_comps=True,
        )

        assert setup.bulge_prior_model.centre == setup.disk_prior_model.centre
        assert (
            setup.bulge_prior_model.elliptical_comps
            == setup.disk_prior_model.elliptical_comps
        )

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.SphericalSersic),
            disk_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            align_bulge_disk_centre=True,
            align_bulge_disk_elliptical_comps=True,
        )

        assert setup.bulge_prior_model.centre == setup.disk_prior_model.centre

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=None,
            envelope_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            align_bulge_envelope_centre=True,
        )

        assert setup.bulge_prior_model.centre == setup.envelope_prior_model.centre

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            envelope_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            align_bulge_disk_centre=True,
            align_bulge_envelope_centre=True,
        )

        assert setup.bulge_prior_model.centre == setup.disk_prior_model.centre
        assert setup.bulge_prior_model.centre == setup.envelope_prior_model.centre

    def test__light_centre_gaussian_prior_values_input(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.SphericalSersic),
            disk_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            envelope_prior_model=af.PriorModel(ag.lp.EllipticalCoreSersic),
        )

        assert setup.bulge_prior_model.centre_0.mean == 0.0
        assert setup.bulge_prior_model.centre_1.mean == 0.0
        assert setup.bulge_prior_model.centre_0.sigma == 0.3
        assert setup.bulge_prior_model.centre_1.sigma == 0.3
        assert setup.disk_prior_model.centre_0.mean == 0.0
        assert setup.disk_prior_model.centre_1.mean == 0.0
        assert setup.disk_prior_model.centre_0.sigma == 0.3
        assert setup.disk_prior_model.centre_1.sigma == 0.3
        assert setup.envelope_prior_model.centre_0.mean == 0.0
        assert setup.envelope_prior_model.centre_1.mean == 0.0
        assert setup.envelope_prior_model.centre_0.sigma == 0.3
        assert setup.envelope_prior_model.centre_1.sigma == 0.3

        setup = ag.SetupLightParametric(
            light_centre_gaussian_prior_values=(0.1, 0.4),
            bulge_prior_model=af.PriorModel(ag.lp.SphericalSersic),
            disk_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            envelope_prior_model=af.PriorModel(ag.lp.EllipticalCoreSersic),
        )

        assert setup.bulge_prior_model.centre_0.mean == 0.1
        assert setup.bulge_prior_model.centre_1.mean == 0.1
        assert setup.bulge_prior_model.centre_0.sigma == 0.4
        assert setup.bulge_prior_model.centre_1.sigma == 0.4
        assert setup.disk_prior_model.centre_0.mean == 0.1
        assert setup.disk_prior_model.centre_1.mean == 0.1
        assert setup.disk_prior_model.centre_0.sigma == 0.4
        assert setup.disk_prior_model.centre_1.sigma == 0.4
        assert setup.envelope_prior_model.centre_0.mean == 0.1
        assert setup.envelope_prior_model.centre_1.mean == 0.1
        assert setup.envelope_prior_model.centre_0.sigma == 0.4
        assert setup.envelope_prior_model.centre_1.sigma == 0.4


class TestSetupLightInversion:
    def test__pixelization_prior_model__model_depends_on_inversion_pixels_fixed(self):

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.Rectangular),
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )

        assert setup.pixelization_prior_model.cls is ag.pix.Rectangular

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )

        assert setup.pixelization_prior_model.cls is ag.pix.VoronoiBrightnessImage

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            inversion_pixels_fixed=100,
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )

        assert isinstance(setup.pixelization_prior_model, af.PriorModel)
        assert setup.pixelization_prior_model.pixels == 100


class TestAbstractSetupMass:
    def test__align_centre_to_mass_centre(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupMassTotal(mass_centre=(1.0, 2.0))

        mass = source.align_centre_to_mass_centre(mass_prior_model=mass)

        assert mass.centre == (1.0, 2.0)


class TestSetupMassTotal:
    def test__mass_prior_model(self):

        setup = ag.SetupMassTotal(
            mass_prior_model=af.PriorModel(ag.mp.EllipticalPowerLaw)
        )

        assert setup.mass_prior_model.cls is ag.mp.EllipticalPowerLaw

        setup = ag.SetupMassTotal(
            mass_prior_model=af.PriorModel(ag.mp.EllipticalIsothermal)
        )

        assert setup.mass_prior_model.cls is ag.mp.EllipticalIsothermal

    def test__mass_centre_updates_mass_prior_model(self):

        setup = ag.SetupMassTotal(
            mass_prior_model=af.PriorModel(ag.mp.EllipticalIsothermal),
            mass_centre=(0.0, 1.0),
        )

        assert setup.mass_prior_model.centre == (0.0, 1.0)

    def test__align_centre_of_mass_to_light(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupMassTotal(align_bulge_mass_centre=False)

        mass = source.align_centre_of_mass_to_light(
            mass_prior_model=mass, light_centre=(1.0, 2.0)
        )

        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1
        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1

        source = ag.SetupMassTotal(align_bulge_mass_centre=True)

        mass = source.align_centre_of_mass_to_light(
            mass_prior_model=mass, light_centre=(1.0, 2.0)
        )

        assert mass.centre == (1.0, 2.0)


class TestSetupMassLightDark:
    def test__prior_models(self):

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.lmp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lmp.EllipticalExponential),
        )

        assert setup.bulge_prior_model.cls is ag.lmp.EllipticalSersic
        assert setup.disk_prior_model.cls is ag.lmp.EllipticalExponential
        assert setup.envelope_prior_model is None
        assert setup.dark_prior_model.cls is ag.mp.EllipticalNFWMCRLudlow

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.lmp.EllipticalDevVaucouleurs),
            disk_prior_model=af.PriorModel(ag.lmp.SphericalDevVaucouleurs),
            envelope_prior_model=af.PriorModel(ag.lmp.SphericalExponential),
            dark_prior_model=af.PriorModel(ag.mp.EllipticalNFW),
        )

        assert setup.bulge_prior_model.cls is ag.lmp.EllipticalDevVaucouleurs
        assert setup.disk_prior_model.cls is ag.lmp.SphericalDevVaucouleurs
        assert setup.envelope_prior_model.cls is ag.lmp.SphericalExponential
        assert setup.dark_prior_model.cls is ag.mp.EllipticalNFW

    def test__set_chameleon_prior_model_assertions(self):

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalChameleon),
            disk_prior_model=af.PriorModel(ag.lp.SphericalChameleon),
            envelope_prior_model=af.PriorModel(ag.lmp.EllipticalChameleon),
        )

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            envelope_prior_model=setup.envelope_prior_model,
        )

        assert isinstance(
            setup.bulge_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.disk_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.envelope_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.mp.EllipticalChameleon),
            disk_prior_model=af.PriorModel(ag.mp.SphericalChameleon),
            envelope_prior_model=af.PriorModel(ag.lmp.SphericalChameleon),
        )

        setup.set_light_prior_model_assertions(
            bulge_prior_model=setup.bulge_prior_model,
            disk_prior_model=setup.disk_prior_model,
            envelope_prior_model=setup.envelope_prior_model,
        )

        assert isinstance(
            setup.bulge_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.disk_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )
        assert isinstance(
            setup.envelope_prior_model._assertions[0], af.GreaterThanLessThanAssertion
        )

    def test__consstant_mass_to_light_ratio__sets_mass_to_light_ratios_of_light_and_mass_profiles(
        self,
    ):

        setup = ag.SetupMassLightDark(
            bulge_prior_model=ag.lmp.EllipticalSersic,
            disk_prior_model=ag.lmp.EllipticalSersic,
            envelope_prior_model=ag.lmp.EllipticalSersic,
            constant_mass_to_light_ratio=False,
        )

        assert (
            setup.bulge_prior_model.mass_to_light_ratio
            != setup.disk_prior_model.mass_to_light_ratio
        )
        assert (
            setup.bulge_prior_model.mass_to_light_ratio
            != setup.envelope_prior_model.mass_to_light_ratio
        )
        assert (
            setup.disk_prior_model.mass_to_light_ratio
            != setup.envelope_prior_model.mass_to_light_ratio
        )

        setup = ag.SetupMassLightDark(
            bulge_prior_model=ag.lmp.EllipticalSersic,
            disk_prior_model=ag.lmp.EllipticalSersic,
            envelope_prior_model=ag.lmp.EllipticalSersic,
            constant_mass_to_light_ratio=True,
        )

        assert (
            setup.bulge_prior_model.mass_to_light_ratio
            == setup.disk_prior_model.mass_to_light_ratio
        )
        assert (
            setup.bulge_prior_model.mass_to_light_ratio
            == setup.envelope_prior_model.mass_to_light_ratio
        )
        assert (
            setup.disk_prior_model.mass_to_light_ratio
            == setup.envelope_prior_model.mass_to_light_ratio
        )

        setup = ag.SetupMassLightDark(
            bulge_prior_model=ag.lmp.EllipticalSersic,
            disk_prior_model=ag.lmp.EllipticalSersic,
            envelope_prior_model=None,
            constant_mass_to_light_ratio=True,
        )

        assert (
            setup.bulge_prior_model.mass_to_light_ratio
            == setup.disk_prior_model.mass_to_light_ratio
        )

        setup = ag.SetupMassLightDark(
            bulge_prior_model=ag.lmp.EllipticalSersic,
            disk_prior_model=None,
            envelope_prior_model=None,
            constant_mass_to_light_ratio=True,
        )

        assert (
            setup.bulge_prior_model.mass_to_light_ratio
            == setup.bulge_prior_model.mass_to_light_ratio
        )


class TestSMBH:
    def test__smbh_from_centre(self):

        setup = ag.SetupSMBH(smbh_centre_fixed=True)
        smbh = setup.smbh_from_centre(centre=(0.0, 0.0))

        assert isinstance(smbh, af.PriorModel)
        assert smbh.centre == (0.0, 0.0)

        setup = ag.SetupSMBH(smbh_centre_fixed=False)
        smbh = setup.smbh_from_centre(centre=(0.1, 0.2), centre_sigma=0.2)

        assert isinstance(smbh, af.PriorModel)

        assert isinstance(smbh.centre[0], af.GaussianPrior)
        assert smbh.centre[0].mean == 0.1
        assert smbh.centre[0].sigma == 0.2

        assert isinstance(smbh.centre[1], af.GaussianPrior)
        assert smbh.centre[1].mean == 0.2
        assert smbh.centre[1].sigma == 0.2


class TestSetupPipeline:
    def test__setup__passes_light_setup_to_mass_light_setup(self):

        setup_light = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalExponential),
        )

        setup_mass = ag.SetupMassLightDark()

        setup_pipeline = ag.SetupPipeline(
            setup_light=setup_light, setup_mass=setup_mass
        )

        assert (
            setup_pipeline.setup_light.bulge_prior_model.cls is ag.lp.EllipticalSersic
        )
        assert (
            setup_pipeline.setup_light.disk_prior_model.cls
            is ag.lp.EllipticalExponential
        )
        assert setup_pipeline.setup_light.envelope_prior_model is None

        assert setup_pipeline.setup_mass.bulge_prior_model.cls is ag.lp.EllipticalSersic
        assert (
            setup_pipeline.setup_mass.disk_prior_model.cls
            is ag.lp.EllipticalExponential
        )
        assert setup_pipeline.setup_mass.envelope_prior_model is None
