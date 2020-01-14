import pytest

from autoastro.plots import lensing_plotters


class TestInclude:
    def test__critical_curves_from_object(self, lp_0, mp_0):

        include = lensing_plotters.Include(critical_curves=False)

        critical_curves = include.critical_curves_from_obj(obj=mp_0)

        assert critical_curves == None

        include = lensing_plotters.Include(critical_curves=True)

        critical_curves = include.critical_curves_from_obj(obj=lp_0)

        assert critical_curves == None

        include = lensing_plotters.Include(critical_curves=True)

        critical_curves = include.critical_curves_from_obj(obj=mp_0)

        assert critical_curves[0] == pytest.approx(mp_0.critical_curves[0], 1.0e-4)
        assert critical_curves[1] == pytest.approx(mp_0.critical_curves[1], 1.0e-4)

    def test__caustics_from_object(self, lp_0, mp_0):

        include = lensing_plotters.Include(caustics=False)

        caustics = include.caustics_from_obj(obj=mp_0)

        assert caustics == None

        include = lensing_plotters.Include(caustics=True)

        caustics = include.caustics_from_obj(obj=lp_0)

        assert caustics == None

        include = lensing_plotters.Include(caustics=True)

        caustics = include.caustics_from_obj(obj=mp_0)

        assert caustics[0] == pytest.approx(mp_0.caustics[0], 1.0e-4)
        assert caustics[1] == pytest.approx(mp_0.caustics[1], 1.0e-4)

    def test__new_include_with_preloaded_critical_curves_and_caustics(self):

        include = lensing_plotters.Include(mask=True)

        assert include.preloaded_critical_curves == None
        assert include.preloaded_caustics == None

        include = include.new_include_with_preloaded_critical_curves_and_caustics(preloaded_critical_curves=1, preloaded_caustics=2)

        assert include.mask == True
        assert include.preloaded_critical_curves == 1
        assert include.preloaded_caustics == 2