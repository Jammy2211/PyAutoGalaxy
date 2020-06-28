import autogalaxy as ag
import pytest
from autogalaxy import exc


class TestLength:
    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(
        self
    ):
        unit_arcsec = ag.dim.Length(value=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_length == "arcsec"

        unit_arcsec = unit_arcsec.convert(unit_length="arcsec")

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "arcsec"

        unit_kpc = unit_arcsec.convert(unit_length="kpc", kpc_per_arcsec=2.0)

        assert unit_kpc == 4.0
        assert unit_kpc.unit == "kpc"

        unit_kpc = unit_kpc.convert(unit_length="kpc")

        assert unit_kpc == 4.0
        assert unit_kpc.unit == "kpc"

        unit_arcsec = unit_kpc.convert(unit_length="arcsec", kpc_per_arcsec=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "arcsec"

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length="kpc")
            unit_kpc.convert(unit_length="arcsec")
            unit_arcsec.convert(unit_length="lol")


class TestLuminosity:
    def test__conversions_from_eps_and_counts_and_back__errors_raised_if_no_exposure_time(
        self
    ):

        unit_eps = ag.dim.Luminosity(value=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit_luminosity == "eps"

        unit_eps = unit_eps.convert(unit_luminosity="eps")

        assert unit_eps == 2.0
        assert unit_eps.unit == "eps"

        unit_counts = unit_eps.convert(unit_luminosity="counts", exposure_time=2.0)

        assert unit_counts == 4.0
        assert unit_counts.unit == "counts"

        unit_counts = unit_counts.convert(unit_luminosity="counts")

        assert unit_counts == 4.0
        assert unit_counts.unit == "counts"

        unit_eps = unit_counts.convert(unit_luminosity="eps", exposure_time=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit == "eps"

        with pytest.raises(exc.UnitsException):
            unit_eps.convert(unit_luminosity="counts")
            unit_counts.convert(unit_luminosity="eps")
            unit_eps.convert(unit_luminosity="lol")


class TestMass:
    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_no_exposure_time(
        self
    ):

        mass_angular = ag.dim.Mass(value=2.0)

        assert mass_angular == 2.0
        assert mass_angular.unit_mass == "angular"

        # angular -> angular, stays 2.0

        mass_angular = mass_angular.convert(unit_mass="angular")

        assert mass_angular == 2.0
        assert mass_angular.unit == "angular"

        # angular -> solMass, converts to 2.0 * 2.0 = 4.0

        mas_sol_mass = mass_angular.convert(
            unit_mass="solMass", critical_surface_density=2.0
        )

        assert mas_sol_mass == 4.0
        assert mas_sol_mass.unit == "solMass"

        # solMass -> solMass, stays 4.0

        mas_sol_mass = mas_sol_mass.convert(unit_mass="solMass")

        assert mas_sol_mass == 4.0
        assert mas_sol_mass.unit == "solMass"

        # solMass -> angular, stays 4.0

        mass_angular = mas_sol_mass.convert(
            unit_mass="angular", critical_surface_density=2.0
        )

        assert mass_angular == 2.0
        assert mass_angular.unit == "angular"

        with pytest.raises(exc.UnitsException):
            mass_angular.convert(unit_mass="solMass")
            mas_sol_mass.convert(unit_mass="angular")
            mass_angular.convert(unit_mass="lol")


class TestMassOverLuminosity:
    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_critical_mass_density(
        self
    ):

        unit_angular = ag.dim.MassOverLuminosity(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / eps"

        unit_angular = unit_angular.convert(unit_mass="angular", unit_luminosity="eps")

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / eps"

        unit_sol_mass = unit_angular.convert(
            unit_mass="solMass", critical_surface_density=2.0, unit_luminosity="eps"
        )

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == "solMass / eps"

        unit_sol_mass = unit_sol_mass.convert(
            unit_mass="solMass", unit_luminosity="eps"
        )

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == "solMass / eps"

        unit_angular = unit_sol_mass.convert(
            unit_mass="angular", critical_surface_density=2.0, unit_luminosity="eps"
        )

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / eps"

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass="solMass", unit_luminosity="eps")
            unit_sol_mass.convert(unit_mass="angular", unit_luminosity="eps")
            unit_angular.convert(unit_mass="lol", unit_luminosity="eps")

    def test__conversions_from_eps_and_counts_and_back__errors_raised_if_no_exposure_time(
        self
    ):

        unit_eps = ag.dim.MassOverLuminosity(value=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit == "angular / eps"

        unit_eps = unit_eps.convert(unit_mass="angular", unit_luminosity="eps")

        assert unit_eps == 2.0
        assert unit_eps.unit == "angular / eps"

        unit_counts = unit_eps.convert(
            unit_mass="angular", exposure_time=2.0, unit_luminosity="counts"
        )

        assert unit_counts == 1.0
        assert unit_counts.unit == "angular / counts"

        unit_counts = unit_counts.convert(unit_mass="angular", unit_luminosity="counts")

        assert unit_counts == 1.0
        assert unit_counts.unit == "angular / counts"

        unit_eps = unit_counts.convert(
            unit_mass="angular", exposure_time=2.0, unit_luminosity="eps"
        )

        assert unit_eps == 2.0
        assert unit_eps.unit == "angular / eps"

        with pytest.raises(exc.UnitsException):
            unit_eps.convert(unit_mass="angular", unit_luminosity="eps")
            unit_counts.convert(unit_mass="angular", unit_luminosity="eps")
            unit_eps.convert(unit_mass="lol", unit_luminosity="eps")


class TestMassOverLength2:
    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_critical_mass_density(
        self
    ):

        unit_angular = ag.dim.MassOverLength2(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / arcsec^2"

        unit_angular = unit_angular.convert(unit_mass="angular", unit_length="arcsec")

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / arcsec^2"

        unit_sol_mass = unit_angular.convert(
            unit_mass="solMass", critical_surface_density=2.0, unit_length="arcsec"
        )

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == "solMass / arcsec^2"

        unit_sol_mass = unit_sol_mass.convert(unit_mass="solMass", unit_length="arcsec")

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == "solMass / arcsec^2"

        unit_angular = unit_sol_mass.convert(
            unit_mass="angular", critical_surface_density=2.0, unit_length="arcsec"
        )

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / arcsec^2"

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass="solMass", unit_length="eps")
            unit_sol_mass.convert(unit_mass="angular", unit_length="eps")
            unit_angular.convert(unit_mass="lol", unit_length="eps")

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(
        self
    ):

        unit_arcsec = ag.dim.MassOverLength2(value=2.0, unit_mass="solMass")

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "solMass / arcsec^2"

        unit_arcsec = unit_arcsec.convert(unit_length="arcsec", unit_mass="solMass")

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "solMass / arcsec^2"

        unit_kpc = unit_arcsec.convert(
            unit_length="kpc", kpc_per_arcsec=2.0, unit_mass="solMass"
        )

        assert unit_kpc == 2.0 / 2.0 ** 2.0
        assert unit_kpc.unit == "solMass / kpc^2"

        unit_kpc = unit_kpc.convert(unit_length="kpc", unit_mass="solMass")

        assert unit_kpc == 2.0 / 2.0 ** 2.0
        assert unit_kpc.unit == "solMass / kpc^2"

        unit_arcsec = unit_kpc.convert(
            unit_length="arcsec", kpc_per_arcsec=2.0, unit_mass="solMass"
        )

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "solMass / arcsec^2"

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length="kpc", unit_mass="solMass")
            unit_kpc.convert(unit_length="arcsec", unit_mass="solMass")
            unit_arcsec.convert(unit_length="lol", unit_mass="solMass")


class TestMassOverLength3:
    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_critical_mass_density(
        self
    ):

        unit_angular = ag.dim.MassOverLength3(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / arcsec^3"

        unit_angular = unit_angular.convert(unit_mass="angular", unit_length="arcsec")

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / arcsec^3"

        unit_sol_mass = unit_angular.convert(
            unit_mass="solMass", critical_surface_density=2.0, unit_length="arcsec"
        )

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == "solMass / arcsec^3"

        unit_sol_mass = unit_sol_mass.convert(unit_mass="solMass", unit_length="arcsec")

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == "solMass / arcsec^3"

        unit_angular = unit_sol_mass.convert(
            unit_mass="angular", critical_surface_density=2.0, unit_length="arcsec"
        )

        assert unit_angular == 2.0
        assert unit_angular.unit == "angular / arcsec^3"

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass="solMass", unit_length="eps")
            unit_sol_mass.convert(unit_mass="angular", unit_length="eps")
            unit_angular.convert(unit_mass="lol", unit_length="eps")

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(
        self
    ):

        unit_arcsec = ag.dim.MassOverLength3(value=2.0, unit_mass="solMass")

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "solMass / arcsec^3"

        unit_arcsec = unit_arcsec.convert(unit_length="arcsec", unit_mass="solMass")

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "solMass / arcsec^3"

        unit_kpc = unit_arcsec.convert(
            unit_length="kpc", kpc_per_arcsec=2.0, unit_mass="solMass"
        )

        assert unit_kpc == 2.0 / 2.0 ** 3.0
        assert unit_kpc.unit == "solMass / kpc^3"

        unit_kpc = unit_kpc.convert(unit_length="kpc", unit_mass="solMass")

        assert unit_kpc == 2.0 / 2.0 ** 3.0
        assert unit_kpc.unit == "solMass / kpc^3"

        unit_arcsec = unit_kpc.convert(
            unit_length="arcsec", kpc_per_arcsec=2.0, unit_mass="solMass"
        )

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "solMass / arcsec^3"

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length="kpc", unit_mass="solMass")
            unit_kpc.convert(unit_length="arcsec", unit_mass="solMass")
            unit_arcsec.convert(unit_length="lol", unit_mass="solMass")


class MockDimensionsProfile(ag.dim.DimensionsProfile):
    def __init__(
        self,
        position: ag.dim.Position = None,
        param_float: float = None,
        length: ag.dim.Length = None,
        luminosity: ag.dim.Luminosity = None,
        mass: ag.dim.Mass = None,
        mass_over_luminosity: ag.dim.MassOverLuminosity = None,
    ):

        super(MockDimensionsProfile, self).__init__()

        self.position = position
        self.param_float = param_float
        self.luminosity = luminosity
        self.length = length
        self.mass = mass
        self.mass_over_luminosity = mass_over_luminosity


class TestDimensionsProfile:
    class TestUnitProperties:
        def test__extracts_length_correctly__raises_error_if_different_lengths_input(
            self
        ):
            profile = MockDimensionsProfile(
                position=(
                    ag.dim.Length(value=3.0, unit_length="arcsec"),
                    ag.dim.Length(value=3.0, unit_length="arcsec"),
                ),
                length=ag.dim.Length(3.0, "arcsec"),
            )

            assert profile.unit_length == "arcsec"

            profile = MockDimensionsProfile(
                position=(
                    ag.dim.Length(value=3.0, unit_length="kpc"),
                    ag.dim.Length(value=3.0, unit_length="kpc"),
                ),
                length=ag.dim.Length(3.0, "kpc"),
            )

            assert profile.unit_length == "kpc"

            with pytest.raises(exc.UnitsException):
                profile = MockDimensionsProfile(
                    position=(
                        ag.dim.Length(value=3.0, unit_length="kpc"),
                        ag.dim.Length(value=3.0, unit_length="kpc"),
                    ),
                    length=ag.dim.Length(3.0, "arcsec"),
                )

                profile.unit_length

        def test__extracts_luminosity_correctly__raises_error_if_different_luminosities(
            self
        ):
            profile = MockDimensionsProfile(
                luminosity=ag.dim.Luminosity(3.0, "eps"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=1.0, unit_luminosity="eps"
                ),
            )

            assert profile.unit_luminosity == "eps"

            profile = MockDimensionsProfile(
                luminosity=ag.dim.Luminosity(3.0, "counts"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=1.0, unit_luminosity="counts"
                ),
            )

            assert profile.unit_luminosity == "counts"

            with pytest.raises(exc.UnitsException):
                profile = MockDimensionsProfile(
                    luminosity=ag.dim.Luminosity(3.0, "eps"),
                    mass_over_luminosity=ag.dim.MassOverLuminosity(
                        value=1.0, unit_luminosity="counts"
                    ),
                )

                profile.unit_luminosity

        def test__extracts_mass_correctly__raises_error_if_different_mass(self):
            profile = MockDimensionsProfile(
                mass=ag.dim.Mass(3.0, "angular"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=1.0, unit_mass="angular"
                ),
            )

            assert profile.unit_mass == "angular"

            profile = MockDimensionsProfile(
                mass=ag.dim.Mass(3.0, "solMass"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=1.0, unit_mass="solMass"
                ),
            )

            assert profile.unit_mass == "solMass"

            with pytest.raises(exc.UnitsException):
                profile = MockDimensionsProfile(
                    mass=ag.dim.Mass(3.0, "angular"),
                    mass_over_luminosity=ag.dim.MassOverLuminosity(
                        value=1.0, unit_mass="solMass"
                    ),
                )

                profile.unit_mass

    class TestUnitConversions:
        def test__arcsec_to_kpc_conversions_of_length__float_and_tuple_length__conversion_converts_values(
            self
        ):

            profile_arcsec = MockDimensionsProfile(
                position=(ag.dim.Length(1.0, "arcsec"), ag.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=ag.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=ag.dim.Luminosity(value=4.0, unit_luminosity="eps"),
                mass=ag.dim.Mass(value=5.0, unit_mass="angular"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=6.0, unit_luminosity="eps", unit_mass="angular"
                ),
            )

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit_length == "arcsec"
            assert profile_arcsec.position[1].unit_length == "arcsec"
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit_length == "arcsec"
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit_luminosity == "eps"
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == "angular"
            assert profile_arcsec.mass_over_luminosity == 6.0
            assert profile_arcsec.mass_over_luminosity.unit == "angular / eps"

            profile_arcsec = profile_arcsec.new_object_with_units_converted(
                unit_length="arcsec"
            )

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit == "arcsec"
            assert profile_arcsec.position[1].unit == "arcsec"
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit == "arcsec"
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit == "eps"
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == "angular"
            assert profile_arcsec.mass_over_luminosity == 6.0
            assert profile_arcsec.mass_over_luminosity.unit == "angular / eps"

            profile_kpc = profile_arcsec.new_object_with_units_converted(
                unit_length="kpc", kpc_per_arcsec=2.0
            )

            assert profile_kpc.position == (2.0, 4.0)
            assert profile_kpc.position[0].unit == "kpc"
            assert profile_kpc.position[1].unit == "kpc"
            assert profile_kpc.param_float == 2.0
            assert profile_kpc.length == 6.0
            assert profile_kpc.length.unit == "kpc"
            assert profile_kpc.luminosity == 4.0
            assert profile_kpc.luminosity.unit == "eps"
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == "angular"
            assert profile_kpc.mass_over_luminosity == 6.0
            assert profile_kpc.mass_over_luminosity.unit == "angular / eps"

            profile_kpc = profile_kpc.new_object_with_units_converted(unit_length="kpc")

            assert profile_kpc.position == (2.0, 4.0)
            assert profile_kpc.position[0].unit == "kpc"
            assert profile_kpc.position[1].unit == "kpc"
            assert profile_kpc.param_float == 2.0
            assert profile_kpc.length == 6.0
            assert profile_kpc.length.unit == "kpc"
            assert profile_kpc.luminosity == 4.0
            assert profile_kpc.luminosity.unit == "eps"
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == "angular"
            assert profile_kpc.mass_over_luminosity == 6.0
            assert profile_kpc.mass_over_luminosity.unit == "angular / eps"

            profile_arcsec = profile_kpc.new_object_with_units_converted(
                unit_length="arcsec", kpc_per_arcsec=2.0
            )

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit == "arcsec"
            assert profile_arcsec.position[1].unit == "arcsec"
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit == "arcsec"
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit == "eps"
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == "angular"
            assert profile_arcsec.mass_over_luminosity == 6.0
            assert profile_arcsec.mass_over_luminosity.unit == "angular / eps"

        def test__conversion_requires_kpc_per_arcsec_but_does_not_supply_it_raises_error(
            self
        ):

            profile_arcsec = MockDimensionsProfile(
                position=(ag.dim.Length(1.0, "arcsec"), ag.dim.Length(2.0, "arcsec"))
            )

            with pytest.raises(exc.UnitsException):
                profile_arcsec.new_object_with_units_converted(unit_length="kpc")

            profile_kpc = profile_arcsec.new_object_with_units_converted(
                unit_length="kpc", kpc_per_arcsec=2.0
            )

            with pytest.raises(exc.UnitsException):
                profile_kpc.new_object_with_units_converted(unit_length="arcsec")

        def test__eps_to_counts_conversions_of_luminosity__conversions_convert_values(
            self
        ):

            profile_eps = MockDimensionsProfile(
                position=(ag.dim.Length(1.0, "arcsec"), ag.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=ag.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=ag.dim.Luminosity(value=4.0, unit_luminosity="eps"),
                mass=ag.dim.Mass(value=5.0, unit_mass="angular"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=6.0, unit_luminosity="eps", unit_mass="angular"
                ),
            )

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == "arcsec"
            assert profile_eps.position[1].unit_length == "arcsec"
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == "arcsec"
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == "eps"
            assert profile_eps.mass == 5.0
            assert profile_eps.mass.unit_mass == "angular"
            assert profile_eps.mass_over_luminosity == 6.0
            assert profile_eps.mass_over_luminosity.unit == "angular / eps"

            profile_eps = profile_eps.new_object_with_units_converted(
                unit_luminosity="eps"
            )

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == "arcsec"
            assert profile_eps.position[1].unit_length == "arcsec"
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == "arcsec"
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == "eps"
            assert profile_eps.mass == 5.0
            assert profile_eps.mass.unit_mass == "angular"
            assert profile_eps.mass_over_luminosity == 6.0
            assert profile_eps.mass_over_luminosity.unit == "angular / eps"

            profile_counts = profile_eps.new_object_with_units_converted(
                unit_luminosity="counts", exposure_time=10.0
            )

            assert profile_counts.position == (1.0, 2.0)
            assert profile_counts.position[0].unit_length == "arcsec"
            assert profile_counts.position[1].unit_length == "arcsec"
            assert profile_counts.param_float == 2.0
            assert profile_counts.length == 3.0
            assert profile_counts.length.unit_length == "arcsec"
            assert profile_counts.luminosity == 40.0
            assert profile_counts.luminosity.unit_luminosity == "counts"
            assert profile_counts.mass == 5.0
            assert profile_counts.mass.unit_mass == "angular"
            assert profile_counts.mass_over_luminosity == pytest.approx(0.6, 1.0e-4)
            assert profile_counts.mass_over_luminosity.unit == "angular / counts"

            profile_counts = profile_counts.new_object_with_units_converted(
                unit_luminosity="counts"
            )

            assert profile_counts.position == (1.0, 2.0)
            assert profile_counts.position[0].unit_length == "arcsec"
            assert profile_counts.position[1].unit_length == "arcsec"
            assert profile_counts.param_float == 2.0
            assert profile_counts.length == 3.0
            assert profile_counts.length.unit_length == "arcsec"
            assert profile_counts.luminosity == 40.0
            assert profile_counts.luminosity.unit_luminosity == "counts"
            assert profile_counts.mass == 5.0
            assert profile_counts.mass.unit_mass == "angular"
            assert profile_counts.mass_over_luminosity == pytest.approx(0.6, 1.0e-4)
            assert profile_counts.mass_over_luminosity.unit == "angular / counts"

            profile_eps = profile_counts.new_object_with_units_converted(
                unit_luminosity="eps", exposure_time=10.0
            )

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == "arcsec"
            assert profile_eps.position[1].unit_length == "arcsec"
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == "arcsec"
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == "eps"
            assert profile_eps.mass == 5.0
            assert profile_eps.mass.unit_mass == "angular"
            assert profile_eps.mass_over_luminosity == pytest.approx(6.0, 1.0e-4)
            assert profile_eps.mass_over_luminosity.unit == "angular / eps"

        def test__luminosity_conversion_requires_exposure_time_but_does_not_supply_it_raises_error(
            self
        ):

            profile_eps = MockDimensionsProfile(
                position=(ag.dim.Length(1.0, "arcsec"), ag.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=ag.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=ag.dim.Luminosity(value=4.0, unit_luminosity="eps"),
                mass=ag.dim.Mass(value=5.0, unit_mass="angular"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=6.0, unit_luminosity="eps", unit_mass="angular"
                ),
            )

            with pytest.raises(exc.UnitsException):
                profile_eps.new_object_with_units_converted(unit_luminosity="counts")

            profile_counts = profile_eps.new_object_with_units_converted(
                unit_luminosity="counts", exposure_time=10.0
            )

            with pytest.raises(exc.UnitsException):
                profile_counts.new_object_with_units_converted(unit_luminosity="eps")

        def test__angular_to_solMass_conversions_of_mass__conversions_convert_values(
            self
        ):

            profile_angular = MockDimensionsProfile(
                position=(ag.dim.Length(1.0, "arcsec"), ag.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=ag.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=ag.dim.Luminosity(value=4.0, unit_luminosity="eps"),
                mass=ag.dim.Mass(value=5.0, unit_mass="angular"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=6.0, unit_luminosity="eps", unit_mass="angular"
                ),
            )

            assert profile_angular.position == (1.0, 2.0)
            assert profile_angular.position[0].unit_length == "arcsec"
            assert profile_angular.position[1].unit_length == "arcsec"
            assert profile_angular.param_float == 2.0
            assert profile_angular.length == 3.0
            assert profile_angular.length.unit_length == "arcsec"
            assert profile_angular.luminosity == 4.0
            assert profile_angular.luminosity.unit_luminosity == "eps"
            assert profile_angular.mass == 5.0
            assert profile_angular.mass.unit_mass == "angular"
            assert profile_angular.mass_over_luminosity == 6.0
            assert profile_angular.mass_over_luminosity.unit == "angular / eps"

            profile_angular = profile_angular.new_object_with_units_converted(
                unit_mass="angular"
            )

            assert profile_angular.position == (1.0, 2.0)
            assert profile_angular.position[0].unit_length == "arcsec"
            assert profile_angular.position[1].unit_length == "arcsec"
            assert profile_angular.param_float == 2.0
            assert profile_angular.length == 3.0
            assert profile_angular.length.unit_length == "arcsec"
            assert profile_angular.luminosity == 4.0
            assert profile_angular.luminosity.unit_luminosity == "eps"
            assert profile_angular.mass == 5.0
            assert profile_angular.mass.unit_mass == "angular"
            assert profile_angular.mass_over_luminosity == 6.0
            assert profile_angular.mass_over_luminosity.unit == "angular / eps"

            profile_solMass = profile_angular.new_object_with_units_converted(
                unit_mass="solMass", critical_surface_density=10.0
            )

            assert profile_solMass.position == (1.0, 2.0)
            assert profile_solMass.position[0].unit_length == "arcsec"
            assert profile_solMass.position[1].unit_length == "arcsec"
            assert profile_solMass.param_float == 2.0
            assert profile_solMass.length == 3.0
            assert profile_solMass.length.unit_length == "arcsec"
            assert profile_solMass.luminosity == 4.0
            assert profile_solMass.luminosity.unit_luminosity == "eps"
            assert profile_solMass.mass == 50.0
            assert profile_solMass.mass.unit_mass == "solMass"
            assert profile_solMass.mass_over_luminosity == pytest.approx(60.0, 1.0e-4)
            assert profile_solMass.mass_over_luminosity.unit == "solMass / eps"

            profile_solMass = profile_solMass.new_object_with_units_converted(
                unit_mass="solMass"
            )

            assert profile_solMass.position == (1.0, 2.0)
            assert profile_solMass.position[0].unit_length == "arcsec"
            assert profile_solMass.position[1].unit_length == "arcsec"
            assert profile_solMass.param_float == 2.0
            assert profile_solMass.length == 3.0
            assert profile_solMass.length.unit_length == "arcsec"
            assert profile_solMass.luminosity == 4.0
            assert profile_solMass.luminosity.unit_luminosity == "eps"
            assert profile_solMass.mass == 50.0
            assert profile_solMass.mass.unit_mass == "solMass"
            assert profile_solMass.mass_over_luminosity == pytest.approx(60.0, 1.0e-4)
            assert profile_solMass.mass_over_luminosity.unit == "solMass / eps"

            profile_angular = profile_solMass.new_object_with_units_converted(
                unit_mass="angular", critical_surface_density=10.0
            )

            assert profile_angular.position == (1.0, 2.0)
            assert profile_angular.position[0].unit_length == "arcsec"
            assert profile_angular.position[1].unit_length == "arcsec"
            assert profile_angular.param_float == 2.0
            assert profile_angular.length == 3.0
            assert profile_angular.length.unit_length == "arcsec"
            assert profile_angular.luminosity == 4.0
            assert profile_angular.luminosity.unit_luminosity == "eps"
            assert profile_angular.mass == 5.0
            assert profile_angular.mass.unit_mass == "angular"
            assert profile_angular.mass_over_luminosity == pytest.approx(6.0, 1.0e-4)
            assert profile_angular.mass_over_luminosity.unit == "angular / eps"

        def test__mass_conversion_requires_critical_surface_density_but_does_not_supply_it_raises_error(
            self
        ):

            profile_angular = MockDimensionsProfile(
                position=(ag.dim.Length(1.0, "arcsec"), ag.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=ag.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=ag.dim.Luminosity(value=4.0, unit_luminosity="eps"),
                mass=ag.dim.Mass(value=5.0, unit_mass="angular"),
                mass_over_luminosity=ag.dim.MassOverLuminosity(
                    value=6.0, unit_luminosity="eps", unit_mass="angular"
                ),
            )

            with pytest.raises(exc.UnitsException):
                profile_angular.new_object_with_units_converted(unit_mass="solMass")

            profile_solMass = profile_angular.new_object_with_units_converted(
                unit_mass="solMass", critical_surface_density=10.0
            )

            with pytest.raises(exc.UnitsException):
                profile_solMass.new_object_with_units_converted(unit_mass="angular")
