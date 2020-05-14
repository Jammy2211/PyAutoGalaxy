from test import galaxy_redshifts

from test_autogalaxy.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_galaxy_redshifts(self):
        run_a_mock(galaxy_redshifts)
