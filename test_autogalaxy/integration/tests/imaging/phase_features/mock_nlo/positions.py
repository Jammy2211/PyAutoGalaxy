from test import positions

from test_autogalaxy.integration.tests.imaging.runner import run_a_mock


class TestCase:
    def _test_positions(self):
        run_a_mock(positions)
