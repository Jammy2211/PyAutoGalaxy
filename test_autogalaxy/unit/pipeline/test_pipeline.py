import builtins

import autofit as af
import autogalaxy as ag
import numpy as np
import pytest
from autofit import Paths


class MockAnalysis:
    def __init__(self, number_galaxies, shape, value):
        self.number_galaxies = number_galaxies
        self.shape = shape
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.full(self.shape, self.value)]


class MockMask:
    pass


class Optimizer:
    def __init__(self, phase_name="dummy_phase"):
        self.phase_name = phase_name
        self.phase_path = ""


class DummyPhaseImaging(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name, phase_tag=""):
        super().__init__(Paths(name=phase_name, tag=phase_tag))
        self.dataset = None
        self.results = None
        self.mask = None

        self.search = Optimizer(phase_name)

    def run(self, dataset, results, mask=None, info=None):
        self.save_metadata(dataset)
        self.dataset = dataset
        self.results = results
        self.mask = mask
        return af.Result(af.ModelInstance(), 1)


class MockImagingData(af.Dataset):
    def __init__(self, metadata=None):
        self._metadata = metadata or dict()

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def name(self) -> str:
        return "data_name"


class MockFile:
    def __init__(self):
        self.text = None
        self.filename = None

    def write(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(name="mock_files", autouse=True)
def make_mock_file(monkeypatch):
    files = []

    def mock_open(filename, flag, *args, **kwargs):
        assert flag in ("w+", "w+b", "a")
        file = MockFile()
        file.filename = filename
        files.append(file)
        return file

    monkeypatch.setattr(builtins, "open", mock_open)
    yield files


class TestMetaData:
    def test_files(self, mock_files):
        pipeline = ag.PipelineDataset(
            "pipeline_name", DummyPhaseImaging(phase_name="phase_name")
        )
        pipeline.run(dataset=MockImagingData(), mask=MockMask())

        assert (
            mock_files[2].text
            == "phase=phase_name\nphase_tag=\npipeline=pipeline_name\npipeline_tag=\nnon_linear_search=search\ndataset_name=data_name"
        )

        assert "phase_name////non_linear.pickle" in mock_files[3].filename


class TestPassMask:
    def test_pass_mask(self):
        mask = MockMask()
        phase1 = DummyPhaseImaging("one")
        phase2 = DummyPhaseImaging("two")
        pipeline = ag.PipelineDataset("", phase_1, phase_2)
        pipeline.run(dataset=MockImagingData(), mask=mask)

        assert phase1.mask is mask
        assert phase2.mask is mask


class TestPipelineImaging:
    def test_run_pipeline(self):
        phase1 = DummyPhaseImaging("one")
        phase2 = DummyPhaseImaging("two")

        pipeline = ag.PipelineDataset("", phase_1, phase_2)

        pipeline.run(dataset=MockImagingData(), mask=MockMask())

        assert len(phase2.results) == 2

    def test_addition(self):
        phase1 = DummyPhaseImaging("one")
        phase2 = DummyPhaseImaging("two")
        phase3 = DummyPhaseImaging("three")

        pipeline1 = ag.PipelineDataset("", phase_1, phase_2)
        pipeline2 = ag.PipelineDataset("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases


class DummyPhasePositions(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name):
        super().__init__(Paths(name=phase_name, tag=""))
        self.results = None
        self.pixel_scales = None
        self.search = Optimizer(phase_name)

    def run(self, pixel_scales, results):
        self.save_metadata(MockImagingData())
        self.pixel_scales = pixel_scales
        self.results = results
        return af.Result(af.ModelInstance(), 1)
