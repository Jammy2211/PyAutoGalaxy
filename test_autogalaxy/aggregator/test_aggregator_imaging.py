from os import path

from autoconf import conf
import autofit as af
import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean


def test__imaging_generator_from_aggregator(imaging_7x7, mask_2d_7x7, samples, model):
    path_prefix = "aggregator_imaging_gen"

    database_file = path.join(conf.instance.output_path, "imaging.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(
            grid_class=ag.Grid2DIterate,
            grid_pixelization_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
        )
    )

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    imaging_agg = ag.agg.ImagingAgg(aggregator=agg)
    imaging_gen = imaging_agg.dataset_gen_from()

    for imaging in imaging_gen:
        assert (imaging.image == masked_imaging_7x7.image).all()
        assert isinstance(imaging.grid, ag.Grid2DIterate)
        assert isinstance(imaging.grid_pixelization, ag.Grid2DIterate)
        assert imaging.grid.sub_steps == [2]
        assert imaging.grid.fractional_accuracy == 0.5

    clean(database_file=database_file, result_path=result_path)
