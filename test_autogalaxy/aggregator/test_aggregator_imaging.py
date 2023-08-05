from os import path

from autoconf import conf
import autofit as af
import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean


def test__dataset_generator_from_aggregator__analysis_has_single_dataset(imaging_7x7, mask_2d_7x7, samples, model):
    path_prefix = "aggregator_dataset_gen"

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

    dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:

        assert (dataset_list[0].data == masked_imaging_7x7.data).all()
        assert isinstance(dataset_list[0].grid, ag.Grid2DIterate)
        assert isinstance(dataset_list[0].grid_pixelization, ag.Grid2DIterate)
        assert dataset_list[0].grid.sub_steps == [2]
        assert dataset_list[0].grid.fractional_accuracy == 0.5
        assert isinstance(dataset_list[0].settings, ag.SettingsImaging)

    clean(database_file=database_file, result_path=result_path)


def test__dataset_generator_from_aggregator__analysis_multi(analysis_imaging_7x7, samples, model):
    path_prefix = "aggregator_dataset_gen"

    database_file = path.join(conf.instance.output_path, "imaging.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = analysis_imaging_7x7 + analysis_imaging_7x7

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:

        assert (dataset_list[0].data == analysis_imaging_7x7.dataset.data).all()
        assert (dataset_list[1].data == analysis_imaging_7x7.dataset.data).all()

    clean(database_file=database_file, result_path=result_path)