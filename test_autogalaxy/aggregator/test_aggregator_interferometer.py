from os import path

from autoconf import conf
import autofit as af
import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean


def test__interferometer_generator_from_aggregator__analysis_has_single_dataset(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
    samples,
    model,
):
    path_prefix = "aggregator_interferometer"

    database_file = path.join(conf.instance.output_path, "interferometer.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    interferometer_7 = ag.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        settings=ag.SettingsInterferometer(
            grid_class=ag.Grid2DIterate,
            grid_pixelization_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
            transformer_class=ag.TransformerDFT,
        ),
    )

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    dataset_agg = ag.agg.InterferometerAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == interferometer_7.data).all()
        assert (dataset_list[0].real_space_mask == mask_2d_7x7).all()
        assert isinstance(dataset_list[0].grid, ag.Grid2DIterate)
        assert isinstance(dataset_list[0].grid_pixelization, ag.Grid2DIterate)
        assert dataset_list[0].grid.sub_steps == [2]
        assert dataset_list[0].grid.fractional_accuracy == 0.5
        assert isinstance(dataset_list[0].transformer, ag.TransformerDFT)
        assert isinstance(dataset_list[0].settings, ag.SettingsInterferometer)

    clean(database_file=database_file, result_path=result_path)


def test__interferometer_generator_from_aggregator__analysis_multi(
    analysis_interferometer_7,
    samples,
    model,
):
    path_prefix = "aggregator_interferometer"

    database_file = path.join(conf.instance.output_path, "interferometer.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = analysis_interferometer_7 + analysis_interferometer_7

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    dataset_agg = ag.agg.InterferometerAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:

        assert (dataset_list[0].data == analysis_interferometer_7.dataset.data).all()
        assert (dataset_list[1].data == analysis_interferometer_7.dataset.data).all()

    clean(database_file=database_file, result_path=result_path)
