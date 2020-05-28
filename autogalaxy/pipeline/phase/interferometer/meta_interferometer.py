from autogalaxy.dataset import interferometer
from autogalaxy.pipeline.phase.dataset import meta_dataset


class MetaInterferometer(meta_dataset.MetaDataset):
    def __init__(self, settings, model, real_space_mask, is_hyper_phase=False):
        super().__init__(settings=settings, model=model, is_hyper_phase=is_hyper_phase)
        self.real_space_mask = real_space_mask

    def masked_dataset_from(self, dataset, mask, results, modified_visibilities):

        real_space_mask = self.mask_with_phase_sub_size_from_mask(
            mask=self.real_space_mask
        )

        masked_interferometer = interferometer.MaskedInterferometer(
            interferometer=dataset.modified_visibilities_from_visibilities(
                modified_visibilities
            ),
            visibilities_mask=mask,
            real_space_mask=real_space_mask,
            grid_class=self.settings.grid_class,
            grid_inversion_class=self.settings.grid_inversion_class,
            fractional_accuracy=self.settings.fractional_accuracy,
            sub_steps=self.settings.sub_steps,
            transformer_class=self.settings.transformer_class,
            primary_beam_shape_2d=self.settings.primary_beam_shape_2d,
            inversion_pixel_limit=self.settings.inversion_pixel_limit,
        )

        return masked_interferometer
