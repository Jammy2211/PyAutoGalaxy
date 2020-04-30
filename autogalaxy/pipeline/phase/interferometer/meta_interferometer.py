from autogalaxy.dataset import interferometer
from autogalaxy.pipeline.phase.dataset import meta_dataset


class MetaInterferometer(meta_dataset.MetaDataset):
    def __init__(
        self,
        model,
        real_space_mask,
        transformer_class,
        sub_size=2,
        is_hyper_phase=False,
        inversion_pixel_limit=None,
        primary_beam_shape_2d=None,
        bin_up_factor=None,
    ):
        super().__init__(
            model=model,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            inversion_pixel_limit=inversion_pixel_limit,
        )
        self.real_space_mask = real_space_mask
        self.transformer_class = transformer_class
        self.primary_beam_shape_2d = primary_beam_shape_2d
        self.bin_up_factor = bin_up_factor

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
            transformer_class=self.transformer_class,
            primary_beam_shape_2d=self.primary_beam_shape_2d,
            inversion_pixel_limit=self.inversion_pixel_limit,
        )

        return masked_interferometer
