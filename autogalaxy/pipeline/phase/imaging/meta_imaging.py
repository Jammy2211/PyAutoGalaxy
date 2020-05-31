from autogalaxy.dataset import imaging
from autogalaxy.pipeline.phase.dataset import meta_dataset


class MetaImaging(meta_dataset.MetaDataset):
    def __init__(self, settings, model, is_hyper_phase=False):

        super().__init__(settings=settings, model=model, is_hyper_phase=is_hyper_phase)

    def masked_dataset_from(self, dataset, mask, results):

        mask = self.mask_with_phase_sub_size_from_mask(mask=mask)

        if self.settings.bin_up_factor is not None:

            dataset = dataset.binned_from_bin_up_factor(
                bin_up_factor=self.settings.bin_up_factor
            )

            mask = mask.binned_mask_from_bin_up_factor(
                bin_up_factor=self.settings.bin_up_factor
            )

        if self.settings.signal_to_noise_limit is not None:
            dataset = dataset.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.settings.signal_to_noise_limit
            )

        masked_imaging = imaging.MaskedImaging(
            imaging=dataset,
            mask=mask,
            grid_class=self.settings.grid_class,
            grid_inversion_class=self.settings.grid_inversion_class,
            fractional_accuracy=self.settings.fractional_accuracy,
            sub_steps=self.settings.sub_steps,
            psf_shape_2d=self.settings.psf_shape_2d,
            inversion_pixel_limit=self.settings.inversion_pixel_limit,
        )

        return masked_imaging
