import autofit as af
import pickle


class Analysis(af.Analysis):
    def __init__(self, cosmology, settings):

        self.cosmology = cosmology
        self.settings = settings

    def save_settings(self, paths: af.Paths):
        with open(f"{paths.pickle_path}/settings.pickle", "wb+") as f:
            pickle.dump(self.settings, f)
