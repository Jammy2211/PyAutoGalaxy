import autoarray.plot as aplt


class Include1D(aplt.Include1D):
    def __init__(self, half_light_radius=None, einstein_radius=None):
        """
        Sets which `Visuals1D` are included on a figure plotting 1D data that is plotted using a `Plotter1D`.

        The `Include` object is used to extract the visuals of the plotted 1D data structures so they can be used in
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        half_light_radius
            If `True`, the `half_light_radius` of the plotted light profile is included on the figure.
        einstein_radius
            If `True`, the `einstein_radius` of the plotted mass profile is included on the figure.
        """
        super().__init__()

        self._half_light_radius = half_light_radius
        self._einstein_radius = einstein_radius

    @property
    def half_light_radius(self):
        return self.load(value=self._half_light_radius, name="half_light_radius")

    @property
    def einstein_radius(self):
        return self.load(value=self._einstein_radius, name="einstein_radius")
