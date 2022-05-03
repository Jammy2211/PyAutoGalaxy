.. _overview_3_modeling:

Modeling
========

We can use a ``Plane`` to fit data of a galaxy and quantify its goodness-of-fit via a
*log_likelihood*.

Of course, when observe an image of a galaxy, we have no idea what combination of
``LightProfile``'s will produce a model-image that looks like the galaxy we observed:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/master/docs/overview/images/fitting/image.png
  :width: 400
  :alt: Alternative text

The task of finding these ``LightProfiles``'is called *modeling*.

PyAutoFit
---------

Modeling with **PyAutoGalaxy** uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import it separately to **PyAutoGalaxy**

.. code-block:: python

    import autofit as af

Model Composition
-----------------

We compose the model that we fit to the data using a ``Model`` object, which behaves analogously to the ``Galaxy``,
and ``LightProfile`` used previously, however their parameters are not specified and are instead
determined by a fitting procedure.

.. code-block:: python

    galaxy = af.Model(
        ag.Galaxy, redshift=0.5, bulge=ag.lp.EllSersic, disk=ag.lp.EllExponential
    )

We put the model galaxy above into a `Collection`, which is the model we will fit. Note how we could easily 
extend this object to compose complex models containing many galaxies.

The reason we create separate `Collection`'s for the `galaxies` and `model` is so that the `model`
can be extended to include other components than just galaxies.

.. code-block:: python

    galaxies = af.Collection(galaxy=galaxy)
    model = af.Collection(galaxies=galaxies)

In this example we therefore fit a model where:

 - The galaxy's bulge is a parametric `EllSersic` bulge [7 parameters]. 
 - The galaxy's disk is a parametric `EllExponential` disk [6 parameters].

The redshifts of the galaxy (z=0.5) is fixed.

Non-linear Search
-----------------

We now choose the non-linear search, which is the fitting method used to determine the set of `LightProfile` (e.g.
bulge and disk) parameters that best-fit our data.

In this example we use `dynesty` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm that is
very effective at modeling.

.. code-block:: python

    search = af.DynestyStatic(name="search_example")

**PyAutoGalaxy** supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.

Analysis
--------

We next create an ``AnalysisImaging`` object, which contains the ``log likelihood function`` that the non-linear
search calls to fit the lens model to the data.

.. code-block:: python

    analysis = ag.AnalysisImaging(dataset=imaging)

Model-Fit
---------

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
dynesty samples, model parameters, visualization) to hard-disk.

.. code-block:: python

    result = search.fit(model=model, analysis=analysis)

The non-linear search fits the model by guessing many models over and over iteratively, using the models which
give a good fit to the data to guide it where to guess subsequent model. 

An animation of a non-linear search is shown below, although this is for a strong gravitational lens using
**PyAutoGalaxy**'s child project **PyAutoLens**. Updating the animation for a galaxy is on the **PyAutoGalaxy**
to-do list!

We can see that initial models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true
  :width: 600

**Credit: Amy Etherington**

Results
-------

Once a model-fit is running, **PyAutoGalaxy** outputs the results of the search to hard-disk on-the-fly. This includes
model parameter estimates with errors non-linear samples and the visualization of the best-fit model inferred
by the search so far.

The fit above returns a ``Result`` object, which includes lots of information on the model. 

Below we print the maximum log likelihood model inferred.

.. code-block:: python

    print(result.max_log_likelihood_instance.galaxies.galaxy.bulge)
    print(result.max_log_likelihood_instance.galaxies.galaxy.disk)

This result contains the full posterior information of our non-linear search, including all
parameter samples, log likelihood values and tools to compute the errors on the lens model. 

**PyAutoGalaxy** includes many visualization tools for plotting the results of a non-linear search, for example we can
make a corner plot of the probability density function (PDF):

.. code-block:: python

    dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
    dynesty_plotter.cornerplot()

Here is an example of how a PDF estimated for a model appears:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/master/docs/overview/images/modeling/cornerplot.png
  :width: 600
  :alt: Alternative text

The result also contains the maximum log likelihood ``Plane`` and ``FitImaging`` objects and which can easily be
plotted.

.. code-block:: python

    plane_plotter = aplt.PlanePlotter(plane=result.max_log_likelihood_plane, grid=mask.masked_grid)
    plane_plotter.subplot_plane()

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_imaging_plotter.subplot_fit_imaging()

Here's what the model-fit of the model which maximizes the log likelihood looks like, providing good residuals and
low chi-squared values:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/master/docs/overview/images/modeling/subplot_fit.png
  :width: 600
  :alt: Alternative text

The package ``autogalaxy_workspace/notebooks/results`` contains a full description of all information contained
in a ``Result``.

Model Customization
-------------------

The ``Model`` can be fully customized, making it simple to parameterize and fit many different models
using any combination of light profiles:

.. code-block:: python

    galaxy_model = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp.EllDevVaucouleurs,
        disk = ag.lp.EllSersic,
        bar=ag.lp.EllGaussian,
        clump_0=ag.lp.EllEff,
        clump_1=ag.lp.EllEff,
    )

    """
    This aligns the bulge and disk centres in the galaxy model, reducing the
    number of free parameter fitted for by Dynesty by 2.
    """
    galaxy_model.bulge.centre = galaxy_model.disk.centre

    """
    This fixes the galaxy bulge light profile's effective radius to a value of
    0.8 arc-seconds, removing another free parameter.
    """
    galaxy_model.bulge.effective_radius = 0.8

    """
    This forces the light profile disk's effective radius to be above 3.0.
    """
    galaxy_model.bulge.add_assertion(galaxy_model.disk.effective_radius > 3.0)

Wrap-Up
-------

Chapters 2 and 3 **HowToGalaxy** lecture series give a comprehensive description of modeling, including a
description of what a non-linear search is and strategies to fit complex model to data in efficient and
robust ways.


