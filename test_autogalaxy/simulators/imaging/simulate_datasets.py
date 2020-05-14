from test_autogalaxy.simulators.imaging import simulators

"""
Welcome to the PyAutoGalaxy dataset generator. Here, we'll make the datasets used to test and profile PyAutoGalaxy.

This consists of the following sets of images:

- An image where the galaxy is an elliptical Dev Vaucouleurs profile.
- An image where the galaxy is a bulge (Dev Vaucouleurs) + Envelope (Exponential) profile.
- An image where there are two galaxies both composed of Sersic bulges.
- An image where the galaxy is an elliptical Dev Vaucouleurs profile with a centre offset from the image centre.

Each image is generated for up to 5 instrument, VRO, Euclid, HST, HST, Keck AO.
"""

instruments = ["vro", "euclid", "hst"]  # , "hst_up", "ao"]

# To simulate each galaxy, we pass it an instrument and call its simulate function.

for instrument in instruments:

    simulators.simulate__galaxy_x1__dev_vaucouleurs(instrument=instrument)
    simulators.simulate__galaxy_x1__bulge_disk(instrument=instrument)
    simulators.simulate__galaxy_x2__sersics(instrument=instrument)
    simulators.simulate__galaxy_x1__dev_vaucouleurs__offset_centre(
        instrument=instrument
    )
