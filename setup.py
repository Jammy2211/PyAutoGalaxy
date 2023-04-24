import os
from codecs import open
from os import environ
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.rst"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

version = environ.get("VERSION", "1.0.dev0")
requirements.extend(
    [f"autoconf=={version}", f"autoarray=={version}", f"autofit=={version}"]
)


def config_packages(directory):
    paths = [directory.replace("/", ".")]
    for path, directories, filenames in os.walk(directory):
        for directory in directories:
            paths.append(f"{path}/{directory}".replace("/", "."))
    return paths


setup(
    name="autogalaxy",
    version=version,
    description="Open-Source Multi Wavelength Galaxy Structure & Morphology",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Jammy2211/PyAutoGalaxy",
    author="James Nightingale and Richard Hayes",
    author_email="james.w.nightingale@durham.ac.uk",
    include_package_data=True,
    license="MIT License",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    keywords="cli",
    packages=find_packages(exclude=["docs", "test_autogalaxy", "test_autogalaxy*"])
    + config_packages("autogalaxy/config"),
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
