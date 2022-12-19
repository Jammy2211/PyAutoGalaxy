rm -rf dist
rm -rf build
python3 setup.py bdist_wheel
python3 setup.py sdist
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --username Jammy2211 --password 2K99aw2211!