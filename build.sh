rm -rf build/ dist/
python setup.py bdist_wheel
# python setup.py sdist bdist_wheel
# twine check dist/*
# twine upload dist/*
