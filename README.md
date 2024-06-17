# 3D-Cloth-Registration
The project is about making a database for an AI about garment recognition. The database is a compact representation of 3D garments worn on the body. This project is the code used to create the database based off of CLOTH3D's database.

Here is the link to my report [UNAVAILABLE], it includes previous reports

## Requirements
For the database you might want to download the SMPL bodies from my report and download the CLOTH3D database. For now there is a sample in [`test_dataset`](test_dataset/).

To install the requirements, clone the project then write:
`conda install --file conda_requirements_nvidia.yml`
`conda install --file conda_requirements.yml`
`pip install -r path/to/pip_requirements.txt`

Then manually install the remaining libraries by reading [the manual requirements file](tools/manual_requirements.md). There will be instructions provided in the file. Those are mostly due to them either not working when installing with conda or pip or because installing manually is the only way.

## Run the project
To run the project, the [`run.py`](run.py) file is given in the code. For now there is a default line inside which will successfully run the project. To test with other files, modify the `run.py` file to run the project. If you want to turn on testing, uncomment the lines in the "test" section of the file and/or use the object "tests" already given. Just type `tests.` and different tests will show up. To see the full class, it is [`main_test,py`](main_test.py).

To increase / decrease the logging level, go in [`main.py`](main.py) line 21 and instead of `INFO`, write the level you want. Default is `INFO`.

## Testing the project
All tests are in [`testing/`](testing/). If you want to add tests to a certain function, open the `test_[name]` file, with `[name]` the name of the file that the function is in. For example, testing any functions in `nricp.py` will be in `test_nricp.py`. If you add a function make sure to add it within the `main_test.py` class for it to be run. Then in `run.py` write `tests.test_[module name].py`. For example, the `module` of `chamfer.py` is `distance`.

## Adding files to the project
If you add a file in the project, make sure to add it in the corresponding module. Then in the module's `__init__.py` add the following line:
`from .[new file name] import *`. This will ensure that you have access to the file anywhere in the project. To access your file from for example the `testing` folder, in the test file write `from modules.[module name] import [file name]`. To access within a neighbour file write `from .[file name] import ...`.