# 3D-Cloth-Registration
The project is about making a database for an AI about garment recognition. The database is a compact representation of 3D garments worn on the body. This project is the code used to create the database based off of CLOTH3D's database.

Here is the [link to my report](https://docs.google.com/document/d/1G_IEzwUEEl-3eR2ekM1aVFeGACRdNuclQOZnt125qjg/edit?usp=sharing "Marine Collet's Google Docs Report"), it includes previous reports

## How to use
Clone or download the project, download the SMPL bodies, then download the CLOTH3D database. Modify the `run.py` file to run the project.

To install the requirements, write:
`conda install --file conda_requirements_nvidia.yml`
`conda install --file conda_requirements.yml`
`pip install -r path/to/pip_requirements.txt`

Then manually install the remaining libraries by reading [the manual requirements file](manual_requirements.md). There will be instructions provided in the file. Those are mostly due to them either not working when installing with conda or pip or because installing manually is the only way.