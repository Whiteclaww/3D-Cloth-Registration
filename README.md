# 3D-Cloth-Registration
The project is about making a database for an AI about garment recognition. The database is a compact representation of 3D garments worn on the body. This project is the code used to create the database based off of CLOTH3D's database.

## How to use
Clone or download the project, download the SMPL bodies, then download the CLOTH3D database. Modify the `test.py` file or run `main` with:
* `garment_file` : The file path including the file name to the garment file (has to be a .obj), ex: `"dataset/test_t1/00006/Top.obj"`
* `smpl_file` : The file path including the file name to the body / SMPL (could be a .obj or a .npz file), ex: `"dataset/smpl_highres_female.npz"`
* `aligned_garment_file` : The file path of the file you want to save the garment as (has to include .obj at the end)
* [optional] `i_iterations` : The number of iterations of the ICP (the higher the number of iterations the longer it takes)
* [optional] `j_iterations` : The number of iterations to calculate matrices
* [optional] `alpha` : The weight (the heavier the less the garment will transform)
