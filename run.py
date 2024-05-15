#========[ IMPORTS ]========
import main
import main_test

#========[ FUNCTIONS ]========

tests = main_test.Test_Main()



## NRICP
# Best
#main.main("dataset/test_t1/00006/Top.obj", "dataset/smpl_highres_female.npz", "top_aligned_smpl_female.obj")
#main.main("dataset/test_t1/00001/Jumpsuit.obj", "dataset/smpl_highres_female.npz", "Jumpsuit_aligned_smpl_female.obj")

# Decent
#main.main("dataset/test_t1/00007/Tshirt.obj", "dataset/smpl_highres_female.npz", "tshirt_aligned_smpl_female.obj")

# Worse
#main.main("dataset/test_t1/00007/Trousers.obj", "dataset/smpl_highres_female.npz", "trousers_aligned_smpl_female.obj")
#main.main("dataset/test_t1/00004/Dress.obj", "dataset/smpl_highres_female.npz", "dress_aligned_smpl_female.obj")

## NEAREST NEIGHBOURS
main.main("starterkit/Tshirt.obj", "starterkit/ex_highres_smpl.obj", "starterkit/ts_simplified.obj")