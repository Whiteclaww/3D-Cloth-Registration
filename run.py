#========[ IMPORTS ]========
import main
import main_test

#========[ FUNCTIONS ]========

tests = main_test.Test_Main()
tests.test_plan()

## NRICP
# Best
#main.main("test_dataset/00006/Top.obj", "test_dataset/smpl_highres_female.npz", "top_aligned_smpl_female.obj")
#main.main("dataset/test_t1/00006/Top.obj", "dataset/smpl_highres_female.npz", "top_aligned_smpl_female.obj")
#main.main("dataset/test_t1/00001/Jumpsuit.obj", "dataset/smpl_highres_female.npz", "Jumpsuit_aligned_smpl_female.obj")

# Decent
#main.main("dataset/test_t1/00007/Tshirt.obj", "dataset/smpl_highres_female.npz", "tshirt_aligned_smpl_female.obj")

# Worse
#main.main("dataset/test_t1/00007/Trousers.obj", "dataset/smpl_highres_female.npz", "trousers_aligned_smpl_female.obj")
#main.main("dataset/test_t1/00004/Dress.obj", "dataset/smpl_highres_female.npz", "dress_aligned_smpl_female.obj")

## DISTANCE
#main.main("test_dataset/starterkit/Tshirt.obj", "test_dataset/starterkit/ex_highres_smpl.obj", "test_dataset/starterkit/ts_simplified.obj")
main.main("test_dataset/00006_top_smoothed.obj", "test_dataset/starterkit/ex_highres_smpl.obj")
#main.main("test_dataset/starterkit/Trousers.obj", "test_dataset/starterkit/ex_highres_smpl.obj", "test_dataset/starterkit/ts_simplified.obj")