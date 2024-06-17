#========[ IMPORTS ]========
from modules.utils.classes import Object

#from chamferdist import ChamferDistance
import math
from torch import Tensor#

#========[ FUNCTIONS ]========

'''def chamfer_distance(chamfer:ChamferDistance, source:Object, target:Object):
    torch_garment = Tensor([source.vertices])
    torch_smpl = Tensor([target.vertices])
    result = chamfer.forward(source_cloud=torch_smpl, target_cloud=torch_garment, batch_reduction="mean")
    return result'''