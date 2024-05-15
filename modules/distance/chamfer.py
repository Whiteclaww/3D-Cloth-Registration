#========[ IMPORTS ]========
import sys

from chamferdist import ChamferDistance
import math
import timeit
from torch import Tensor

from modules.utils.classes import SMPLModel, Garment

#========[ CLASSES ]========

class Plan():
    def __init__(self, A:list, B:list, C:list) -> None:
        self.A = A
        self.B = B
        self.C = C
        AB = vector(A, B)
        AC = vector(A, C)
        produit:list = produit_vectoriel(AB, AC)
        self.n = divide(produit, norm(produit))

#========[ FUNCTIONS ]========

def produit_scalaire(vect1:list, vect2:list) -> float:
    result = 0
    for i in range(3):
        result += vect1[i] * vect2[i]
    return result
    
def divide(vect:list, value:float) -> list:
    return [vect[0] / value,
            vect[1] / value,
            vect[2] / value]

def vector(A:list, B:list) -> list:
    return [B[0] - A[0],
            B[1] - A[1],
            B[2] - A[2]]
    
def norm(vect:list) -> float:
    return math.sqrt(vect[0] * vect[0] +
                     vect[1] * vect[1] +
                     vect[2] * vect[2])

def produit_vectoriel(AB:list, AC:list) -> list:
    """Produit vectoriel AB^AC

    Args:
        AB (list)
        AC (list)
    """
    result = [0] * 3
    for i in range(3):
        result[(i + 2) % 3] = AB[i] * AC[(i + 1) % 3] - AB[(i + 1) % 3] * AC[i]
    return result


def chamfer_distance(chamfer:ChamferDistance, source:Garment, target:SMPLModel):
    torch_garment = Tensor([source.vertices])
    torch_smpl = Tensor([target.vertices])
    result = chamfer.forward(source_cloud=torch_smpl, target_cloud=torch_garment, batch_reduction="mean")
    return result
    
def dist_face(M:list, face:list):
    """Calculates the distance between M and the face ABC

    Args:
        M (list): vertex\n
        plan (list):
        -- A (list): first vertex on face
        -- B (list): second vertex on face
        -- C (list): third vertex on face

    Returns:
        list: vector between M and the plan ABC
    """
    start = timeit.timeit()
    A, B, C = face
    plan = Plan(A, B, C)
    n_MA = produit_scalaire(plan.n, vector(M, A))
    dist = abs(n_MA) / norm(plan.n)
    end = timeit.timeit()
    print(end - start)
    return dist