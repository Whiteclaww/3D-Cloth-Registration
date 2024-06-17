#========[ IMPORTS ]========
import logging
import math

#========[ CLASSES ]========

class Plan():
    def __init__(self, face:list = [], A:list = [], B:list = [], C:list = []) -> None:
        if face != []:
            A = face[0]
            B = face[1]
            C = face[2]
        self.A = A
        self.B = B
        self.C = C
        AB = vector(A, B)
        AC = vector(A, C)
        produit:list = produit_vectoriel(AB, AC)
        self.n = divide(produit, norm(produit))

#========[ FUNCTIONS ]========

def produit_vectoriel(AB:list, AC:list) -> list:
    result = [0] * 3
    for i in range(3):
        result[(i + 2) % 3] = AB[i] * AC[(i + 1) % 3] - AB[(i + 1) % 3] * AC[i]
    return result

def produit_scalaire(vect1:list, vect2:list) -> float:
    result = 0
    for i in range(3):
        result += vect1[i] * vect2[i]
    return result

def divide(vect:list, value:float) -> list:
    if value == 0:
        return None
    return [vect[0] / value,
            vect[1] / value,
            vect[2] / value]

def vector(A:list, B:list) -> list:
    return [B[0] - A[0],
            B[1] - A[1],
            B[2] - A[2]]

def mul(vect:list, value) -> list:
    return [vect[0] * value,
            vect[1] * value,
            vect[2] * value]
    
def norm(vect:list) -> float:
    return math.sqrt(vect[0] * vect[0] +
                     vect[1] * vect[1] +
                     vect[2] * vect[2])

def sum_vect(point:list, vector:list, value = 1) -> list:
    return [point[0] + value * vector[0],
            point[1] + value * vector[1],
            point[2] + value * vector[2]]

def dist_face(M:list, face:list) -> float:
    """Calculates the distance between M and the face ABC

    Args:
        M (list): vertex\n
        plan (list):isn
        -- A (list): first vertex on face
        -- B (list): second vertex on face
        -- C (list): third vertex on face

    Returns:
        list: vector between M and the plan ABC
    """
    A, B, C = face
    plan = Plan(A = A, B = B, C = C)
    n_MA = produit_scalaire(plan.n, vector(M, A))
    dist = abs(n_MA)
    return dist

def dist_triangle(M:list, face:list) -> bool:
    A, B, C = face
    nMA = norm(vector(M, A))
    nMB = norm(vector(M, B))
    nMC = norm(vector(M, C))
    if (nMA + nMB + nMC) / 3 > 0.02:
        return False
    return True
    
def same_norm_as_plan(vect1, plan):
    n1 = divide(vect1, norm(vect1))
    for i in range(3):
        if not math.isclose(n1[i], plan.n[i], abs_tol = 0.2):
            return False
    return True

def dist_proj_to_triangle(P:list, face:list) -> bool:
    for i in range(3):
        niP = norm(vector(face[i], P))
        niB = norm(vector(face[i], face[(i + 1) % 3]))
        niC = norm(vector(face[i], face[(i + 2) % 3]))
        if niP > niB and niP > niC:
            return False
    return True

#def is_above_face(M:list, face:list)

def find_vertex(M:list, face:list, dist):
    if dist_face(M, face) > 0.1:
        #logging.error("Find_vertex: the vertex is too far!")
        return None
    # norm(plan.n) should be equal to 1, but because of imprecicsion we will take 1 instead
    A, B, C = face
    plan = Plan(A = A, B = B, C = C)
    projected_point = sum_vect(M, plan.n, produit_scalaire(plan.n, vector(M, A)))
    a = dist_face(projected_point, face)
    if not same_norm_as_plan(vector(M, projected_point), plan):
        return None
    #if not math.isclose(a, 0, abs_tol=1e-7):
        #logging.error("Find_vertex: the projected point is not close to the face!")
    if not dist_triangle(projected_point, face):
        return None
    if not dist_proj_to_triangle(projected_point, face):
        return None
    #if dist_face(projected_point, face) > 0.001:
    #    return None
    return projected_point

def find_center(plan:Plan):
    centroid = []
    for i in range(3):
        centroid.append((plan.A[i] + plan.B[i] + plan.C[i]) / 3)
    return centroid
