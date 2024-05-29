#========[ IMPORTS ]========
from tools.kaolin.kaolin.metrics.trianglemesh import point_to_mesh_distance

import torch

#========[ FUNCTIONS ]========

def kaolin_point_to_mesh(cloud, face_n_vertices):
    vertices = torch.tensor(cloud, device='cuda')
    mesh = torch.tensor(face_n_vertices, device='cuda')
    a = point_to_mesh_distance(vertices, mesh)
    if a is not None:
        print("a")
    print("b")
    return a