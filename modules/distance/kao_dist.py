#========[ IMPORTS ]========

from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces
import modules.utils.conversion as convert
import torch

#========[ FUNCTIONS ]========

def kaolin_point_to_mesh(cloud, face_coordinates, face_indices):
    if (type(cloud) is not torch.Tensor):
        vertices = torch.tensor([cloud], dtype=torch.float32, device='cuda')
    if (type(face_coordinates) is not torch.Tensor):
        mesh = torch.tensor([face_coordinates], dtype=torch.float32, device='cuda')
    if (type(face_indices) is not torch.Tensor):
        mesh_indices = torch.tensor(face_indices, dtype=torch.long, device='cuda')

    faces = index_vertices_by_faces(mesh, mesh_indices)

    distance, indices, dist_type = point_to_mesh_distance(vertices, faces)
    return distance.tolist(), indices.tolist(), dist_type.tolist()