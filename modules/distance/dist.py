#========[ IMPORTS ]========

from .plan import *
from .kao_dist import kaolin_point_to_mesh

#========[ FUNCTIONS ]========

def associate_faces(projected_garment_vertices, garment_faces, smpl_vertices):
    link = points_to_mesh(smpl_vertices, projected_garment_vertices, garment_faces)

def points_to_faces(smpl_vertices, garment_vertices, garment_faces, links):
    return 0

def corresponding_vertices(A_vertices:list, distance:list, indices:list, dist_type:list, B_faces_indices:list, B_vertices:list):
    correspond = []
    new_vertices = []
    for i in range(len(A_vertices)):
        vertex = A_vertices[i]
        face = B_faces_indices[indices[i]]
        face_coordinates = []
        for index in face:
            face_coordinates.append(B_vertices[index])
        dist = distance[i]
        dtype = dist_type[i]
        if dtype > 0 and dtype < 4:
            new_vertices.append(face_coordinates[dtype - 1])
            correspond.append(vertex)
        else:
            new = find_vertex(vertex, face_coordinates, dist)
            if new == None:
                continue
            new_vertices.append(new)
            correspond.append(vertex)
    return new_vertices, correspond

def points_to_mesh(cloud, face_coordinates, face_indices):
    dist, indices, dist_type = kaolin_point_to_mesh(cloud, face_coordinates, face_indices)
    #zero_smpl, new_smpl = simplify(cloud, dist)
    #dist, indices, dist_type = kaolin_point_to_mesh(new_smpl, face_coordinates, face_indices)
    new_verts, corresponding_smpl = corresponding_vertices(cloud, dist[0], indices[0], dist_type[0], face_indices, face_coordinates)
    return new_verts, dist, cloud, corresponding_smpl

def calculate_norms(faces, vertices):
    norms = []
    for i in faces:
        face = []
        for index in i:
            face.append(vertices[index])
        norms.append(Plan(face = face).n)
    return norms

def calculate_vectors(cloud1, cloud2) -> list:
    """
    Args:
        cloud1 (list): list of M
        cloud2 (list): list of projected points

    Returns:
        list
    """
    vectors = []
    length = len(cloud1)
    if length != len(cloud2):
        return []
    for i in range(length):
        vectors.append(vector(cloud1[i], cloud2[i]))
    return vectors

def calculate_centers(faces, vertices):
    centroids = []
    for i in faces:
        face = []
        for index in i:
            face.append(vertices[index])
        centroids.append(find_center(Plan(face = face)))
    return centroids

def simplify(smpl, distances):
    new_smpl = []
    distances = distances[0]
    length = len(smpl)
    for i in range(length):
        if distances[i] > 0.005:
            smpl[i] = [0, 0, 0]
        else:
            new_smpl.append(smpl[i])
    return smpl, new_smpl