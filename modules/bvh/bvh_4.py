#========[ IMPORTS ]========
import numpy as np
from modules.distance.chamfer import dist_face

#========[ FUNCTIONS ]========

class BoundingBox:
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)

    def intersects(self, other):
        return np.all(self.min_point <= other.max_point) and np.all(self.max_point >= other.min_point)

    def distance_to_point(self, point):
        clamped_point = np.maximum(self.min_point, np.minimum(self.max_point, point))
        return np.linalg.norm(clamped_point - point)

class BVHNode:
    def __init__(self, faces, bounding_box):
        self.faces = faces
        self.bounding_box:BoundingBox = bounding_box
        self.left = None
        self.right = None

def build_bvh(faces:list, depth = 0):
    if len(faces) == 0:
        return None
    
    if len(faces) == 1:
        bounding_box = compute_bounding_box(faces)
        return BVHNode(faces, bounding_box)

    axis = depth % 3
    
    faces.sort(key=lambda face: face_centroid(face)[axis])
    mid = len(faces) // 2

    left_faces = faces[:mid]
    right_faces = faces[mid:]

    left_node = build_bvh(left_faces, depth + 1)
    right_node = build_bvh(right_faces, depth + 1)

    combined_faces = left_faces + right_faces
    bounding_box = compute_bounding_box(combined_faces)

    node = BVHNode(combined_faces, bounding_box)
    node.left = left_node
    node.right = right_node

    return node

def compute_bounding_box(faces):
    vertices = np.concatenate([face for face in faces])
    min_point = np.min(vertices, axis=0)
    max_point = np.max(vertices, axis=0)
    return BoundingBox(min_point, max_point)

def face_centroid(face) -> list:
    return np.mean(face, axis=0)

def distance_point_to_face(face, point):
    A, B, C, D = face  # Now it handles faces with 4 vertices
    # Calculate the normal vector of the face
    normal = np.cross(B - A, C - A)
    normal = (normal / np.linalg.norm(normal))
    distance_to_plane = np.dot(point - A, normal)
    projected_point = point - distance_to_plane * normal

    # Check if the projected point is inside the face
    if is_point_in_quad(projected_point, A, B, C, D):
        return np.linalg.norm(point - projected_point)
    else:
        # If the point is not inside the face, compute distance to the face edges
        return min(distance_point_to_segment(point, A, B),
                   distance_point_to_segment(point, B, C),
                   distance_point_to_segment(point, C, D),
                   distance_point_to_segment(point, D, A))

def is_point_in_quad(p, A, B, C, D):
    return is_point_in_triangle(p, A, B, C) or is_point_in_triangle(p, A, C, D)

def is_point_in_triangle(p, A, B, C):
    v0 = C - A
    v1 = B - A
    v2 = p - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v < 1)

def distance_point_to_segment(point, A, B):
    AB = B - A
    t = np.dot(point - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0, 1)
    projection = A + t * AB
    return np.linalg.norm(point - projection)

def find_closest_face(bvh, point):
    def search_bvh(node:BVHNode, point, best_face, best_distance):
        if node is None:
            return best_face, best_distance

        if node.bounding_box.distance_to_point(point) > best_distance:
            return best_face, best_distance

        for face in node.faces:
            distance = distance_point_to_face(face, point)
            if distance < best_distance:
                best_face = face
                best_distance = distance

        best_face, best_distance = search_bvh(node.left, point, best_face, best_distance)
        best_face, best_distance = search_bvh(node.right, point, best_face, best_distance)

        return best_face, best_distance

    return search_bvh(bvh, point, None, float('inf'))

def example():
    # Exemple d'utilisation
    # Définir les faces du maillage C comme des tableaux numpy de sommets
    faces_C = [
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0], [1, -1, 0]]),
        # Ajouter d'autres faces
    ]

    # Construire le BVH
    bvh_root = build_bvh(faces_C)

    # Définir les sommets du maillage B comme des points numpy
    vertices_B = [
        np.array([0.1, 0.1, 0.1]),
        np.array([0.5, 0.5, 0.5]),
        # Ajouter d'autres sommets
    ]

    # Trouver les faces les plus proches pour chaque sommet de B
    result = []
    for vertex in vertices_B:
        closest_face = find_closest_face(bvh_root, vertex)
        result.append([vertex, closest_face])
        print(f"Le sommet {vertex} a la face la plus proche\n{closest_face}")
    
    return result

def bvh(faces:list[np.ndarray], vertices:list[np.ndarray]):
    # Construire le BVH
    bvh_root = build_bvh(faces)

    # Trouver les faces les plus proches pour chaque sommet de B
    result = []
    for vertex in vertices:
        closest_face = find_closest_face(bvh_root, vertex)
        result.append([vertex, closest_face])
        print(f"Le sommet {vertex} a la face la plus proche\n{closest_face}")
    
    return result