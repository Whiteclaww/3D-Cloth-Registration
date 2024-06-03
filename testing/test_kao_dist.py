#========[ IMPORTS ]========
from modules.distance import kao_dist

#========[ FUNCTIONS ]========

'''def test_kaolin_point_to_mesh_1():
    cloud = [[0, 0, 0], [1, 1, 1]]
    mesh = [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 2, 0], [0, 0, 0]]]

    kao_dist.kaolin_point_to_mesh(cloud, mesh)'''

def test_kaolin_point_to_mesh_2():
    cloud = [[0, 0.5, 0.5], [3., 4., 5.]]
    mesh = [[0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [4, 4, 4],
            [0, 4, 4],
            [4, 0, 4]]
    face_indices = [[0, 1, 2], [3, 4, 5]]

    dist, index, dist_type = kao_dist.kaolin_point_to_mesh(cloud, mesh, face_indices)

    assert dist.tolist() == [[0., 1.]]
    assert index.tolist() == [[0, 1]]
    assert dist_type.tolist() == [[5, 4]]