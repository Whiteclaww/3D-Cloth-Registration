#========[ IMPORTS ]========
from modules.distance import kao_dist

#========[ FUNCTIONS ]========

def test_kaolin_point_to_mesh():
    cloud = [[0, 0, 0], [1, 1, 1]]
    mesh = [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 2, 0], [0, 0, 0]]]

    kao_dist.kaolin_point_to_mesh(cloud, mesh)