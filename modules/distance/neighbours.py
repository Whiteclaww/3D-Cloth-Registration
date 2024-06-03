#========[ IMPORTS ]========
from math import dist
import numpy as np
from sklearn.neighbors import NearestNeighbors

#========[ FUNCTIONS ]========

def get_nearest(vertex, possible_values:np.ndarray):
    nearest_index = -1
    nearest_norm = 5
    length = len(possible_values)
    for i in range(length):
        norm = dist(vertex, possible_values[i])
        if norm < 0.04 and nearest_norm > norm:
            nearest_norm = norm
            nearest_index = i
    return nearest_index

def find(array:np.ndarray, vertex):
    result = -1
    for i in range(len(array)):
        if array[i, 1] >= vertex[1] and array[i, 1] <= vertex[1] + 0.000002:
            result = i
    return result


def nearest_neighbours_using_sklearn(garment:np.ndarray, smpl:np.ndarray):
    """
    Applies nearest-neighbours algorithm from garment to smpl

    Args:
        garment (np.ndarray)
        smpl (np.ndarray)

    Returns:
        list: list of indexes (smpl_index, garment_index) of linked points
    """
    # garment: (0.091150, -0.037136, 1.4421)
    # SMPL: (0.093775, -0.037374, 1.4353)
    # found = find(garment, np.array((0.091150, -0.037136, 1.4421)))
    linked:list = []
    length = len(smpl)
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(smpl)
    dist, index = nn.kneighbors(garment)
    print(len(smpl))
    print(len(garment))
    print(len(dist))
    print(nn.kneighbors_graph(garment).toarray())

    #for i in range(length):
        #linked.append([i, get_nearest(smpl[i], garment)])
        
    #    linked.append((garment[i], smpl[index]))
    #return linked



def nearest_neighbours(source:np.ndarray, target:np.ndarray):
    length_source = len(source)
    length_target = len(target)

    source_multiple_correspondances:list = []
    target_no_correspondances:list = []
    
    hist_target = [[] for _ in range(length_target)]
    
    # Apply Nearest Neighbour algorithm
    nn_target = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target)
    _, source_to_target = nn_target.kneighbors(source) # array of size source, contains indexes to target
    
    for source_index in range(length_source):
        target_index = source_to_target[source_index, 0]
        hist_target[target_index].append(source_index) # Fills the "hist" target for element i in target (like a histogram) with the index related to it
            
    for target_index in range(length_target):
        if len(hist_target[target_index]) > 1:
            source_multiple_correspondances.append(hist_target[target_index]) # if there are multiple target indexes for a source index, 
        elif len(hist_target[target_index]) < 1:
            if target_index == 588:
                print('a')
            target_no_correspondances.append(target_index)
    
    return (source_multiple_correspondances, target_no_correspondances)
    

def nearest_neighbours_generator(garment:np.ndarray, smpl:np.ndarray):
    
    multiple_garment, single_smpl = nearest_neighbours(garment, smpl)
    multiple_smpl, single_garment = nearest_neighbours(smpl, garment)
    
    print(1)
    
    
    '''garment_multiple_correspondances:list = []
    smpl_multiple_correspondances:list = []
    
    garment_no_correspondances:list = []
    smpl_no_correspondances:list = []
    
    hist_cloth = fill_empty(length_garment)
    hist_body = fill_empty(length_smpl)
    
    # Apply Nearest Neighbour algorithm
    nn_garment = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(garment)
    nn_smpl = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(smpl)
    
    _, smpl_to_garment = nn_garment.kneighbors(smpl) # array size of SMPL, contains indexes to garment
    _, garment_to_smpl = nn_smpl.kneighbors(garment)
    
    # Fill the two empty lists
    for i in range(length_smpl):
        j = smpl_to_garment[i, 0] # j index in garment
        hist_cloth[j].append(i) # append index of the smpl vertex
    
    
    for i in range(length_garment):
        if len(hist_cloth[i]) > 1:
            smpl_multiple_correspondances.append(hist_cloth[i])
        elif len(hist_cloth[i]) < 1:
            garment_no_correspondances.append(i)
    '''
        
import numpy as np
from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

'''def chamferdist(source, target):
    #y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    #min_x_to_y = y_nn.kneighbors(x)[0]
    #chamfer_dist = np.mean(min_x_to_y)
    #return chamfer_dist
    
    import point_cloud_utils
    
    cd = pcu.chamfer_distance(source, target)
    return cd'''