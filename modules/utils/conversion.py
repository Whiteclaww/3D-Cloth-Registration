#========[ IMPORTS ]========
from .trimesh import *
from .vtk_additions import *

import logging
import scipy.sparse as sp
import torch
from vtk.util.numpy_support import numpy_to_vtk     # Ignore the error

#========[ FUNCTIONS ]========

def trimesh_to_vtk(trimesh: TriMesh):
    r"""Return a `vtkPolyData` representation of a :map:`TriMesh` instance

    Parameters
    ----------
    trimesh : :map:`TriMesh`
        The menpo :map:`TriMesh` object that needs to be converted to a
        `vtkPolyData`

    Returns
    -------
    `vtk_mesh` : `vtkPolyData`
        A VTK mesh representation of the Menpo :map:`TriMesh` data

    Raises
    ------
    ValueError:
        If the input trimesh is not 3D.
    """
    if trimesh.n_dims != 3:
        raise ValueError("trimesh_to_vtk() only works on 3D TriMesh instances")
    logging.info("|----1: entered trimesh_to_vtk")

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(trimesh.points, deep=1))
    mesh.SetPoints(points)

    cells = vtk.vtkCellArray()
    logging.info("|----2: initialisation")

    # Create VTK cells manually
    for tri in trimesh.trilist:
        vtk_cell = vtk.vtkTriangle()
        vtk_cell.GetPointIds().SetId(0, tri[0])
        vtk_cell.GetPointIds().SetId(1, tri[1])
        vtk_cell.GetPointIds().SetId(2, tri[2])
        cells.InsertNextCell(vtk_cell)
    logging.info("|----3: created VTK cells manually")

    # Set cells
    mesh.SetPolys(cells)
    logging.info("|----4: set cells")
    return mesh


def numpyToVTK(data, multi_component = False, type = 'float'):
    """
    multi_components: rgb has 3 components
    type：float or char
    """
    if type == 'float':
        data_type = vtk.VTK_FLOAT
    elif type == 'char':
        data_type = vtk.VTK_UNSIGNED_CHAR
    else:
        raise RuntimeError('unknown type')
    if multi_component == False:
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        flat_data_array = data.transpose(2,1,0).flatten()
        vtk_data = numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
        shape = data.shape
    else:
        assert len(data.shape) == 3, 'only test for 2D RGB'
        flat_data_array = data.transpose(1, 0, 2)
        flat_data_array = np.reshape(flat_data_array, newshape=[-1, data.shape[2]])
        vtk_data = numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
        shape = [data.shape[0], data.shape[1], 1]
    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    return img

def node_arc_incidence_matrix(source):
    unique_edge_pairs = source.unique_edge_indices()
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    row = np.hstack((np.arange(m), np.arange(m)))
    col = unique_edge_pairs.T.ravel()
    data = np.hstack((-1 * np.ones(m), np.ones(m)))
    return sp.coo_matrix((data, (row, col))), unique_edge_pairs

def toarray_without_loss(sparse_matrix):
    # Get the shape of the sparse matrix
    m, n = sparse_matrix.shape
    
    # Create a dense array of zeros with the same shape as the sparse matrix
    dense_array = np.zeros((m, n), dtype=sparse_matrix.dtype)
    
    # Retrieve the nonzero entries from the sparse matrix
    row_indices, col_indices = sparse_matrix.nonzero()
    data = sparse_matrix.data
    
    for i in range(len(row_indices)):
        dense_array[row_indices[i], col_indices[i]] = data[i]
    
    return dense_array

def toarray(matrix):
    if matrix.shape[0] != 1:
        return False
    result = np.zeros(matrix.shape[1])
    for i, val in zip(matrix.indices, matrix.data):
        result[i] = val
    return result

def tolist(list):
    result = []
    for i in list:
        result.append(i)
    return result

def tofloat(tensor:torch.Tensor):
    string = str(tensor)
    only_str = ""
    for i in string:
        if i >= '0' and i <= '9' or i == '.':
            only_str += i
    return float(only_str)