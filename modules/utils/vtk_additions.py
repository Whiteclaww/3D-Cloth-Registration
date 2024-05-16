#========[ IMPORTS ]========
from menpo.transform.homogeneous.affine import DiscreteAffine
from menpo.transform.homogeneous.similarity import Similarity
import numpy as np
import vtk

#========[ CLASSES ]========

class Translation(DiscreteAffine, Similarity):
    r"""
    An ``n_dims``-dimensional translation transform.

    Parameters
    ----------
    translation : ``(n_dims,)`` `ndarray`
        The translation in each axis.
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """

    def __init__(self, translation, skip_checks=False):
        translation = np.asarray(translation)
        h_matrix = np.eye(translation.shape[0] + 1)
        h_matrix[:-1, -1] = translation
        Similarity.__init__(self, h_matrix, copy=False, skip_checks=skip_checks)

    def pseudoinverse(self):
        r"""
        The inverse translation (negated).

        :type: :class:`Translation`
        """
        return Translation(-self.translation_component, skip_checks=True)
    
class VTKClosestPointLocator(object):
    r"""A callable that can be used to find the closest point on a given
    `vtkPolyData` for a query point.

    Parameters
    ----------
    vtk_mesh : `vtkPolyData`
        The VTK mesh that will be queried for finding closest points. A
        data structure will be initialized around this mesh which will enable
        efficient future lookups.
    """

    def __init__(self, vtk_mesh):
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(vtk_mesh)
        cell_locator.BuildLocator()
        self.cell_locator = cell_locator

        # prepare some private properties that will be filled in for us by VTK
        self._c_point = np.zeros(3)
        self._cell_id = vtk.mutable(0)
        self._sub_id = vtk.mutable(0)
        self._distance = vtk.mutable(0.0)
        
    def __call__(self, points):
        r"""Return the nearest points on the mesh and the index of the nearest
        triangle for a collection of points. This is a lower-level algorithm
        and operates directly on a numpy array rather than an pointcloud.

        Parameters
        ----------
        points : ``(n_points, 3)`` `ndarray`
            Query points

        Returns
        -------
        `nearest_points`, `tri_indices` : ``(n_points, 3)`` `ndarray`, ``(n_points,)`` `ndarray`
            A tuple of the nearest points on the `vtkPolyData` and the triangle
            indices of the triangles that the nearest point is located inside of.
        """
        snapped_points = np.zeros_like(points)
        indices = np.zeros(len(points), dtype=int)
        
        for i, point in enumerate(points):
            self.cell_locator.FindClosestPoint(point, self._c_point, self._cell_id, self._sub_id, self._distance)
            snapped_points[i] = self._c_point[:]
            indices[i] = self._cell_id.get()
        
        return snapped_points, indices