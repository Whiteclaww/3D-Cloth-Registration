import os
import sys
from warnings import warn
import vtk
from contextlib import contextmanager

import numpy as np
import logging
import scipy.sparse as sp
from io import UnsupportedOperation
from menpo.transform import UniformScale
from menpo.transform.homogeneous.affine import DiscreteAffine
from menpo.transform.homogeneous.similarity import Similarity
from menpo.shape import PointCloud
from scipy import sparse
from menpo.shape.adjacency import mask_adjacency_array, reindex_adjacency_array
#from functools import reduce

'''class Affine(Homogeneous):
    r"""
    Base class for all ``n``-dimensional affine transformations. Provides
    methods to break the transform down into its constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transform with other affine transformations.

    Parameters
    ----------
    h_matrix : ``(n_dims + 1, n_dims + 1)`` `ndarray`
        The homogeneous matrix of the affine transformation.
    copy : `bool`, optional
        If ``False`` avoid copying ``h_matrix`` for performance.
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """

    def __init__(self, h_matrix, copy=True, skip_checks=False):
        Homogeneous.__init__(self, h_matrix, copy=copy, skip_checks=skip_checks)

    @classmethod
    def init_identity(cls, n_dims):
        r"""
        Creates an identity matrix Affine transform.

        Parameters
        ----------
        n_dims : `int`
            The number of dimensions.

        Returns
        -------
        identity : :class:`Affine`
            The identity matrix transform.
        """
        return cls(np.eye(n_dims + 1), copy=False, skip_checks=True)

    @classmethod
    def init_from_2d_shear(cls, phi, psi, degrees=True):
        r"""
        Convenience constructor for 2D shear transformations about the origin.

        Parameters
        ----------
        phi : `float`
            The angle of shearing in the X direction.
        psi : `float`
            The angle of shearing in the Y direction.
        degrees : `bool`, optional
            If ``True`` phi and psi are interpreted as degrees.
            If ``False``, phi and psi are interpreted as radians.

        Returns
        -------
        shear_transform : :map:`Affine`
            A 2D shear transform.
        """
        if degrees:
            phi = np.deg2rad(phi)
            psi = np.deg2rad(psi)
        # Create shear matrix
        h_matrix = np.eye(3)
        h_matrix[0, 1] = np.tan(phi)
        h_matrix[1, 0] = np.tan(psi)
        return cls(h_matrix, skip_checks=True)

    @property
    def h_matrix(self):
        r"""
        The homogeneous matrix defining this transform.

        :type: ``(n_dims + 1, n_dims + 1)`` `ndarray`
        """
        return self._h_matrix

    def _set_h_matrix(self, value, copy=True, skip_checks=False):
        r"""
        Updates the `h_matrix`, performing sanity checks.

        Parameters
        ----------
        value : `ndarray`
            The new homogeneous matrix to set
        copy : `bool`, optional
            If ``False`` do not copy the h_matrix. Useful for performance.
        skip_checks : `bool`, optional
            If ``True`` skip sanity checks on the matrix. Useful for performance.
        """
        if not skip_checks:
            shape = value.shape
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("You need to provide a square homogeneous " "matrix")
            if self.h_matrix is not None:
                # already have a matrix set! The update better be the same size
                if self.n_dims != shape[0] - 1:
                    raise ValueError(
                        "Trying to update the homogeneous "
                        "matrix to a different dimension"
                    )
            if shape[0] - 1 not in [2, 3]:
                raise ValueError("Affine Transforms can only be 2D or 3D")
            if not (np.allclose(value[-1, :-1], 0) and np.allclose(value[-1, -1], 1)):
                raise ValueError("Bottom row must be [0 0 0 1] or [0, 0, 1]")
        if copy:
            value = value.copy()
        self._h_matrix = value

    @property
    def linear_component(self):
        r"""
        The linear component of this affine transform.

        :type: ``(n_dims, n_dims)`` `ndarray`
        """
        return self.h_matrix[:-1, :-1]

    @property
    def translation_component(self):
        r"""
        The translation component of this affine transform.

        :type: ``(n_dims,)`` `ndarray`
        """
        return self.h_matrix[:-1, -1]

    def decompose(self):
        r"""
        Decompose this transform into discrete Affine Transforms.

        Useful for understanding the effect of a complex composite transform.

        Returns
        -------
        transforms : `list` of :map:`DiscreteAffine`
            Equivalent to this affine transform, such that

            .. code-block:: python

                reduce(lambda x, y: x.chain(y), self.decompose()) == self

        """
        from .rotation import Rotation
        from .translation import Translation
        from .scale import Scale

        U, S, V = np.linalg.svd(self.linear_component)
        rotation_2 = Rotation(U)
        rotation_1 = Rotation(V)
        scale = Scale(S)
        translation = Translation(self.translation_component)
        return [rotation_1, scale, rotation_2, translation]

    def _transform_str(self):
        r"""
        A string representation explaining what this affine transform does.
        Has to be implemented by base classes.

        Returns
        -------
        str : `str`
            String representation of transform.
        """
        header = "Affine decomposing into:"
        list_str = [t._transform_str() for t in self.decompose()]
        return header + reduce(lambda x, y: x + "\n" + "  " + y, list_str, "  ")

    def _apply(self, x, **kwargs):
        r"""
        Applies this transform to a new set of vectors.

        Parameters
        ----------
        x : ``(N, D)`` `ndarray`
            Array to apply this transform to.

        Returns
        -------
        transformed_x : ``(N, D)`` `ndarray`
            The transformed array.
        """
        return np.dot(x, self.linear_component.T) + self.translation_component

    @property
    def n_parameters(self):
        r"""
        ``n_dims * (n_dims + 1)`` parameters - every element of the matrix but
        the homogeneous part.

        :type: int

        Examples
        --------
        2D Affine: 6 parameters::

            [p1, p3, p5]
            [p2, p4, p6]


        3D Affine: 12 parameters::

            [p1, p4, p7, p10]
            [p2, p5, p8, p11]
            [p3, p6, p9, p12]

        """
        return self.n_dims * (self.n_dims + 1)

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. This does not
        include the homogeneous part of the warp. Note that it flattens using
        Fortran ordering, to stay consistent with Matlab.

        **2D**

        ========= ===========================================
        parameter definition
        ========= ===========================================
        p1        Affine parameter
        p2        Affine parameter
        p3        Affine parameter
        p4        Affine parameter
        p5        Translation in `x`
        p6        Translation in `y`
        ========= ===========================================

        3D and higher transformations follow a similar format to the 2D case.

        Returns
        -------
        params : ``(n_parameters,)`` `ndarray`
            The values that parametrise the transform.
        """
        params = self.h_matrix - np.eye(self.n_dims + 1)
        return params[: self.n_dims, :].ravel(order="F")

    def _from_vector_inplace(self, p):
        r"""
        Updates this Affine in-place from the new parameters. See
        from_vector for details of the parameter format
        """
        h_matrix = None
        if p.shape[0] == 6:  # 2D affine
            h_matrix = np.eye(3)
            h_matrix[:2, :] += p.reshape((2, 3), order="F")
        elif p.shape[0] == 12:  # 3D affine
            h_matrix = np.eye(4)
            h_matrix[:3, :] += p.reshape((3, 4), order="F")
        else:
            ValueError(
                "Only 2D (6 parameters) or 3D (12 parameters) "
                "homogeneous matrices are supported."
            )
        self._set_h_matrix(h_matrix, copy=False, skip_checks=True)

    @property
    def composes_inplace_with(self):
        r"""
        :class:`Affine` can swallow composition with any other :class:`Affine`.
        """
        return Affine

class Similarity(Affine):
    r"""
    Specialist version of an :map:`Affine` that is guaranteed to be a
    Similarity transform.

    Parameters
    ----------
    h_matrix : ``(n_dims + 1, n_dims + 1)`` `ndarray`
        The homogeneous matrix of the affine transformation.
    copy : `bool`, optional
        If ``False`` avoid copying ``h_matrix`` for performance.
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """

    def __init__(self, h_matrix, copy=True, skip_checks=False):
        Affine.__init__(self, h_matrix, copy=copy, skip_checks=skip_checks)

    @classmethod
    def init_identity(cls, n_dims):
        r"""
        Creates an identity transform.

        Parameters
        ----------
        n_dims : `int`
            The number of dimensions.

        Returns
        -------
        identity : :class:`Similarity`
            The identity matrix transform.
        """
        return cls(np.eye(n_dims + 1), copy=False, skip_checks=True)

    def _transform_str(self):
        r"""
        A string representation explaining what this similarity transform does.

        Returns
        -------
        string : `str`
            String representation of transform.
        """
        header = "Similarity decomposing into:"
        list_str = [t._transform_str() for t in self.decompose()]
        return header + reduce(lambda x, y: x + "\n" + "  " + y, list_str, "  ")

    @property
    def n_parameters(self):
        r"""Number of parameters of Similarity

        2D Similarity - 4 parameters ::

            [(1 + a), -b,      tx]
            [b,       (1 + a), ty]

        3D Similarity: Currently not supported

        Returns
        -------
        n_parameters : `int`
            The transform parameters

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.
        """
        if self.n_dims == 2:
            return 4
        elif self.n_dims == 3:
            raise NotImplementedError(
                "3D similarity transforms cannot be " "vectorized yet."
            )
        else:
            raise ValueError(
                "Only 2D and 3D Similarity transforms " "are currently supported."
            )

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order ``[a, b, tx, ty]``, given that
        ``a = k cos(theta) - 1`` and ``b = k sin(theta)`` where ``k`` is a
        uniform scale and `theta` is a clockwise rotation in radians.

        **2D**

        ========= ===========================================
        parameter definition
        ========= ===========================================
        a         `a = k cos(theta) - 1`
        b         `b = k sin(theta)`
        tx        Translation in `x`
        ty        Translation in `y`
        ========= ===========================================

        .. note::

            Only 2D transforms are currently supported.

        Returns
        -------
        params : ``(P,)`` `ndarray`
            The values that parameterise the transform.

        Raises
        ------
        DimensionalityError, NotImplementedError
            If the transform is not 2D
        """
        n_dims = self.n_dims
        if n_dims == 2:
            params = self.h_matrix - np.eye(n_dims + 1)
            # Pick off a, b, tx, ty
            params = params[:n_dims, :].ravel(order="F")
            # Pick out a, b, tx, ty
            return params[[0, 1, 4, 5]]
        elif n_dims == 3:
            raise NotImplementedError(
                "3D similarity transforms cannot be " "vectorized yet."
            )
        else:
            raise ValueError(
                "Only 2D and 3D Similarity transforms " "are currently supported."
            )

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Similarity: 4 parameters ::

            [a, b, tx, ty]

        Parameters
        ----------
        p : ``(P,)`` `ndarray`
            The array of parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.
        """
        if p.shape[0] == 4:
            homog = np.eye(3)
            homog[0, 0] += p[0]
            homog[1, 1] += p[0]
            homog[0, 1] = -p[1]
            homog[1, 0] = p[1]
            homog[:2, 2] = p[2:]
            self._set_h_matrix(homog, skip_checks=True, copy=False)
        elif p.shape[0] == 7:
            raise NotImplementedError(
                "3D similarity transforms cannot be " "vectorized yet."
            )
        else:
            raise ValueError(
                "Only 2D and 3D Similarity transforms " "are currently supported."
            )

class DiscreteAffine(object):
    r"""
    A discrete Affine transform operation (such as a :meth:`Scale`,
    :class:`Translation` or :meth:`Rotation`). Has to be invertable. Make sure
    you inherit from :class:`DiscreteAffine` first, for optimal
    `decompose()` behavior.
    """

    def decompose(self):
        r"""
        A :class:`DiscreteAffine` is already maximally decomposed -
        return a copy of self in a `list`.

        Returns
        -------
        transform : :class:`DiscreteAffine`
            Deep copy of `self`.
        """
        return [self.copy()]

class UniformScale(DiscreteAffine, Similarity):
    r"""
    An abstract similarity scale transform, with a single scale component
    applied to all dimensions. This is abstracted out to remove unnecessary
    code duplication.

    Parameters
    ----------
    scale : ``(n_dims,)`` `ndarray`
        A scale for each axis.
    n_dims : `int`
        The number of dimensions
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """

    def __init__(self, scale, n_dims, skip_checks=False):
        if not skip_checks:
            if n_dims > 3 or n_dims < 2:
                raise ValueError(
                    "UniformScale can only be 2D or 3D" ", not {}".format(n_dims)
                )
        h_matrix = np.eye(n_dims + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        Similarity.__init__(self, h_matrix, copy=False, skip_checks=True)

    @classmethod
    def init_identity(cls, n_dims):
        r"""
        Creates an identity transform.

        Parameters
        ----------
        n_dims : `int`
            The number of dimensions.

        Returns
        -------
        identity : :class:`UniformScale`
            The identity matrix transform.
        """
        return UniformScale(1, n_dims)

    @property
    def scale(self):
        r"""
        The single scale value.

        :type: `float`
        """
        return self.h_matrix[0, 0]

    def _transform_str(self):
        message = "UniformScale by {}".format(self.scale)
        return message

    @property
    def n_parameters(self):
        r"""
        The number of parameters: 1

        :type: `int`
        """
        return 1

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order ``[s]``.

        +----------+--------------------------------+
        |parameter | definition                     |
        +==========+================================+
        |s         | The scale across each axis     |
        +----------+--------------------------------+

        Returns
        -------
        s : `float`
            The scale across each axis.
        """
        return np.asarray(self.scale)

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Parameters
        ----------
        p : `float`
            The parameter
        """
        np.fill_diagonal(self.h_matrix, p)
        self.h_matrix[-1, -1] = 1

    @property
    def composes_inplace_with(self):
        r"""
        :class:`UniformScale` can swallow composition with any other
        :class:`UniformScale`.
        """
        return UniformScale

    def pseudoinverse(self):
        r"""
        The inverse scale.

        :type: :class:`UniformScale`
        """
        return UniformScale(1.0 / self.scale, self.n_dims, skip_checks=True)
'''
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

    @classmethod
    def init_identity(cls, n_dims):
        r"""
        Creates an identity transform.

        Parameters
        ----------
        n_dims : `int`
            The number of dimensions.

        Returns
        -------
        identity : :class:`Translation`
            The identity matrix transform.
        """
        return Translation(np.zeros(n_dims))

    def _transform_str(self):
        message = "Translation by {}".format(self.translation_component)
        return message

    @property
    def n_parameters(self):
        r"""
        The number of parameters: ``n_dims``

        :type: `int`
        """
        return self.n_dims

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order ``[t0, t1, ...]``.

        +-----------+--------------------------------------------+
        |parameter | definition                                  |
        +==========+=============================================+
        |t0        | The translation in the first axis           |
        |t1        | The translation in the second axis          |
        |...       | ...                                         |
        |tn        | The translation in the nth axis             |
        +----------+---------------------------------------------+

        Returns
        -------
        ts : ``(n_dims,)`` `ndarray`
            The translation in each axis.
        """
        return self.h_matrix[:-1, -1]

    def _from_vector_inplace(self, p):
        r"""
        Updates the :class:`Translation` inplace.

        Parameters
        ----------
        vector : ``(n_dims,)`` `ndarray`
            The array of parameters.
        """
        self.h_matrix[:-1, -1] = p

    def pseudoinverse(self):
        r"""
        The inverse translation (negated).

        :type: :class:`Translation`
        """
        return Translation(-self.translation_component, skip_checks=True)


















































































































def _normalize(v):
    magnitude = np.sqrt((v ** 2).sum(axis=1, keepdims=True))
    # Check for zero or NaN magnitude
    zero_or_nan_mask = np.logical_or(np.isnan(magnitude), magnitude == 0)
    # Set zero or NaN magnitudes to 1 to avoid division by zero
    magnitude[zero_or_nan_mask] = 1
    normalized_v = v / magnitude
    # Replace NaN values with 0
    normalized_v = np.nan_to_num(normalized_v)
    return normalized_v


def compute_face_normals(points, trilist):
    """
    Compute per-face normals of the vertices given a list of
    faces.

    Parameters
    ----------
    points : (N, 3) float32/float64 ndarray
        The list of points to compute normals for.
    trilist : (M, 3) int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    face_normal : (M, 3) float32/float64 ndarray
        The normal per face.
    :return:
    """
    pt = points[trilist]
    a, b, c = pt[:, 0], pt[:, 1], pt[:, 2]
    norm = np.cross(b - a, c - a)
    return _normalize(norm)


def compute_vertex_normals(points, trilist):
    """
    Compute the per-vertex normals of the vertices given a list of
    faces.

    Parameters
    ----------
    points : (N, 3) float32/float64 ndarray
        The list of points to compute normals for.
    trilist : (M, 3) int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) float32/float64 ndarray
        The normal per vertex.
    """
    face_normals = compute_face_normals(points, trilist)

    vertex_normals = np.zeros(points.shape, dtype=points.dtype)
    np.add.at(vertex_normals, trilist[:, 0], face_normals)
    np.add.at(vertex_normals, trilist[:, 1], face_normals)
    np.add.at(vertex_normals, trilist[:, 2], face_normals)

    return _normalize(vertex_normals)

class TriMesh(PointCloud):
    def __init__(self, points, trilist=None, copy = True):
        super(TriMesh, self).__init__(points, copy = copy)
        if trilist is None:
            from scipy.spatial import Delaunay  # expensive import

            trilist = Delaunay(points).simplices
        if not copy:
            if not trilist.flags.c_contiguous:
                warn(
                    "The copy flag was NOT honoured. A copy HAS been made. "
                    "Please ensure the data you pass is C-contiguous."
                )
                print(type(trilist))
                trilist = np.array(trilist, copy = True, order = "C")
        else:
            trilist = np.array(trilist, copy = True, order = "C")
        self.trilist = trilist
    
    @property
    def n_dims(self):
        r"""
        The number of dimensions in the pointcloud.

        :type: `int`
        """
        return self.points.shape[1]

    def vertex_normals(self):
        r"""
        Compute the per-vertex normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Returns
        -------
        normals : ``(n_points, 3)`` `ndarray`
            Normal at each point.

        Raises
        ------
        ValueError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise ValueError("Normals are only valid for 3D meshes")
        return compute_vertex_normals(self.points, self.trilist)
    
    def unique_edge_indices(self):
        r"""An unordered index into points that rebuilds the unique edges of
        this :map:`TriMesh`.

        Note that each physical edge will only be counted once in this method
        (i.e. edges shared between neighbouring triangles are only counted once
        not twice). The ordering should be considered random.

        Returns
        -------
        unique_edge_indices : ``(n_unique_edges, 2)`` `ndarray`
            Return a point index that rebuilds all edges present in this
            :map:`TriMesh` only once.
        """
        # Get a sorted list of edge pairs. sort ensures that each edge is
        # ordered from lowest index to highest.
        edge_pairs = np.sort(self.edge_indices())

        # We want to remove duplicates - this is a little hairy: basically we
        # get a view on the array where each pair is considered by numpy to be
        # one item
        edge_pair_view = np.ascontiguousarray(edge_pairs).view(
            np.dtype((np.void, edge_pairs.dtype.itemsize * edge_pairs.shape[1]))
        )
        # Now we can use this view to ask for only unique edges...
        unique_edge_index = np.unique(edge_pair_view, return_index=True)[1]
        # And use that to filter our original list down
        return edge_pairs[unique_edge_index]

    def __str__(self):
        return "{}, n_tris: {}".format(PointCloud.__str__(self), self.n_tris)

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: `int`
        """
        return len(self.trilist)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the TriMesh. This is then broadcast across the dimensions
        of the mesh and returns a new mesh containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        mesh : :map:`TriMesh`
            A new mesh that has been masked.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError(
                "Mask must be a 1D boolean array of the same "
                "number of entries as points in this TriMesh."
            )

        tm = self.copy()
        if np.all(mask):  # Fast path for all true
            return tm
        else:
            # Recalculate the mask to remove isolated vertices
            isolated_mask = self._isolated_mask(mask)
            # Recreate the adjacency array with the updated mask
            masked_adj = mask_adjacency_array(isolated_mask, self.trilist)
            tm.trilist = reindex_adjacency_array(masked_adj)
            tm.points = tm.points[isolated_mask, :]
            return tm

    def _isolated_mask(self, mask):
        # Find the triangles we need to keep
        masked_adj = mask_adjacency_array(mask, self.trilist)
        # Find isolated vertices (vertices that don't exist in valid
        # triangles)
        isolated_indices = np.setdiff1d(np.nonzero(mask)[0], masked_adj)

        # Create a 'new mask' that contains the points the use asked
        # for MINUS the points that we can't create triangles for
        new_mask = mask.copy()
        new_mask[isolated_indices] = False
        return new_mask

    def tri_normals(self):
        r"""
        Compute the triangle face normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Returns
        -------
        normals : ``(n_tris, 3)`` `ndarray`
            Normal at each triangle face.

        Raises
        ------
        ValueError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise ValueError("Normals are only valid for 3D meshes")
        return compute_face_normals(self.points, self.trilist)

    def boundary_tri_index(self):
        r"""Boolean index into triangles that are at the edge of the TriMesh.
        The boundary vertices can be visualized as follows
        ::

            tri_mask = mesh.boundary_tri_index()
            boundary_points = mesh.points[mesh.trilist[tri_mask].ravel()]
            pc = menpo.shape.PointCloud(boundary_points)
            pc.view()

        Returns
        -------
        boundary_tri_index : ``(n_tris,)`` `ndarray`
            For each triangle (ABC), returns whether any of it's edges is not
            also an edge of another triangle (and so this triangle exists on
            the boundary of the TriMesh)
        """
        # Compute the edge indices so that we can find duplicated edges
        edge_indices = self.edge_indices()
        # Compute the triangle indices and repeat them so that when we loop
        # over the edges we get the correct triangle index per edge
        # (e.g. [0, 0, 0, 1, 1, 1, ...])
        tri_indices = np.arange(self.trilist.shape[0]).repeat(3)

        # Loop over the edges to find the "lonely" triangles that have an edge
        # that isn't shared with another triangle. Due to the definition of a
        # triangle and the careful ordering chosen above, each edge will be
        # seen either exactly once or exactly twice.
        # Note that some triangles may appear more than once as it's possible
        # for a triangle to only share one edge with the rest of the mesh (so
        # it would have two "lonely" edges
        lonely_triangles = {}
        for edge, t_i in zip(edge_indices, tri_indices):
            # Sorted the edge indices since we may see an edge (0, 1) and then
            # see it again as (1, 0) when in fact that is the same edge
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in lonely_triangles:
                lonely_triangles[sorted_edge] = t_i
            else:
                # If we've already seen the edge the we will never see it again
                # so we can just remove it from the candidate set
                del lonely_triangles[sorted_edge]

        mask = np.zeros(self.n_tris, dtype=np.bool_)
        mask[np.array(list(lonely_triangles.values()))] = True
        return mask

    def edge_indices(self):
        r"""An unordered index into points that rebuilds the edges of this
        :map:`TriMesh`.

        Note that there will be two edges present in cases where two triangles
        'share' an edge. Consider :meth:`unique_edge_indices` for a single index
        for each physical edge on the :map:`TriMesh`.

        Returns
        -------
        edge_indices : ``(n_tris * 3, 2)`` `ndarray`
            For each triangle (ABC), returns the pair of point indices that
            rebuild AB, BC, CA. All edge indices are concatenated for a total
            of ``n_tris * 3`` edge_indices. The ordering is done so that each
            triangle is returned in order
            e.g. [AB_1, BC_1, CA_1, AB_2, BC_2, CA_2, ...]
        """
        tl = self.trilist
        return np.hstack((tl[:, [0, 1]], tl[:, [1, 2]], tl[:, [2, 0]])).reshape(-1, 2)
    
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
        self._c_point = [0.0, 0.0, 0.0]
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
        snapped_points, indices = [], []
        for p in points:
            snapped, index = self._find_single_closest_point(p)
            snapped_points.append(snapped)
            indices.append(index)

        return np.array(snapped_points), np.array(indices)

    def _find_single_closest_point(self, point):
        r"""Return the nearest point on the mesh and the index of the nearest
        triangle

        Parameters
        ----------
        point : ``(3,)`` `ndarray`
            Query point

        Returns
        -------
        `nearest_point`, `tri_index` : ``(3,)`` `ndarray`, ``int``
            A tuple of the nearest point on the `vtkPolyData` and the triangle
            index of the triangle that the nearest point is located inside of.
        """
        self.cell_locator.FindClosestPoint(
            point, self._c_point, self._cell_id, self._sub_id, self._distance
        )
        return self._c_point[:], self._cell_id.get()
