#========[ IMPORTS ]========
from menpo.shape import PointCloud
import numpy as np
from warnings import warn

#========[ FUNCTIONS ]========

def _normalize(v:np.ndarray):
    magnitude = np.sqrt((v ** 2).sum(axis=1, keepdims=True))
    
    # Check for zero or NaN magnitude
    zero_or_nan_mask = np.logical_or(np.isnan(magnitude), magnitude == 0)
    
    # Set zero or NaN magnitudes to 1 to avoid division by zero
    magnitude[zero_or_nan_mask] = 1
    normalized_v = v / magnitude
    
    # Replace NaN values with 0
    normalized_v = np.nan_to_num(normalized_v)
    return normalized_v

class TriMesh(PointCloud):
    def __init__(self, points:np.ndarray, trilist = None, copy = True):
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
        point = self.points[self.trilist]
        a, b, c = point[:, 0], point[:, 1], point[:, 2]
        norm = np.cross(b - a, c - a)
        face_normals = _normalize(norm)

        vertex_normals = np.zeros(self.points.shape, dtype=self.points.dtype)
        np.add.at(vertex_normals, self.trilist[:, 0], face_normals)
        np.add.at(vertex_normals, self.trilist[:, 1], face_normals)
        np.add.at(vertex_normals, self.trilist[:, 2], face_normals)

        return _normalize(vertex_normals)
    
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

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: `int`
        """
        return len(self.trilist)

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
        point = self.points[self.trilist]
        a, b, c = point[:, 0], point[:, 1], point[:, 2]
        norm = np.cross(b - a, c - a)
        return _normalize(norm)

    def boundary_tri_index(self) -> np.ndarray:
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