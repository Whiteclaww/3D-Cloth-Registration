from imports import *

from contextlib import contextmanager
from io import UnsupportedOperation
import logging
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr
import os
import sys
from vtk.util.numpy_support import numpy_to_vtk
#--------------------------

@contextmanager
def stdout_redirected(to=os.devnull):
    r"""
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    try:
        fd = sys.stdout.fileno()
    except UnsupportedOperation:
        # It's possible this is being run in an interpreter like an IPython
        # notebook where stdout doesn't behave the same as in a "normal" python
        # interpreter and in this case we cannot treat stdout like a file
        # descriptor
        warn(
            "Unable to duplicate stdout file descriptor, likely due "
            "to stdout having been replaced (e.g. a notebook)"
        )
        yield
    else:
        # assert that Python and C stdio write using the same file descriptor
        # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

        def redirect_stdout(to):
            sys.stdout.close()  # + implicit flush()
            os.dup2(to.fileno(), fd)  # fd writes to 'to' file
            sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

        with os.fdopen(os.dup(fd), "w") as old_stdout:
            with open(to, "w") as file:
                redirect_stdout(to=file)
            try:
                yield  # allow code to be run with the redirected stdout
            finally:
                # restore stdout.
                # buffering and flags such as CLOEXEC may be different
                redirect_stdout(to=old_stdout)

#--------------------------

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

def non_rigid_icp(source, target, eps = 1e-03, iterations = 1, generate_instances = True):
    # call the generator version of NICP, always returning a generator
    generator = non_rigid_icp_generator(source, target, threshold = eps, i_iterations = iterations)
    
    # the handler decides whether the user get's details and each iteration returned, or just the final result.
    return non_rigid_icp_generator_handler(generator, generate_instances)

def non_rigid_icp_generator_handler(generator, generate_instances):
    if generate_instances:
        # the user wants to inspect results per-iteration - return the iterator
        # directly to them
        return generator
    else:
        # the user is not interested in per-iteration results. Exhaust the
        # generator ourselves and return the last result only.
        while True:
            try:
                instance = next(generator)
            except StopIteration:
                return instance[0]

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

def spsolve(A, B):
    r"""
    Solve A*X = B (matrices)
    Since B is much smaller than A, we divide B in multiple vectors, calculate A*X = b then append every X to the result

    Args:
        A (sparse matrix)
        B (sparse matrix)
    """
    X = []
    tol = 0#1e-17
    for count in range(B.shape[1]):
        # Extract the i-th row of b
        b = B[:, count].reshape(1, -1)
        
        # Solve the linear system for the i-th right-hand side vector
        (solved_row_b, stop, _, _, _, _, _, _) = lsmr(A, toarray_without_loss(b), atol=tol, btol=tol, damp=0, conlim=0)
        
        if stop != 5 and stop != 2:
            logging.error("Error Solving Ax=B")
        
        # Append the solution to the list of solutions
        X.append(solved_row_b)
        
    X = sp.csr_matrix(X).tocsr().T
    
    '''maybeB:sp.csr_matrix = A.dot(X)
    
    abs_diff = np.abs(toarray_without_loss(maybeB) - toarray_without_loss(B))
    if np.all(abs_diff <= tolerance).all() and maybeB.shape == B.shape:
        logging.info(" |  | Verified that A * X = B")
    else:
        logging.error("=====Error when verifying A * X = B, precision error")'''
        
    return X











def non_rigid_icp_generator(source:np.ndarray, target:np.ndarray, threshold:float, i_iterations:int = 5, j_iterations:int = 15, alpha = 1, verbose = False):
    """
    Deforms the source trimesh to align with to optimally the target.
    """
    log_counter:int = 1
    logging.info(" ----------- Started Non-Rigid ICP  -----------  ")
    # Scale factors completely change the behavior of the algorithm - always
    # rescale the source down to a sensible size (so it fits inside box of
    # diagonal 1) and is centred on the origin. We'll undo this after the fit
    # so the user can use whatever scale they prefer.
    trimesh_source:TriMesh = TriMesh(source)
    trimesh_target:TriMesh = TriMesh(target)
    log_counter += 1
    logging.info(f"|-{log_counter}: Transformed meshes again (applied translation)")

    # Homogeneous dimension (1 extra for translation effects)
    n_dims = trimesh_source.n_dims
    h_dims = n_dims + 1
    current_deformed_points, trilist = trimesh_source.points, trimesh_source.trilist
    n = current_deformed_points.shape[0]  # record number of points
    
    #edge_triangles = trimesh_source.boundary_tri_index()
    log_counter += 1
    logging.info(f"|-{log_counter}: Boundary triangle indices")

    incidence_matrix, unique_edge_pairs = node_arc_incidence_matrix(trimesh_source)
    log_counter += 1
    logging.info(f"|-{log_counter}: Node-arc incidence matrix")

    # weight matrix
    weight_matrix = np.identity(n_dims + 1)

    heavier_incidence_matrix = sp.kron(incidence_matrix, weight_matrix)
    log_counter += 1
    logging.info(f"|-{log_counter}: Weight matrix")

    # build octree for finding closest points on target.
    target_vtk = trimesh_to_vtk(trimesh_target)
    log_counter += 1
    logging.info(f"|-{log_counter}: Trimesh to VTK conversion")

    closest_points_on_target = VTKClosestPointLocator(target_vtk)
    log_counter += 1
    logging.info(f"|-{log_counter}: Created VTK closest point locator")

    # save out the target normals. We need them for the weight matrix.
    target_tri_normals = trimesh_target.tri_normals()
    log_counter += 1
    logging.info(f"|-{log_counter}: Saved target normals")

    # init transformation
    previousX = np.tile(np.zeros((n_dims, h_dims)), n).T
    log_counter += 1
    logging.info(f"|-{log_counter}: Initialized transformation")

    row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(), np.arange(n)))
    x = np.arange(n * h_dims).reshape((n, h_dims))
    col = np.hstack((x[:, :n_dims].ravel(), x[:, n_dims]))
    array_of_ones = np.ones(n)
    log_counter += 1
    logging.info(f"|-{log_counter}: Parsing matrix")

    for i in range(0, i_iterations):
        print(i)
        logging.info(f"*** Iteration {i} ***")

        '''alpha_is_per_vertex = isinstance(alpha, np.ndarray)
        if alpha_is_per_vertex:
            # stiffness is provided per-vertex
            if alpha.shape[0] != trimesh_source.n_points:
                raise ValueError()
            alpha_per_edge = alpha[unique_edge_pairs].mean(axis=1)
            alpha_M_s = sp.diags(alpha_per_edge).dot(incidence_matrix)
            alpha_times_incidence_heavier_matrix = sp.kron(alpha_M_s, weight_matrix)
            logging.info(" | Alpha is per vertex")
        else:
            # stiffness is global - just a scalar multiply. Note that here
            # we don't have to recalculate M_kron_G_s'''
        if alpha != 1:
            alpha_times_incidence_heavier_matrix = alpha * heavier_incidence_matrix
        else:
            alpha_times_incidence_heavier_matrix = heavier_incidence_matrix
        logging.info(" | Alpha is not per vertex (stiffness is global)")
        
        j = 0
        while j < j_iterations:  # iterate until convergence                          SUPPOSED TO BE WHILE TRUE
            j += 1  # track the iterations for this alpha/landmark weight   
            logging.info(" | *** Iterate until convergence ***")
            # find nearest neighbour and the normals
            print(f"|-{j}")
            
            # this step is very slow
            closest_points, triangle_indices = closest_points_on_target(current_deformed_points)

            # ---- WEIGHTS ----
            # 1.  Edges
            # Are any of the corresponding tris on the edge of the target?
            # Where they are we return a false weight (we *don't* want to
            # include these points in the solve)
            #mask_edges = np.in1d(triangle_indices, edge_triangles, invert=True)
            logging.info(" |  | If points are already on the mesh, in edge_point_weights is false")
            # 2. Normals
            # Calculate the normals of the current deformed points
            current_deformed_points_trimesh = TriMesh(current_deformed_points, trilist = trilist, copy = False)
            current_deformed_points_normals = current_deformed_points_trimesh.vertex_normals()
            logging.info(" |  | Calculated normals")
            # Extract the corresponding normals from the target
            normal_corresponding_triangles_in_target = target_tri_normals[triangle_indices]
            # If the dot of the normals is lt 0.9 don't contrib to deformation
            mask_normals = (normal_corresponding_triangles_in_target * current_deformed_points_normals).sum(axis = 1) > 0.9

            # Form the overall w_i from the normals, edge case
            # for now disable the edge constraint (it was noisy anyway)
            mask_all = mask_normals
            logging.info(" |  | Variable definitions")

            #prop_omitted = (n - mask_all.sum() * 1.0) / n
            #prop_omitted_norms = (n - mask_normals.sum() * 1.0) / n
            #prop_omitted_edges = (n - mask_edges.sum() * 1.0) / n
            logging.info(" |  | Divided values")

            # Build the sparse diagonal weight matrix
            sparse_diagonal_weight_matrix = sp.diags(mask_all.astype(np.float64)[None, :], [0]) # type: ignore
            logging.info(" |  | Building sparse diagonal weight matrix...")

            data = np.hstack((current_deformed_points.ravel(), array_of_ones))
            data_sparse_matrix = sp.coo_matrix((data, (row, col)))

            matrixA = sp.vstack([alpha_times_incidence_heavier_matrix, sparse_diagonal_weight_matrix.dot(data_sparse_matrix)]).tocsr()
            matrixB = sp.vstack([np.zeros((alpha_times_incidence_heavier_matrix.shape[0], n_dims)),
                          closest_points * mask_all[:, None]]).tocsr()  # nullify nearest points by combined_weights
            logging.info(" |  | Building done!")
            
            # those two steps are very slow
            solved = spsolve(matrixA, matrixB)

            # deform template
            #previous_deformed_points = current_deformed_points
            current_deformed_points = toarray_without_loss(data_sparse_matrix.dot(solved))
            #deformation_per_step = current_deformed_points - previous_deformed_points
            logging.info(" |  | Deformed template")
            
            
            

            error_delta = np.linalg.norm(toarray_without_loss(previousX) - solved, ord="fro")
            stop_criterion = error_delta / np.sqrt(np.size(previousX))
            
            #error_delta = np.linalg.norm(solved - previousX_initial, ord="fro")
            #stop_criterion = error_delta / np.sqrt(np.size(previousX))

            # Update previous solution
            #previousX_initial = solved.copy()

            
            
            
            logging.info(f" |  | stop criterion: {stop_criterion}, threshold: {threshold}")

            previousX = solved
            logging.info(f" |  | Finished iteration {j}")
            logging.info("-----------------")
            
            # track the progress of the algorithm per-iteration
            '''info_dict = {
                "alpha": alpha,
                "iteration": j,
                "prop_omitted": prop_omitted,
                "prop_omitted_norms": prop_omitted_norms,
                "prop_omitted_edges": prop_omitted_edges,
                "mask_normals": mask_normals,
                "mask_edges": mask_edges,
                "mask_all": mask_all,
                "nearest_points": restore.apply(closest_points),
                "deformation_per_step": deformation_per_step,
                "error_delta": error_delta
            }

            logging.info(f"|{'-' * i} {info_dict}")'''

            if stop_criterion < threshold:
                break
        
        logging.info(f" | Finished iteration {i}")
        
    logging.info("Non-Rigid ICP done!")
    return current_deformed_points