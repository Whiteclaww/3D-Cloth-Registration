#========[ IMPORTS ]========
from modules.utils import *
from .solver import spsolve

from contextlib import contextmanager
from io import UnsupportedOperation
import logging
import os
import scipy.sparse as sp
import sys

#========[ FUNCTIONS ]========

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

def non_rigid_icp(source, target, i_iterations = 1, j_iterations = 10, alpha = 5, eps = 1e-03, generate_instances = True):
    # call the generator version of NICP, always returning a generator
    generator = non_rigid_icp_generator(source, target, threshold = eps, i_iterations = i_iterations, j_iterations = j_iterations, alpha = alpha)
    
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

def non_rigid_icp_generator(source:np.ndarray, target:np.ndarray, threshold:float, i_iterations:int, j_iterations:int, alpha, verbose = False):
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