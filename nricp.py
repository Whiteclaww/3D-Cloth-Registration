from imports import *
from sksparse.cholmod import cholesky_AAt
import scipy
from scipy.sparse import csr_matrix
from vtk.util.numpy_support import numpy_to_vtk

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

def custom_solver(a, b, max_iter = 1000, threshold = 1e-6):
    """
    Solve the linear system Ax = b using the conjugate gradient method.
    """
    # Initialize x
    x = np.zeros((a.shape[1], b.shape[1]))
    residual = b
    initial_residual_norm_squared = np.dot(residual.T, residual)

    # Iterate for a maximum number of iterations
    for _ in range(max_iter):
        Ap = np.dot(a.T, residual.T)
        alpha = np.dot(residual.T, Ap) / np.dot(Ap.T, Ap)
        x += alpha * residual
        residual -= alpha * np.dot(a, Ap)
        new_residual_norm_squared = np.linalg.norm(residual)**2

        # Check for convergence
        if np.sqrt(new_residual_norm_squared) < threshold:
            return x
        beta = new_residual_norm_squared / initial_residual_norm_squared
        residual += beta * residual
        initial_residual_norm_squared = new_residual_norm_squared
    return x

def node_arc_incidence_matrix(source):
    unique_edge_pairs = source.unique_edge_indices()
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    row = np.hstack((np.arange(m), np.arange(m)))
    col = unique_edge_pairs.T.ravel()
    data = np.hstack((-1 * np.ones(m), np.ones(m)))
    return sp.coo_matrix((data, (row, col))), unique_edge_pairs


def validate_weights(label, weights, n_points, n_iterations=None, verbose=False):
    if n_iterations is not None and len(weights) != n_iterations:
        raise ValueError(
            "Invalid {label}: - due to other weights there are "
            "{n_iterations} iterations but {n_weights} {label} "
            "were provided".format(
                label=label, n_iterations=n_iterations, n_weights=len(weights)
            )
        )
    invalid = []
    for i, weight in enumerate(weights):
        is_per_vertex = isinstance(weight, np.ndarray)
        if is_per_vertex and weight.shape != (n_points,):
            invalid.append("({}): {}".format(i, weight.shape[0]))

    if verbose and len(weights) >= 1:
        is_per_vertex = isinstance(weights[0], np.ndarray)
        if is_per_vertex:
            print("Using per-vertex {label}".format(label=label))
        else:
            print("Using global {label}".format(label=label))

    if len(invalid) != 0:
        raise ValueError(
            "Invalid {label}: expected shape ({n_points},) "
            "got: {invalid_cases}".format(
                label=label,
                n_points=n_points,
                invalid_cases="{}".format(", ".join(invalid)),
            )
        )

def non_rigid_icp(source, target, eps = 1e-3, iterations = 10, generate_instances = True):
    # call the generator version of NICP, always returning a generator
    generator = non_rigid_icp_generator(source, target, threshold = eps, iterations = iterations)
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

def transform(source:np.ndarray, target:np.ndarray):
    trimesh_source:TriMesh = TriMesh(source)
    trimesh_target:TriMesh = TriMesh(target)
    
    translation:Translation = Translation(-1 * trimesh_source.centre())
    
    trimesh_source:TriMesh = translation.apply(trimesh_source) # type: ignore (TriMesh is indeed the same as the returned object)
    trimesh_target:TriMesh = translation.apply(trimesh_target) # type: ignore
    
    return(trimesh_source, trimesh_target, translation)

from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d

# Example non-linear function
def nonlinear_function(x):
    return np.sin(x)

num_intervals = 10
intervals = np.linspace(0, 2*np.pi, num_intervals)

# Divide the domain into intervals
num_intervals = 10
intervals = np.linspace(0, 2*np.pi, num_intervals)

# Approximate the non-linear function within each interval
def linear_approximation(x):
    return np.sin(intervals[np.searchsorted(intervals, x)])

# Construct the piecewise linear function
def piecewise_linear_function(x):
    return nonlinear_function(x)

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

def solve_with_dense(matrixA, matrixB):
    # Convert sparse matrices to dense arrays without data loss
    dense_A = toarray_without_loss(matrixA)
    dense_B = toarray_without_loss(matrixB)

    # Check if matrices have the same number of rows
    if dense_A.shape[0] != dense_B.shape[0]:
        raise ValueError("Matrices must have the same number of rows for solving.")

    # Perform the solving algorithm using dense arrays
    # For example, let's use NumPy's linear algebra solver for demonstration
    # You can replace this with any solving algorithm that works with dense arrays
    solution = np.linalg.lstsq(dense_A, dense_B, rcond=None)[0]

    return solution

def toarray(matrix):
    if matrix.shape[0] != 1:
        return False
    result = np.zeros(matrix.shape[1])
    for i, val in zip(matrix.indices, matrix.data):
        result[i] = val
    return result









def non_rigid_icp_generator(source:np.ndarray, target:np.ndarray, threshold:float = 1e-3, iterations:int = 10):
    """
    Deforms the source trimesh to align with to optimally the target.
    """
    log_counter:int = 1
    logging.info(" ----------- Started Non-Rigid ICP  -----------  ")
    # Scale factors completely change the behavior of the algorithm - always
    # rescale the source down to a sensible size (so it fits inside box of
    # diagonal 1) and is centred on the origin. We'll undo this after the fit
    # so the user can use whatever scale they prefer.
    
    (trimesh_source, trimesh_target, translation) = transform(source, target)
    log_counter += 1
    logging.info(f"|-{log_counter}: Transformed meshes again (applied translation)")
    
    restore:Translation = translation.pseudoinverse()
    log_counter += 1
    logging.info(f"|-{log_counter}: Pseudoinverse on translation applied")

    n_dims = trimesh_source.n_dims
    # Homogeneous dimension (1 extra for translation effects)
    h_dims = n_dims + 1
    points, trilist = trimesh_source.points, trimesh_source.trilist
    n = points.shape[0]  # record number of points

    edge_triangles = trimesh_source.boundary_tri_index()
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
    current_deformed_points = points
    log_counter += 1
    logging.info(f"|-{log_counter}: Initialized transformation")

    stiffness_weights = [0] * iterations
    log_counter += 1
    logging.info(f"|-{log_counter}: Created stiffness weights")

    data_weights:list = [None] * iterations

    log_counter += 1
    logging.info(f"|-{log_counter}: Instantiated data and stiffness weights")

    row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(), np.arange(n)))

    x = np.arange(n * h_dims).reshape((n, h_dims))
    col = np.hstack((x[:, :n_dims].ravel(), x[:, n_dims]))
    array_of_ones = np.ones(n)
    log_counter += 1
    logging.info(f"|-{log_counter}: Parsing matrix")

    for i, (alpha, gamma) in enumerate(zip(stiffness_weights, data_weights), 1):
        logging.info(f"*** Iteration {i} ***")

        alpha_is_per_vertex = isinstance(alpha, np.ndarray)
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
            # we don't have to recalculate M_kron_G_s
            alpha_times_incidence_heavier_matrix = alpha * heavier_incidence_matrix
            logging.info(" | Alpha is not per vertex (stiffness is global)")

        j = 0
        while True:  # iterate until convergence
            j += 1  # track the iterations for this alpha/landmark weight
            logging.info(" | *** Iterate until convergence ***")
            # find nearest neighbour and the normals
            closest_points, triangle_indices = closest_points_on_target(current_deformed_points)

            # ---- WEIGHTS ----
            # 1.  Edges
            # Are any of the corresponding tris on the edge of the target?
            # Where they are we return a false weight (we *don't* want to
            # include these points in the solve)
            mask_edges = np.in1d(triangle_indices, edge_triangles, invert=True)
            logging.info(" |  | If points are already on the mesh, in edge_point_weights is false")
            # 2. Normals
            # Calculate the normals of the current v_i
            current_deformed_points_trimesh = TriMesh(current_deformed_points, trilist=trilist, copy=False)
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

            prop_omitted = (n - mask_all.sum() * 1.0) / n
            prop_omitted_norms = (n - mask_normals.sum() * 1.0) / n
            prop_omitted_edges = (n - mask_edges.sum() * 1.0) / n
            logging.info(" |  | Divided values")

            if gamma is not None:
                mask_all = mask_all * gamma
                logging.info(" |  | Gamma is not null")

            # Build the sparse diagonal weight matrix
            sparse_diagonal_weight_matrix = sp.diags(mask_all.astype(np.float64)[None, :], [0])
            logging.info(" |  | Building sparse diagonal weight matrix...")

            data = np.hstack((current_deformed_points.ravel(), array_of_ones))
            data_sparse_matrix = sp.coo_matrix((data, (row, col)))

            to_stack_A = [alpha_times_incidence_heavier_matrix, sparse_diagonal_weight_matrix.dot(data_sparse_matrix)]
            to_stack_B = [np.zeros((alpha_times_incidence_heavier_matrix.shape[0], n_dims)),
                          closest_points * mask_all[:, None]]  # nullify nearest points by combined_weights
            logging.info(" |  | Building done!")

            matrixA = sp.vstack(to_stack_A).tocsr()
            matrixB = sp.vstack(to_stack_B).tocsr()
            
            matrixB_transposed = matrixB.transpose()
            
            a = matrixA
            b = matrixB
            #a:np.ndarray = toarray_without_loss(matrixA)
            #b = toarray_without_loss(matrixB_transposed)
            
            
            #from scipy.sparse.linalg import lsqr
            # Solve the linear system
            #solved = spsolve(toarray_without_loss(matrixA), toarray_without_loss(matrixB_transposed))
            #solved = scipy_spsolve(toarray_without_loss(matrixA), toarray_without_loss(matrixB_transposed))
            from scipy.sparse.linalg import lsmr

            # Iterate over each column of b
            '''solved = []
            for i in range(b.shape[0]):
                solution = lsmr(a, toarray(b[i, :]))
                solved.append(solution)'''
            
            '''residuals = []
            for i in range(b.shape[0]):
                # Obtain the solution
                solution = lsmr(a, toarray(b[i, :]))
                
                # Calculate the residual using the original equation
                residual = np.linalg.norm(a.dot(solution) - toarray(b[i, :]))
                
                # Append the residual to the list
                residuals.append(residual)

            # Check if all residuals are close to zero
            tolerance = 1e-6  # Define a tolerance level
            all_correct = all(residual < tolerance for residual in residuals)

            if all_correct:
                print("All solutions are correct.")
            else:
                print("Some solutions are not accurate.")'''
            
            #solved = lsmr(a, b.T)
            
            solution = []
            for i in range(b.shape[1]):
                # Extract the i-th row of b
                b_row = b[:, i].reshape(1, -1)
                qwertyuiop = toarray_without_loss(b_row)
                
                # Solve the linear system for the i-th right-hand side vector
                (sol, _, _, _, _, _, _, _) = lsmr(a, qwertyuiop)
                
                # Append the solution to the list of solutions
                solution.append(sol)
                
            # Combine the solutions into a single array
            #solved = np.hstack(solution)
            solved = solution
            
            #SCALE = 1000
            #solved = custom_solver(a, b)
            #solved = np.linalg.solve(toarray_without_loss(a), toarray_without_loss(b))
            #A_pinv = np.linalg.pinv(toarray_without_loss(a))
            #solved = np.dot(A_pinv, toarray_without_loss(b))
            
            #P, L, U = scipy.linalg.lu(atest)
            #print("LU")
            #y = scipy.linalg.solve_triangular(L, np.dot(P, btest), lower=True)
            #print("triangles")

            # Solve for x: Ux = y
            #solved = scipy.linalg.solve_triangular(U, y, lower=False)
            logging.info(" |  | Applied scipy_spsolve")

            # deform template
            previous_deformed_points = current_deformed_points
            current_deformed_points = data_sparse_matrix.T.dot(solved)
            deformation_per_step = current_deformed_points - previous_deformed_points
            logging.info(" |  | Deformed template")

            error_delta = np.linalg.norm(previousX - solved, ord="fro")
            stop_criterion = error_delta / np.sqrt(np.size(previousX))

            previousX = solved

            current_instance = trimesh_source.copy()
            current_instance.points = current_deformed_points.copy()
            logging.info(f" |  | Finished iteration {i}")
            
            # track the progress of the algorithm per-iteration
            info_dict = {
                "alpha": alpha,
                "iteration": j,
                "prop_omitted": prop_omitted,
                "prop_omitted_norms": prop_omitted_norms,
                "prop_omitted_edges": prop_omitted_edges,
                "delta": error_delta,
                "mask_normals": mask_normals,
                "mask_edges": mask_edges,
                "mask_all": mask_all,
                "nearest_points": restore.apply(closest_points),
                "deformation_per_step": deformation_per_step,
            }

            logging.info(f"|{'-' * i} {info_dict}")

            if stop_criterion < threshold:
                break
            
    logging.info("Non-Rigid ICP done!")
    return current_deformed_points