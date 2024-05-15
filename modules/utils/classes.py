#========[ IMPORTS ]========
import numpy as np

#========[ CLASSES ]========

class Garment():
    def __init__(self, V = [], F = [], UV_V = [], UV_F = []) -> None:
        """
        Initialise a garment object

        Args:
            V (list): vertices
            F (list): face_indices
            UV_V (list): UV_vertices
            UV_F (list): UV_face_indices
        """
        self.vertices = V
        self.face_indices = F
        self.UV_vertices = UV_V
        self.UV_face_indices = UV_F
    
    def load_obj(self, filename):
        vertices, face_indices, UV_vertices, UV_face_indices = [], [], [], []
        with open(filename, 'r') as f:
            file = f.readlines()
        for line in file:
            # 3D vertex
            if line.startswith('v '):
                coord = [float(n) for n in line.replace('v ','').split(' ')]
                vertices += [coord]
            # UV vertex
            elif line.startswith('vt '):
                coord = [float(n) for n in line.replace('vt ','').split(' ')]
                UV_vertices += [coord]
            # Face
            elif line.startswith('f '):
                idx = [n.split('/') for n in line.replace('f ','').split(' ')]
                f = [int(n[0]) - 1 for n in idx]
                face_indices += [f]
                # UV face
                if '/' in line:
                    f = [int(n[1]) - 1 for n in idx]
                    UV_face_indices += [f]
        vertices = np.array(vertices, np.float32)
        UV_vertices = np.array(UV_vertices, np.float32)
        if UV_face_indices:
            assert len(face_indices) == len(UV_face_indices), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
        else:
            UV_vertices, UV_face_indices = None, None
        self.__init__(vertices, face_indices, UV_vertices, UV_face_indices)
    
    def save_obj(self, save_file_name):
        if self.UV_vertices is not None:
            assert len(self.face_indices) == len(self.UV_face_indices), 'Inconsistent data, mesh and UV map do not have the same number of faces'
            
        with open(save_file_name, 'w') as file:
            # Vertices
            for vertex in self.vertices:
                line = 'v ' + ' '.join([str(_) for _ in vertex]) + '\n'
                file.write(line)
            # UV verts
            if not self.UV_vertices is None:
                for vertex in self.UV_vertices:
                    line = 'vt ' + ' '.join([str(_) for _ in vertex]) + '\n'
                    file.write(line)
            # 3D Faces / UV faces
            if self.UV_face_indices:
                face_indices = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(self.face_indices,self.UV_face_indices)]
            else:
                face_indices = [[str(i + 1) for i in f] for f in self.face_indices]		
            for f in face_indices:
                line = 'f ' + ' '.join(f) + '\n'
                file.write(line)

class SMPLModel():
    def __init__(self, J_regressor = np.empty(0), weights = np.empty(0), posedirs = np.empty(0), v_template = np.empty(0), shapedirs = np.empty(0), faces = np.empty(0), kintree_table = np.empty(0),
                 pose_shape = [], beta_shape = [], trans_shape = [], vertices = np.empty(0)):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        self.J_regressor = J_regressor
        self.weights = weights
        self.posedirs = posedirs
        self.v_template = v_template
        self.shapedirs = shapedirs
        self.faces = faces
        self.kintree_table = kintree_table

        # id_to_col = {
        #   self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        # }
        # self.parent = {
        #   i: id_to_col[self.kintree_table[0, i]]
        #   for i in range(1, self.kintree_table.shape[1])
        # }
        
        self.parent = np.array(self.kintree_table).astype(np.int32)

        self.pose_shape = pose_shape
        self.beta_shape = beta_shape
        self.trans_shape = trans_shape

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        # Matrices
        self.vertices = vertices
        self.joint = None
        self.rotation = None
    
    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        dot = np.matmul(A, r_hat)
        return cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    
    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.
        
        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))
    
    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))
    
    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        # rotation matrix for each joint
        self.R = self.rodrigues(self.pose.reshape((-1, 1, 3)))
        I_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), (self.R.shape[0]-1, 3, 3))
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        # G = np.empty((self.kintree_table.shape[1], 4, 4))
        G = np.empty((len(self.parent), 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        # for i in range(1, self.kintree_table.shape[1]):
        for i in range(1, len(self.parent)):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(np.hstack([self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]))
            )
        G = G - self.pack(
            np.matmul(G, np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1]))
        )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]]) # type: ignore
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        self.vertices = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        #+ self.trans.reshape([1, 3])
    
    def load_npz(self, model_path):
        params = np.load(model_path)
        self.__init__(params['J_regressor'], params['weights'], params['posedirs'], params['v_template'],
                      params['shapedirs'], params['f'], params['kintree_table'], 
                      [24, 3], [10], [3])
        
        self.update()
        self.J_regressor = params['J_regressor']
    
    def load_obj(self, filename):
        vertices, face_indices, UV_vertices, UV_face_indices = [], [], [], []
        with open(filename, 'r') as f:
            file = f.readlines()
        for line in file:
            # 3D vertex
            if line.startswith('v '):
                coord = [float(n) for n in line.replace('v ','').split(' ')]
                vertices += [coord]
            # UV vertex
            elif line.startswith('vt '):
                coord = [float(n) for n in line.replace('vt ','').split(' ')]
                UV_vertices += [coord]
            # Face
            elif line.startswith('f '):
                idx = [n.split('/') for n in line.replace('f ','').split(' ')]
                f = [int(float(n[0])) - 1 for n in idx]
                face_indices += [f]
                # UV face
                if '/' in line:
                    f = [int(n[1]) - 1 for n in idx]
                    UV_face_indices += [f]
        vertices = np.array(vertices, np.float32)
        UV_vertices = np.array(UV_vertices, np.float32)
        if UV_face_indices:
            assert len(face_indices) == len(UV_face_indices), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
        else:
            UV_vertices, UV_face_indices = None, None
        #self.__init__(vertices=vertices, face_indices, UV_vertices, UV_face_indices)
        self.update()