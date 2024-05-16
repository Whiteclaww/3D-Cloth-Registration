#========[ IMPORTS ]========
import numpy as np

#========[ CLASSES ]========

class Garment():
    def __init__(self, V = [], F = [], UV_V = [], UV_F = []) -> None:
        """
        Initialise a garment object

        Args:
            V (list): vertices
            F (list): faces
            UV_V (list): UV_vertices
            UV_F (list): UV_faces
        """
        self.vertices = V
        self.faces = F
        self.UV_vertices = UV_V
        self.UV_faces = UV_F
    
    def load_obj(self, filename):
        vertices, faces, UV_vertices, UV_faces = [], [], [], []
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
                faces += [f]
                # UV face
                if '/' in line:
                    f = [int(n[1]) - 1 for n in idx]
                    UV_faces += [f]
        vertices = np.array(vertices, np.float32)
        UV_vertices = np.array(UV_vertices, np.float32)
        if UV_faces:
            assert len(faces) == len(UV_faces), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
        else:
            UV_vertices, UV_faces = None, None
        self.__init__(vertices, faces, UV_vertices, UV_faces)
    
    def save_obj(self, save_file_name):
        if self.UV_vertices is not None:
            assert len(self.faces) == len(self.UV_faces), 'Inconsistent data, mesh and UV map do not have the same number of faces'
            
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
            if self.UV_faces:
                faces = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(self.faces,self.UV_faces)]
            else:
                faces = [[str(i + 1) for i in f] for f in self.faces]		
            for f in faces:
                line = 'f ' + ' '.join(f) + '\n'
                file.write(line)

class SMPLModel():
    def __init__(self, weights = np.empty(0), v_template = np.empty(0), shapedirs = np.empty(0), faces = np.empty(0),
                 vertices = np.empty(0)):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        self.weights = weights
        self.v_template = v_template
        self.shapedirs = shapedirs
        self.faces = faces

        # Matrices
        self.vertices = vertices
    
    def load_npz(self, model_path):
        params = np.load(model_path)
        self.__init__(weights=params['weights'], v_template=params['v_template'],
                      shapedirs=params['shapedirs'], faces=params['f'])
    
    def load_obj(self, filename):
        vertices, faces, UV_vertices, UV_faces = [], [], [], []
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
                faces += [f]
                # UV face
                if '/' in line:
                    f = [int(n[1]) - 1 for n in idx]
                    UV_faces += [f]
        vertices = np.array(vertices, np.float64)
        UV_vertices = np.array(UV_vertices, np.float64)
        faces = np.array(vertices, np.float64)
        UV_faces = np.array(UV_vertices, np.float64)
        if UV_faces:
            assert len(faces) == len(UV_faces), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
        else:
            UV_vertices, UV_faces = None, None
        self.__init__(vertices=vertices, faces=faces)

class Linked_Node():
    def __init__(self, vertex, face, distance):
        self.vertex = vertex
        self.face = face
        self.distance = distance

class Linked():
    def __init__(self, length):
        self.linked = [None] * length
    
    def add_node(self, vertex, face, distance):
        self.linked[vertex] = Linked_Node(vertex, face, distance) # type: ignore