#========[ IMPORTS ]========
import numpy as np

#========[ CLASSES ]========

class Object():
    def __init__(self, vertices = np.empty(0), faces = np.empty(0)):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        self.faces = faces
        self.vertices = vertices
    
    def load_npz(self, model_path):
        params = np.load(model_path)
        self.vertices = params['v_template']
        self.faces = params['f']
    
    def load_obj(self, filename):
        vertices, face_indices = [], []
        with open(filename, 'r') as f:
            file = f.readlines()
        for line in file:
            # 3D vertex
            if line.startswith('v '):
                coord = [float(n) for n in line.replace('v ','').split(' ')]
                vertices += [coord]
            # Face
            elif line.startswith('f '):
                idx = [n.split('/') for n in line.replace('f ','').split(' ')]
                f = [int(float(n[0])) - 1 for n in idx]
                face_indices += [f]

        self.vertices = np.array(vertices, np.float32)
        self.faces = np.array(face_indices, np.float32)
    
    def save_obj(self, save_file_name):
        with open(save_file_name, 'w') as file:
            # Vertices
            for vertex in self.vertices:
                line = 'v ' + ' '.join([str(_) for _ in vertex]) + '\n'
                file.write(line)
            # 3D Faces
            faces = [[str(i + 1) for i in f] for f in self.faces]		
            for f in faces:
                line = 'f ' + ' '.join(f) + '\n'
                file.write(line)
    
    def link_vertices_to_faces(self):
        full_faces = []
        for face in self.faces:
            face_coordinates = []
            for index in face:
                index = int(index)
                face_coordinates.append(self.vertices[index])
            full_faces.append(face_coordinates)
        return full_faces