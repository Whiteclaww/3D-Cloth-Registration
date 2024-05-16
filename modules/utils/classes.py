#========[ IMPORTS ]========
import numpy as np

#========[ CLASSES ]========

class Object():
    def __init__(self, filename:str = ""):
        """
        Either the SMPL or Garment

        Args:
            filename (str): Name of the object's file.
        """
        self.faces = []
        self.vertices = []
        
        if filename != "":
            if filename[-3:] == "npz":
                self.load_npz(filename)
            elif filename[-3:] == "obj":
                self.load_obj(filename)
            else:
                raise Exception("Incompatible data type")
    
    def load_npz(self, model_path):
        params = np.load(model_path)
        self.faces = params['f'].tolist()
        self.vertices = params['v_template'].tolist()
    
    def load_obj(self, filename):
        vertices, face_indices = [], []
        with open(filename, 'r') as f:
            file = f.readlines()
        for line in file:
            # 3D vertex
            if line.startswith('v '):
                count = 0
                coord = np.empty(3, np.float32)
                for n in line.replace('v ','').split(' '):
                    coord[count] = np.float32(n)
                    count += 1
                vertices.append(coord)
            # Face
            elif line.startswith('f '):
                count = 0
                face = []
                for n in line.replace('f ','').split(' '):
                    face.append(np.array(n.split('/'), np.float32))
                    count += 1
                face = np.array(face, np.float32)
                count = 0
                f = np.empty(len(face), int)
                for n in face:
                    f[count] = int(n[0]) - 1
                    count += 1
                face_indices.append(f)
        self.faces = face_indices
        self.vertices=vertices
    
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