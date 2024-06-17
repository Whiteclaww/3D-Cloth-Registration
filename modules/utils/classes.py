#========[ IMPORTS ]========
import logging
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
            if filename[-4:] == ".npz":
                self.load_npz(filename)
            elif filename[-4:] == ".obj":
                self.load_obj(filename)
            else:
                logging.error(' Object_Initialisation: Wrong file type, has to be "", "[...].npz" or "[...].obj"')
    
    def load_npz(self, model_path):
        params = np.load(model_path)
        self.faces = params['f'].tolist()
        self.vertices = params['v_template'].tolist()
    
    def load_obj(self, filename):
        with open(filename, 'r') as face_indices:
            file = face_indices.readlines()
            
        vertices, face_indices = [], []
        for line in file:
            # 3D vertex
            if line.startswith('v '):
                vertices_3 = []
                for n in line.replace('v ','').split(' '):
                    vertices_3.append(np.float32(n))
                vertices.append(vertices_3)
            # Face
            elif line.startswith('f '):
                count = 0
                face = []
                for n in line.replace('f ','').split(' '):
                    face.append(np.array(n.split('/'), np.float32))
                    count += 1
                faces = []
                for n in face:
                    faces.append(int(n[0]) - 1)
                face_indices.append(faces)
        self.faces = face_indices
        self.vertices = vertices
    
    def save_obj(self, save_file_name):
        if save_file_name != "":
            if save_file_name[-4:] != ".obj":
                save_file_name += ".obj"
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
            return 0 # Saved successfully
        return 1 # Saved unsuccesssfully
    
    def link_vertices_to_faces(self):
        full_faces = []
        for face in self.faces:
            face_coordinates = []
            for index in face:
                index = int(index)
                face_coordinates.append(self.vertices[index])
            full_faces.append(face_coordinates)
        return full_faces
    
    def convert_to_triangular(self):
        triangular_mesh = []
        for face in self.faces:
            if len(face) == 4:  # Si la face est carrée
                # Diviser la face carrée en deux triangles
                triangular_mesh.append([face[0], face[1], face[2]])
                triangular_mesh.append([face[0], face[2], face[3]])
            elif len(face) == 3:  # Si la face est déjà un triangle
                triangular_mesh.append(face)
            else:
                logging.error("Attention : La face n'est ni un carré ni un triangle.")
        self.trifaces = triangular_mesh