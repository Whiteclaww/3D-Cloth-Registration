import numpy as np
import matplotlib.pyplot as plt
import logging

def load_obj(file_path:str):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = line.strip().split()[1:]
                vertices.append([float(coord) for coord in vertex])
    logging.info(f"|- : Loaded OBJ object {file_path}")
    return np.array(vertices)

def save_simplified_smpl(file_path:str, return_file_path:str):
    smpl = load_obj(file_path)
    result_smpl = []
    for i in smpl:
        if i[1] < 0.35 and i[1] > -1.05 and abs(i[0]) < 0.7:
            result_smpl.append(i)
    save_obj(return_file_path, result_smpl)

def load_smpl(file_path:str):
    with np.load(file_path) as data:
        logging.info(f"|- : Loaded SMPL object {file_path}")
        return data['v_template']
    
def save_obj(file_path:str, vertices) -> None:
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {' '.join(map(str, vertex))}\n")
    logging.info(f"|- : Saved object at {file_path}")

def fillYZ(garment):
    Y, Z = [], []
    for i in garment:
        Y.append(i[1])
        Z.append(i[2])
    logging.info("|- : Filled Y and Z values using garment's Y and Z values")
    return (Y, Z)

def replace(garment, Y:list, Z:list):
    """Replace in the given numpy ndarray the Y and Z coordinates

    Args:
        garment: given array in which we replace
        Y (list): Y list
        Z (list): Z list
    """
    length:int = len(Y)
    for i in range(length):
        garment[i] = [garment[i][0], Y[i] - 0.25, Z[i]]
    logging.info("|- : Filled garment replacing Y and Z values")
    return garment

def plot_alignment_2(garment_vertices, smpl_vertices, angle:int) -> None:
    """Display the graph in a new window

    Args:
        garment_vertices
        smpl_vertices
        angle(int): angle of vision on the XY plan, around the Z axis
    """
    # Setting the axes
    plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.set_xlim(-0.3, 0.5)
    ax.set_ylim(-0.3, 0.5)
    ax.set_zlim(-0.3, 0.5) # type: ignore
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    
    # Debugging
    #ax.scatter([0], [0], [0.6], color='green', label='test') Z axis
    #ax.scatter([0], [0.6], [0], color='yellow', label='test') Y axis
    #ax.scatter([0.6], [0], [0], color='orange', label='test') X axis
    
    # Display the garment and the body
    ax.scatter(smpl_vertices[:, 0], smpl_vertices[:, 2], smpl_vertices[:, 1], c='blue', label='SMPL Body')
    ax.scatter(garment_vertices[:, 0], garment_vertices[:, 2], garment_vertices[:, 1], color='red', label='Garment')
    ax.legend()
    
    # Camera view
    ax.view_init(elev = 20, azim = angle) # type: ignore
    
    plt.show()
    logging.info("|- : Displayed an element!")
    
def plot_alignment_1(garment_vertices, angle:int) -> None:
    """Display the graph in a new window

    Args:
        garment_vertices
        angle(int): angle of vision on the XY plan, around the Z axis
    """
    # Setting the axes
    plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.set_xlim(-0.3, 0.5)
    ax.set_ylim(-0.3, 0.5)
    ax.set_zlim(-0.3, 0.5) # type: ignore
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    
    # Debugging
    #ax.scatter([0], [0], [0.6], color='green', label='test') Z axis
    #ax.scatter([0], [0.6], [0], color='yellow', label='test') Y axis
    #ax.scatter([0.6], [0], [0], color='orange', label='test') X axis
    
    # Display the garment and the body
    ax.scatter(garment_vertices[:, 0], garment_vertices[:, 2], garment_vertices[:, 1], color='red', label='Garment')
    ax.legend()
    
    # Camera view
    ax.view_init(elev = 20, azim = angle) # type: ignore
    
    plt.show()
    logging.info("|- : Displayed an element!")