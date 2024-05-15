#========[ IMPORTS ]========
import logging
import numpy as np
import matplotlib.pyplot as plt

#========[ FUNCTIONS ]========

def plot_alignment_multiple(garment_vertices, smpl_vertices, angle:int) -> None:
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
    
def plot_alignment_single(garment_vertices, angle:int) -> None:
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