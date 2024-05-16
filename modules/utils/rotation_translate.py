#========[ IMPORTS ]========
import logging
import math
import numpy as np

#========[ FUNCTIONS ]========

# Rotation on X axis, on the centre of the garment
def axis_rotation(garmentY:list, garmentZ:list, degree:int):
    """
    Transformation on plan YZ, rotation on X axis

    Args:
        garmentY (list): list of all Y coordinates
        garmentZ (list): list of all Z coordinates
        degree (int): angle we have to rotate

    Returns:
        tuple: Returns the new Y and Z lists
    """
    logging.info("|-1: Entered Axis Rotation function")
    # Convert angle to radians
    angle = degree * math.pi / 180

    # Calculating rotation matrix
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0],
                                [0, 0, 1]])

    # Combine Y and Z coordinates into a single array
    coordinates = np.column_stack((garmentY, garmentZ, np.zeros_like(garmentY)))

    # Rotate coordinates
    mean = np.mean(coordinates, axis=0)
    rotated_coordinates = np.dot(coordinates - mean, rotation_matrix) + mean
    logging.info("|-2: Rotated coordinates")

    return rotated_coordinates[:, 0], rotated_coordinates[:, 1]

# Rotation on X axis, based on the origin
def origin_rotation(garmentY:list, garmentZ:list, degree:int):
    """
    Transformation on plan YZ, rotation on X axis

    Args:
        garmentY (list): list of all Y coordinates
        garmentZ (list): list of all Z coordinates
        degree (int): angle we have to rotate

    Returns:
        tuple: Returns the new Y and Z lists
    """
    logging.info("|-1: Entered Axis Rotation function")
    # Convert angle to radians
    angle:float = degree * math.pi / 180

    # Calculate rotation matrix for rotation around the X-axis
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0],
                                [0, 0, 1]])

    # Combine Y and Z coordinates into a single array
    coordinates = np.column_stack((garmentY, garmentZ, np.ones_like(garmentY)))

    # Apply translation to move rotation center to the origin, then rotation, then translation back (Although translation is identity matrix)
    rotated_coordinates = np.dot(coordinates, rotation_matrix)
    logging.info("|-2: Rotated coordinates")

    return rotated_coordinates[:, 0], rotated_coordinates[:, 1]