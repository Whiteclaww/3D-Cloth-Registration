#========[ IMPORTS ]========
import logging
from modules.utils.rotation_translate import origin_rotation

#========[ FUNCTIONS ]========

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

def pretreating(garmentVertices):
    # Fill the Y and Z lists
    logging.info("Attempting to fill the lists Y and Z before rotation of OBJ (always at a 90 degree angle)")
    garmentY:list = []
    garmentZ:list = []
    (garmentY, garmentZ) = fillYZ(garmentVertices)
    
    # Rotate the Y and Z lists
    logging.info("Attempting to rotate the garment")
    (garmentY, garmentZ) = origin_rotation(garmentY, garmentZ, 90)
    
    # Replace the Y and Z lists in the numpy ndarray
    logging.info("Attempting to replace the Y and Z elements in the garment list")
    garmentVertices = replace(garmentVertices, garmentY, garmentZ)