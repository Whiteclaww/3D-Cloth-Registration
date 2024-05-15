#========[ IMPORTS ]========
from . import distance, nricp, utils
import logging
import numpy as np
import torch
# Setting up so that in the logs the file name shows up
#logger = logging.getLogger(__name__)

#========[ FUNCTIONS ]========

def pretreating(garmentVertices):
    # Fill the Y and Z lists
    logging.info("Attempting to fill the lists Y and Z before rotation of OBJ (always at a 90 degree angle)")
    garmentY:list = []
    garmentZ:list = []
    (garmentY, garmentZ) = utils.tools.fillYZ(garmentVertices)
    
    # Rotate the Y and Z lists
    logging.info("Attempting to rotate the garment")
    (garmentY, garmentZ) = utils.origin_rotation(garmentY, garmentZ, 90)
    
    # Replace the Y and Z lists in the numpy ndarray
    logging.info("Attempting to replace the Y and Z elements in the garment list")
    garmentVertices = utils.tools.replace(garmentVertices, garmentY, garmentZ)

def main(garment_file:str, smpl_file:str, aligned_garment_file:str):
    # Setting up logging, the level corresponds to what will be shown, options are:
    #   - NOTSET
    #   - DEBUG
    #   - INFO
    #   - WARN
    #   - ERROR
    #   - CRITICAL
    # The logger only displays its level of logging and all the ones below but not the ones on top
    # Ex: WARNING(or WARN) is the default, will print out only WARN, ERROR, CRITICAL
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: ==' + garment_file + '== %(message)s', datefmt='%I:%M:%S %p')
    
    # Viewpoint angle on XY plan, optimal are: 100: front, 180: side
    ANGLE = 100
    
    # Load garment and SMPL body
    logging.info("Attempting to load OBJ and SMPL")
    #garment = utils.Garment()
    #smpl = utils.SMPLModel()
    
    #garment.load_obj(garment_file)#utils.load_obj(garment_file)
    #if smpl_file[-3:] == "npz":
    #    smpl.load_npz(smpl_file)
    #elif smpl_file[-3:] == "obj":
    #    smpl.load_obj(smpl_file)
    #else:
    #    raise Exception("Incompatible data type")
    
    #pretreating(garment_vertices)+
    #utils.plot_alignment_2(garment_vertices, smpl_vertices, ANGLE)

    # Non-rigid ICP
    '''logging.info("Attempting to apply a Non-Rigid ICP from garment to SMPL")
    aligned_garment = non_rigid_icp(garment_vertices, smpl_vertices, i_iterations = i_iterations, j_iterations = j_iterations, alpha = alpha)
    logging.info("Success applying Non-Rigid ICP")'''
    
    #result = neighbours.nearest_neighbours_generator(garment.vertices, smpl.vertices)
    
    #cham = ChamferDistance()
    #torch_garment = torch.Tensor([garment.vertices])
    #torch_smpl = torch.Tensor([smpl.vertices])
            
    dist = distance.dist_face([1, 1, 1], [[0, 0, 0], [2, -1, 0], [5, 2, 0]])
    #print(result.detach().cpu().item())
    '''count = 0
    for i in result:
        if i[1] != -1:
            count += 1
            garment_vertices[i[1]] = np.zeros(3)'''
    
    #a = neighbours.chamferdist(garment.vertices, smpl_vertices)
    
    #utils.plot_alignment_2(aligned_garment, smpl_vertices, ANGLE)

    # Save aligned garment
    logging.info("Attempting to save the aligned garment in OBJ")
    #utils.save_obj(aligned_garment_file, garment_vertices)