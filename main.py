#========[ IMPORTS ]========
from modules import distance, nricp, utils
import logging

# Setting up so that in the logs the file name shows up
#logger = logging.getLogger(__name__)

#========[ FUNCTIONS ]========

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%I:%M:%S %p')
    
    # Viewpoint angle on XY plan, optimal are: 100: front, 180: side
    ANGLE = 100
    
    # Load garment and SMPL body
    logging.info("Attempting to load OBJ and SMPL")
    garment = utils.Garment()
    smpl = utils.SMPLModel()
    
    '''garment.load_obj(garment_file)
    if smpl_file[-3:] == "npz":
        smpl.load_npz(smpl_file)
    elif smpl_file[-3:] == "obj":
        smpl.load_obj(smpl_file)
    else:
        raise Exception("Incompatible data type")''' # Does not work
    
    utils.pretreat.pretreating(garment.vertices)
    #utils.plot_alignment_2(garment_vertices, smpl_vertices, ANGLE)

    # Non-rigid ICP
    logging.info("Attempting to apply a Non-Rigid ICP from garment to SMPL")
    #aligned_garment = nricp.nricp.non_rigid_icp(garment.vertices, smpl.vertices)
    logging.info("Success applying Non-Rigid ICP")
    
    #result = neighbours.nearest_neighbours_generator(garment.vertices, smpl.vertices)
    
    #cham = ChamferDistance()
    #torch_garment = torch.Tensor([garment.vertices])
    #torch_smpl = torch.Tensor([smpl.vertices])
    
    #utils.plot_alignment_2(aligned_garment, smpl_vertices, ANGLE)

    # Save aligned garment
    logging.info("Attempting to save the aligned garment in OBJ")
    #utils.save_obj(aligned_garment_file, garment_vertices)