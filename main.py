import utils
import rotation_translate as rt
import logging
from nricp import non_rigid_icp

# Setting up so that in the logs the file name shows up
#logger = logging.getLogger(__name__)

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
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s: ==' + garment_file + '== %(message)s', datefmt='%I:%M:%S %p')
    
    # Load garment and SMPL body
    logging.info("Attempting to load OBJ and SMPL")
    garmentVertices = utils.load_obj(garment_file)
    smplVertices = utils.load_smpl(smpl_file)
    
    # Viewpoint angle on XY plan, optimal are: 100: front, 180: side
    ANGLE = 100
    #utils.plot_alignment_2(garmentVertices, smplVertices, ANGLE)
    
    # Fill the Y and Z lists
    logging.info("Attempting to fill the lists Y and Z before rotation of OBJ (always at a 90 degree angle)")
    garmentY:list = []
    garmentZ:list = []
    (garmentY, garmentZ) = utils.fillYZ(garmentVertices)
    
    # Rotate the Y and Z lists
    logging.info("Attempting to rotate the garment")
    (garmentY, garmentZ) = rt.origin_rotation(garmentY, garmentZ, 90)
    
    # Replace the Y and Z lists in the numpy ndarray
    logging.info("Attempting to replace the Y and Z elements in the garment list")
    garmentVertices = utils.replace(garmentVertices, garmentY, garmentZ)
    
    #utils.plot_alignment_2(garmentVertices, smplVertices, ANGLE)

    # Non-rigid ICP
    logging.info("Attempting to apply a Non-Rigid ICP from garment to SMPL")
    aligned_garment = non_rigid_icp(garmentVertices, smplVertices, iterations = 5)
    logging.info("Success applying Non-Rigid ICP")
    
    #utils.plot_alignment_2(aligned_garment, smplVertices, ANGLE)

    # Save aligned garment
    logging.info("Attempting to save the aligned garment in OBJ")
    utils.save_obj(aligned_garment_file, aligned_garment)