#========[ IMPORTS ]========
from modules import bvh, distance, nricp, utils

import logging

# Setting up so that in the logs the file name shows up
#logger = logging.getLogger(__name__)

#========[ FUNCTIONS ]========

def main(garment_file:str, smpl_file:str, aligned_garment_file:str = ""):
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
    garment = utils.classes.Object(garment_file)
    smpl = utils.classes.Object(smpl_file)
    
    #utils.pretreat.pretreating(garment.vertices)
    #utils.plot_alignment_2(garment_vertices, smpl_vertices, ANGLE)

    # Non-rigid ICP
    #logging.info("Attempting to apply a Non-Rigid ICP from garment to SMPL")
    #aligned_garment = nricp.nricp.non_rigid_icp(garment.vertices, smpl.vertices)
    #logging.info("Success applying Non-Rigid ICP")
    
    # Distance
    # Step 1: smoothing garment

    # Step 2: garment to SMPL
    projected_garment = distance.points_to_mesh(garment.vertices, smpl.faces, smpl.vertices)

    # Step 3: find garment face
    links = distance.associate_faces(projected_garment, garment.faces, smpl.vertices)

    # Step 4: SMPL to garment
    projected_smpl = distance.points_to_faces(smpl.vertices, garment.vertices, garment.faces, links)
    
    #new_vertices, dist, new_smpl, why = distance.point_to_mesh(smpl.vertices, garment.vertices, garment.faces)
    #new_vertices, dist, new_smpl = distance.point_to_mesh(new_smpl, garment.vertices, garment.faces)
    #new = utils.Object()
    #new.vertices = new_vertices
    #new.save_obj(aligned_garment_file)

    #vectors = distance.calculate_vectors(why, new_vertices)
    #norms = distance.calculate_norms(garment.faces, garment.vertices)
    #centers = distance.calculate_centers(garment.faces, garment.vertices)

    #plot = utils.k3d_display.representation(garment, new_vertices, new_smpl, dist, vectors, norms, centers, why)
    #utils.k3d_display.save("hello", plot)

    #utils.plot_alignment_2(aligned_garment, smpl_vertices, ANGLE)

    # Save aligned garment
    logging.info("Attempting to save the aligned garment in OBJ")
    #utils.save_obj(aligned_garment_file, garment_vertices)