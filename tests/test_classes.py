#========[ IMPORTS ]========
from modules.utils import classes

import logging
import numpy as np
import os

#========[ FUNCTIONS ]========

def catch_exception(text, condition):
    try:    
        condition()
    except FileNotFoundError:
        logging.error(f"File not found: {text}")
    except Exception as e:
        logging.error(f"An exception has occurred: {e}")

# Class Object

def test_Object():
    logging.debug('Testing Object: initialization function || 0/5 Passed') # not testing with a valid extension
    test_init()
    logging.debug('Testing Object: loading obj function || 1/5 Passed')
    obj = test_load_obj()
    logging.debug('Testing Object: loading npz function || 2/5 Passed')
    test_load_npz()
    logging.debug('Testing Object: saving obj function || 3/5 Passed')
    test_save_obj(obj)
    logging.debug('Testing Object: vertices to face function || 4/5 Passed')
    test_vertices_to_face(obj)
    logging.debug('Testing Object: =====[ 5/5 PASSED ]=====')

# Functions within Object

def test_init() -> None:
    # Variables
    empty = ""
    text = "hello"
    wrong_extension = "a.abc"
    
    empty_object = classes.Object(empty)
    text_object = classes.Object(text)
    wrong_extension_object = classes.Object(wrong_extension)
    
    assert [] == empty_object.faces, "Testing Object initialization: When initializing an empty OBJECT the list of faces is not empty"
    assert [] == text_object.faces, "Testing Object initialization: When initializing with a wrong string the list of faces is not empty"
    assert [] == wrong_extension_object.faces, "Testing Object initialization: When initializing an invalid extension the list of faces is not empty"

def test_load_npz() -> None:
    false_npz = "a.npz"
    smpl_fem = "3D-Cloth-Registration/test_dataset/smpl_highres_female.npz"
    
    #catch_exception(text = false_npz, condition = lambda: classes.Object(false_npz))
    
    npz = classes.Object(smpl_fem)
    
    # Tests
    correct_npz_faces = np.load(smpl_fem)['f'].tolist()
    assert correct_npz_faces[0] == npz.faces[0], "Testing Object loading npz: When initializing an empty OBJECT the list of faces is not correct"
    
def test_load_obj() -> classes.Object:
    false_obj = "a.obj"
    garment = "3D-Cloth-Registration/test_dataset/00006/Top.obj"
    
    #catch_exception(text = false_obj, condition=lambda: classes.Object(false_obj))
    
    obj = classes.Object(garment)
    
    # Tests
    with open(garment, 'r') as f:
            file = f.readlines()
    for line in file:
        if line.startswith('f '):
            count = 0
            face_obj = []
            for n in line.replace('f ','').split(' '):
                face_obj.append(np.array(n.split('/'), np.float32))
                count += 1
            face_obj = np.array(face_obj, np.float32)
            count = 0
            correct_obj_faces_first = np.empty(len(face_obj), int)
            for n in face_obj:
                correct_obj_faces_first[count] = int(n[0]) - 1
                count += 1
            break

    assert np.all(np.array(correct_obj_faces_first) == np.array(obj.faces[0])), "Testing Object loading obj: When initializing an empty OBJECT the list of faces is not correct"
    return obj

def test_save_obj(obj:classes.Object) -> None:
    empty_str = ""
    text = "hello"
    invalid = "a.abc"
    valid = "valid.obj"
    
    def save_empty():
        empty = classes.Object()
        
        assert empty.save_obj(empty_str) == 1, "Testing Object save obj: Should not save a file with a empty name"
        assert empty.save_obj(text) == 0, "Testing Object save obj: Should save and add .obj at the end of the filename"
        assert empty.save_obj(invalid) == 0, "Testing Object save obj: Should save and add .obj at the end of the filename even though it is the wrong extension"
        assert empty.save_obj(valid) == 0, "Testing Object save obj: Should have saved a file with a valid extension"
    
    def save_full():
        full = classes.Object()
    
        assert full.save_obj(empty_str) == 1, "Testing Object save obj: Should not save a file with a empty name"
        assert full.save_obj(text) == 0, "Testing Object save obj: Should save and add .obj at the end of the filename"
        assert full.save_obj(invalid) == 0, "Testing Object save obj: Should save and add .obj at the end of the filename even though it is the wrong extension"
        assert full.save_obj(valid) == 0, "Testing Object save obj: Should have saved a file with a valid extension"
    
    save_empty()
    save_full()
    
    os.remove(text + ".obj")
    os.remove(invalid + ".obj")
    os.remove(valid)

def test_vertices_to_face(obj:classes.Object) -> None:
    classes.Object.link_vertices_to_faces