#========[ IMPORTS ]========
from modules.distance import plan

#========[ FUNCTIONS ]========

def test_dist_face():
    vertex0 = [0, 0, 0]
    vertex1 = [1, 0, 0]
    vertex2 = [2, 0, 0]

    face = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]

    dist0 = plan.dist_face(vertex0, face)
    dist1 = plan.dist_face(vertex1, face)
    dist2 = plan.dist_face(vertex2, face)

    assert dist0 == 1, "Testing plan distance to face: distance found is incorrect"
    assert dist1 == 0, "Testing plan distance to face: distance found is incorrect"
    assert dist2 == 1, "Testing plan distance to face: distance found is incorrect"

def test_divide():
    vertex2 = [2, 2, 2]
    vertex2_0 = [2, 0, 0]

    value2 = 2
    value0 = 0

    found0 = plan.divide(vertex2, value2)
    found1 = plan.divide(vertex2_0, value2)

    not_found0 = plan.divide(vertex2, value0)
    not_found1 = plan.divide(vertex2_0, value0)

    assert found0 == [1, 1, 1], "Testing plan divide: Division does not return the correct value"
    assert found1 == [1, 0, 0], "Testing plan divide: Division does not return the correct value"

    assert not_found0 == None, "Testing plan divide: Division by 0"
    assert not_found1 == None, "Testing plan divide: Division by 0"

'''def test_find_vertex():
    vertex0 = [0, 0, 0]
    vertex1 = [1, 0, 0]
    vertex2 = [2, 0, 0]

    face1 = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]

    found0 = plan.find_vertex(vertex0, face1, 1)
    found1 = plan.find_vertex(vertex1, face1, 0)
    found2 = plan.find_vertex(vertex2, face1, 1)

    assert plan.norm(plan.vector(vertex0, found0)) == 1, "Testing plan find vertex: The new vertex found is incorrect, distance not equal"
    assert plan.norm(plan.vector(vertex1, found1)) == 0, "Testing plan find vertex: The new vertex found is incorrect, distance not equal"
    assert plan.norm(plan.vector(vertex2, found2)) == 1, "Testing plan find vertex: The new vertex found is incorrect, distance not equal"

    #assert not_found0 == None, "Testing plan find vertex: a vertex not found should be None"'''

def test_mul():
    pass

def test_norm():
    vertex0 = [2, 1, -2]

    norm0 = plan.norm(vertex0)

    assert norm0 == 3, "Testing plan norm: The norm is incorrect"

def test_Plan():
    A0 = [0, 0, 1]
    B0 = [0, 1, 1]
    C0 = [0, 1, 0]

    plan0 = plan.Plan(A = A0, B = B0, C = C0)

    correct_n = [-1, 0, 0]

    assert plan0.n == correct_n, "Testing plan face normal: The normal of the face is incorrect"

def test_produit_scalaire():
    vect0 = [1, 2, 3]
    vect1 = [4, 5, 6]

    produit0 = plan.produit_scalaire(vect0, vect1)

    assert produit0 == 32, "Testing plan produit scalaire: incorrect value"

def test_produit_vectoriel():
    vect0 = [1, 2, 3]
    vect1 = [4, 5, 6]

    produit0 = plan.produit_vectoriel(vect0, vect1)

    correct_produit0 = [-3, 6, -3]

    assert produit0 == correct_produit0, "Testing plan produit vectoriel: incorrect value"

def test_sum_vect():
    pass

def test_vector():
    vertex0 = [1, 2, 3]
    vertex1 = [-4, 5, 25]

    vect01 = plan.vector(vertex0, vertex1)

    correct_vect01 = [-5, 3, 22]

    assert vect01 == correct_vect01, "Testing plan vector: incorrect vector"

def test_norm2():
    vect0 = [1, 0, 0]
    vect1 = [-1, 0, 0]
    vect2 = [2, 0, 0]

    plan1 = plan.Plan(A = [0, 0, 0], B = [0, 1, 0], C = [0, 0, 1])

    value0 = plan.same_norm_as_plan(vect0, plan1)
    value1 = plan.same_norm_as_plan(vect1, plan1)
    value2 = plan.same_norm_as_plan(vect2, plan1)
    
    assert value0 == True
    assert value1 == False
    assert value2 == True