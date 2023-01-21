import math

import trimesh
import numpy as np
import ppf_fast
import ppf


def test_vector_angle():
    np.random.seed(123)
    
    for i in range(50):
        vecA = np.random.uniform(size=3)
        vecB = np.random.uniform(size=3)

        vecA /= np.linalg.norm(vecA)
        vecB /= np.linalg.norm(vecB)
        print(vecA)
        print(vecB)

        anglepy = trimesh.geometry.vector_angle([(vecA, vecB)])[0]
        print(anglepy)

        anglecpp = ppf_fast.vector_angle(vecA, vecB)
        print(anglecpp)

        assert math.isclose(anglepy, anglecpp, abs_tol=1e-6)

def test_compute_feature():
    step_rad = 0.1
    step_dist = 0.1

    np.random.seed(123)
    for i in range(50):
        vertA = np.random.uniform(-10,10, size=3)
        vertB = np.random.uniform(-10,10, size=3)
        normA = np.random.uniform(size=3)
        normB = np.random.uniform(size=3)
        normA /= np.linalg.norm(normA)
        normB /= np.linalg.norm(normB)

        Fpy = ppf.compute_feature(vertA, vertB, normA, normB, step_rad, step_dist)
        print("Fpy", Fpy)

        Fcpp = ppf_fast.compute_feature(vertA, vertB, normA, normB, step_rad, step_dist)
        print("Fcpp", Fcpp)

        assert np.allclose(Fcpp, Fpy)