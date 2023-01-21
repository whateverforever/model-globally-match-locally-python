import math

import trimesh
import numpy as np
import ppf_fast
import ppf


def test_vector_angle():
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

    vertA = np.array([0,0,0])
    vertB = np.array([1,1,1])
    normA = np.array([1,0,0])
    normB = np.array([0,1,0])

    Fpy = ppf.compute_feature(vertA, vertB, normA, normB, step_rad, step_dist)
    print("Fpy", Fpy)

    Fcpp = ppf_fast.compute_feature(vertA, vertB, normA, normB, step_rad, step_dist)
    print("Fcpp", Fcpp)

    assert np.allclose(Fcpp, Fpy)