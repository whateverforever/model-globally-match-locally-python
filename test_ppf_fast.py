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

def test_compute_ppf():
    mesh = trimesh.load("./example_models/model_small.ply")
    step_rad = 0.1
    step_dist = 0.1

    _, pairs_model_py, _ = ppf.compute_ppf(
        mesh, step_rad, step_dist, alphas=False
    )

    # Current bug in nanobind: Doesn't recognize write protected arrays
    F_verts = np.asfortranarray(mesh.vertices)
    F_verts.setflags(write=True)
    F_norms = np.asfortranarray(mesh.vertex_normals)
    F_norms.setflags(write=True)

    pairs_model_cpp = ppf_fast.compute_ppf(
        F_verts, F_norms, step_rad, step_dist
    )

    for key in pairs_model_cpp:
        pypair = pairs_model_py[key[0]][key[1]]
        if pypair is None:
            continue
        
        # Allow result to be off by one bin in any of the features
        assert np.sum(np.subtract(pypair, pairs_model_cpp[key])) <= 1
