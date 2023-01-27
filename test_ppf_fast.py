import math

import trimesh
import trimesh.transformations as tf
import numpy as np
import ppf_fast
import ppf


def test_align_vectors():
    np.random.seed(123)

    for _ in range(50):
        vecA = np.random.uniform(size=3)
        vecB = np.random.uniform(size=3)

        vecA /= np.linalg.norm(vecA)
        vecB /= np.linalg.norm(vecB)

        R_py = ppf.align_vectors(vecA, vecB)
        R_cc = ppf_fast.align_vectors(vecA, vecB)
        print("ref\n",R_py)
        print("cc\n", R_cc)

        assert np.allclose(R_cc, R_py)


def test_vector_angle():
    np.random.seed(123)

    for _ in range(50):
        vecA = np.random.uniform(size=3)
        vecB = np.random.uniform(size=3)

        vecA /= np.linalg.norm(vecA)
        vecB /= np.linalg.norm(vecB)
        print(vecA)
        print(vecB)

        anglepy = trimesh.geometry.vector_angle([(vecA, vecB)])[0]
        anglecpp = ppf_fast.vector_angle(vecA, vecB)
        assert math.isclose(anglepy, anglecpp, abs_tol=1e-6)

        anglepy_signed = ppf.vector_angle_signed_x(vecA, vecB)
        anglecpp_signed = ppf_fast.vector_angle_signed_x(vecA, vecB)
        assert math.isclose(anglepy_signed, anglecpp_signed)


def test_compute_feature():
    step_rad = 0.1
    step_dist = 0.1

    np.random.seed(123)
    for i in range(50):
        vertA = np.random.uniform(-10, 10, size=3)
        vertB = np.random.uniform(-10, 10, size=3)
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

    _, pairs_model_py, alphas_py = ppf.compute_ppf(mesh.vertices, mesh.vertex_normals, step_rad, step_dist, alphas=True)

    # Current bug in nanobind: Doesn't recognize write protected arrays
    F_verts = np.asfortranarray(mesh.vertices)
    F_verts.setflags(write=True)
    F_norms = np.asfortranarray(mesh.vertex_normals)
    F_norms.setflags(write=True)

    _, pairs_model_cpp, alphas_cpp = ppf_fast.compute_ppf(F_verts, F_norms, step_rad, step_dist, True)

    for ref in pairs_model_cpp:
        for other in pairs_model_cpp[ref]:
            pypair = pairs_model_py[ref][other]
            print("pypair", pypair)
            if pypair is None:
                continue
            ccpair = pairs_model_cpp[ref][other]
            print("ccpair", ccpair)

            # Allow result to be off by one bin in any of the features
            assert np.sum(np.subtract(ccpair, pypair)) <= 1
            assert np.isclose(alphas_py[(ref, other)], alphas_cpp[(ref, other)])

def test_pdist_rot():
    rotmats = []
    
    for _ in range(50):
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.uniform(size=3)
        axis /= np.linalg.norm(axis)
        rotmat = tf.rotation_matrix(angle, axis)
        
        rotmats.append(np.asfortranarray(rotmat[:3, :3]))
    
    dists_py = ppf.pdist_rot(rotmats)
    dists_cc = ppf_fast.pdist_rot(rotmats)

    print("python dists", dists_py)
    print("c++ dists", dists_cc)
    print(f"python side {rotmats[0].ctypes.data:x}")

    print("lens", len(dists_py), len(dists_cc))
    print("diff", np.sum(dists_py - dists_cc))

    assert np.allclose(dists_py, dists_cc)