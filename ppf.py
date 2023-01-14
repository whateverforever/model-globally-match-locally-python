#!/usr/bin/env python3
"""
This script computes the point-pair features of a given
model and tries to find the model in a given scene.
"""

import os
import random
import time
import argparse
import pickle
import math
from collections import defaultdict

import trimesh
import trimesh.transformations as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

VIS = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to the model pointcloud")
    parser.add_argument("scene", help="Path to the scene pointcloud")
    # parser.add_argument("scene_vis", help="Path to a version of the scene that is better for visualization")
    args = parser.parse_args()

    model: trimesh.Trimesh = trimesh.load(args.model)
    scene: trimesh.Trimesh = trimesh.load(args.scene)

    model.visual.vertex_colors = [[255, 0, 0, 128] for _ in model.vertices]
    scene.visual.vertex_colors = [[0, 255, 0, 255] for _ in model.vertices]

    if False and VIS:
        vis = trimesh.Scene([model, scene])
        vis.show()

    # 1. compute ppfs of all vertex pairs in model, store in hash table
    angle_num = 30
    angle_step = np.radians(360 / angle_num)
    dist_step = 0.05 * model.scale

    modelbase, _ = os.path.splitext(args.model)
    model_ppf_path = f"{modelbase}.ppf"
    if not os.path.exists(model_ppf_path):
        print("Computing model ppfs features")
        t_start = time.perf_counter()

        ppfs_model, pairs_model = compute_ppf(model, angle_step, dist_step)

        t_end = time.perf_counter()
        print(
            f"Computing ppfs for {len(model.vertices)} verts took {t_end - t_start:.2f}s"
        )

        with open(model_ppf_path, "wb") as fh:
            pickle.dump((ppfs_model, pairs_model), fh)
    else:
        print("Loading model features from", model_ppf_path)
        with open(model_ppf_path, "rb") as fh:
            ppfs_model, pairs_model = pickle.load(fh)

    # 2. choose reference points in scene, compute their ppfs
    scenebase, _ = os.path.splitext(args.scene)
    scene_ppf_path = f"{scenebase}.ppf"
    if not os.path.exists(scene_ppf_path):
        print("Computing scene ppfs features")
        t_start = time.perf_counter()
        ppfs_scene, pairs_scene = compute_ppf(scene, angle_step, dist_step, 1 / 5)
        t_end = time.perf_counter()
        print(f"Computing ppfs for the scene took {t_end - t_start:.2f}s")

        with open(scene_ppf_path, "wb") as fh:
            pickle.dump((ppfs_scene, pairs_scene), fh)
    else:
        print("Loading scene features from", scene_ppf_path)
        with open(scene_ppf_path, "rb") as fh:
            ppfs_scene, pairs_scene = pickle.load(fh)

    homog = lambda x: [*x, 1]

    # 3. go through scene ppfs, look up in table if we find model ppf
    skipped_features = 0
    model_alphas = {}
    # discretization for the alpha rotation
    alpha_num = 30
    assert 360 % alpha_num == 0
    alpha_step = np.radians(360 / alpha_num)

    poses_path = f"{modelbase}_{scenebase}.poses"
    if not os.path.exists(poses_path):
        print("Couldn't find cached poses. Computing anew.")
        idx_scene = 0
        poses = []

        print("Num reference verts", len(pairs_scene))
        for sA in pairs_scene:
            print(f"{len(pairs_scene[sA])} paired verts for ref {sA}", " " * 20)

            # one accumulator per reference vert
            accumulator = np.zeros((len(model.vertices), alpha_num))
            for sB in pairs_scene[sA]:
                if sA == sB:
                    continue

                s_feature = pairs_scene[sA][sB]

                if s_feature not in ppfs_model:
                    skipped_features += 1
                    continue

                # print(s_feature, s_pairs)
                s_r = scene.vertices[sA]
                s_i = scene.vertices[sB]
                s_normal = scene.vertex_normals[sA]

                R_scene2glob = np.eye(4)
                R_scene2glob[:3, :3] = align_vectors(s_normal, [1, 0, 0])
                T_scene2glob = R_scene2glob @ tf.translation_matrix(-s_r)

                s_ig = (T_scene2glob @ homog(s_i))[:3]
                alpha_s = np.arccos(np.dot(s_ig, [0, 0, -1]) / np.linalg.norm(s_ig))

                print(
                    "Found",
                    len(ppfs_model[s_feature]),
                    "matching pairs in model",
                    " " * 20,
                    end="\r",
                )
                for m_pair in ppfs_model[s_feature]:
                    mA, mB = m_pair
                    m_r = model.vertices[mA]
                    m_i = model.vertices[mB]
                    m_normal = model.vertex_normals[mA]

                    R_model2glob = np.eye(4)
                    R_model2glob[:3, :3] = align_vectors(m_normal, [1, 0, 0])
                    T_model2glob = R_model2glob @ tf.translation_matrix(-m_r)

                    # TODO: precompute
                    if m_pair not in model_alphas:
                        m_ig = (T_model2glob @ homog(m_i))[:3]
                        alpha_m = np.arccos(
                            np.dot(m_ig, [0, 0, -1]) / np.linalg.norm(m_ig)
                        )
                        model_alphas[m_pair] = alpha_m

                    alpha_m = model_alphas[m_pair]
                    alpha = alpha_m - alpha_s
                    # print("Alpha", np.degrees(alpha), "model:", np.degrees(alpha_m), "scene:", np.degrees(alpha_s))

                    alpha_disc = int(alpha // alpha_step)
                    accumulator[mA, alpha_disc] += 1

            peak_cutoff = np.quantile(accumulator.reshape(-1), 0.99)
            idxs_peaks = np.argwhere(accumulator > peak_cutoff)

            # import matplotlib.pyplot as plt
            # plt.imshow(accumulator.T)
            # plt.show()

            # TODO: vectorize
            for best_mr, best_alpha in idxs_peaks:
                R_model2glob = np.eye(4)
                R_model2glob[:3, :3] = align_vectors(
                    model.vertex_normals[best_mr], [1, 0, 0]
                )
                T_model2glob = R_model2glob @ tf.translation_matrix(
                    -model.vertices[best_mr]
                )

                R_alpha = tf.rotation_matrix(
                    alpha_step * best_alpha, [1, 0, 0], [0, 0, 0]
                )
                # TODO: invert homog
                T_model2scene = np.linalg.inv(T_scene2glob) @ R_alpha @ T_model2glob
                poses.append((T_model2scene, best_mr, accumulator[best_mr, best_alpha]))
            # break

            idx_scene += 1
            if idx_scene > 50:
                break

        with open(poses_path, "wb") as fh:
            pickle.dump(poses, fh)
    else:
        print("Loading poses from", poses_path)
        with open(poses_path, "rb") as fh:
            poses = pickle.load(fh)

    print("Skipped", skipped_features, "features")
    # TODO: take more peaks, cluster poses

    best_score = np.max(list(zip(*poses))[2])
    print("Best score", best_score)

    poses.sort(key=lambda thing: thing[2], reverse=True)

    # maybe easiest to sample pts on object and cluster on location
    # of these pts

    # or: first cluster locations
    # then split clusters by clustering them by orientation
    pose_clusters = cluster_poses(poses, dist_max=model.scale * 0.1)
    poses = pose_clusters

    cam_trafo = None
    for T_model2scene, m_r, score in poses:
        model_vis = model.copy()
        model_vis.apply_transform(T_model2scene)
        model_ref = trimesh.PointCloud([model_vis.vertices[m_r]])
        vis = trimesh.Scene([model_vis, scene, model_ref])
        if cam_trafo is not None:
            vis.camera_transform = cam_trafo
        vis.show()
        cam_trafo = vis.camera_transform


def align_vectors(a, b):
    """
    Computes rotation matrix that rotates a into b
    """

    v = np.cross(a, b)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def compute_feature(vertA, vertB, normA, normB, angle_step=None, dist_step=None):
    """
    angle_step: Angle step in radians
    """

    diffvec = vertA - vertB

    F1 = np.linalg.norm(diffvec)
    F2, F3, F4 = trimesh.geometry.vector_angle(
        [(-diffvec / F1, normA), (diffvec / F1, normB), (normA, normB)]
    )

    if dist_step and angle_step:
        prev = (F1, F2, F3, F4)
        F1 //= dist_step
        F2 //= angle_step
        F3 //= angle_step
        F4 //= angle_step
        try:
            res = tuple(int(x) for x in [F1, F2, F3, F4])
        except ValueError as e:
            print(e, "F1", F1, "F2", F2, "F3", F3, "F4", F4)
            print("prev", prev)
            return None
        return res

    return (F1, F2, F3, F4)


def compute_ppf(
    mesh: trimesh.Trimesh, angle_step: float, dist_step: float, fraction_pts=1.0
):
    table = defaultdict(list)
    ref2feature = defaultdict(dict)

    idxs = range(len(mesh.vertices))

    num_pts = int(fraction_pts * len(mesh.vertices))
    idxsA = random.sample(idxs, k=num_pts)

    num = 0
    for ivertA in idxsA:
        for ivertB, vertB in enumerate(mesh.vertices):
            if ivertA == ivertB:
                continue
            num += 1
            if num < 500 or num % 10000 == 0:
                print(num, end="\r")

            vertA = mesh.vertices[ivertA]
            normA = mesh.vertex_normals[ivertA]
            normB = mesh.vertex_normals[ivertB]

            F = compute_feature(
                vertA, vertB, normA, normB, angle_step=angle_step, dist_step=dist_step
            )

            if F is None:
                continue

            table[F].append((ivertA, ivertB))
            ref2feature[ivertA][ivertB] = F
    return table, ref2feature


def cluster_poses(poses, dist_max=0.5, rot_max_deg=10):
    rots = [T_m2s[:3, :3] for T_m2s, _, _ in poses]
    locs = [T_m2s[:3, 3] for T_m2s, _, _ in poses]
    scores = np.array([score for _, _, score in poses])

    # 1) cluster by location
    dist_dists = pdist(locs)
    dist_dendro = linkage(dist_dists, "complete")
    dist_clusters = fcluster(dist_dendro, dist_max, criterion="distance")

    # 2) split each cluster, by clustering the contents by orientation
    # rot_dists = None # TODO
    # rot_dendor = linkage(rot_dists, "complete")
    # rot_clusters = fcluster(rot_dendor, np.radians(rot_max_deg), criterion="distance")

    # TODO figure out how to combine both
    # final clusters = two pts clustered together in distance AND rotation
    # i.e. for each pt, look at cluster buddies in dist and rot
    # if two pts are buddies in either, they form a cluster
    # if either of them is already in a final cluster, the other is taken into that one

    cluster_scores = np.zeros(np.max(dist_clusters) + 1)
    for score, cluster in zip(scores, dist_clusters):
        cluster_scores[cluster] += score

    best_cluster = np.argmax(cluster_scores)
    print(
        "Best cluster",
        best_cluster,
        cluster_scores[best_cluster],
        np.count_nonzero(dist_clusters == best_cluster),
    )

    cluster_score_thresh = np.quantile(cluster_scores, 0.99)
    import matplotlib.pyplot as plt
    plt.hist(cluster_scores, histtype="stepfilled", bins=100)
    plt.title("Cluster Scores Histogram")
    plt.show()

    out_ts = defaultdict(list)
    out_Rs = defaultdict(list)
    num_skipped_clusters = 0
    for pose_idx, cluster_idx in enumerate(dist_clusters):
        if cluster_scores[cluster_idx] < cluster_score_thresh:
            num_skipped_clusters += 1
            continue

        out_ts[cluster_idx].append(locs[pose_idx])
        out_Rs[cluster_idx].append(rots[pose_idx])

    print("Skipped", num_skipped_clusters, "clusters because score too low")

    out_poses = []
    for cluster_idx in out_ts:
        ts = out_ts[cluster_idx]
        Rs = out_Rs[cluster_idx]

        avg_t = np.mean(ts, axis=0)
        avg_R = average_rotations(Rs)
        out_T = np.eye(4)
        out_T[:3, :3] = avg_R[:3, :3]
        out_T[:3, 3] = avg_t
        out_poses.append((out_T, 0, None))
    
    print("Returning", len(out_poses), "clustered and averaged poses")

    return out_poses


def average_rotations(rotations):
    Q = np.zeros((4, len(rotations)))

    for i, rot in enumerate(rotations):
        quat = tf.quaternion_from_matrix(rot)
        Q[:, i] = quat

    _, v = np.linalg.eigh(Q @ Q.T)
    quat_avg = v[:, -1]

    return tf.quaternion_matrix(quat_avg)


# trimesh todo
# - point cloud normals
# - scipy not in dependencies

# Validation

vertA = np.array([0, 0, 0])
vertB = np.array([1, 0, 0])
normA = np.array([0, 1, 0])
normB = np.array([0, 1, 0])
F1, F2, F3, F4 = compute_feature(vertA, vertB, normA, normB)
assert np.isclose(F1, 1)
assert np.isclose(F2, F3)
assert np.isclose(F2, np.radians(90)), f"F2={np.degrees(F2)}"
assert np.isclose(F4, 0)

vertA = np.array([0, 0, 0])
vertB = np.array([1, 0, 0])
normA = np.array([1, 1, 0])
normA = normA / 2 * 1.414
normB = np.array([-1, 1, 0])
normB = normB / 2 * 1.414
F1, F2, F3, F4 = compute_feature(vertA, vertB, normA, normB)
assert np.isclose(F1, 1)
assert np.isclose(F2, F3)
assert np.isclose(F2, np.radians(45), rtol=1e-3), f"F2={np.degrees(F2)}"
assert np.isclose(F4, np.radians(90), rtol=1e-3)

if __name__ == "__main__":
    main()
