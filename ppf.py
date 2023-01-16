#!/usr/bin/env python3
"""
This script computes the point-pair features of a given
model and tries to find the model in a given scene.

Note: Currently, trimesh doesn't support pointcloud with normals. To combat this, you need to
      reconstruct some surface between the points (e.g. ball pivoting)
"""

import os
import random
import time
import argparse
import pickle
import itertools
import math
from collections import defaultdict

import trimesh
import trimesh.transformations as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

VIS = True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="Path to the model pointcloud")
    parser.add_argument("scene", help="Path to the scene pointcloud")
    parser.add_argument(
        "--scene-pts-fraction",
        default=0.2,
        type=float,
        help="Fraction of scene points to use as reference",
    )
    parser.add_argument(
        "--scene-pts-abs",
        type=float,
        help="Max absolute number of scene pts. Between fraction and abs, smaller value is taken.",
    )
    parser.add_argument(
        "--cluster-max-angle",
        type=float,
        default=30,
        help="Maximal angle between poses after which they don't belong to same cluster anymore. [degrees]"
    )
    args = parser.parse_args()

    model = trimesh.load(args.model)
    scene = trimesh.load(args.scene)

    assert hasattr(
        model, "vertex_normals"
    ), "model has no vertex normals (might be trimesh loading bug)"
    assert hasattr(
        scene, "vertex_normals"
    ), "scene has no vertex normals (might be trimesh loading bug)"

    # if no abs number given, use all scene verts
    args.scene_pts_abs = args.scene_pts_abs or len(scene.vertices)

    print("Model has", len(model.vertices), "vertices")
    print("Scene has", len(scene.vertices), "vertices")

    model.visual.vertex_colors = [[255, 0, 0, 255] for _ in model.vertices]
    scene.visual.vertex_colors = [[150, 200, 150, 240] for _ in scene.vertices]

    if VIS:
        vis = trimesh.Scene([model, scene])
        vis.show()

    # 1. compute ppfs of all vertex pairs in model, store in hash table
    angle_num = 30
    angle_step = np.radians(360 / angle_num)
    dist_step = 0.05 * model.scale

    modelbase, _ = os.path.splitext(args.model)
    model_ppf_path = f"{modelbase}.model.ppf"
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
    scene_ppf_path = f"{scenebase}.scene.ppf"
    if not os.path.exists(scene_ppf_path):
        print("Computing scene ppfs features")
        t_start = time.perf_counter()
        ppfs_scene, pairs_scene = compute_ppf(
            scene,
            angle_step,
            dist_step,
            args.scene_pts_fraction,
            args.scene_pts_abs,
            model.scale/2,
        )
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

    scene_tree = KDTree(scene.vertices)

    poses_path = os.path.join(
        os.path.dirname(modelbase),
        f"{os.path.basename(modelbase)}_{os.path.basename(scenebase)}.poses",
    )
    if not os.path.exists(poses_path):
        print("Couldn't find cached poses. Computing anew.")
        poses = []
        # one accumulator per reference vert
        accumulator = np.zeros((len(model.vertices), alpha_num))

        print("Num reference verts", len(pairs_scene))
        for idx_ref, sA in enumerate(pairs_scene):
            print(f"{idx_ref+1}/{len(pairs_scene)}: {len(pairs_scene[sA])} paired verts for ref {sA}", " " * 20)

            # one accumulator per reference vert, we set it to zero instead of re-initializing
            accumulator[...] = 0

            s_r = scene.vertices[sA]
            s_normal = scene.vertex_normals[sA]

            R_scene2glob = np.eye(4)
            R_scene2glob[:3, :3] = align_vectors(s_normal, [1, 0, 0])
            T_scene2glob = R_scene2glob @ tf.translation_matrix(-s_r)

            for sB in pairs_scene[sA]:
                if sA == sB:
                    continue

                s_feature = pairs_scene[sA][sB]

                if s_feature not in ppfs_model:
                    skipped_features += 1
                    continue

                s_i = scene.vertices[sB]
                s_ig = (T_scene2glob @ homog(s_i))[:3]
                s_ig /= np.linalg.norm(s_ig)
                alpha_s = vector_angle_signed_x(s_ig, [0, 0, -1])

                # print(
                #     "Found",
                #     len(ppfs_model[s_feature]),
                #     "matching pairs in model",
                #     " " * 20,
                #     end="\r",
                # )
                for m_pair in ppfs_model[s_feature]:
                    # TODO: precompute
                    if m_pair not in model_alphas:
                        mA, mB = m_pair
                        m_r = model.vertices[mA]
                        m_i = model.vertices[mB]
                        m_normal = model.vertex_normals[mA]

                        R_model2glob = np.eye(4)
                        R_model2glob[:3, :3] = align_vectors(m_normal, [1, 0, 0])
                        T_model2glob = R_model2glob @ tf.translation_matrix(-m_r)
                    
                        m_ig = (T_model2glob @ homog(m_i))[:3]
                        m_ig /= np.linalg.norm(m_ig)
                        alpha_m = vector_angle_signed_x(m_ig, [0, 0, -1])
                        model_alphas[m_pair] = alpha_m

                    alpha_m = model_alphas[m_pair]
                    alpha = alpha_m - alpha_s

                    alpha_disc = int(alpha // alpha_step)
                    # print("Alpha", np.degrees(alpha), "model:", np.degrees(alpha_m), "scene:", np.degrees(alpha_s), "alpha disc:", alpha_disc)
                    accumulator[mA, alpha_disc] += 1

            # peak_cutoff = np.quantile(accumulator.reshape(-1), 0.99)
            peak_cutoff = np.max(accumulator) * 0.9
            idxs_peaks = np.argwhere(accumulator > peak_cutoff)

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

        with open(poses_path, "wb") as fh:
            pickle.dump(poses, fh)
    else:
        print("Loading poses from", poses_path)
        with open(poses_path, "rb") as fh:
            poses = pickle.load(fh)

    print(f"Got {len(poses)} poses after matching")
    print("Skipped", skipped_features, "features")

    best_score = np.max(list(zip(*poses))[2])
    print("Best score", best_score)

    poses.sort(key=lambda thing: thing[2], reverse=True)

    bad_score_thresh = np.quantile(list(zip(*poses))[2], 0.1)
    poses = list(filter(lambda thing: thing[2] > bad_score_thresh, poses))
    print(f"Got {len(poses)} poses after filtering (>{bad_score_thresh})")

    t_cluster_start = time.perf_counter()
    pose_clusters = cluster_poses(poses, dist_max=model.scale * 0.5, rot_max_deg=args.cluster_max_angle)
    poses = pose_clusters
    t_cluster_end = time.perf_counter()
    print(f"Clustering took {t_cluster_end - t_cluster_start:.1f}s")

    # cam_trafo = None
    # for T_model2scene, m_r, score in poses:
    #     model_vis = model.copy()
    #     model_vis.apply_transform(T_model2scene)
    #     # model_ref = trimesh.PointCloud([model_vis.vertices[m_r]])
    #     scene_ref = trimesh.PointCloud(
    #         [scene.vertices[idx] for idx in list(pairs_scene.keys())[:100]]
    #     )
    #     vis = trimesh.Scene([model_vis, scene, scene_ref])
    #     if cam_trafo is not None:
    #         vis.camera_transform = cam_trafo
    #     vis.show()
    #     cam_trafo = vis.camera_transform
    scene_refs = trimesh.PointCloud(
        [scene.vertices[idx] for idx in list(pairs_scene.keys())[:100]]
    )
    vis = trimesh.Scene([scene, scene_refs])
    for T_model2scene, m_r, score in poses:
        model_vis = model.copy()
        model_vis.apply_transform(T_model2scene)
        vis.add_geometry(model_vis)
    vis.show()


def vector_angle_signed_x(vecA, vecB):
    assert np.isclose(np.linalg.norm(vecA), 1)
    assert np.isclose(np.linalg.norm(vecB), 1)
    return np.arctan2(np.dot(np.cross(vecA, vecB), [1, 0, 0]), np.dot(vecA, vecB))


assert np.isclose(vector_angle_signed_x([0, 1, 0], [0, 0, 1]), np.pi / 2)
assert np.isclose(vector_angle_signed_x([0, 0, 1], [0, 1, 0]), -np.pi / 2)


def align_vectors(a, b):
    """
    Computes rotation matrix that rotates a into b
    """

    v = np.cross(a, b)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    R = np.eye(3) + Vmat + (Vmat.dot(Vmat) * h)
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
    mesh: trimesh.Trimesh,
    angle_step: float,
    dist_step: float,
    ref_fraction=1.0,
    ref_abs=None,
    max_dist=np.inf,
):
    table = defaultdict(list)
    ref2feature = defaultdict(dict)

    idxs = range(len(mesh.vertices))

    num_pts = int(ref_fraction * len(mesh.vertices))
    num_pts = min(num_pts, ref_abs or len(mesh.vertices))
    idxsA = random.sample(idxs, k=num_pts)
    print(
        f"Going for {num_pts} reference pts ({num_pts/len(mesh.vertices) * 100:.0f}%)"
    )

    # without KDTREE: Computing ppfs for the scene took 2134.7s
    # with KDTree:                                       814.9s
    vert_tree = KDTree(mesh.vertices)

    num = 0
    for ivertA in idxsA:
        vertA = mesh.vertices[ivertA]

        for ivertB in vert_tree.query_ball_point(vertA, max_dist):
            if ivertA == ivertB:
                continue

            normA = mesh.vertex_normals[ivertA]
            normB = mesh.vertex_normals[ivertB]
            vertB = mesh.vertices[ivertB]

            F = compute_feature(
                vertA, vertB, normA, normB, angle_step=angle_step, dist_step=dist_step
            )

            if F is None:
                continue

            num += 1
            if num < 500 or num % 10000 == 0:
                print("pair", num, end="\r")

            table[F].append((ivertA, ivertB))
            ref2feature[ivertA][ivertB] = F
    return table, ref2feature


def bernstein(vala, valb):
    """thanks to special sauce https://stackoverflow.com/a/34006336/10059727"""
    h = 1009
    h = h * 9176 + vala
    h = h * 9176 + valb
    return h


def rotation_between(rotmatA, rotmatB):
    """thanks to JonasVautherin https://math.stackexchange.com/q/2113634"""
    assert rotmatA.shape == (3, 3)
    assert rotmatB.shape == (3, 3)

    r_oa_t = np.transpose(rotmatA)
    r_ab = r_oa_t @ rotmatB
    return np.arccos((np.trace(r_ab) - 1) / 2)


matA = tf.rotation_matrix(np.pi / 4, [1, 0, 0])[:3, :3]
matB = tf.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
assert np.isclose(rotation_between(matA, matB), np.pi / 4), rotation_between(matA, matB)


def pdist_rot(rot_mats):
    """Returns a distance matrix like pdist, but in rotation space"""

    # we save distance in degrees and use uint8 for smaller memory footprint
    dists = np.zeros((len(rot_mats), len(rot_mats)), dtype=np.uint8)
    print("dists shape", dists.shape)

    # XXX instead of distance matrix, compute condensed form to save memory
    mat_idxs = np.arange(len(rot_mats))
    for idxA, idxB in itertools.combinations(mat_idxs, 2):
        dist = np.degrees(rotation_between(rot_mats[idxA], rot_mats[idxB])).astype(
            np.uint8
        )
        dists[idxA, idxB] = dist
        dists[idxB, idxA] = dist

    return squareform(dists).astype(float)


def cluster_poses(poses, dist_max=0.5, rot_max_deg=10):
    rots = np.array([T_m2s[:3, :3] for T_m2s, _, _ in poses])
    locs = np.array([T_m2s[:3, 3] for T_m2s, _, _ in poses])
    scores = np.array([score for _, _, score in poses])

    # 1) cluster by location
    dist_dists = pdist(locs)
    dist_dendro = linkage(dist_dists, "complete")
    dist_clusters = fcluster(dist_dendro, dist_max, criterion="distance")

    # 2) cluster by rotations
    # XXX optimize, we can make more smaller cluster problems, since
    # a cluster across distant poses doesn't make sense
    rot_dists = pdist_rot(rots)
    rot_dendor = linkage(rot_dists, "complete")
    rot_clusters = fcluster(rot_dendor, rot_max_deg, criterion="distance")

    # Combine the two clusterings, by creating new clusters
    # if two poses are in same cluster in loc and rot, they will be in new
    # common cluster (hash of both cluster ids)
    pose_clusters = bernstein(dist_clusters, rot_clusters)

    # remap the ludicrous hash values to range 0..num
    _, pose_clusters = np.unique(pose_clusters, return_inverse=True)

    cluster_scores = np.zeros(np.max(pose_clusters) + 1)
    for pose_score, pose_cluster in zip(scores, pose_clusters):
        cluster_scores[pose_cluster] += pose_score

    best_cluster = np.argmax(cluster_scores)
    print(
        "Best cluster",
        best_cluster,
        cluster_scores[best_cluster],
        np.count_nonzero(pose_clusters == best_cluster),
    )

    import matplotlib.pyplot as plt
    plt.hist(cluster_scores, histtype="stepfilled", bins=100)
    plt.title("Cluster Scores Histogram")
    plt.show()

    out_ts = defaultdict(list)
    out_Rs = defaultdict(list)
    num_skipped_clusters = 0
    for pose_idx, cluster_idx in enumerate(pose_clusters):
        out_ts[cluster_idx].append(locs[pose_idx])
        out_Rs[cluster_idx].append(rots[pose_idx])

    sorted_clusters = np.argsort(cluster_scores)[::-1]

    for idx_cluster, top_cluster in enumerate(sorted_clusters[:10]):
        print(f"{idx_cluster} contains {len(out_ts[top_cluster])} poses")

    
    best_score = np.max(cluster_scores)
    best_rel_thresh = 0.5

    out_poses = []
    for cluster_idx in sorted_clusters:
        if cluster_scores[cluster_idx] < best_rel_thresh * best_score:
            continue

        print("cluster idx", cluster_idx, cluster_scores[cluster_idx])
        ts = out_ts[cluster_idx]
        Rs = out_Rs[cluster_idx]

        avg_t = np.mean(ts, axis=0)
        avg_R = average_rotations(Rs)
        out_T = np.eye(4)
        out_T[:3, :3] = avg_R[:3, :3]
        out_T[:3, 3] = avg_t
        out_poses.append((out_T, 0, cluster_scores[cluster_idx]))

    print("Returning", len(out_poses), "clustered and averaged poses")

    return out_poses


def average_rotations(rotations):
    """thanks to jonathan https://stackoverflow.com/a/27410865/10059727"""
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
