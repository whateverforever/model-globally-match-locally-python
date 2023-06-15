#!/usr/bin/env python3
"""
This script computes the point-pair features of a given
model and tries to find the model in a given scene.

Note: Currently, trimesh doesn't support pointcloud with normals. To combat this, you need to
      reconstruct some surface between the points (e.g. ball pivoting)
"""

import random
import time
import argparse
import itertools
from collections import defaultdict

import trimesh
import pyglet
import trimesh.viewer
import trimesh.creation
import trimesh.transformations as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="Path to the model pointcloud")
    parser.add_argument("scene", help="Path to the scene pointcloud")
    parser.add_argument(
        "--model-vis",
        help="Path to model mesh to use for visualization instead of pointcloud",
    )
    parser.add_argument(
        "--scene-vis",
        help="Path to scene mesh to use for visualization instead of pointcloud"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use the c++ extension for speeeeeed"
    )
    parser.add_argument(
        "--scene-pts-fraction",
        default=0.2,
        type=float,
        help="Fraction of scene points to use as reference",
    )
    parser.add_argument(
        "--ppf-num-angles",
        default=30,
        type=int,
        help="Number of angle steps used to discretize feature angles.",
    )
    parser.add_argument(
        "--ppf-rel-dist-step",
        default=0.05,
        type=float,
        help="Discretization step of feature distances, relative to model diameter.",
    )
    parser.add_argument(
        "--alpha-num-angles",
        default=30,
        type=int,
        help="Number of angle steps used to discretize the rotation angle alpha.",
    )
    parser.add_argument(
        "--cluster-max-angle",
        type=float,
        default=30,
        help="Maximal angle between poses after which they don't belong to same cluster anymore. [degrees]",
    )
    parser.add_argument(
        "--cluster-max-rel-dist",
        type=float,
        default=0.25,
        help="Maximal distance for candidate poses to be clustered together. Relative to object diameter.",
    )
    args = parser.parse_args()

    if args.fast:
        import ppf_fast

        _compute_ppf = ppf_fast.compute_ppf
        _pdist_rot = ppf_fast.pdist_rot
        print("Using fast c++ mode")
    else:
        _compute_ppf = compute_ppf
        _pdist_rot = pdist_rot
        print("Using slow python mode")

    model = trimesh.load(args.model)
    scene = trimesh.load(args.scene)

    if hasattr(model, "vertex_normals") and hasattr(scene, "vertex_normals"):
        _model_orig = model.copy()
        _scene_vis = scene.copy()
    else:
        try:
            print(
                "Your trimesh version still can't load pointclouds with normals. Trying open3d..."
            )
            import open3d as o3d

            model = o3d.io.read_point_cloud(args.model)
            scene = o3d.io.read_point_cloud(args.scene)

            # model.estimate_normals()
            # scene.estimate_normals()

            model = trimesh.Trimesh(
                np.asarray(model.points), vertex_normals=np.asarray(model.normals)
            )
            scene = trimesh.Trimesh(
                np.asarray(scene.points), vertex_normals=np.asarray(scene.normals)
            )

            _model_orig = trimesh.PointCloud(vertices=model.vertices)
            _scene_vis = trimesh.PointCloud(vertices=scene.vertices)
        except ImportError as e:
            print("Fallback to open3d failed", e)
            quit()

    model_normals = (
        model.vertex_normals / np.linalg.norm(model.vertex_normals, axis=1)[:, None]
    )
    scene_normals = (
        scene.vertex_normals / np.linalg.norm(scene.vertex_normals, axis=1)[:, None]
    )

    model = trimesh.Trimesh(model.vertices, vertex_normals=model_normals)
    scene = trimesh.Trimesh(scene.vertices, vertex_normals=scene_normals)

    if args.model_vis:
        _model_vis = trimesh.load(args.model_vis)
    if args.scene_vis:
        _scene_vis = trimesh.load(args.scene_vis)

    print("Model has", len(model.vertices), "vertices. scale=", model.scale)
    print("Scene has", len(scene.vertices), "vertices. scale=", scene.scale)

    # trimesh doesn't support .scale for point clouds
    model_diag = np.linalg.norm(
        np.max(model.vertices, axis=0) - np.min(model.vertices, axis=0)
    )
    print("modelscale", model_diag)

    _model_orig.visual.vertex_colors = (255, 0, 0, 255)
    _model_vis.visual.vertex_colors = (255, 0, 0, 255)

    vis = trimesh.Scene([_model_orig, _model_vis, _scene_vis])
    vis.add_geometry(_model_orig, geom_name="pre_clustering0")
    viewer = Viewer(vis)

    ## 1. compute ppfs of all vertex pairs in model, store in hash table
    angle_step = float(np.radians(360 / args.ppf_num_angles))
    dist_step = args.ppf_rel_dist_step * model_diag
    print(f"angle_step={np.rad2deg(angle_step):.1f}d dist_step={dist_step:.2f}")

    print("Computing model ppfs features")
    t_start = time.perf_counter()
    ppfs_model, _, model_alphas = _compute_ppf(
        to_nanobind(model.vertices),
        to_nanobind(model.vertex_normals),
        angle_step,
        dist_step,
    )
    t_end = time.perf_counter()
    print(f"Computing ppfs for {len(model.vertices)} verts took {t_end - t_start:.2f}s")

    features = np.array(list(ppfs_model.keys()))
    print("Model features min/max", np.min(features, axis=0), np.max(features, axis=0))
    fig, axs = plt.subplots(ncols=4)
    axs[0].hist(features[:, 0])
    axs[1].hist(features[:, 1])
    axs[2].hist(features[:, 2])
    axs[3].hist(features[:, 3])
    fig.suptitle("Feature Component Histograms")
    plt.show()

    ## 2. choose reference points in scene, compute their ppfs
    t_start = time.perf_counter()
    _, pairs_scene, scene_alphas = _compute_ppf(
        to_nanobind(scene.vertices),
        to_nanobind(scene.vertex_normals),
        angle_step,
        dist_step,
        max_dist=model_diag,
        ref_fraction=args.scene_pts_fraction,
    )
    t_end = time.perf_counter()
    print(f"Computing all scene ppfs took {t_end - t_start:.1f}s")

    ## 3. go through scene ppfs, look up in table if we find model ppf
    skipped_features = 0

    # discretization for the alpha rotation
    alpha_step = np.radians(360 / args.alpha_num_angles)

    poses = []
    # accumulator we're going to reuse for each reference vert
    accumulator = np.zeros((len(model.vertices), args.alpha_num_angles))

    print("Num reference verts", len(pairs_scene))
    for idx_ref, sA in enumerate(pairs_scene):
        print(
            f"{idx_ref+1}/{len(pairs_scene)}: {len(pairs_scene[sA])} paired verts for ref {sA}",
            " " * 20,
            end="\r",
        )

        # one accumulator per reference vert, we set it to zero instead of re-initializing
        accumulator[...] = 0

        for sB in pairs_scene[sA]:
            if sA == sB:
                continue

            s_feature = pairs_scene[sA][sB]
            if s_feature not in ppfs_model:
                skipped_features += 1
                continue

            alpha_s = scene_alphas[(sA, sB)]

            for m_pair in ppfs_model[s_feature]:
                mA, mB = m_pair
                alpha_m = model_alphas[m_pair]
                alpha = alpha_m - alpha_s

                alpha_disc = int(alpha // alpha_step)
                accumulator[mA, alpha_disc] += 1
                accumulator[mA, (alpha_disc - 1) % args.alpha_num_angles] += 1
                accumulator[mA, (alpha_disc + 1) % args.alpha_num_angles] += 1

        # We kick out any model ref that doesn't have at least 10% of expected
        # matches
        where_lowscore = np.sum(accumulator, axis=1) < 0.1 * len(model.vertices) * 3
        accumulator[where_lowscore] = 0

        # NMS
        # For each model ref, we only keep alpha with highest score
        where_peak_neighbor = accumulator < np.max(accumulator, axis=1)[:, None]
        accumulator[where_peak_neighbor] = 0

        # We can take a high threshold here, because this step is per scene ref
        # If we're left with a single candidate per reference vert, so be it
        peak_cutoff = np.max(accumulator) * 0.95
        idxs_peaks = np.argwhere((accumulator > peak_cutoff) & (accumulator > 0))

        s_r = scene.vertices[sA]
        s_normal = scene.vertex_normals[sA]

        R_scene2glob = np.eye(4)
        R_scene2glob[:3, :3] = align_vectors(s_normal, [1, 0, 0])
        assert np.isclose(np.linalg.det(R_scene2glob), 1), np.linalg.det(R_scene2glob)

        T_scene2glob = R_scene2glob @ tf.translation_matrix(-s_r)
        assert np.isclose(np.linalg.det(T_scene2glob), 1), np.linalg.det(T_scene2glob)

        for best_mr, best_alpha in idxs_peaks:
            R_model2glob = np.eye(4)
            R_model2glob[:3, :3] = align_vectors(
                model.vertex_normals[best_mr], [1, 0, 0]
            )
            assert np.isclose(np.linalg.det(R_model2glob), 1), np.linalg.det(
                R_model2glob
            )

            T_model2glob = R_model2glob @ tf.translation_matrix(
                -model.vertices[best_mr]
            )
            assert np.isclose(np.linalg.det(T_model2glob), 1), np.linalg.det(
                T_model2glob
            )

            R_alpha = tf.rotation_matrix(alpha_step * best_alpha, [1, 0, 0], [0, 0, 0])
            # TODO: invert homog
            T_model2scene = np.linalg.inv(T_scene2glob) @ R_alpha @ T_model2glob
            poses.append((T_model2scene, accumulator[best_mr, best_alpha]))

            if False:
                tmp_scene = trimesh.Scene()
                tmp_scene.add_geometry(_scene_vis, transform=T_scene2glob)
                tmp_scene.add_geometry(_model_vis, transform=T_model2glob)
                tmp_scene.show()

    print(f"Got {len(poses)} poses after matching", " " * 20)
    print("Skipped", skipped_features, "scene pairs, not found in model")

    poses_orig = poses[:100].copy()
    t_cluster_start = time.perf_counter()
    poses = cluster_poses(
        poses,
        dist_max=model_diag * args.cluster_max_rel_dist,
        rot_max_deg=args.cluster_max_angle,
        pdist_rot=_pdist_rot,
        _scene_vis=_scene_vis,
        _model_vis=_model_vis
    )
    t_cluster_end = time.perf_counter()
    print(f"Clustering took {t_cluster_end - t_cluster_start:.1f}s")

    ## Visualize result
    scene_refs = trimesh.PointCloud(
        [scene.vertices[idx] for idx in list(pairs_scene.keys())]
    )
    vis = trimesh.Scene([_scene_vis, scene_refs])
    for idx, (T_model2scene, score) in enumerate(poses):
        vis.add_geometry(_model_orig, transform=T_model2scene, geom_name=f"match_pts_{idx}")
        vis.add_geometry(_model_vis, transform=T_model2scene, geom_name=f"match_{idx}")

        print("Score", score)
        print(np.around(T_model2scene, decimals=2))
        print()

    prior_model = _model_vis.copy()
    prior_model.visual.vertex_colors = (128, 128, 128)
    for T_model2scene, score in poses_orig:
        vis.add_geometry(
            prior_model, transform=T_model2scene, geom_name="pre_clustering"
        )
    Viewer(vis, line_settings={'point_size':10})


def to_nanobind(arr):
    """
    Workaround for current bug in nanobind: arrays need to be writable to be recognized
    https://github.com/wjakob/nanobind/issues/42
    """
    F_arr = np.asfortranarray(arr)
    F_arr.setflags(write=True)
    return F_arr


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

    assert np.isclose(np.linalg.norm(a), 1), np.linalg.norm(a)
    assert np.isclose(np.linalg.norm(b), 1), np.linalg.norm(b)

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


def homog(vec3):
    return [*vec3, 1]


def compute_ppf(
    vertices,
    normals,
    angle_step: float,
    dist_step: float,
    ref_fraction=1.0,
    ref_abs=None,
    max_dist=np.inf,
    alphas=False,
):
    table = defaultdict(list)
    ref2feature = defaultdict(dict)
    model_alphas = {}

    idxs = range(len(vertices))

    num_pts = int(ref_fraction * len(vertices))
    num_pts = min(num_pts, ref_abs or len(vertices))
    idxsA = random.sample(idxs, k=num_pts)
    print(f"Going for {num_pts} reference pts ({num_pts/len(vertices) * 100:.0f}%)")

    # without KDTREE: Computing ppfs for the scene took 2134.7s
    # with KDTree:                                       814.9s
    vert_tree = KDTree(vertices)

    num = 0
    for ivertA in idxsA:
        vertA = vertices[ivertA]

        for ivertB in vert_tree.query_ball_point(vertA, max_dist):
            if ivertA == ivertB:
                continue

            normA = normals[ivertA]
            normB = normals[ivertB]
            vertB = vertices[ivertB]

            F = compute_feature(
                vertA, vertB, normA, normB, angle_step=angle_step, dist_step=dist_step
            )

            if F is None:
                continue

            num += 1
            if num < 500 or num % 10000 == 0:
                print("pair", num, f"{num/(len(vertices)**2)*100:.0f}%", end="\r")

            table[F].append((ivertA, ivertB))
            ref2feature[ivertA][ivertB] = F

            # precompute the model angles
            if alphas:
                m_r = vertA
                m_i = vertB
                m_normal = normA

                R_model2glob = np.eye(4)
                R_model2glob[:3, :3] = align_vectors(m_normal, [1, 0, 0])
                T_model2glob = R_model2glob @ tf.translation_matrix(-m_r)

                m_ig = (T_model2glob @ homog(m_i))[:3]
                m_ig /= np.linalg.norm(m_ig)
                alpha_m = vector_angle_signed_x(m_ig, [0, 0, -1])
                model_alphas[(ivertA, ivertB)] = alpha_m

    return table, ref2feature, model_alphas


def rotation_between(rotmatA, rotmatB):
    """thanks to JonasVautherin https://math.stackexchange.com/q/2113634"""
    assert rotmatA.shape == (3, 3)
    assert rotmatB.shape == (3, 3)

    r_oa_t = np.transpose(rotmatA)
    r_ab = r_oa_t @ rotmatB
    val = (np.trace(r_ab) - 1) / 2
    return np.arccos(np.clip(val, -1, 1))


matA = tf.rotation_matrix(np.pi / 4, [1, 0, 0])[:3, :3]
matB = tf.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
assert np.isclose(rotation_between(matA, matB), np.pi / 4), rotation_between(matA, matB)


def pdist_rot(rot_mats):
    """Returns the condensed distance matrix like pdist, but in rotation space"""
    m = len(rot_mats)
    idx = lambda i, j: m * i + j - ((i + 2) * (i + 1)) // 2
    print("Index for last pair", idx(m, m) + 1)

    # we save distance in degrees and use uint8 for smaller memory footprint
    dists = np.zeros(idx(m, m) + 1, dtype=np.uint8)
    print("dists shape", dists.shape)

    mat_idxs = np.arange(m)
    # Note: combinations() doesn't give (i,i) pairs
    # Note: combinations() keeps original ascending index order
    for idxA, idxB in itertools.combinations(mat_idxs, 2):
        dist = np.degrees(rotation_between(rot_mats[idxA], rot_mats[idxB])).astype(
            np.uint8
        )
        dists[idx(idxA, idxB)] = dist

    return dists.astype(float)


def cluster_poses(poses, dist_max=0.5, rot_max_deg=10, pdist_rot=None, _scene_vis=None, _model_vis=None):
    rots = np.array([T_m2s[:3, :3] for T_m2s, _ in poses])
    locs = np.array([T_m2s[:3, 3] for T_m2s, _ in poses])
    pose_scores = np.array([score for _, score in poses])

    method = "centroid"

    # 1) cluster by location
    dist_dists = pdist(locs)
    dist_dendro = linkage(dist_dists, method)
    dist_clusters = fcluster(dist_dendro, dist_max, criterion="distance")

    # 2) cluster by rotations
    rot_dists = pdist_rot(np.asfortranarray(rots))
    rot_dendro = linkage(rot_dists, method)
    rot_clusters = fcluster(rot_dendro, rot_max_deg, criterion="distance")

    # Combine the two clusterings, by creating new clusters
    # if two poses are in same cluster in loc and rot, they will be in new
    # common cluster (hash of both cluster ids)
    pose_clusters = [
        int(''.join(f'{ord(char)}' for char in f"{dist},{rot}"))
        for dist, rot in zip(dist_clusters, rot_clusters)
    ]

    # remap the ludicrous hash values to range 0..num
    _, pose_clusters = np.unique(pose_clusters, return_inverse=True)

    cluster_score = np.zeros_like(pose_clusters)
    for pose_score, pose_cluster in zip(pose_scores, pose_clusters):
        cluster_score[pose_cluster] += pose_score

    cluster_ts = defaultdict(list)
    cluster_Rs = defaultdict(list)
    cluster_size = defaultdict(int)
    for pose_idx, cluster_idx in enumerate(pose_clusters):
        cluster_ts[cluster_idx].append(locs[pose_idx])
        cluster_Rs[cluster_idx].append(rots[pose_idx])
        cluster_size[cluster_idx] += 1

    print("Info about top-10 clusters:")
    for cluster_idx in np.argsort(cluster_score)[::-1][:10]:
        print(
            f"    cluster {cluster_idx} contains {cluster_size[cluster_idx]} poses, score={cluster_score[cluster_idx]}"
        )

    geomean = lambda x, y: np.sqrt(x * y)
    harmmean = lambda x, y: 2 / (1 / (x + 1e-8) + 1 / (y + 1e-8))

    cluster_geoscore = [
        geomean(cluster_size[cluster_idx], cluster_score[cluster_idx])
        for cluster_idx in pose_clusters
    ]
    cluster_harmscore = [
        harmmean(cluster_size[cluster_idx], cluster_score[cluster_idx])
        for cluster_idx in pose_clusters
    ]
    cluster_uscore = [cluster_score[cluster_idx] ** 2 for cluster_idx in pose_clusters]

    fig, axs = plt.subplots(ncols=3)
    axs[0].hist(cluster_score, histtype="stepfilled", bins=100)
    axs[0].set_yscale("log")
    axs[0].set_title("Cluster Scores Histogram (Y-Axis logged)")
    axs[1].hist(cluster_geoscore, histtype="stepfilled", bins=100)
    axs[1].set_title("Geometric score + number poses")
    axs[2].hist(cluster_harmscore, histtype="stepfilled", bins=100)
    axs[2].set_title("Harmonic score + number poses")

    # axs[2].hist(cluster_uscore, histtype="stepfilled", bins=100)
    # axs[2].set_title("Combined score with num cluster with that score (bin)")
    plt.show()

    # best_cluster_idx = np.argmax(cluster_harmscore)
    best_cluster_idx = np.argmax(cluster_score)
    best_geo_score = cluster_harmscore[best_cluster_idx]
    best_rel_thresh = 0.6

    print(f"Best score={best_geo_score}, min_needed={best_rel_thresh * best_geo_score}")
    print("Info about final clusters:")
    out_poses = []
    for cluster_idx in set(pose_clusters):
        ts, Rs, score, size = (
            cluster_ts[cluster_idx],
            cluster_Rs[cluster_idx],
            cluster_score[cluster_idx],
            cluster_size[cluster_idx],
        )
        # geo_score = cluster_harmscore[cluster_idx]
        # if geo_score < best_rel_thresh * best_geo_score:
        geo_score = score
        if score < best_rel_thresh * cluster_score[best_cluster_idx]:
            continue

        print(
            "    cluster",
            cluster_idx,
            "plain score",
            cluster_score[cluster_idx],
            "geoscore",
            geo_score,
        )

        #sc = trimesh.Scene([_scene_vis])
        #model = _model_vis.copy()
        #model.visual.vertex_colors = (255, 255, 0, 128)
        for R, t in zip(Rs, ts):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            # sc.add_geometry(model, transform=T)

        out_T = average_rotations(Rs)
        out_T[:3, 3] = np.mean(ts, axis=0)
        #sc.add_geometry(_model_vis, transform=out_T)
        #sc.show()
        assert np.isclose(np.linalg.det(out_T), 1), np.linalg.det(out_T)
        out_poses.append((out_T, geo_score))

    # XXX todo: after clustering, cluster again on the surviving parts,
    # this time ignoring orientation, in order to merge symmetric objs

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


def average_rotations_so3(rotmats, n_steps=10):
    """
    Better average orientation by iteratively moving to the
    mean orientation. Gives better results than the quaternion
    method, when tested against global optimization using genetic algo.
    """

    from scipy.spatial.transform import Rotation

    mat2vec = lambda mat: Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    vec2mat = lambda vec: Rotation.from_rotvec(vec).as_matrix()

    rotvecs = np.array([mat2vec(mat) for mat in rotmats])
    vec = [0, 0, 0]

    for _ in range(n_steps):
        diffvecs = rotvecs - vec
        vec += np.mean(diffvecs, axis=0)

    out = np.eye(4)
    out[:3, :3] = vec2mat(vec)
    return out


class Viewer(trimesh.viewer.SceneViewer):
    TOGGLE_PRE_CLUSTER = ord("i")
    TOGGLE_MATCHES = ord("m")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, start_loop=False)

        self.pre_cluster_visib = True
        self.matches_visib = True

        print("### Debug Viewer")
        print(
            "### To show/hide pre-cluster poses, press", chr(Viewer.TOGGLE_PRE_CLUSTER)
        )
        print("### To show/hide matches, press", chr(Viewer.TOGGLE_MATCHES))
        pyglet.app.run()

    def on_key_press(self, symbol, modifiers):
        if symbol == Viewer.TOGGLE_PRE_CLUSTER:
            geoms = [n for n in self.scene.graph.nodes if n.startswith("pre_cluster")]
            for nodename in geoms:
                if self.pre_cluster_visib:
                    self.hide_geometry(nodename)
                else:
                    self.unhide_geometry(nodename)
            self.pre_cluster_visib = not self.pre_cluster_visib

        elif symbol == Viewer.TOGGLE_MATCHES:
            geoms = [n for n in self.scene.graph.nodes if n.startswith("match")]
            for nodename in geoms:
                if self.matches_visib:
                    self.hide_geometry(nodename)
                else:
                    self.unhide_geometry(nodename)
            self.matches_visib = not self.matches_visib

        elif symbol == ord("q"):
            super().on_key_press(symbol, modifiers)


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
