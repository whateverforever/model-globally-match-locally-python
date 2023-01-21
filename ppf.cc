#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/tensor.h>

#include <Eigen/Dense>

#include <iostream>
#include <map>
#include <tuple>

namespace nb = nanobind;
using namespace nb::literals;

using Feature = std::tuple<int, int, int, int>;
using Pair2Feature = std::map<std::pair<int, int>, Feature>;
using Vec3 = Eigen::Vector3d;
using Vecs3 = Eigen::MatrixX3d;

/*
 *  Computes the angle between two vectors originating from the same location
 */
double vectorAngle(const Vec3 &vecA, const Vec3 &vecB) {
  const Vec3 normedA = vecA.normalized();
  const Vec3 normedB = vecB.normalized();

  return std::acos(normedA.dot(normedB));
}

/**
 * Computes the Point-Pair-Feature for two given points and their normals.
 */
Feature computeFeature(const Vec3 &vecA, const Vec3 &vecB, const Vec3 &normA,
                       const Vec3 &normB, double step_rad, double step_dist) {
  const Vec3 diffvec = vecA - vecB;
  double F1 = diffvec.norm();
  double F2 = vectorAngle(-diffvec / F1, normA);
  double F3 = vectorAngle(diffvec / F1, normB);
  double F4 = vectorAngle(normA, normB);

  F1 = std::floor(F1 / step_dist);
  F2 = std::floor(F2 / step_rad);
  F3 = std::floor(F3 / step_rad);
  F4 = std::floor(F4 / step_rad);

  return {F1, F2, F3, F4};
}

double vectorAngleSignedX(const Vec3 &vecA, const Vec3 &vecB) {
  assert(std::abs(vecA.norm() - 1) <= 1e-3);
  assert(std::abs(vecB.norm() - 1) <= 1e-3);
  return std::atan2(vecA.cross(vecB).dot(Vec3(1, 0, 0)), vecA.dot(vecB));
}

Eigen::Matrix3d alignVectors(const Vec3 &a, const Vec3 &b) {
  const Vec3 v = a.cross(b);
  const double c = a.dot(b);
  const double h = 1 / (1 + c);

  const double v1 = v(0);
  const double v2 = v(1);
  const double v3 = v(2);

  Eigen::Matrix3d Vmat;
  Vmat << 0, -v3, v2, v3, 0, -v1, -v2, v1, 0;

  return Eigen::Matrix3d::Identity() + Vmat + Vmat * Vmat * h;
}

Pair2Feature computePPF(const Vecs3 &verts, const Vecs3 &normals,
                        double step_rad, double step_dist,
                        bool alphas = false) {
  Pair2Feature ref2feature;

  // std::cout << "verts" << verts << "\n";
  // std::cout << "normals" << normals << "\n";

  const int num_verts = verts.rows();

  for (int i = 0; i < num_verts; i++) {
    const auto &vertA = verts.row(i);
    const auto &normA = normals.row(i);

    for (int j = 0; j < num_verts; j++) {
      const auto &vertB = verts.row(j);
      const auto &normB = normals.row(j);

      if (i == j)
        continue;

      const auto F =
          computeFeature(vertA, vertB, normA, normB, step_rad, step_dist);

      if (std::get<0>(F) == 0)
        continue;

      ref2feature[{i, j}] = F;

      if (alphas) {
        const auto &m_r = vertA;
        const auto &m_i = vertB;
        const auto &m_normal = normA;

        Eigen::Matrix3d R_model2glob = Eigen::Matrix3d::Identity();
        // R_model2glob.block<3, 3>(0, 0) = alignVectors(m_normal, Vec3(1, 0,
        // 0));
      }
      // R_model2glob = np.eye(4)
      // R_model2glob[:3, :3] = align_vectors(m_normal, [1, 0, 0])
      // T_model2glob = R_model2glob @ tf.translation_matrix(-m_r)

      // m_ig = (T_model2glob @ homog(m_i))[:3]
      // m_ig /= np.linalg.norm(m_ig)
      // alpha_m = vector_angle_signed_x(m_ig, [0, 0, -1])
      // model_alphas[(ivertA, ivertB)] = alpha_m
    }
  }

  return ref2feature;
}

// Bindings

using nbMatX3 = nb::tensor<double, nb::shape<nb::any, 3>, nb::f_contig>;
using nbVec3 = nb::tensor<double, nb::shape<3>, nb::f_contig>;

const auto toMatX3 = [](const nbMatX3 &mat) {
  return Eigen::Map<const Vecs3>(mat.data(), mat.shape(0), mat.shape(1));
};

const auto toVec3 = [](const nbVec3 &vec) {
  return Eigen::Map<const Vec3>(vec.data(), 1, 3);
};

NB_MODULE(ppf_fast, m) {
  m.def("compute_ppf", [](const nbMatX3 &verts, const nbMatX3 &normals,
                          double step_rad, double step_dist) {
    return computePPF(toMatX3(verts), toMatX3(normals), step_rad, step_dist);
  });

  m.def("vector_angle", [](const nbVec3 &vecA, const nbVec3 &vecB) {
    return vectorAngle(toVec3(vecA), toVec3(vecB));
  });

  m.def("vector_angle_signed_x", [](const nbVec3 &vecA, const nbVec3 &vecB) {
    return vectorAngleSignedX(toVec3(vecA), toVec3(vecB));
  });

  m.def("align_vectors", [&](const nbVec3 &vecA, const nbVec3 &vecB) {
    size_t shape[2] = {3, 3};
    auto R = alignVectors(toVec3(vecA), toVec3(vecB));

    // Ugly construction to preserve lifetime while python uses it
    // not sure how to make this nicer
    double *data = new double[9];
    std::memcpy(data, R.data(), sizeof(data[0]) * 9);
    nb::capsule owner(data, [](void *p) noexcept { delete[](double *) p; });

    // Since eigen is column major, we need to jump three entries to get one
    // place to the right. And one to jump down.
    int64_t strides[2] = {1, 3};

    return nb::tensor<nb::numpy, double, nb::shape<3, 3>, nb::f_contig>(
        data, 2, shape, owner, strides);
  });

  m.def("compute_feature",
        [](const nbVec3 &vecA, const nbVec3 &vecB, const nbVec3 &normA,
           const nbVec3 &normB, double step_rad, double step_dist) {
          return computeFeature(toVec3(vecA), toVec3(vecB), toVec3(normA),
                                toVec3(normB), step_rad, step_dist);
        });
}
