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

Pair2Feature computePPF(const Vecs3 &verts, const Vecs3 &normals,
                        double step_rad, double step_dist) {
  Pair2Feature ref2feature;

  std::cout << "verts" << verts << "\n";
  std::cout << "normals" << normals << "\n";

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

const auto toVec3f = [](const nbVec3 &vec) {
  return Eigen::Map<const Vec3>(vec.data(), 1, 3);
};

NB_MODULE(ppf_fast, m) {
  m.def("compute_ppf", [](const nbMatX3 &verts, const nbMatX3 &normals,
                          double step_rad, double step_dist) {
    return computePPF(toMatX3(verts), toMatX3(normals), step_rad, step_dist);
  });

  m.def("vector_angle", [](const nbVec3 &vecA, const nbVec3 &vecB) {
    return vectorAngle(toVec3f(vecA), toVec3f(vecB));
  });

  m.def("compute_feature",
        [](const nbVec3 &vecA, const nbVec3 &vecB, const nbVec3 &normA,
           const nbVec3 &normB, double step_rad, double step_dist) {
          return computeFeature(toVec3f(vecA), toVec3f(vecB), toVec3f(normA),
                                toVec3f(normB), step_rad, step_dist);
        });
}

// correct order
// Eigen::Map<const Vecs3, Eigen::Unaligned, Eigen::Stride<1, 3>>(verts.data(),
// verts.shape(0), verts.shape(1)),