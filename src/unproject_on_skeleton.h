#ifndef IGL_UNPROJECT_ON_SKELETON_H
#define IGL_UNPROJECT_ON_SKELETON_H

#include <Eigen/Dense>

  // Given a screen space point (u,v) and the current projection matrix (e.g.
  // gl_proj * gl_modelview) and viewport, _unproject_ the point into the scene
  // so that it lies on given line (origin and dir) and projects as closely as
  // possible to the given screen space point.
  //
  // Inputs:
  //   UV  2-long uv-coordinates of screen space point
  //   M  4 by 4 projection matrix
  //   VP  4-long viewport: (corner_u, corner_v, width, height)
  //   origin  point on line
  //   dir  vector parallel to line
  // Output:
  //   t  line parameter so that closest poin on line to viewer ray through UV
  //     lies at origin+t*dir
  template <
    typename DerivedUV,
    typename DerivedM,
    typename DerivedVP>
    bool unproject_on_skeleton(
      const Eigen::MatrixBase<DerivedUV> & UV,
      const Eigen::MatrixBase<DerivedM> & M,
      const Eigen::MatrixBase<DerivedVP> & VP,
      const Eigen::MatrixXd &skel_V,
      const Eigen::MatrixXi &skel_E,
      //  const Eigen::MatrixBase<Derivedorigin> & origin,
      //  const Eigen::MatrixBase<Deriveddir> & dir,
      //  typename DerivedUV::Scalar & t)
      Eigen::RowVector3d &p);

//#include "unproject_on_skeleton.cpp"

#endif
