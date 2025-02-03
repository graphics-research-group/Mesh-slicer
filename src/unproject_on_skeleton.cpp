#include "unproject_on_skeleton.h"
#include <igl/projection_constraint.h>
#include <map>

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
  Eigen::RowVector3d &p)
{
  using namespace Eigen;
  typedef typename DerivedUV::Scalar Scalar;
  Matrix<Scalar,2,3> A;
  Matrix<Scalar,2,1> B;
  igl::projection_constraint(UV,M,VP,A,B);

  std::map<double, std::pair<int, double> > projections; // This will be sorted on the energy term.
  // min_z,t ‖Az - B‖²  subject to z = origin + t*dir
  // min_t  ‖A(origin + t*dir) - B‖²
  // min_t  ‖A*t*dir + A*origin - B‖²
  // min_t  ‖D*t + C‖²
  // t = -(D'D)\(D'*C)

  // For all skeleton segments, calculate nearest point and error
  double energy;
  for(int i = 0; i < skel_E.rows(); i++)
  {
    auto origin = skel_V.row(skel_E(i,0));
    auto dir = skel_V.row(skel_E(i,1)) - skel_V.row(skel_E(i,0));
    auto C = A*origin.transpose().template cast<Scalar>() - B;
    auto D = A*dir.transpose().template cast<Scalar>();
    // Solve least squares system directly
    const Matrix<Scalar,1,1> t_mat = D.jacobiSvd(ComputeFullU | ComputeFullV).solve(-C);
    double t = t_mat(0,0);
    if (t < 0. || t > 1.) continue; // Point is outside segment; discard.
    energy = (D*t_mat + C).norm();
    projections.insert(std::make_pair(energy, std::make_pair(i, t)));
  }

  int idx = projections.begin()->second.first;
  double t = projections.begin()->second.second;

  p = skel_V.row(skel_E(idx, 0)) + t*(skel_V.row(skel_E(idx, 1)) - skel_V.row(skel_E(idx, 0)));

  // TODO: add sensitivity threshold in selection. Compute distance of nearest point from unprojected ray 
  // and reject based on threshold distance.
  return true;
}

template bool unproject_on_skeleton<
    Eigen::Matrix<float, 2, 1, 0, 2, 1>, 
    Eigen::Product<Eigen::Matrix<float, 4, 4, 0, 4, 4>, Eigen::Matrix<float, 4, 4, 0, 4, 4>, 0>, 
    Eigen::Matrix<float, 4, 1, 0, 4, 1> >(
    Eigen::MatrixBase<Eigen::Matrix<float, 2, 1, 0, 2, 1> > const&, 
    Eigen::MatrixBase<Eigen::Product<Eigen::Matrix<float, 4, 4, 0, 4, 4>, Eigen::Matrix<float, 4, 4, 0, 4, 4>, 0> > const&, 
    Eigen::MatrixBase<Eigen::Matrix<float, 4, 1, 0, 4, 1> > const&, 
    Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, 
    Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, 
    Eigen::Matrix<double, 1, 3, 1, 1, 3>&);
