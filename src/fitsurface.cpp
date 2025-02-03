#include <iostream>
#include <cmath>
#include <float.h>

#include <Eigen/Core>

#include "fitsurface.h"

FitSurface::FitSurface(const Eigen::MatrixXd &points, const size_t minRes, const Eigen::MatrixXd &mesh_V) : 
  m_worldPoints(points),
  m_minRes(minRes),
  m_mesh_V(mesh_V)
{
  transformToLocal();
  createMeshGrid();
}

void FitSurface::transformToLocal()
{
  //Calculate mean 
  Eigen::RowVectorXd mean = m_worldPoints.colwise().mean();
  Eigen::MatrixXd points_zeromean = m_worldPoints.rowwise() - mean;

  // Compute PCA
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(points_zeromean, Eigen::ComputeFullV);
  Eigen::MatrixXd v = svd.matrixV();

  // Matrix to transform to local basis. Z-axis has minimum variance
  m_worldToLocal = Eigen::Matrix4d::Identity();
  m_worldToLocal.block<3, 3>(0, 0) = v.transpose();// R
  m_worldToLocal.block<3, 1>(0, 3) = -v.transpose()*mean.transpose();//-Rt

  // Transform point cloud to local coordinates
  m_localPoints = (m_worldPoints * m_worldToLocal.block<3, 3>(0, 0).transpose()).rowwise() + m_worldToLocal.block<3, 1>(0, 3).transpose();
}

void FitSurface::createMeshGrid()
{
  // Calculate bounding box of mesh w.r.t. the local space
  // Note: all bbox related calculations are in local XY plane, and localMeshV is Nx2
  Eigen::MatrixXd localMeshV = (m_mesh_V * m_worldToLocal.block<2, 3>(0, 0).transpose()).rowwise() + m_worldToLocal.block<2, 1>(0, 3).transpose();
  Eigen::RowVectorXd bbox_min = localMeshV.colwise().minCoeff();
  Eigen::RowVectorXd bbox_max = localMeshV.colwise().maxCoeff();
  // Calculate bbox dim abd increase bbox by 10% as a safety measure so that all parts of projected mesh are inside
  Eigen::RowVectorXd bbox_center = (bbox_min + bbox_max)/2.0;
  Eigen::RowVectorXd dims = 1.1*(bbox_max - bbox_min).block<1,2>(0,0);  
  bbox_min = bbox_center - dims/2.0;
  bbox_max = bbox_center + dims/2.0;

  //Calculate mesh resolutions along X and Y 
  float gridSize = dims.minCoeff()/(m_minRes - 1);
  size_t xres = round(dims(0)/gridSize) + 1;
  size_t yres = round(dims(1)/gridSize) + 1;

  // Generate grid points for X and Y
  Eigen::RowVectorXd seq_x = Eigen::RowVectorXd::LinSpaced(xres, bbox_min(0), bbox_max(0));
  Eigen::VectorXd seq_y = Eigen::VectorXd::LinSpaced(yres, bbox_min(1), bbox_max(1));
  m_X = seq_x.replicate(seq_y.rows(), 1);
  m_Y = seq_y.replicate(1, seq_x.cols());
  m_Z.resize(m_X.rows(), m_X.cols()); // Resize m_Z to the correct size
}

void FitSurface::assembleMesh(Eigen::MatrixXd &meshV, Eigen::MatrixXi &meshF)
{
  // Generate topology for the mesh (this is fixed).
  // Matrices in Eigen are stored in column-major format.
  size_t nrows = m_X.rows();
  size_t ncols = m_X.cols();
  size_t ntriangles = 2*(nrows-1)*(ncols-1);
  meshF.resize(ntriangles, 3);
  size_t k = 0;
  for(size_t j=0; j<(ncols-1); j++)
    for(size_t i=0; i<(nrows-1); i++)
    {
      meshF.row(k++) << (i + (j+1)*nrows), (i+ j*nrows), (i+1 + j*nrows);
      meshF.row(k++) << (i+1 + j*nrows), (i+1 + (j+1)*nrows), (i + (j+1)*nrows);
    }
   assert(meshF.rows() == ntriangles);

  // Transform coordinates m_X, m_Y, and m_Z to World space and store into meshV
  size_t npoints = nrows*ncols;
  meshV.resize(npoints, 3);
  Eigen::Map<Eigen::VectorXd> v_X(m_X.data(), m_X.size());
  Eigen::Map<Eigen::VectorXd> v_Y(m_Y.data(), m_Y.size());
  Eigen::Map<Eigen::VectorXd> v_Z(m_Z.data(), m_Z.size());
  Eigen::MatrixXd localToWorld_tr = m_worldToLocal.inverse().transpose();
  for(k=0; k<npoints; k++) {
    meshV.row(k) << v_X.row(k)*localToWorld_tr.block<1,3>(0,0) +  
                    v_Y.row(k)*localToWorld_tr.block<1,3>(1,0) +
                    v_Z.row(k)*localToWorld_tr.block<1,3>(2,0) +
                    localToWorld_tr.block<1,3>(3,0);
  }
}

void FitSurface::fitPlane(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F)
{
  // Set Z = 0 (PCA space Z=0 plane)
  m_Z.setZero();
  m_coefficients = Eigen::VectorXd::Zero(1);

  assembleMesh(surface_V, surface_F);
}

void FitSurface::fitPoly33(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F)
{
  Eigen::MatrixXd A(m_localPoints.rows(), 10);
  A.col(0) = Eigen::VectorXd::Ones(m_localPoints.rows()); // 1
  A.col(1) = m_localPoints.col(0); // x
  A.col(2) = m_localPoints.col(1); // y
  A.col(3) = m_localPoints.col(0).array()*m_localPoints.col(1).array(); // xy
  A.col(4) = m_localPoints.col(0).array().square(); // x^2 
  A.col(5) = m_localPoints.col(1).array().square(); // y^2
  A.col(6) = m_localPoints.col(0).array().square()*m_localPoints.col(1).array(); // x^2y
  A.col(7) = m_localPoints.col(1).array().square()*m_localPoints.col(0).array(); // xy^2
  A.col(8) = m_localPoints.col(0).array().cube(); // x^3
  A.col(9) = m_localPoints.col(1).array().cube(); // y^3

  // Least-squares solution
  Eigen::VectorXd b = m_localPoints.col(2);
  m_coefficients = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

  // Calculate m_Z
  m_Z = m_coefficients(0) + m_coefficients(1)*m_X.array() + m_coefficients(2)*m_Y.array() + 
    m_coefficients(3)*m_X.array()*m_Y.array() + 
    m_coefficients(4)*m_X.array().square() + 
    m_coefficients(5)*m_Y.array().square() + 
    m_coefficients(6)*m_X.array().square()*m_Y.array() + 
    m_coefficients(7)*m_X.array()*m_Y.array().square() + 
    m_coefficients(8)*m_X.array().cube() + 
    m_coefficients(9)*m_Y.array().cube();

  assembleMesh(surface_V, surface_F);
}


void FitSurface::fitBiharmonic(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F)
/* Based on [Sandwell, 1987]:
   D.T. Sandwell, Biharmonic spline interpolation of GEOS-3 and SEASAT
   altimeter data, Geophysica Research Letters 14: 139–142, 1987.
*/
{
    // std::cout << "\nInside fitBiharmonic():" << std::flush;
    Eigen::MatrixXd A(m_localPoints.rows(), m_localPoints.rows());
    double rij2 = 0.0;
    m_Z.setZero();
    m_coefficients = Eigen::VectorXd::Zero(1);

    // Compute Green's functions bases matrix for Local points
    for (int i = 0; i < m_localPoints.rows(); i++) {
        for (int j = i + 1; j < m_localPoints.rows(); j++) {
            rij2 = (std::pow(m_localPoints(i, 0) - m_localPoints(j, 0), 2) +
                    std::pow(m_localPoints(i, 1) - m_localPoints(j, 1), 2));
            if (rij2 < TOL) 
                A(i, j) = 0.0;
            else 
                A(i, j) = (std::log(std::sqrt(rij2)) - 1) * rij2;
            A(j, i) = A(i, j);
        }
        A(i, i) = 0.0;
    }

    /* Compute coefficients for Green's basis functions for fitting to
       Local data */
    Eigen::VectorXd b = m_localPoints.col(2);
    m_coefficients = A.partialPivLu().solve(b);

    double m_X_ij, m_Y_ij, m_local_x, m_local_y;
    for (int i = 0; i < m_X.rows(); i++) 
        for (int j = 0; j < m_X.cols(); j++) {
            m_X_ij = m_X(i, j);    // Obtain local x and y coordinates
            m_Y_ij = m_Y(i, j);    // of current vertex in mesh
            for (int k = 0; k < m_localPoints.rows(); k++) {
                // Obtain local coordinates of k-th data point
                m_local_x = m_localPoints(k, 0);
                m_local_y = m_localPoints(k, 1);
                // Compute Green's function contribution at mesh vertex
                rij2 = (std::pow(m_X_ij - m_local_x, 2) +
                        std::pow(m_Y_ij - m_local_y, 2));
                if (rij2 < TOL) 
                    continue;
                m_Z(i, j) += (m_coefficients(k) *
                              (std::log(std::sqrt(rij2)) - 1) * rij2);
            }
        }

    assembleMesh(surface_V, surface_F);
}

void FitSurface::fit(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F, FittingType ftype)
{
  switch(ftype)
  {
    case FittingType_Plane:
      fitPlane(surface_V, surface_F);
      break;
    case FittingType_Poly33:
      fitPoly33(surface_V, surface_F);
      break;
    case FittingType_Biharmonic:
      fitBiharmonic(surface_V, surface_F);
      break;
  }
}


