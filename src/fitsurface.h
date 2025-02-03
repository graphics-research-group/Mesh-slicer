#ifndef _FITSURFACE_H_
#define _FITSURFACE_H_

#include <Eigen/Dense>

#define TOL 1e-9

enum FittingType {FittingType_Plane, FittingType_Poly33, FittingType_Biharmonic};

class FitSurface
{
    private:
    const size_t m_minRes;
    const Eigen::MatrixXd &m_worldPoints; // Input points in World space
    Eigen::MatrixXd m_localPoints; // Local points after PCA
    Eigen::Matrix4d m_worldToLocal; // World to Local rigid transformation matrix

    Eigen::MatrixXd m_X, m_Y; // Local grid of X, Y coordinates for fit surface evaluation
    Eigen::MatrixXd m_Z; // Local grid of Z values calculated by evaluating the fit surface

    const Eigen::MatrixXd &m_mesh_V;
    Eigen::VectorXd m_coefficients;

    void transformToLocal();
    void createMeshGrid();
    void assembleMesh(Eigen::MatrixXd &meshV, Eigen::MatrixXi &meshE);
    void fitPlane(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_E);
    void fitPoly33(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_E);
    void fitBiharmonic(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_E);

    public:
    FitSurface(const Eigen::MatrixXd &points, const size_t minRes, const Eigen::MatrixXd &mesh_V);
    void fit(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F, FittingType ftype);
    Eigen::MatrixXd getWorldToLocal() { return m_worldToLocal;}
    Eigen::VectorXd getCoefficients() { return m_coefficients;}
};

#endif
