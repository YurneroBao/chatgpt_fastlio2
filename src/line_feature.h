#pragma once
#include <Eigen/Eigen>
#include "common_lib.h"

struct LineParam
{
    V3D p0;
    V3D d;
    Eigen::Matrix<double,2,3> N;
};

inline bool esti_line_pca(LineParam &out,
                   const PointVector &points_near,
                   double max_mse,
                   double min_linearity)
{
    int m = points_near.size();
    if(m < 5) return false;

    V3D mu(0,0,0);
    for(int i=0;i<m;i++)
        mu += V3D(points_near[i].x,
                  points_near[i].y,
                  points_near[i].z);
    mu /= m;

    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    for(int i=0;i<m;i++)
    {
        V3D d = V3D(points_near[i].x,
                    points_near[i].y,
                    points_near[i].z) - mu;
        C += d*d.transpose();
    }
    C /= m;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
    if(es.info()!=Eigen::Success) return false;

    Eigen::Vector3d eval = es.eigenvalues();
    Eigen::Matrix3d evec = es.eigenvectors();

    double l1 = eval(2);
    double l2 = eval(1);

    if((l1-l2)/l1 < min_linearity) return false;

    out.p0 = mu;
    out.d  = evec.col(2).normalized();

    V3D a = fabs(out.d.z())<0.9?V3D(0,0,1):V3D(0,1,0);
    V3D n1 = out.d.cross(a).normalized();
    V3D n2 = out.d.cross(n1).normalized();

    out.N.row(0) = n1.transpose();
    out.N.row(1) = n2.transpose();

    return true;
}