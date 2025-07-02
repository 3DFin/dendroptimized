#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unsupported/Eigen/NonLinearOptimization>

#include "types.hpp"

namespace dendroptimized
{

template <typename real_t>
struct EigenCircleFitFunctor
{
    RefCloud2<real_t> data;

    EigenCircleFitFunctor(RefCloud2<real_t> points) : n_values(points.rows()), data(points) {}

    // Evaluate residual
    int operator()(const Eigen::VectorX<real_t>& x, Eigen::VectorX<real_t>& fvec) const
    {
        const real_t a = x(0);  // TODO cache computation
        const real_t b = x(1);
        const real_t r = x(2);
        for (Eigen::Index i = 0; i < data.rows(); ++i)
        {
            const real_t dx = data(i, 0) - a;
            const real_t dy = data(i, 1) - b;
            fvec(i)         = std::sqrt(dx * dx + dy * dy) - r;
        }
        return 0;
    }

    // Compute jacobian
    int df(const Eigen::VectorX<real_t> x, Eigen::MatrixX<real_t>& fjac) const
    {
        const real_t a = x(0);
        const real_t b = x(1);
        const real_t r = x(2);
        for (Eigen::Index i = 0; i < data.rows(); ++i)
        {
            // TODO cache computation
            const real_t dx = data(i, 0) - a;
            const real_t dy = data(i, 1) - b;
            const real_t d  = std::sqrt(dx * dx + dy * dy);

            if (d < real_t(1e-10))  // add robustness (avoid division by zero)
            {
                fjac(i, 0) = fjac(i, 1) = 0;
            }
            else
            {
                fjac(i, 0) = -dx / d;
                fjac(i, 1) = -dy / d;
            }
            fjac(i, 2) = real_t(-1.0);
        }
        return 0;
    }

    int n_values = 0;
    int inputs() const { return 3; }
    int values() const { return n_values; }
};

template <typename real_t>
Eigen::Vector3<real_t> LMCircleFit(RefCloud2<real_t> xy)
{
    if (xy.rows() < 3) throw std::invalid_argument("Circle fit need at least 3 points");
    // Initial guess: center at centroid
    Eigen::Vector2<real_t> centroid = xy.colwise().mean();
    real_t                 r0       = ((xy.rowwise() - centroid.transpose()).rowwise().norm()).mean();

    Eigen::VectorX<real_t> x0(3);
    x0 << centroid(0), centroid(1), r0;

    EigenCircleFitFunctor                                    functor(xy);
    Eigen::LevenbergMarquardt<EigenCircleFitFunctor<real_t>> lm(functor);
    // Max function evaluations.
    // 100 iters should be enough.
    // See H. Abdul-Rahman and N. Chernov, 2013. : "The GN and LM normally converge in 5–10 iterations."
    // We could use algebraic fit (i.e. Taubin) to improve convergence and results in difficult cases.
    lm.parameters.maxfev = 100;
    lm.parameters.xtol   = 1.4e-8;

    int status = lm.minimize(x0); //TODO: check status
    return {Eigen::Vector3<real_t>(x0(0), x0(1), x0(2))};
}

}  // namespace dendroptimized
