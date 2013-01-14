/* ----------------------------------------------------------------------- *//**
 *
 * @file elastic_net.hpp
 *
 *//* ----------------------------------------------------------------------- */

#ifndef MADLIB_MODULES_CONVEX_TASK_ELASTIC_NET_HPP_
#define MADLIB_MODULES_CONVEX_TASK_ELASTIC_NET_HPP_

#include <dbconnector/dbconnector.hpp>
#include <limits>

namespace madlib {
namespace modules {
namespace convex {

// Use Eigen
using namespace madlib::dbal::eigen_integration;

template <class Model>
class ElasticNet
{
  public:
    typedef Model model_type;
    
    static void gradient(
        const model_type& model, const double& lambda, const double& alpha,
        const int& row_num, const double& stepsize, model_type& gradient);
    
    // likelihood actually
    static double loss(const model_type& model, const double& lambda,
                       const double& alpha);

  private:
    static double sign(const double & x)
    {
        if (x == 0.) { return 0.; }
        else { return x > 0. ? 1. : -1.; }
    }
};

template <class Model>
void
ElasticNet<Model>::gradient(
    const model_type& model, const double& lambda, const double& alpha,
    const int& row_num, const double& stepsize, model_type& gradient)
{
    Index i;
    // the last model coefficient is intercept, which
    // will not be regularized
    for (i = 0; i < model.rows() - 1; i ++)
    {
        if (std::abs(model(i)) <= std::numeric_limits<double>::denorm_min())
        {
            // soft thresholding
            if (std::abs(gradient(i)) > lambda) {
                gradient(i) -= alpha * lambda * sign(gradient(i));
                // gradient(i) = - gradient(i) / stepsize + alpha * model(i) * row_num / stepsize;
            } else {
                gradient(i) = alpha * model(i) * row_num / stepsize;
            }
        }
        else
        {
            gradient(i) += alpha * lambda * sign(model(i));
        }

        gradient(i) += (1 - alpha) * lambda * model(i);
    }
}

template <class Model>
double 
ElasticNet<Model>::loss(const model_type& model, const double& lambda, const double& alpha)
{
    double norm = 0.;
    Index i;
    // the last model coefficient is intercept, which
    // will not be regularized
    for (i = 0; i < model.rows() - 1; i ++)
        norm += alpha * std::abs(model(i)) +
            (1 - alpha) * model(i) * model(i) * 0.5;
    return lambda * norm;
}

} // namespace convex
} // namespace modules
} // namespace madlib

#endif

