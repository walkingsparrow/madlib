/* ----------------------------------------------------------------------- *//**
 *
 * @file logit.hpp
 *
 *//* ----------------------------------------------------------------------- */

#ifndef MADLIB_MODULES_CONVEX_TASK_LOGIT_HPP_
#define MADLIB_MODULES_CONVEX_TASK_LOGIT_HPP_

#include <dbconnector/dbconnector.hpp>

namespace madlib {

namespace modules {

namespace convex {

// Use Eigen
using namespace madlib::dbal::eigen_integration;

template <class Model, class Tuple>
class Logit {
public:
    typedef Model model_type;
    typedef Tuple tuple_type;
    typedef typename Tuple::independent_variables_type 
        independent_variables_type;
    typedef typename Tuple::dependent_variable_type dependent_variable_type;

    static void gradient(
            const model_type                    &model,
            const independent_variables_type    &x,
            const dependent_variable_type       &y, 
            model_type                          &gradient);

    static void gradientInPlace(
            model_type                          &model, 
            const independent_variables_type    &x,
            const dependent_variable_type       &y, 
            const double                        &stepsize);
    
    static double loss(
            const model_type                    &model, 
            const independent_variables_type    &x, 
            const dependent_variable_type       &y);
    
    static dependent_variable_type predict(
            const model_type                    &model, 
            const independent_variables_type    &x);

private:
    static double sigma(double x) {
        return 1. / (1. + std::exp(-x));
    }
};

template <class Model, class Tuple>
void
Logit<Model, Tuple>::gradient(
        const model_type                    &model,
        const independent_variables_type    &x,
        const dependent_variable_type       &y,
        model_type                          &gradient) {
    double wx = dot(model, x);
    double sig = sigma(-wx * y);
    double c = -sig * y; // minus for "-loglik"
    gradient += c * x;
}

template <class Model, class Tuple>
void
Logit<Model, Tuple>::gradientInPlace(
        model_type                          &model,
        const independent_variables_type    &x, 
        const dependent_variable_type       &y, 
        const double                        &stepsize) {
    double wx = dot(model, x);
    double sig = sigma(-wx * y);
    double c = -sig * y; // minus for "-loglik"
    model -= stepsize * c * x;
}

template <class Model, class Tuple>
double 
Logit<Model, Tuple>::loss(
        const model_type                    &model, 
        const independent_variables_type    &x, 
        const dependent_variable_type       &y) {
    double wx = dot(model, x);
    return log(1. + std::exp(-wx * y)); //  = -log(sigma(y * wx))
}

// Not currently used.
template <class Model, class Tuple>
typename Tuple::dependent_variable_type
Logit<Model, Tuple>::predict(
        const model_type                    &model, 
        const independent_variables_type    &x) {
    double wx = dot(model, x);
    return sigma(wx);
}

} // namespace convex

} // namespace modules

} // namespace madlib

#endif

