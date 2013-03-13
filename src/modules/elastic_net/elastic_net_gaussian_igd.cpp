
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_igd.hpp"
#include "state/igd.hpp"
#include <limits>

namespace madlib {
namespace modules {
namespace elastic_net {

// sign of a number
static double sign(const double & x) {
    if (x == 0.) { return 0.; }
    else { return x > 0. ? 1. : -1.; }
}

// ------------------------------------------------------------------------
// Need divided-by-zero type check

static double p_abs (ColumnVector v, double r)
{
    double sum = 0;
    for (int i = 0; i < v.size(); i++)
        sum += pow(fabs(v(i)), r);
    return pow(sum, 1./r);
}

// p-form link function, q = p/(p-1)
// For inverse function, jut replace w with theta and q with p 
static ColumnVector link_fn (ColumnVector w, double q)
{
    ColumnVector theta(w.size());
    double abs_w = p_abs(w, q);
    if (abs_w == 0)
    {
        for (int i = 0; i < w.size(); i++)
            theta(i) = 0;
        
        return theta;
    }

    double denominator = pow(abs_w, q - 2);

    for (int i = 0; i < w.size(); i++)
        if (w(i) == 0) theta(i) = 0;
        else
            theta(i) = sign(w(i)) * pow(fabs(w(i)), q - 1)
                / denominator;
    
    return theta;
}

// ------------------------------------------------------------------------

/**
   @brief Perform IGD transition step

   It is called for each tuple.

   The input AnyType has 9 args: state. ind_var, dep_var,
   pre_state, lambda, alpha, dimension, stepsize, totalrows
*/
AnyType
gaussian_igd_transition::run (AnyType& args)
{
    IgdState<MutableArrayHandle<double> > state = args[0];
    
    // initialize the state if working on the first tuple
    if (state.algo.numRows == 0)
    {
        if (!args[3].isNull())
        {
            IgdState<ArrayHandle<double> > pre_state = args[3];
            state.allocate(*this, pre_state.task.dimension);
            state = pre_state;
        }
        else
        {
            double lambda = args[4].getAs<double>();
            double alpha = args[5].getAs<double>();
            int dimension = args[6].getAs<int>();
            double stepsize = args[7].getAs<double>();
            int total_rows = args[8].getAs<int>();

            state.allocate(*this, dimension);
            state.task.stepsize = stepsize;
            state.task.lambda = lambda;
            state.task.alpha = alpha;
            state.task.totalRows = total_rows;
            // dual vector theta
            state.algo.theta.setzeros();
            state.algo.p = 2 * log(state.task.dimension);
            state.algo.q = p / (p - 1);
        }
      
        state.algo.loss = 0.;
        state.algo.incrCoef = state.task.coef;
        state.algo.incrIntercept = state.task.intercept;
    }

    // tuple
    // using madlib::dbal::eigen_integration::MappedColumnVector;
   
    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    // Now do the transition step
    double wx = dot(state.algo.incrCoef, x) + state.algo.incrIntercept;
    double r = wx - y;

    ColumnVector gradient(state.task.dimension);
    state.algo.gradient = r * x;
    
    for (uint32_t i = 0; i < state.task.dimension; i++)
    {
        gradient(i) += (1 - state.task.alpha) * state.task.lambda
            * state.task.coef(i);
        // step 1
        state.algo.theta(i) -= state.task.stepsize * gradient(i)
            / state.task.totalRows;
        double step1_sign = sign(state.algo.theta(i));
        // step 2
        state.algo.theta(i) -= state.task.stepsize * state.task.alpha
            * state.task.lambda * sign(state.task.coef(i))
            / state.task.totalRows;
        // set to 0 if the value crossed zero during the two steps
        if (step1_sign != sign(state.algo.theta(i))) state.algo.theta(i);
    }
    
    stata.algo.incrCoef = link_fn(state.algo.theta, state.algo.p);

    state.algo.incrIntercept = ymean - dot(state.algo.incrCoef, xmean);
    
    // OLSENRegularizedIGDAlgorithm::transition(state, tuple);
    state.algo.loss += r * r / 2.;
    // OLSLossAlgorithm::transition(state, tuple);
    state.algo.numRows ++;

    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
gaussian_igd_merge::run (AnyType& args)
{
    IgdState<MutableArrayHandle<double> > state1 = args[0];
    IgdState<ArrayHandle<double> > state2 = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (state1.algo.numRows == 0) { return state2; }
    else if (state2.algo.numRows == 0) { return state1; }

    // Merge states together
    double totalNumRows = static_cast<double>(state1.algo.numRows + state2.algo.numRows);
    state1.algo.theta *= static_cast<double>(state1.algo.numRows) /
        static_cast<double>(state2.algo.numRows);
    state1.algo.theta += state2.algo.incrCoef;
    state1.algo.theta *= static_cast<double>(state2.algo.numRows) /
        static_cast<double>(totalNumRows);

    // the following two lines might not be necessary, since incrCoef is
    // not used in merge, only in final function
    // this can be put into final
    stata1.algo.incrCoef = link_fn(state1.algo.theta, state1.algo.p);
    state1.algo.incrIntercept = ymean - dot(state1.algo.incrCoef, xmean);
    
    state1.algo.loss += state2.algo.loss;
    
    // The following numRows update, cannot be put above, because the coef
    // averaging depends on their original values
    state1.algo.numRows += state2.algo.numRows;

    return state1;
}

// ------------------------------------------------------------------------

/**
 * @brief Perform the final step
 */
AnyType
gaussian_igd_final::run (AnyType& args)
{
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    IgdState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.algo.numRows == 0) return Null(); 

    // finalizing
    state.task.coef = state.algo.incrCoef;
    state.task.intercept = state.algo.incrIntercept;

    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType
__gaussian_igd_state_diff::run (AnyType& args)
{
    IgdState<ArrayHandle<double> > state1 = args[0];
    IgdState<ArrayHandle<double> > state2 = args[1];

    // double diff = 0;    
    // Index i;
    // int n = state1.task.coef.rows();
    // for (i = 0; i < n; i++)
    // {
    //     diff += std::abs(state1.task.coef(i) - state2.task.coef(i));
    // }

    // return diff / n;

    return std::abs((state1.algo.loss - state2.algo.loss) / state2.algo.loss);
}

// ------------------------------------------------------------------------

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
__gaussian_igd_result::run (AnyType& args)
{
    IgdState<ArrayHandle<double> > state = args[0];
    double norm = 0;

    for (Index i = 0; i < state.task.coef.rows() - 1; i ++) {
        double m = state.task.coef(i);
        norm += state.task.alpha * std::abs(m) + (1 - state.task.alpha) * m * m * 0.5;
    }
    norm *= state.task.lambda;
        
    AnyType tuple;
    tuple << static_cast<double>(state.task.intercept)
          << state.task.coef 
          << static_cast<double>(state.algo.loss) + norm;// +
        // (double)(GLMENRegularizer::loss(state.task.coef,
        //                                 state.task.lambda,
        //                                 state.task.alpha));

    return tuple;
}

// ------------------------------------------------------------------------

// /**
//  * @brief Compute w \dot x, where w is the vector of coefficients
//  */
// AnyType
// gaussian_igd_predict::run (AnyType& args)
// {
//     using madlib::dbal::eigen_integration::MappedColumnVector;
//     MappedColumnVector coef = args[0].getAs<MappedColumnVector>();
//     double intercept = args[1].getAs<double>();
//     MappedColumnVector indVar = args[2].getAs<MappedColumnVector>();

//     return OLS<MappedColumnVector, GLMTuple>::predict(coef, intercept, indVar);
// }
 
} // namespace elastic_net 
} // namespace modules
} // namespace madlib
