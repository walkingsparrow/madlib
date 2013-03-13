
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
    if (state.numRows == 0)
    {
        if (!args[3].isNull())
        {
            IgdState<ArrayHandle<double> > pre_state = args[3];
            state.allocate(*this, pre_state.dimension);
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
            state.stepsize = stepsize;
            state.lambda = lambda;
            state.alpha = alpha;
            state.totalRows = total_rows;
            state.xmean = args[9].getAs<MappedColumnVector>();
            state.ymean = args[10].getAs<double>();
            // dual vector theta
            state.theta.setZero();
            state.p = 2 * log(state.dimension);
            state.q = state.p / (state.p - 1);
        }
      
        state.loss = 0.;
        state.incrCoef = state.coef;
        state.incrIntercept = state.intercept;
    }

    // tuple
    // using madlib::dbal::eigen_integration::MappedColumnVector;
   
    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    // Now do the transition step
    double wx = dot(state.incrCoef, x) + state.incrIntercept;
    double r = wx - y;

    ColumnVector gradient(state.dimension);
    gradient = r * x;
    
    for (uint32_t i = 0; i < state.dimension; i++)
    {
        gradient(i) += (1 - state.alpha) * state.lambda
            * state.coef(i);
        // step 1
        state.theta(i) -= state.stepsize * gradient(i)
            / state.totalRows;
        double step1_sign = sign(state.theta(i));
        // step 2
        state.theta(i) -= state.stepsize * state.alpha
            * state.lambda * sign(state.coef(i))
            / state.totalRows;
        // set to 0 if the value crossed zero during the two steps
        if (step1_sign != sign(state.theta(i))) state.theta(i);
    }
    
    state.incrCoef = link_fn(state.theta, state.p);

    state.incrIntercept = state.ymean - dot(state.incrCoef, state.xmean);
    
    // OLSENRegularizedIGDAlgorithm::transition(state, tuple);
    state.loss += r * r / 2.;
    // OLSLossAlgorithm::transition(state, tuple);
    state.numRows ++;

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
    if (state1.numRows == 0) { return state2; }
    else if (state2.numRows == 0) { return state1; }

    // Merge states together
    double totalNumRows = static_cast<double>(state1.numRows + state2.numRows);
    state1.theta *= static_cast<double>(state1.numRows) /
        static_cast<double>(state2.numRows);
    state1.theta += state2.incrCoef;
    state1.theta *= static_cast<double>(state2.numRows) /
        static_cast<double>(totalNumRows);

    // the following two lines might not be necessary, since incrCoef is
    // not used in merge, only in final function
    // this can be put into final
    state1.incrCoef = link_fn(state1.theta, state1.p);
    state1.incrIntercept = state1.ymean - dot(state1.incrCoef, state1.xmean);
    
    state1.loss += state2.loss;
    
    // The following numRows update, cannot be put above, because the coef
    // averaging depends on their original values
    state1.numRows += state2.numRows;

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
    if (state.numRows == 0) return Null(); 

    // finalizing
    state.coef = state.incrCoef;
    state.intercept = state.incrIntercept;

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
    // int n = state1.coef.rows();
    // for (i = 0; i < n; i++)
    // {
    //     diff += std::abs(state1.coef(i) - state2.coef(i));
    // }

    // return diff / n;

    return std::abs((state1.loss - state2.loss) / state2.loss);
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

    for (Index i = 0; i < state.coef.rows() - 1; i ++) {
        double m = state.coef(i);
        norm += state.alpha * std::abs(m) + (1 - state.alpha) * m * m * 0.5;
    }
    norm *= state.lambda;
        
    AnyType tuple;
    tuple << static_cast<double>(state.intercept)
          << state.coef 
          << static_cast<double>(state.loss) + norm;// +
        // (double)(GLMENRegularizer::loss(state.coef,
        //                                 state.lambda,
        //                                 state.alpha));

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
