
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

static ColumnVector p_abs (ColumnVector v, double r)
{
    double sum = 0;
    for (int i = 0; i < v.size(); i++)
        sum += pow(abs(v(i)), r);
    return pow(sum, 1./r);
}

// p-form link function, q = p/(p-1)
static ColumnVector link_fn (ColumnVector w, double q)
{
    
}

// inverse of p-form link function
static ColumnVector inverse_link_fn (ColumnVector theta, uint32_t size, double p)
{

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
        }
      
        state.algo.loss = 0.;
        state.algo.incrModel = state.task.model;
    }

    // tuple
    using madlib::dbal::eigen_integration::MappedColumnVector;
   
    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    // Now do the transition step
    double wx = dot(state.task.model, x);
    double r = wx - y;
    state.algo.gradient += r * x;

    for (Index i = 0; i < state.task.model.rows() - 1; i ++)
    {
        if (std::abs(state.task.model(i)) <= std::numeric_limits<double>::denorm_min())
        {
            // soft thresholding
            if (std::abs(state.algo.gradient(i)) > state.task.lambda * state.task.alpha)
            {
                state.algo.gradient(i) -= state.task.alpha * state.task.lambda
                    * sign(state.algo.gradient(i));

                state.algo.gradient(i) = - state.algo.gradient(i) / state.task.stepsize
                    + state.task.alpha * state.task.model(i) * state.task.totalRows
                    / state.task.stepsize;
            }
            else
            {
                state.algo.gradient(i) = state.task.alpha * state.task.model(i)
                    * state.task.totalRows / state.task.stepsize;
                //gradient(i) = 0;
            }
        }
        else
        {
            state.algo.gradient(i) += state.task.alpha * state.task.lambda
                * sign(state.task.model(i));
        }

        state.algo.gradient(i) += (1 - state.task.alpha) * state.task.lambda
            * state.task.model(i);
    }
    
    state.algo.incrModel -= state.task.stepsize * state.algo.gradient
        / state.task.totalRows;
    
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
    state1.algo.incrModel *= static_cast<double>(state1.algo.numRows) /
        static_cast<double>(state2.algo.numRows);
    state1.algo.incrModel += state2.algo.incrModel;
    state1.algo.incrModel *= static_cast<double>(state2.algo.numRows) /
        static_cast<double>(totalNumRows);

    state1.algo.loss += state2.algo.loss;
    
    // The following numRows update, cannot be put above, because the model
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
    state.task.model = state.algo.incrModel;

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
    // int n = state1.task.model.rows();
    // for (i = 0; i < n; i++)
    // {
    //     diff += std::abs(state1.task.model(i) - state2.task.model(i));
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

    for (Index i = 0; i < state.task.model.rows() - 1; i ++) {
        double m = state.task.model(i);
        norm += state.task.alpha * std::abs(m) + (1 - state.task.alpha) * m * m * 0.5;
    }
    norm *= state.task.lambda;
        
    AnyType tuple;
    tuple << state.task.model
          << static_cast<double>(state.algo.loss) + norm;// +
        // (double)(GLMENRegularizer::loss(state.task.model,
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
//     MappedColumnVector model = args[0].getAs<MappedColumnVector>();
//     double intercept = args[1].getAs<double>();
//     MappedColumnVector indVar = args[2].getAs<MappedColumnVector>();

//     return OLS<MappedColumnVector, GLMTuple>::predict(model, intercept, indVar);
// }
 
} // namespace elastic_net 
} // namespace modules
} // namespace madlib
