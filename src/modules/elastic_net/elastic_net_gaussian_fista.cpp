
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_fista.hpp"
#include "state/fista.hpp"

namespace madlib {
namespace modules {
namespace elastic_net {

/**
   @brief Perform FISTA transition step

   It is called for each tuple of (x, y)
*/
AnyType gaussian_fista_transition::run (AnyType& args)
{
    FistaState<MutableArrayHandle<double> > state = args[0];

    // initialize the state if working on the first tuple
    if (state.algo.numRows == 0)
    {
        if (!args[3].isNull())
        {
            FistaState<ArrayHandle<double> > pre_state = args[3];
            state.allocate(*this, pre_state.task.dimension);
            state = pre_state;
        }
        else
        {
            double lambda = args[4].getAs<double>();
            double alpha = args[5].getAs<double>();
            double stepsize = args[6].getAs<double>();
            uint32_t dimension = args[7].getAs<uint32_t>();
            MappedColumnVector xmean = args[8].getAs<MappedColumnVector>();
            double ymean = args[9].getAs<double>();
            double tk = args[10].getAs<double>();
            int totalRows = args[11].getAs<int>();
            
            state.allocate(*this, dimension);
            state.task.lambda = lambda;
            state.task.alpha = alpha;
            state.task.stepsize = stepsize;
            state.task.totalRows = totalRows;
            state.task.ymean = ymean;
            state.task.tk = tk;

            for (uint32_t i = 0; i < dimension; i++)
            {
                // initial values
                state.task.coef(i) = 0;
                state.task.coef_y(i) = 0;
                state.task.xmean(i) = xmean(i);
                state.algo.gradient(i) = 0;
            }
            
            state.task.intercept = ymean;
            state.task.intercept_y = ymean;
        }
    }

    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    state.algo.gradient += - x * (y - state.task.intercept_y);
    
    for (uint32_t i = 0; i < state.task.dimension; i++)
        if (state.task.coef_y(i) != 0)
            for (uint32_t j = 0; j < state.task.dimension; j++)
                state.algo.gradient(j) += x(j) * state.task.coef_y(i) * x(i);

    state.algo.numRows++;

    return state;
}

/**
   @brief Perform Merge transition steps
*/
AnyType gaussian_fista_merge::run (AnyType& args)
{
    FistaState<MutableArrayHandle<double> > state1 = args[0];
    FistaState<MutableArrayHandle<double> > state2 = args[1];

    if (state1.algo.numRows == 0) return state2;
    else if (state2.algo.numRows == 0) return state1;

    state1.algo.gradient += state2.algo.gradient;
    state1.algo.numRows += state2.algo.numRows;

    return state1;
}

/**
   @brief Perform the final computation
*/
AnyType gaussian_fista_final::run (AnyType& args)
{
    FistaState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null
    if (state.algo.numRows == 0) return Null();

    state.algo.gradient = state.algo.gradient / state.task.totalRows
        + state.task.lambda * (1 - state.task.alpha) * state.task.coef_y;

    ColumnVector u = state.task.coef_y - state.task.stepsize * state.algo.gradient;

    double effective_lambda = state.task.lambda * state.task.alpha * state.task.tk;

    // update tk
    double old_tk = state.task.tk;
    state.task.tk = 0.5 * (1 + sqrt(1 + 4*state.task.tk));

    double old_coef;
    state.task.intercept_y = ymean;
    state.task.intercept = ymean;
    for (uint32_t i = 0; i < state.task.dimension; i++)
    {
        old_coef = state.task.coef(i);
        // soft thresholding with respective to effective_lambda
        if (u(i) > effective_lambda)
            state.task.coef(i) = u(i) - effective_lambda;
        else if (u(i) < - effective_lambda)
            state.task.coef(i) = u(i) + effective_lambda;
        else
            state.task.coef(i) = 0;

        // update coef_y
        state.task.coef_y(i) = state.task.coef(i) + (old_tk - 1) *
            (state.task.coef(i) - old_coef) / state.task.tk;

        // update intercept_y and intercept
        state.task.intercept -= state.task.coef(i) * xmean(i);
        state.task.intercept_y -= state.task.coef_y(i) * xmean(i);
    }

    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType
__gaussian_fista_state_diff::run (AnyType& args)
{
    FistaState<ArrayHandle<double> > state1 = args[0];
    FistaState<ArrayHandle<double> > state2 = args[1];

    double diff = 0;    
    int n = state1.task.coef.rows();
    for (uint32_t i = 0; i < n; i++)
        diff += std::abs(state1.task.coef(i) - state2.task.coef(i));

    return diff / n;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
__gaussian_fista_result::run (AnyType& args)
{
    FistaState<ArrayHandle<double> > state = args[0];
    AnyType tuple;
    
    tuple << state.task.intercept << state.task.coef;

    return tuple;
}

}
}
}
