
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_fista.hpp"
#include "state/fista.hpp"

namespace madlib {
namespace modules {
namespace elastic_net {

/*
  The proxy function, in this case it is just the soft thresholding
 */
static ColumnVector proxy (ColumnVector y, ColumnVector gradient_y,
                           double stepsize, double lambda)
{
    ColumnVector x(y.size());
    ColumnVector u = y - stepsize * gradient_y;
    for (uint32_t i = 0; i < y.size(); i++)
    {
        if (u(i) > lambda)
            x(i) = u(i) - lambda;
        else if (u(i) < - lambda)
            x(i) = u(i) + lambda;
        else
            x(i) = 0;
    }
    return x;
}

/**
   @brief Perform FISTA transition step

   It is called for each tuple of (x, y)
*/
AnyType gaussian_fista_transition::run (AnyType& args)
{
    FistaState<MutableArrayHandle<double> > state = args[0];

    // initialize the state if working on the first tuple
    if (state.numRows == 0)
    {
        if (!args[3].isNull())
        {
            FistaState<ArrayHandle<double> > pre_state = args[3];
            state.allocate(*this, pre_state.dimension);
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
            state.lambda = lambda;
            state.alpha = alpha;
            state.stepsize = stepsize;
            state.totalRows = totalRows;
            state.ymean = ymean;
            state.tk = tk;

            for (uint32_t i = 0; i < dimension; i++)
            {
                // initial values
                state.coef(i) = 0;
                state.coef_y(i) = 0;
                state.xmean(i) = xmean(i);
            }
            
            state.intercept = ymean;
            state.intercept_y = ymean;
        }

        state.gradient.setZero();
    }

    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    state.gradient += - (x - state.xmean) * (y - state.intercept_y);
    
    for (uint32_t i = 0; i < state.dimension; i++)
        if (state.coef_y(i) != 0)
            for (uint32_t j = 0; j < state.dimension; j++)
                state.gradient(j) += (x(j) - state.xmean(j)) * state.coef_y(i) * x(i);

    state.numRows++;

    return state;
}

/**
   @brief Perform Merge transition steps
*/
AnyType gaussian_fista_merge::run (AnyType& args)
{
    FistaState<MutableArrayHandle<double> > state1 = args[0];
    FistaState<MutableArrayHandle<double> > state2 = args[1];

    if (state1.numRows == 0)
        return state2;
    else if (state2.numRows == 0)
        return state1;

    state1.gradient += state2.gradient;
    state1.numRows += state2.numRows;

    return state1;
}

/**
   @brief Perform the final computation
*/
AnyType gaussian_fista_final::run (AnyType& args)
{
    FistaState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null
    if (state.numRows == 0) return Null();

    state.gradient = state.gradient / state.totalRows
        + state.lambda * (1 - state.alpha) * state.coef_y;

    ColumnVector u = state.coef_y - state.stepsize * state.gradient;

    double effective_lambda = state.lambda * state.alpha * state.stepsize;

    // update tk
    double old_tk = state.tk;
    state.tk = 0.5 * (1 + sqrt(1 + 4 * old_tk * old_tk));

    double old_coef_i;
    state.intercept_y = state.ymean;
    state.intercept = state.ymean;
    for (uint32_t i = 0; i < state.dimension; i++)
    {
        old_coef_i = state.coef(i);
        // soft thresholding with respective to effective_lambda
        if (u(i) > effective_lambda)
            state.coef(i) = u(i) - effective_lambda;
        else if (u(i) < - effective_lambda)
            state.coef(i) = u(i) + effective_lambda;
        else
            state.coef(i) = 0;
        
        // update coef_y
        state.coef_y(i) = state.coef(i) + (old_tk - 1) *
            (state.coef(i) - old_coef_i) / state.tk;

        // update intercept_y and intercept
        state.intercept -= state.coef(i) * state.xmean(i);
        state.intercept_y -= state.coef_y(i) * state.xmean(i);
    }

    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType __gaussian_fista_state_diff::run (AnyType& args)
{
    FistaState<ArrayHandle<double> > state1 = args[0];
    FistaState<ArrayHandle<double> > state2 = args[1];

    // double diff = 0;    
    // uint32_t n = state1.coef.rows();
    // for (uint32_t i = 0; i < n; i++)
    //     diff += std::abs(state1.coef(i) - state2.coef(i));

    // return diff / n;

    double diff_max = 0;
    uint32_t n = state1.coef.rows();
    for (uint32_t i = 0; i < n; i++)
    {
        double diff = std::abs(state1.coef(i) - state2.coef(i));
        if (diff > diff_max)
            diff_max = diff;
    }

    return diff_max;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType __gaussian_fista_result::run (AnyType& args)
{
    FistaState<ArrayHandle<double> > state = args[0];
    AnyType tuple;
    
    tuple << static_cast<double>(state.intercept)
          << state.coef
          << 0.;

    return tuple;
}

}
}
}
