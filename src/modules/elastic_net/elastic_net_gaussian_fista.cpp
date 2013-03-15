
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
    for (int i = 0; i < y.size(); i++)
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
            uint32_t dimension = args[6].getAs<uint32_t>();
            MappedColumnVector xmean = args[7].getAs<MappedColumnVector>();
            double ymean = args[8].getAs<double>();
            double tk = args[9].getAs<double>();
            int totalRows = args[10].getAs<int>();
            
            state.allocate(*this, dimension);
            state.lambda = lambda;
            state.alpha = alpha;
            state.totalRows = totalRows;
            state.ymean = ymean;
            state.tk = tk;
            state.backtracking = 0; // the first iteration is always non-backtracking
            state.L0 = args[11].getAs<double>();
            state.eta = args[12].getAs<double>();

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

        if (state.backtracking == 0)
            state.gradient.setZero();
        else
        {
            state.fn = 0;
            if (state.backtracking == 1) state.Qfn = 0;
        }
    }

    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    if (state.backtracking == 0)
    {
        state.gradient += - (x - state.xmean) * (y - state.intercept_y);
        
        for (uint32_t i = 0; i < state.dimension; i++)
            if (state.coef_y(i) != 0)
                for (uint32_t j = 0; j < state.dimension; j++)
                    state.gradient(j) += (x(j) - state.xmean(j)) * state.coef_y(i) * x(i);
    }
    // during backtracking, always use b_coef and b_intercept
    else 
    {
        double r = y - state.b_intercept - dot(state.b_coef, x);
        state.fn += r * r * 0.5;
        // Qfn only need to be calculated once in each backtracking
        if (state.backtracking == 1)
        {
            r = y - state.intercept_y - dot(state.coef_y, x);
            state.Qfn += r * r * 0.5;
        }
    }

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

    if (state1.backtracking == 0)
        state1.gradient += state2.gradient;
    else
    {
        state1.fn += state2.fn;

        // Qfn only need to be calculated once in each backtracking
        if (state1.backtracking == 1)
            state1.Qfn += state2.Qfn;
    }
    
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

    if (state.backtracking == 0) 
    {
        state.gradient = state.gradient / state.totalRows
            + state.lambda * (1 - state.alpha) * state.coef_y;

        // compute the first set of coef for backtracking
        state.stepsize = 1. / state.L0;
        double effective_lambda = state.lambda * state.alpha * state.stepsize;
        state.b_coef = proxy(state.coef_y, state.gradient, state.stepsize,
                           effective_lambda);
        state.b_intercept = state.ymean - dot(state.b_coef, state.xmean);

        state.backtracking = 1; // will do backtracking
    }
    else
    {
        // finish computing fn and Qfn if needed
        state.fn = state.fn / state.totalRows + 0.5 * state.lambda * (1 - state.alpha)
            * dot(state.b_coef, state.b_coef);
        
        if (state.backtracking == 1)
            state.Qfn = state.Qfn / state.totalRows + 0.5 * state.lambda * (1 - state.alpha)
                * dot(state.coef_y, state.coef_y);

        ColumnVector r = state.b_coef - state.coef_y;
        double extra_Q = dot(r, state.gradient) + 0.5 * dot(r, r) / state.stepsize;
        
        if (state.fn <= state.Qfn + extra_Q) { // use last backtracking coef
            // update coef and intercept
            ColumnVector old_coef = state.coef;
            state.coef = state.b_coef;
            state.intercept = state.b_intercept;

            // update tk
            double old_tk = state.tk;
            state.tk = 0.5 * (1 + sqrt(1 + 4 * old_tk * old_tk));

            // update coef_y and intercept_y
            state.coef_y = state.coef + (old_tk - 1) * (state.coef - old_coef)
                / state.tk;
            state.intercept_y = state.ymean - dot(state.xmean, state.coef_y);
            
            state.backtracking = 0; // stop backtracking
        }
        else
        {
            state.stepsize = state.stepsize / state.eta;
            double effective_lambda = state.lambda * state.alpha * state.stepsize;
            state.b_coef = proxy(state.coef_y, state.gradient, state.stepsize,
                                 effective_lambda);
            state.b_intercept = state.ymean - dot(state.b_coef, state.xmean);

            state.backtracking++;
        }
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

    // during backtracking, do not comprae the coefficients
    // of two consecutive states
    if (state2.backtracking > 0) return 1e6;
    
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
          << static_cast<double>(state.lambda);

    return tuple;
}

}
}
}
