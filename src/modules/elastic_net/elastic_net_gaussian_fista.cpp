
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_fista.cpp"
#include "state/fista.hpp"

namespace madlib {
namespace modules {
namespace elastic_net {

/**
   @brief Perform FISTA transition step

   It is called for each tuple of (x, y)
*/
AnyType gaussian_fista_transition::run (Anytype& args)
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
            double lipschitz = args[6].getAs<double>();
            int dimension = args[7].getAs<double>();
            MappedColumnVector xmean = args[8].getAs<MappedColumnVector>();
            double ymean = args[9].getAs<double>();
            double tk = args[10].getAs<double>();
            int totalRows = args[11].getAs<int>();
            
            state.allocate(*this, dimension);
            state.task.lambda = lambda;
            state.task.alpha = alpha;
            state.task.lipschitz = lipschitz;
            state.task.totalRows = totalRows;
            state.task.ymean = ymean;
            state.task.tk = tk;

            for (Index i = 0; i < dimension; i++)
            {
                state.task.coef(i) = 0;
                state.task.xmean(i) = xmean(i);
                state.algo.gradient(i) = 0;
            }
            state.task.intercept = ymean;
        }
    }

    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    for (Index j = 0; j < dimension; j++)
        state.algo.gradient(j) += - x(j) * y;
    
    for (Index i = 0; i < dimension; i++)
        if (state.task.coef(i) != 0)
            for (Index j = 0; j < dimension; j++)
                state.algo.gradient(j) += x(j) * state.task.coef(i) * x(i);

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
}

}
}
}
