
#include <dbconnector/dbconnector.hpp>
#include "elastic_net_gaussian_cd.hpp"

// #include "task/ols.hpp"
// #include "task/elastic_net.hpp"
// #include "algo/igd.hpp"
// #include "algo/regularized_igd.hpp"
// #include "type/tuple.hpp"
// #include "type/model.hpp"
// #include "type/state.hpp"
// #include "algo/loss.hpp"
#include "state/cd.hpp"

#include <ctime>
#include <cstdlib>

namespace madlib {
namespace modules {
namespace elastic_net {

// This 4 classes contain public static methods that can be called
// typedef ElasticNet<GLMModel > GLMENRegularizer;

// typedef ENRegularizedIGD<ENRegularizedGLMIGDState<MutableArrayHandle<double> >,
//                          OLS<GLMModel, GLMTuple >,
//                          GLMENRegularizer > OLSENRegularizedIGDAlgorithm;

// typedef IGD<ENRegularizedGLMIGDState<MutableArrayHandle<double> >, 
//             ENRegularizedGLMIGDState<ArrayHandle<double> >,
//             OLS<GLMModel, GLMTuple > > OLSIGDAlgorithm;

// typedef Loss<ENRegularizedGLMIGDState<MutableArrayHandle<double> >, 
//              ENRegularizedGLMIGDState<ArrayHandle<double> >,
//              OLS<GLMModel, GLMTuple > > OLSLossAlgorithm;

// ------------------------------------------------------------------------

/**
   @brief Perform BCD transition step

   It is called for each tuple.

   The input AnyType has 9 args: state. ind_var, dep_var,
   pre_state, lambda, alpha, dimension, stepsize, totalrows
*/
AnyType
gaussian_cd_transition::run (AnyType& args)
{
    CdState<MutableArrayHandle<double> > state = args[0];
    double lambda = args[4].getAs<double>();
    double alpha = args[5].getAs<double>();
    int dimension = args[6].getAs<int>();
    int total_rows = args[9].getAs<int>();
    
    // initialize the state if working on the first tuple
    if (state.algo.numRows == 0)
    {
        if (!args[3].isNull())
        {
            CdState<ArrayHandle<double> > pre_state = args[3];
            state.allocate(*this, pre_state.task.dimension);
            state = pre_state;
            state.task.lambda = lambda;
        }
        else
        {            
            state.allocate(*this, dimension);
            state.task.lambda = lambda;
            state.task.alpha = alpha;
            state.task.totalRows = total_rows;
            
            MappedColumnVector means = args[7].getAs<MappedColumnVector>();
            MappedColumnVector sq = args[8].getAs<MappedColumnVector>();
            
            // srand ( time(NULL) );
            for (Index i = 0; i < dimension; i++)
            {
                state.task.means(i) = means(i);
                state.task.sq(i) = sq(i);
                state.task.model(i) = 0.; //(rand() * 2 / RAND_MAX - 1) / sq(i);
            }
            state.task.model(dimension - 1) = means(dimension - 1);
        }
        // state.algo.loss = 0;

        // use incrModel to accumulate the changes
        // but the initial values for incrModel are
        // always 0
        for (Index i = 0; i < dimension; i++) state.algo.incrModel(i) = 0;
    }

    // tuple
    using madlib::dbal::eigen_integration::MappedColumnVector;

    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();
    double wv = y - dot(state.task.model, x);

    for (Index i = 0; i < dimension - 1; i++)
        state.algo.incrModel(i) += x(i) * (wv + state.task.model(i) * x(i));

    // state.algo.loss += 0.5 * wv * wv;

    state.algo.numRows++;
    
    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
gaussian_cd_merge::run (AnyType& args)
{
    CdState<MutableArrayHandle<double> > stateLeft = args[0];
    CdState<MutableArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.algo.numRows == 0) return stateRight; 
    else if (stateRight.algo.numRows == 0) return stateLeft;

    stateLeft.algo.incrModel += stateRight.algo.incrModel;
    // stateLeft.algo.loss += stateRight.algo.loss;
    
    // The following numRows update, cannot be put above, because the model
    // averaging depends on their original values
    stateLeft.algo.numRows += stateRight.algo.numRows;

    return stateLeft;
}

// ------------------------------------------------------------------------

/**
 * @brief Perform the final step
 */
AnyType
gaussian_cd_final::run (AnyType& args)
{
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    CdState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.algo.numRows == 0) return Null(); 

    double la = state.task.lambda * state.task.alpha;
    double shrink = state.task.lambda * (1 - state.task.alpha) / state.task.totalRows;
    int dimension = state.task.dimension;

    for (Index i = 0; i < dimension - 1; i++)
    {
        if (state.algo.incrModel(i) > la) {
            state.task.model(i) = (state.algo.incrModel(i) - la) /
                (state.task.totalRows * (state.task.sq(i) + shrink));
        } else if (state.algo.incrModel(i) < -la) {
            state.task.model(i) = (state.algo.incrModel(i) + la) /
                (state.task.totalRows * (state.task.sq(i) + shrink));
        } else
            state.task.model(i) = 0;
    }
    
    double intercept = state.task.means(dimension - 1);
    for (Index i = 0; i < dimension - 1; i++) intercept -= state.task.model(i) * state.task.means(i);
    state.task.model(dimension - 1) = intercept;
    
    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType
__gaussian_cd_state_diff::run (AnyType& args)
{
    CdState<ArrayHandle<double> > state1 = args[0];
    CdState<ArrayHandle<double> > state2 = args[1];

    double diff = 0;    
    Index i;
    int n = state1.task.model.rows();
    for (i = 0; i < n; i++)
    {
        diff += std::abs(state1.task.model(i) - state2.task.model(i));
    }

    return diff / n;

    // return std::abs((state1.algo.loss - state2.algo.loss) / state2.algo.loss);
}

// ------------------------------------------------------------------------

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
__gaussian_cd_result::run (AnyType& args)
{
    CdState<ArrayHandle<double> > state = args[0];
    // double norm = 0;

    // for (Index i = 0; i < state.task.model.rows() - 1; i ++) {
    //     double m = state.task.model(i);
    //     norm += state.task.alpha * std::abs(m) + (1 - state.task.alpha) * m * m * 0.5;
    // }
    // norm *= state.task.lambda;
        
    AnyType tuple;
    tuple << state.task.model << 0.;
    // << static_cast<double>(state.algo.loss) + norm; 

    return tuple;
}
 
} // namespace elastic_net
} // namespace modules
} // namespace madlib
