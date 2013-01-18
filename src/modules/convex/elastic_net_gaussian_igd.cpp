
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_igd.hpp"

#include "task/ols.hpp"
#include "task/elastic_net.hpp"
#include "algo/igd.hpp"
#include "algo/regularized_igd.hpp"
#include "type/tuple.hpp"
#include "type/model.hpp"
#include "type/state.hpp"
#include "algo/loss.hpp"

namespace madlib {
namespace modules {
namespace convex {

// This 4 classes contain public static methods that can be called
typedef ElasticNet<GLMModel > GLMENRegularizer;

typedef ENRegularizedIGD<ENRegularizedGLMIGDState<MutableArrayHandle<double> >,
                         OLS<GLMModel, GLMTuple >,
                         GLMENRegularizer > OLSENRegularizedIGDAlgorithm;

typedef IGD<ENRegularizedGLMIGDState<MutableArrayHandle<double> >, 
            ENRegularizedGLMIGDState<ArrayHandle<double> >,
            OLS<GLMModel, GLMTuple > > OLSIGDAlgorithm;

typedef Loss<ENRegularizedGLMIGDState<MutableArrayHandle<double> >, 
             ENRegularizedGLMIGDState<ArrayHandle<double> >,
             OLS<GLMModel, GLMTuple > > OLSLossAlgorithm;

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
    ENRegularizedGLMIGDState<MutableArrayHandle<double> > state = args[0];
    
    // initialize the state if working on the first tuple
    if (state.algo.numRows == 0)
    {
        if (!args[3].isNull())
        {
            ENRegularizedGLMIGDState<ArrayHandle<double> > pre_state = args[3];
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
        state.reset();
    }

    // // tuple
    // using madlib::dbal::eigen_integration::MappedColumnVector;
    // GLMTuple tuple;
    // MappedColumnVector indVar = args[1].getAs<MappedColumnVector>();
    // tuple.indVar.rebind(indVar.memoryHandle(), indVar.size());
    // tuple.depVar = args[2].getAs<double>();

    // // Now do the transition step
    // OLSENRegularizedIGDAlgorithm::transition(state, tuple);
    // OLSLossAlgorithm::transition(state, tuple);
    // state.algo.numRows ++;

    using madlib::dbal::eigen_integration::MappedColumnVector;
    MappedColumnVector x = args[1].getAs<MappedColumnVector>();
    double y = args[2].getAs<double>();

    // MSE part & ridge part gradients
    double wv = state.task.stepsize * (dot(state.algo.incrModel, x) - y) / state.task.totalRows;
    double ridge = state.task.stepsize * (1 - state.task.alpha) * state.task.lambda / state.task.totalRows;
    int n = state.task.dimension;
    for (Index i = 0; i < n - 1; i++)
        state.algo.incrModel(i) -= x(i) * wv + ridge * state.algo.incrModel(i);

    // update intercept, which is not regularized
    state.algo.incrModel(n - 1) -= state.task.stepsize * wv;

    // LASSO part update, soft threshold
    double lasso = state.task.stepsize * state.task.alpha * state.task.lambda / state.task.totalRows;
    for (Index i = 0; i < n - 1; i++)
    {
        if (state.algo.incrModel(i) > lasso)
            state.algo.incrModel(i) -= lasso;
        else if (state.algo.incrModel(i) < -lasso)
            state.algo.incrModel(i) += lasso;
        else
            state.algo.incrModel(i) = 0;
    }
    
    // compute loss (MSE) value using model (not incrModel)
    wv = dot(state.task.model, x) - y;
    state.algo.loss += 0.5 * wv * wv;

    state.algo.numRows++;

    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
gaussian_igd_merge::run (AnyType& args)
{
    ENRegularizedGLMIGDState<MutableArrayHandle<double> > stateLeft = args[0];
    ENRegularizedGLMIGDState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.algo.numRows == 0) { return stateRight; }
    else if (stateRight.algo.numRows == 0) { return stateLeft; }

    // Merge states together
    OLSIGDAlgorithm::merge(stateLeft, stateRight);
    OLSLossAlgorithm::merge(stateLeft, stateRight);
    
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
gaussian_igd_final::run (AnyType& args)
{
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    ENRegularizedGLMIGDState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.algo.numRows == 0) return Null(); 

    // finalizing
    //OLSIGDAlgorithm::final(state);

    state.task.model = state.algo.incrModel;

    return state;
}

// ------------------------------------------------------------------------

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType
internal_gaussian_igd_state_diff::run (AnyType& args)
{
    ENRegularizedGLMIGDState<ArrayHandle<double> > state1 = args[0];
    ENRegularizedGLMIGDState<ArrayHandle<double> > state2 = args[1];

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
internal_gaussian_igd_result::run (AnyType& args)
{
    ENRegularizedGLMIGDState<ArrayHandle<double> > state = args[0];
    double norm = 0;
    
    // the model values used here is the new values
    // where the values of model used to compute loss
    // in the aggregate are the old ones.
    // But loss difference is tiny, so this does not matter
    for (Index i = 0; i < state.task.model.rows() - 1; i++) {
        double m = state.task.model(i);
        norm += state.task.alpha * std::abs(m) + (1 - state.task.alpha) * m * m * 0.5;
    }
    norm *= (state.task.lambda * state.task.totalRows);
        
    AnyType tuple;
    tuple << state.task.model
          << static_cast<double>(state.algo.loss) + norm;

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
 
} // namespace convex 
} // namespace modules
} // namespace convex
