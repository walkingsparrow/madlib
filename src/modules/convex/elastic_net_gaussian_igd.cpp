
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_igd.hpp"

namespace madlib {
namespace modules {
namespace convex {

// This 4 classes contain public static methods that can be called
typedef ElasticNet<GLMModel > GLMENRegularizer;

typedef RegularizedIGD<RegularizedGLMIGDState<MutableArrayHandle<double> >,
                       OLS<GLMModel, GLMTuple >,
                       GLMENRegularizer > OLSENRegularizedIGDAlgorithm;

typedef IGD<RegularizedGLMIGDState<MutableArrayHandle<double> >, 
            RegularizedGLMIGDState<ArrayHandle<double> >,
            OLS<GLMModel, GLMTuple > > OLSIGDAlgorithm;

typedef Loss<RegularizedGLMIGDState<MutableArrayHandle<double> >, 
             RegularizedGLMIGDState<ArrayHandle<double> >,
             OLS<GLMModel, GLMTuple > > OLSLossAlgorithm;

/**
   @brief Perform IGD transition step

   It is called for each tuple.

   The input AnyType has 9 args: in_state. ind_var, dep_var,
   pre_state, lambda, alpha, dimension, stepsize, totalrows
*/
AnyType
gaussian_igd_transition::run (AnyType& args)
{
    RegularizedGLMIGDState<MutableArrayHandle<double> > in_state = args[0];
    
    // initialize the state if working on the first tuple
    if (state.algo.numRows == 0)
    {
        if (!args[3].isNull())
        {
            RegularizedGLMIGDState<ArrayHandle<double> > pre_state = args[3];
            in_state.allocate(*this, pre_state.task.dimension);
            in_state = pre_state;
        }
        else
        {
            
        }
    }
}
    
} // namespace convex 
} // namespace modules
} // namespace convex
