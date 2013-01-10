
#include "dbconnector/dbconnector.hpp"
#include "elastic_net_gaussian_igd.hpp"

namespace madlib {
namespace modules {
namespace convex {
            
// This 4 classes contain public static methods that can be called
typedef L1<GLMModel > GLML1Regularizer;

typedef RegularizedIGD<RegularizedGLMIGDState<MutableArrayHandle<double> >,
                       OLS<GLMModel, GLMTuple >,
                       GLML1Regularizer > OLSL1RegularizedIGDAlgorithm;
            
typedef IGD<RegularizedGLMIGDState<MutableArrayHandle<double> >, 
            RegularizedGLMIGDState<ArrayHandle<double> >,
            OLS<GLMModel, GLMTuple > > OLSIGDAlgorithm;
            
typedef Loss<RegularizedGLMIGDState<MutableArrayHandle<double> >, 
             RegularizedGLMIGDState<ArrayHandle<double> >,
             OLS<GLMModel, GLMTuple > > OLSLossAlgorithm;

/**
   @brief Perform IGD transition step

   It is called for each tuple.
*/
AnyType
gaussian_igd_transition::run ()
    
}
}
}
