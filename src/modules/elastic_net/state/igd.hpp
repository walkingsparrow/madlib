/**
   @file igd.hpp

   This file contains the definitions for IGD state of
   user-defined aggregates
*/

#ifndef MADLIB_MODULES_ELASIC_NET_STATE_IGD_
#define MADLIB_MODULES_ELASIC_NET_STATE_IGD_

#include "dbconnector/dbconnector.hpp"
#include "modules/shared/HandleTraits.hpp"

namespace madlib {
namespace modules {
namespace elastic_net {

using namespace madlib::dbal::eigen_integration;

template <class Handle>
class IgdState
{
    template <class OtherHandle> friend class IgdState;

  public:
    IgdState (const AnyType& inArray):
        mStorage(inArray.getAs<Handle>())
    {
        rebind();
    }

    /**
     * @brief Convert to backend representation
     *
     * We define this function so that we can use State in the
     * argument list and as a return type.
     */
    inline operator AnyType() const {
        return mStorage;
    }

    /**
     * @brief Allocating the incremental gradient state.
     */
    inline void allocate(const Allocator& inAllocator, uint32_t inDimension)
    {
        mStorage = inAllocator.allocateArray<double,
                                             dbal::AggregateContext,
                                             dbal::DoZero,
                                             dbal::ThrowBadAlloc>(
                                                 arraySize(inDimension));

        task.dimension.rebind(&mStorage[0]);
        task.dimension = inDimension;
        rebind();
    }

    /**
     * @brief Support for assigning the previous state
     */
    template <class OtherHandle>
    IgdState &operator= (const IgdState<OtherHandle>& inOtherState)
    {
        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
     * @brief Total size of the state object
     */
    static inline uint32_t arraySize (const uint32_t inDimension)
    {
        return 12 + 4 * inDimension;
    }

  protected:
    void rebind ()
    {
        task.dimension.rebind(&mStorage[0]);
        task.stepsize.rebind(&mStorage[1]);
        task.lambda.rebind(&mStorage[2]);
        task.alpha.rebind(&mStorage[3]);
        task.totalRows.rebind(&mStorage[4]);
        task.intercept.rebind(&mStorage[5]);
        task.ymean.rebind(&mStorage[6]);
        task.xmean.rbind(&mStorage[7], task.dimension);
        task.coef.rebind(&mStorage[7 + task.dimension], task.dimension);

        algo.numRows.rebind(&mStorage[7 + 2 * task.dimension]);
        algo.loss.rebind(&mStorage[8 + 2 * task.dimension]);
        algo.p.rebind(&mStorage[9 + 2 * task.dimension]);
        algo.q.rebind(&mStorage[10 + 2 * task.dimension]);
        algo.incrIntercept.rebind(&mStorage[11 + 2 * task.dimension]);
        algo.incrCoef.rebind(&mStorage[12 + 2 * task.dimension], task.dimension);
        algo.theta.rebind(&mStorage[12 + 3 * task.dimension], task.dimension);
    }

    Handle mStorage;

  public:
    /*
      intercept and coef are updated after each scan of the data

      During the scan of the data, incrIntercept and incrCoef are used for recording
      changes.

      With this setting, other quantities such as loss can be computed using
      intercept and coef during the scan of the data.

      xmean and ymean are used to compute the intercept
     */
    struct TaskState
    {
        typename HandleTraits<Handle>::ReferenceToUInt32 dimension;
        typename HandleTraits<Handle>::ReferenceToDouble stepsize;
        typename HandleTraits<Handle>::ReferenceToDouble lambda; // regularization control
        typename HandleTraits<Handle>::ReferenceToDouble alpha; // elastic net control
        typename HandleTraits<Handle>::ReferenceToUInt64 totalRows;
        typename HandleTraits<Handle>::ReferenceToDouble intercept;
        typename HandleTraits<Handle>::ReferenceToDouble ymean;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap xmean;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;
    } task;

    struct AlgoState
    {
        typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
        typename HandleTraits<Handle>::ReferenceToDouble loss;
        typename HandleTraits<Handle>::ReferenceToDoubl p; // used for mirror truncation
        typename HandleTraits<Handle>::ReferenceToDoubl q; // used for mirror truncation
        typename HandleTraits<Handle>::ReferenceToDouble incrIntercept; 
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap incrCoef;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap theta; // used for mirror truncation
    } algo;
};

}
}
}

#endif
