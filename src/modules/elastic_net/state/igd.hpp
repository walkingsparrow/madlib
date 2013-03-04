/**
   @file igd.hpp

   This file contains the definitions for IGD state of
   user-defined aggregates
*/

#ifndef MADLIB_MODULES_ELASIC_NET_STATE_IGD_
#define MADLIB_MODULES_ELASIC_NET_STATE_IGD_

#include "dbconnector/dbconnector.hpp"

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
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(
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
        return 7 + 3 * inDimension;
    }

  protected:
    void rebind ()
    {
        task.dimension.rebind(&mStorage[0]);
        task.stepsize.rebind(&mStorage[1]);
        task.lambda.rebind(&mStorage[2]);
        task.alpha.rebind(&mStorage[3]);
        task.totalRows.rebind(&mStorage[4]);
        task.model.rebind(&mStorage[5], task.dimension);

        algo.numRows.rebind(&mStorage[5 + task.dimension]);
        algo.loss.rebind(&mStorage[6 + task.dimension]);
        algo.incrModel.rebind(&mStorage[7 + task.dimension], task.dimension);
        algo.gradient.rebind(&mStorage[7 + 2 * task.dimension], task.dimension);
    }

    Handle mStorage;

  public:
    struct TaskState
    {
        typename HandleTraits<Handle>::ReferenceToUInt32 dimension;
        typename HandleTraits<Handle>::ReferenceToDouble stepsize;
        typename HandleTraits<Handle>::ReferenceToDouble lambda;
        typename HandleTraits<Handle>::ReferenceToDouble alpha;
        typename HandleTraits<Handle>::ReferenceToUInt64 totalRows;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap model;
    } task;

    struct AlgoState
    {
        typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
        typename HandleTraits<Handle>::ReferenceToDouble loss;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap incrModel;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap gradient;
    } algo;
};

}
}
}

#endif
