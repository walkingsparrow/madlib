/**
   @file cd.hpp

   This file contains the definitions for Conjugate Gradient
   state of user-defined aggregates
*/

#ifndef MADLIB_MODULES_ELASIC_NET_STATE_CD_
#define MADLIB_MODULES_ELASIC_NET_STATE_CD_

#include "dbconnector/dbconnector.hpp"
#include "modules/shared/HandleTraits.hpp"

namespace madlib {
namespace modules {
namespace elastic_net {

using namespace madlib::dbal::eigen_integration;

template <class Handle>
class CdState
{
    template <class OtherHandle> friend class CdState;

  public:
    CdState (const AnyType& inArray):
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
    CdState& operator= (const CdState<OtherHandle>& inOtherState)
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
        return 6 + 5 * inDimension;
    }

  protected:
    void rebind()
    {
        task.dimension.rebind(&mStorage[0]);
        task.lambda.rebind(&mStorage[1]);
        task.alpha.rebind(&mStorage[2]);
        task.totalRows.rebind(&mStorage[3]);
        task.means.rebind(&mStorage[4], task.dimension); // means of x (without intercept) and y
        task.sq.rebind(&mStorage[4 + task.dimension], task.dimension); // mean of square of x (without intercept)
        task.model.rebind(&mStorage[4 + 2 * task.dimension], task.dimension);

        algo.numRows.rebind(&mStorage[4 + 3 * task.dimension]);
        algo.loss.rebind(&mStorage[5 + 3 * task.dimension]);
        algo.incrModel.rebind(&mStorage[6 + 3 * task.dimension], task.dimension);
        algo.gradient.rebind(&mStorage[6 + 4 * task.dimension], task.dimension);
    }

    Handle mStorage;

  public:
    struct TaskState
    {
        typename HandleTraits<Handle>::ReferenceToUInt32 dimension;
        typename HandleTraits<Handle>::ReferenceToDouble lambda;
        typename HandleTraits<Handle>::ReferenceToDouble alpha;
        typename HandleTraits<Handle>::ReferenceToUInt64 totalRows;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap model;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap means;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap sq;
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
