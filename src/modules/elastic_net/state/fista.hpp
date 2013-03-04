
/**
   @file fista.hpp

   This file contains the definitions for FISTA state of
   user-defined aggreagtes.
*/

#ifndef MADLIB_MODULES_ELASIC_NET_STATE_FISTA_
#define MADLIB_MODULES_ELASIC_NET_STATE_FISTA_

#include "dbconnector/dbconnector.hpp"
// #include "convex/type/model.hpp"

namespace madlib {
namespace modules {
namespace elastic_net {

using namespace madlib::dbal::eigen_integration;

template <class Handle>
class FistaState
{
    template <class OtherHandle> friend class FistaState;
    
  public:
    FistaState (const AnyType& inArray):
        mStorage(inArray.getAs<Handle>())
    {
        rebind();
    }

    /**
       @brief Convert to backend representation

       Define this function so that we can use State in the argument
       list and as a return type.
    */
    inline operator AnyType () const
    {
        return mStorage;
    }

    /**
       @brief Allocating the needed memory blocks
    */
    inline void allocate (const Allocator& inAllocator,
                          uint32_t inDimension)
    {
        mStorage = inAllocator.allocateArray<double,
                                             dbal::AggregateContext,
                                             dbal::DoZero,
                                             dbal::ThrowBadAlloc>
            (arraySize(inDimension));
        task.dimension.rebind(&mStorage[0]);
        task.dimension = inDimension;
        rebind();
    }

    /**
       @brief We need to support assigning the previous state
    */
    template <class OtherHandle>
    FistaState& operator= (const FistaState<OtherHandle>& inOtherState)
    {
        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
       @brief Total size of the state object
    */
    static inline uint32_t arraySize (const uint32_t inDimension)
    {
        return 10 + 4 * inDimension;
    }

  protected:
    void rebind ()
    {
        task.dimension.rebind(&mStorage[0]);
        task.lambda.rebind(&mStorage[1]);
        task.alpha.rebind(&mStorage[2]);
        task.stepsize.rebind(&mStorage[3]);
        task.totalRows.rebind(&mStorage[4]);
        task.intercept.rebind(&mStorage[5]);
        task.intercept_y.rebind(&mStorage[6]);
        task.coef.rebind(&mStorage[7], task.dimension);
        task.coef_y.rebind(&mStorage[7 + task.dimension], task.dimension);
        task.xmean.rebind(&mStorage[7 + 2 * task.dimension], task.dimension);
        task.ymean.rebind(&mStorage[7 + 3 * task.dimension]);
        task.tk.rebind(&mStorage[8 + 3 * task.dimension]);

        algo.numRows.rebind(&mStorage[9 + 3 * task.dimension]);
        algo.gradient.rebind(&mStorage[10 + 3 * task.dimension], task.dimension);
    }

    Handle mStorage;

  public:
    struct TaskState
    {
        typename HandleTraits<Handle>::ReferenceToUInt32 dimension;
        typename HandleTraits<Handle>::ReferenceToDouble lambda;
        typename HandleTraits<Handle>::ReferenceToDouble alpha;
        typename HandleTraits<Handle>::ReferenceToDouble stepsize;
        typename HandleTraits<Handle>::ReferenceToUInt64 totalRows;
        typename HandleTraits<Handle>::ReferenceToDouble intercept;
        typename HandleTraits<Handle>::ReferenceToDouble intercept_y;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef_y;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap xmean;
        typename HandleTraits<Handle>::ReferenceToDouble ymean;
    } task;

    struct AlgoState
    {
        typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap gradient;
    } algo;
};

}
}
}

#endif
