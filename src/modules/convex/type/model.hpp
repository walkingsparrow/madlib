/* ----------------------------------------------------------------------- *//**
 *
 * @file model.hpp
 *
 * This file contians classes of coefficients (or model), which usually has
 * fields that maps to transition states for user-defined aggregates.
 *
 *//* ----------------------------------------------------------------------- */

#ifndef MADLIB_MODULES_CONVEX_TYPE_MODEL_HPP_
#define MADLIB_MODULES_CONVEX_TYPE_MODEL_HPP_

#include <fstream> /////////

namespace madlib {

namespace modules {

namespace convex {

// The necessity of this wrapper is to allow classes in algo/ and task/ to
// have a type that they can template over
template <class Handle>
struct LMFModel {
    typename HandleTraits<Handle>::MatrixTransparentHandleMap matrixU;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap matrixV;

    /**
     * @brief Space needed.
     *
     * Extra information besides the values in the matrix, like dimension is
     * necessary for a matrix, so that it can perform operations. These are
     * stored in the HandleMap.
     */
    static inline uint32_t arraySize(const uint16_t inRowDim, 
            const uint16_t inColDim, const uint16_t inMaxRank) {
        return (inRowDim + inColDim) * inMaxRank;
    }

    /**
     * @brief Initialize the model randomly with a user-provided scale factor
     */
    void initialize(const double &inScaleFactor) {
        // using madlib::dbconnector::$database::NativeRandomNumberGenerator
        NativeRandomNumberGenerator rng;
        int i, j, rr;
        double base = rng.min();
        double span = rng.max() - base;
        for (rr = 0; rr < matrixU.cols(); rr ++) {
            for (i = 0; i < matrixU.rows(); i ++) {
                matrixU(i, rr) = inScaleFactor * (rng() - base) / span; 
            }
        }
        for (rr = 0; rr < matrixV.cols(); rr ++) {
            for (j = 0; j < matrixV.rows(); j ++) {
                matrixV(j, rr) = inScaleFactor * (rng() - base) / span; 
            }
        }
    }

    /*
     *  Some operator wrappers for two matrices.
     */
    LMFModel &operator*=(const double &c) {
		std::ofstream of;
		of.open("/Users/qianh1/workspace/tests/madlib-756/log.txt", std::ios::app);
		of << "before * " << matrixU.rows() << " " << matrixU.cols() << " "
		   << matrixV.rows() << " " << matrixV.cols() << std::endl;
		
        matrixU *= c;
        matrixV *= c;

		of << "after * " << matrixU.rows() << " " << matrixU.cols() << " "
		   << matrixV.rows() << " " << matrixV.cols() << std::endl;
		of.close();
		
        return *this;
    }

    template<class OtherHandle>
    LMFModel &operator-=(const LMFModel<OtherHandle> &inOtherModel) {
		std::ofstream of;
		of.open("/Users/qianh1/workspace/tests/madlib-756/log.txt", std::ios::app);
		of << "before - " << matrixU.rows() << " " << matrixU.cols() << " "
		   << matrixV.rows() << " " << matrixV.cols() << std::endl;
		
        matrixU -= inOtherModel.matrixU;
        matrixV -= inOtherModel.matrixV;

		of << "after - " << matrixU.rows() << " " << matrixU.cols() << " "
		   << matrixV.rows() << " " << matrixV.cols() << std::endl;
		of.close();
		
        return *this;
    }

    template<class OtherHandle>
    LMFModel &operator+=(const LMFModel<OtherHandle> &inOtherModel) {
		std::ofstream of;
		of.open("/Users/qianh1/workspace/tests/madlib-756/log.txt", std::ios::app);
		of << "before + " << matrixU.rows() << " " << matrixU.cols() << " "
		   << matrixV.rows() << " " << matrixV.cols() << std::endl;

        matrixU += inOtherModel.matrixU;
        matrixV += inOtherModel.matrixV;

		of << "after + " << matrixU.rows() << " " << matrixU.cols() << " "
		   << matrixV.rows() << " " << matrixV.cols() << std::endl;
		of.close();
		
        return *this;
    }

    template<class OtherHandle>
    LMFModel &operator=(const LMFModel<OtherHandle> &inOtherModel) {
        matrixU = inOtherModel.matrixU;
        matrixV = inOtherModel.matrixV;

        return *this;
    }
};

} // namespace convex

} // namespace modules

} // namespace madlib

#endif

