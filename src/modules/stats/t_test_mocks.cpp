#define BOOST_TEST_MODULE Mock example
#include <boost/test/included/unit_test.hpp>
//#include <gtest/gtest.h>
//#include <gmock/gmock.h>
//#include "t_test.cpp"

using namespace std;
//using namespace testing;
//using namespace madlib::modules::stats;
using namespace madlib::dbconnector::postgres;
 
namespace madlib {
namespace modules{
namespace stats{
/*class BoostTestAdapter: public EmptyTestEventListener {
 
    virtual void OnTestStart(const TestInfo& {
    }
 
    virtual void OnTestPartResult(
            const TestPartResult& testPartResult) {
        if (testPartResult.failed()) {
            stringstream s;
            s << "Mock test failed (file = '"
              << testPartResult.file_name()
              << "', line = "
              << testPartResult.line_number()
              << "): "
              << testPartResult.summary();
            BOOST_FAIL(s.str());
        }
    }
 
    virtual void OnTestEnd(
            const ::testing::TestInfo&) {
    }
 
};

class TestFixture {
public:
 
    TestFixture() {
 
        InitGoogleMock(
                &boost::unit_test::framework::master_test_suite().argc,
                boost::unit_test::framework::master_test_suite().argv);
        TestEventListeners &listeners =
                UnitTest::GetInstance()->listeners();
        // this removes the default error printer
        delete listeners.Release(
                listeners.default_result_printer());
        listeners.Append(new BoostTestAdapter);
 
    }
 
    ~TestFixture() {
        // nothing to tear down
    }
 
};
BOOST_GLOBAL_FIXTURE(TestFixture);
*/
/* Fixture to represent AnyType input block */
struct AnyTypeFixture {
    AnyTypeFixture() { 
		double dummy = 1;
		input << dummy << dummy << dummy << dummy << dummy;
	}
     ~ AnyTypeFixture() { delete &input; }
    AnyType input;
};

/*template <typename Handle>
class MockTransitionState: public TTestTransitionState<Handle> {
public:
	//MockTransitionState(const AnyType &inArray):TTestTransitionState(const AnyType &inArray){}
	MockTransitionState(const AnyType &inArray){}
	MOCK_CONST_METHOD0_T(ReturnAnyType, Handle());
	inline operator AnyType() const {
		return ReturnAnyType();
	}
};*/

BOOST_FIXTURE_TEST_CASE(test_ttransition, AnyTypeFixture) {
	//using ::testing::Return;

    //MockTransitionState<MutableArrayHandle <double> > mstate = input;
    //double x = args[1].getAs<double>();
	double x = 0;
	double temp = 1;

    //EXPECT_CALL(mstate, ReturnAnyType()).Times(1);

    updateCorrectedSumOfSquares(
        temp, temp, temp,
        1, x, 0);

	//EXPECT_EQ(mstateOut,mstate);
	//EXPECT_EQ(state.numX,2);
	//EXPECT_EQ(state.x_sum,2);
	//EXPECT_EQ(state.correctedX_square_sum,1);
	//EXPECT_EQ(state.numY,1);
	//EXPECT_EQ(state.y_sum,1);
	//EXPECT_EQ(state.correctedY_square_sum,1);

	//return mstate;
}
}
}
}
