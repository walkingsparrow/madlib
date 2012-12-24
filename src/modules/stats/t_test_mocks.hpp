#ifndef _T_TEST_MOCKS_
#define _T_TEST_MOCKS_

namespace madlib {
namespace modules{
namespace stats{


    
void
updateCorrectedSumOfSquares(double &ioLeftWeight, double &ioLeftSum,
                            double &ioLeftCorrectedSumSquares,
                            double inRightWeight, double inRightSum,
                            double inRightCorrectedSumSquares);

}}}
#endif
