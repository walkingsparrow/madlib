#set(PORT "PostgreSQL")
#string(TOUPPER ${PORT} PORT_UC)
#set(IN_PORT_VERSION "")
#string(REPLACE "." "_" PORT_VERSION_UNDERSCORE "${IN_PORT_VERSION}")
#set(DBMS_UC "${PORT_UC}_${PORT_VERSION_UNDERSCORE}")
#include_directories(SYSTEM
#        ${${DBMS_UC}_SERVER_INCLUDE_DIR}


find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_definitions(-DBOOST_TEST_DYN_LINK)
link_directories("/opt/local/lib/")
include_directories(system "/opt/local/include/postgresql92/server")
include_directories(
    "/opt/local/include"
    "../.."
    "../../ports/postgres"
    "../../../build/third_party/src/EP_eigen")

#add_executable(t_test_mocks t_test_mocks.cpp ${MAD_HPP_SOURCES})
add_executable(t_test_test t_test_test.cpp ${MAD_HPP_SOURCES})
#enable_testing()
#add_test(t_test_mocks t_test_mocks)
target_link_libraries(t_test_test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}) #"/Users/qianh1/workspace/madlib/build/src/ports/postgres/9.2/lib/libmadlib.so")
