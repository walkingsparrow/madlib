#set(PORT "PostgreSQL")
#string(TOUPPER ${PORT} PORT_UC)
#set(IN_PORT_VERSION "")
#string(REPLACE "." "_" PORT_VERSION_UNDERSCORE "${IN_PORT_VERSION}")
#set(DBMS_UC "${PORT_UC}_${PORT_VERSION_UNDERSCORE}")
#include_directories(SYSTEM
#        ${${DBMS_UC}_SERVER_INCLUDE_DIR}


find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_definitions(-DBOOST_TEST_DYN_LINK)
link_directories("/data/home/gpdbchina/hai/local/lib/"
"/data/home/gpdbchina/hai/local/lib/postgresql")
include_directories(
    "/data/home/gpdbchina/hai/local/include/postgresql/server"
    "../../../build/third_party/src/EP_boost"
    "../.."
    "../../ports/postgres"
    "../../../build/third_party/src/EP_eigen")

add_executable(t_test_mocks t_test_mocks.cpp)
#enable_testing()
message(status "*********** ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}")
#add_test(t_test_mocks t_test_mocks)
target_link_libraries(t_test_mocks ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
"/data/home/gpdbchina/hai/local/lib/libpq.so"
"/data/home/gpdbchina/hai/madlib-new/build/src/ports/postgres/9.2/lib/libmadlib.so")
#"libpq.so")
