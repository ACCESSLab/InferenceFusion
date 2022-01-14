cmake_minimum_required(VERSION 3.10.2)
project(mAP)

set(CMAKE_CXX_STANDARD 17)
set(Boost_USE_STATIC_LIBS   ON)

#Set Linker flags
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc  -static-libstdc++")


find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(
        ${Boost_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        include)


add_executable(mAP main.cpp include/FileParser.h include/EvaluationMetrics.h)
target_link_libraries(mAP
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        -pthread
        ${Boost_LIBRARIES}
        )

option(BUILD_MAP_TEST "will download gtest and test TEST_MAP.cpp code" OFF)

if (BUILD_MAP_TEST)
    include(cmake/CPM.cmake)
    CPMAddPackage(
            NAME googletest
            GITHUB_REPOSITORY google/googletest
            GIT_TAG release-1.10.0
            VERSION 1.10.0
            OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
    )

    # ---- Create binary ----

    add_executable(map_test TEST_MAP.cpp)
    target_link_libraries(map_test
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_SYSTEM_LIBRARY}
            -pthread
            gtest gtest_main gmock)
    target_compile_features(map_test PRIVATE cxx_std_17)

    # ---- Enable testing ----

    enable_testing()
    add_test(map_test map_test)
endif ()
