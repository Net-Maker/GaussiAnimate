cmake_minimum_required(VERSION 3.16)
project(gaussianimate_curve_skeleton LANGUAGES CXX)

find_package(CGAL REQUIRED)
find_package(Eigen3 3.2 REQUIRED)

add_executable(curve_skeleton main.cpp)
target_compile_features(curve_skeleton PRIVATE cxx_std_17)
target_compile_definitions(curve_skeleton PRIVATE CGAL_EIGEN3_ENABLED)
target_link_libraries(curve_skeleton PRIVATE CGAL::CGAL)
if(TARGET Eigen3::Eigen)
    target_link_libraries(curve_skeleton PRIVATE Eigen3::Eigen)
else()
    target_include_directories(curve_skeleton PRIVATE "${EIGEN3_INCLUDE_DIR}")
endif()

install(TARGETS curve_skeleton RUNTIME DESTINATION bin)
