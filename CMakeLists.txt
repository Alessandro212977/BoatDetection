cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project ("BoatDetection")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

add_library(ml_libs source/MLUtils.cpp include/MLUtils.h)
add_library(detection source/DetectionUtils.cpp include/DetectionUtils.h)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/source)
add_executable (train source/training.cpp)
add_executable (detector source/detector.cpp)

target_link_libraries(train ml_libs ${OpenCV_LIBS})
target_link_libraries(detector detection ml_libs ${OpenCV_LIBS})
