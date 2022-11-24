cmake_minimum_required(VERSION 3.23)
project(opencvLearn)
set(CMAKE_CXX_STANDARD 17)

# Enable testing
include(CTest)
enable_testing()

# Add openCV library
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
message(DEBUG "${OPENCV_LIBS}")
message(STATUS "OpenCV libs at: ${OpenCV_LIBS}")

add_executable(opencvLearn main.cpp)
target_link_libraries(opencvLearn ${OpenCV_LIBS})