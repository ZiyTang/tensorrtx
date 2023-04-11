cmake_minimum_required(VERSION 3.0)

project(yolov5) # 1
set(OpenCV_DIR "D:/CppLibrary/opencv/build")  #2
set(TRT_DIR "D:/CppLibrary/TensorRT-8.2.5.1")  #3

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0")
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}/include)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}/lib)

add_executable(yolov5 
    ${PROJECT_SOURCE_DIR}/yolov5.cpp 
    ${PROJECT_SOURCE_DIR}/preprocess.cu 
    ${PROJECT_SOURCE_DIR}/preprocess.h 
    ${PROJECT_SOURCE_DIR}/yololayer.cu 
    ${PROJECT_SOURCE_DIR}/yololayer.h
    ${PROJECT_SOURCE_DIR}/segmentlayer.cu 
    ${PROJECT_SOURCE_DIR}/segmentlayer.h
    ${PROJECT_SOURCE_DIR}/radianlayer.cu 
    ${PROJECT_SOURCE_DIR}/radianlayer.h)   #4

target_link_libraries(yolov5 "nvinfer" "nvinfer_plugin")   #5
target_link_libraries(yolov5 ${OpenCV_LIBS})          #6
target_link_libraries(yolov5 ${CUDA_LIBRARIES})   #7
target_link_libraries(yolov5 Threads::Threads)       #8

add_executable(yolov5_seg 
    ${PROJECT_SOURCE_DIR}/yolov5_seg.cpp 
    ${PROJECT_SOURCE_DIR}/preprocess.cu 
    ${PROJECT_SOURCE_DIR}/preprocess.h 
    ${PROJECT_SOURCE_DIR}/yololayer.cu 
    ${PROJECT_SOURCE_DIR}/yololayer.h
    ${PROJECT_SOURCE_DIR}/segmentlayer.cu 
    ${PROJECT_SOURCE_DIR}/segmentlayer.h
    ${PROJECT_SOURCE_DIR}/radianlayer.cu 
    ${PROJECT_SOURCE_DIR}/radianlayer.h)   #4

target_link_libraries(yolov5_seg "nvinfer" "nvinfer_plugin")   #5
target_link_libraries(yolov5_seg ${OpenCV_LIBS})          #6
target_link_libraries(yolov5_seg ${CUDA_LIBRARIES})   #7
target_link_libraries(yolov5_seg Threads::Threads)       #8

add_executable(yolov5_obb 
    ${PROJECT_SOURCE_DIR}/yolov5_obb.cpp 
    ${PROJECT_SOURCE_DIR}/preprocess.cu 
    ${PROJECT_SOURCE_DIR}/preprocess.h 
    ${PROJECT_SOURCE_DIR}/yololayer.cu 
    ${PROJECT_SOURCE_DIR}/yololayer.h
    ${PROJECT_SOURCE_DIR}/segmentlayer.cu 
    ${PROJECT_SOURCE_DIR}/segmentlayer.h
    ${PROJECT_SOURCE_DIR}/radianlayer.cu 
    ${PROJECT_SOURCE_DIR}/radianlayer.h)   #4

target_link_libraries(yolov5_obb "nvinfer" "nvinfer_plugin")   #5
target_link_libraries(yolov5_obb ${OpenCV_LIBS})          #6
target_link_libraries(yolov5_obb ${CUDA_LIBRARIES})   #7
target_link_libraries(yolov5_obb Threads::Threads)       #8


