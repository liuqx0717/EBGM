cmake_minimum_required(VERSION 3.0.0)
project(EBGM)

set(OPENCV_LIB_PATH "/usr/lib64/" CACHE PATH "The directory containing OpenCV binaries")
set(OPENCV_INCLUDE_PATH "/usr/include/" CACHE PATH "The include directory of OpenCV")
set(BOOST_LIB_PATH "/usr/lib64/" CACHE PATH "The directory containing Boost binaries")
set(BOOST_INCLUDE_PATH "/usr/include/" CACHE PATH "The include directory of Boost")
set(EIGEN_INCLUDE_PATH "/usr/include/eigen3/" CACHE PATH "The include directory of Eigen")


find_package(OpenMP REQUIRED)


find_library(LIBOPENCV_CORE_PATH
    NAMES "opencv_core"
    PATHS "${OPENCV_LIB_PATH}"
)
find_library(LIBOPENCV_HIGHGUI_PATH
    NAMES "opencv_highgui"
    PATHS "${OPENCV_LIB_PATH}"
)
find_library(LIBOPENCV_IMGCODECS_PATH
    NAMES "opencv_imgcodecs"
    PATHS "${OPENCV_LIB_PATH}"
)
find_library(LIBOPENCV_IMGPROC_PATH
    NAMES "opencv_imgproc"
    PATHS "${OPENCV_LIB_PATH}"
)

add_library(opencv INTERFACE)
target_include_directories(opencv
    INTERFACE "${OPENCV_INCLUDE_PATH}"
)
target_link_libraries(opencv
    INTERFACE "${LIBOPENCV_CORE_PATH}"
    INTERFACE "${LIBOPENCV_HIGHGUI_PATH}"
    INTERFACE "${LIBOPENCV_IMGCODECS_PATH}"
    INTERFACE "${LIBOPENCV_IMGPROC_PATH}"
)



find_library(LIBBOOST_FILESYSTEM_PATH
    NAMES "boost_filesystem"
    PATHS "${BOOST_LIB_PATH}"
)
find_library(LIBBOOST_SERIALIZATION_PATH
    NAMES "boost_serialization"
    PATHS "${BOOST_LIB_PATH}"
)
find_library(LIBBOOST_SYSTEM_PATH
    NAMES "boost_system"
    PATHS "${BOOST_LIB_PATH}"
)

add_library(boost INTERFACE)
target_include_directories(boost
    INTERFACE "${BOOST_INCLUDE_PATH}"
)
target_link_libraries(boost
    INTERFACE "${LIBBOOST_FILESYSTEM_PATH}"
    INTERFACE "${LIBBOOST_SERIALIZATION_PATH}"
    INTERFACE "${LIBBOOST_SYSTEM_PATH}"
)




add_library(eigen INTERFACE)
target_include_directories(eigen
    INTERFACE "${EIGEN_INCLUDE_PATH}"
)





add_executable(ebgm
    "EBGM/ebgm.cpp"
    "EBGM/kernels.cpp"
    "EBGM/cvutils.cpp"
    "EBGM/alg.cpp"
    "EBGM/utils.cpp"
    "EBGM/tests.cpp"
    "EBGM/gui.cpp"
    "EBGM/iofiles.cpp"
)
target_link_libraries(ebgm
    PRIVATE opencv
    PRIVATE eigen
    PRIVATE boost
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE OpenMP::OpenMP_C
)


