#pragma once

#include <tuple>

#include <opencv2/core.hpp>

class GarborKernel {
private:
    // the dimension of the kernel is kernelSize*kernelSize
    int m_kernelSize = 0;

    // the x component of x
    cv::Mat m_xx;
    // the y component of x
    cv::Mat m_xy;

public:
    // the dimension of the kernel is kernelSize*kernelSize
    void init(int kernelSize);

    std::tuple<cv::Mat/*Re*/,cv::Mat/*Im*/>
    getKernel(
        float sigma,
        float kx,    // the x component of k
        float ky     // the y component of k
    ) const;

    GarborKernel() noexcept {}
    explicit GarborKernel(int kernelSize) noexcept {init(kernelSize);}

};


// Compute the convolution of source matrix and a kernel
// at a specific point. 
// Only support one-channel matrices!
class Convolution {
private:
    // Padded source matrix
    cv::Mat m_src;

    int m_maxKernelRows = 0;
    int m_maxKernelCols = 0;

    int m_origWidth = 0;
    int m_origHeight = 0;

public:

    // Will copy the data of src into an internal Mat in this class, and
    // this internel Mat will be used in other functions of this class.
    // So you do not need to call Mat::clone() when passing src.
    void init(
        const cv::Mat &src,
        int maxKernelRows,  // the max number of rows of kernels
        int maxKernelCols   // the max number of cols of kernels
    );


    // Compute the convolution at a specific point in source matrix. 
    float calcConv(
        const cv::Mat &kernel,
        int x,
        int y
    ) const;

    Convolution() noexcept {}

    // Will copy the data of src into an internel Mat in this class, and
    // this internel Mat will be used in other functions of this class.
    // So you do not need to call Mat::clone() when passing src.
    Convolution(
        const cv::Mat &src,
        int maxKernelRows,  // the max number of rows of kernels
        int maxKernelCols   // the max number of cols of kernels
    ) noexcept 
    {init(src, maxKernelRows, maxKernelCols);}
};