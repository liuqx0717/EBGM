#include "kernels.h"
#include "cvutils.h"

#include <tuple>
//#include <memory>

#include <opencv2/core.hpp>
#include <Eigen/Core>


using namespace std;
using namespace cv;
using namespace Eigen;


void GarborKernel::init(int kernelSize)
{
    assert(kernelSize > 0);

    m_kernelSize = kernelSize;

    m_xx = Mat(m_kernelSize, m_kernelSize, CV_32FC1);

    float *xx_p = (float*)m_xx.data;
    for(int i=0; i<m_kernelSize; i++){
        xx_p[i] =  (float)i - (float)m_kernelSize/2.0F;
    }
    for(int i=1; i<m_kernelSize; i++){
        memcpy(xx_p + i*m_kernelSize, xx_p, sizeof(float)*m_kernelSize);
    }

    m_xy = m_xx.t();

}


// return value: two matrices of size kernelSize*kernelSize
std::tuple<cv::Mat/*Re*/,cv::Mat/*Im*/>
GarborKernel::getKernel(
    float sigma,
    float kx,    // the x component of k
    float ky     // the y component of k
) const
{
    assert(m_kernelSize > 0);

    Mat Re, Im;
    
    float k2 = kx*kx + ky*ky;
    float sigma2 = sigma*sigma;

    Mat coeff = ( m_xx.mul(m_xx) + m_xy.mul(m_xy) ) * ((-0.5F)*k2/sigma2);
    cv::exp(coeff, coeff);
    coeff *= k2/sigma2;

    Re = m_xx*kx + m_xy*ky;
    Im = Re.clone();
    
    int nElements = m_kernelSize*m_kernelSize;
    auto Re_eigen = Map<ArrayXf>((float*)Re.data, nElements);
    auto Im_eigen = Map<ArrayXf>((float*)Im.data, nElements);
    Re_eigen = Re_eigen.cos();
    Im_eigen = Im_eigen.sin();
    //for(int i=0; i<nElements; i++){
        //float *data_p1 = (float*)Re.data;
        //float *data_p2 = (float*)Im.data;
        //data_p1[i] = cosf(data_p1[i]);
        //data_p2[i] = sinf(data_p2[i]);
    //}
    
    Re += Mat::ones(m_kernelSize, m_kernelSize, CV_32FC1) * 
        ( (-1.0F) * expf((-0.5F)*sigma2));

    Re = Re.mul(coeff);
    Im = Im.mul(coeff);

    return make_tuple(Re, Im);
}


// Will copy the data of src into an internel Mat in this class, and
// this internel Mat will be used in other functions of this class.
// So you do not need to call Mat::clone() when passing src.
void Convolution::init(
    const cv::Mat &src,
    int maxKernelRows,  // the max number of rows of kernels
    int maxKernelCols   // the max number of cols of kernels
)
{
    assert(maxKernelRows > 0);
    assert(maxKernelCols > 0);
    assert(CV_TYPE2CHANNELS(src.type()) == 1);

    m_maxKernelRows = maxKernelRows;
    m_maxKernelCols = maxKernelCols;
    m_origWidth = src.cols;
    m_origHeight = src.rows;

    // The lengths of rows have to be "extended".
    // Will pad the following number of zeros before the first elements
    // of all the rows and after the last elements of all the rows.
    int paddingWidthOnRows = m_maxKernelCols / 2U;

    // Same as above, but this is for the lengths of columns.
    int paddingWidthOnCols = m_maxKernelRows / 2U;

    m_src = Mat::zeros(
        src.rows + paddingWidthOnCols*2U,
        src.cols + paddingWidthOnRows*2U,
        src.type()
    );

    // src_sub POINTS TO a sub-area of m_src
    auto src_sub = m_src(Rect(
        paddingWidthOnRows,  // x
        paddingWidthOnCols,  // y
        src.cols,            // width
        src.rows             // height
    ));

    src.copyTo(src_sub);

    //displayMat32FC1(m_src);

}

// Compute the convolution at a specific point in source matrix. 
float Convolution::calcConv(
    const cv::Mat &kernel,
    int x,
    int y
) const
{
    assert(m_maxKernelRows > 0);
    assert(kernel.rows <= m_maxKernelRows);
    assert(kernel.cols <= m_maxKernelCols);
    assert(kernel.type() == m_src.type());
    assert(x >= 0 && x < m_origWidth);
    assert(y >= 0 && y < m_origHeight);

    // src_sub POINTS TO a sub-area of m_src
    auto src_sub = m_src(Rect(
        x + (m_maxKernelCols - (int)kernel.cols)/2U,  // x
        y + (m_maxKernelRows - (int)kernel.rows)/2U,  // y
        kernel.cols,                                     // width
        kernel.rows                                      // height
    ));

    auto mul = src_sub.mul(kernel);

    // Scalar is an OpenCV vector of length 4.
    // cv::sum() calculates the sum of array elements, independently
    // for each channel.
    Scalar mul_sum = cv::sum(mul);
    return mul_sum[0];

}
