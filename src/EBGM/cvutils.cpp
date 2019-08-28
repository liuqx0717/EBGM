#include "cvutils.h"

#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>


#define PI 3.14159265358979323846F

using namespace std;
using namespace cv;
using namespace Eigen;


// Convert real and imaginary parts of complex numbers
// to magnitudes and phases. Phases are within [-0.5pi, 1.5pi).
// The depth of Re and Im must be CV_32F.
std::tuple<cv::Mat/*Magnitude*/,cv::Mat/*Phase*/>
complex2magF(const cv::Mat &Re, const cv::Mat &Im)
{
    assert(CV_TYPE2DEPTH(Re.type()) == CV_32F);
    assert(Re.type() == Im.type());
    assert(Re.size == Im.size);

    Mat mag = Re.mul(Re) + Im.mul(Im);
    cv::pow(mag, 0.5, mag);

    Mat phase = Im/Re;
    int nElements = Re.total();
    float *Re_p = (float*)Re.data;
    float *phase_p = (float*)phase.data;
    auto phase_eigen = Map<ArrayXf>(phase_p, nElements);
    phase_eigen = phase_eigen.atan();
    for(int i=0; i<nElements; i++) {
        if(Re_p[i] < 0.0F){
            phase_p[i] += PI;
        }
    }



    return make_tuple(mag, phase);
}


// mat: two-dimensional, 32FC1
void displayMat32FC1(const cv::Mat &mat)
{
    assert(mat.type() == CV_32FC1);

    double max, min;
    minMaxLoc(mat, &min, &max);

    // In OpenCV, if the image is of floating point type, 
    // then only those pixels can be visualized using imshow, 
    // which have value from 0.0 to 1.0.
    Mat img = 
        ( mat - Mat::ones(mat.rows, mat.cols, CV_32FC1)*(float)min ) /
        (max - min == 0.0 ? 1.0F : (float)(max - min));

    namedWindow("display", WINDOW_AUTOSIZE);
    imshow("display", img);
    waitKey(0);
}


void displayMat(const cv::Mat &mat)
{
    namedWindow("display", WINDOW_AUTOSIZE);
    imshow("display", mat);
    waitKey(0);
}
