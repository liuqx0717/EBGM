#pragma once

#include <tuple>

#include <opencv2/core.hpp>

// Convert Mat::type() to depth (e.g. CV_32F)
#define CV_TYPE2DEPTH(type) (type & CV_MAT_DEPTH_MASK)

// Convert Mat::type() to channels (e.g. 1)
#define CV_TYPE2CHANNELS(type) (1 + (type >> CV_CN_SHIFT))



// Convert real and imaginary parts of complex numbers
// to magnitudes and phases
// The depth of Re and Im must be CV_32F.
std::tuple<cv::Mat/*Magnitude*/,cv::Mat/*Phase*/>
complex2magF(const cv::Mat &Re, const cv::Mat &Im);


// mat: two-dimensional, 32FC1
void displayMat32FC1(const cv::Mat &mat);

void displayMat(const cv::Mat &mat);