#ifndef NDEBUG

#include "tests.h"
#include "alg.h"
#include "kernels.h"
#include "utils.h"
#include "cvutils.h"
#include "jet.hpp"
#include "graph.hpp"
#include "points.hpp"
#include "gui.h"
#include "iofiles.h"
#include "ebgm.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <tuple>



#define PI 3.14159265358979323846F


using namespace std;
using namespace cv;
namespace bf = boost::filesystem;
namespace ba = boost::archive;

// test some matrix operations.
void test1()
{
    Mat m1(2,2,CV_32FC1);
    Mat m2(2,2,CV_32FC1);

    float *m1_p = (float*)m1.data;
    float *m2_p = (float*)m2.data;

    m1_p[0] = 1;
    m1_p[1] = 2;
    m1_p[2] = 3;
    m1_p[3] = 4;
    m2_p[0] = 6;
    m2_p[1] = 7;
    m2_p[2] = 8;
    m2_p[3] = 9;

    cout << m1 << "\n";
    cout << m2 << "\n";
    cout << m2/m1 << "\n";
}

// test one Garbor kernel.
void test2()
{
    GarborKernel garbor(200);
    Mat re,im,mag,ph;
    float kn = 0.5F*PI;
    float phim = 2.0F*(0.125F*PI);

    tie(re, im) = garbor.getKernel(2*PI, kn*cosf(phim), kn*sinf(phim));
    tie(mag, ph) = complex2magF(re, im);

    displayMat32FC1(re);
    displayMat32FC1(im);
    displayMat32FC1(mag);
    displayMat32FC1(ph);
}

// test the generation of 40 Gabor kernels, and
// the convolution of one kernel on one image.
void test3()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    for(int i=0; i<40; i++) {
        displayMat32FC1(kernels.re[i]);
    }

    Mat image;

    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32F);
    displayMat32FC1(image);

    Mat convimage(image.rows, image.cols, image.type());

    Convolution conv(image, 101, 101);
    float *conv_p = (float*)convimage.data;
    for(int i=0; i<image.cols; i++){
        for(int j=0; j<image.rows; j++){
            conv_p[j*image.cols + i] = conv.calcConv(kernels.im[0], i, j);
        }
    }

    displayMat32FC1(convimage);

}

// test the calculation of jets.
void test4()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);
    
    Mat image;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);
    //CalcJet<40> calcJet(re, im, image, 101, 101);
    for(int i=0; i<image.cols; i++){
        for(int j=0; j<image.rows; j++){
            auto jet = calcJet.calcJet(i, j);
        }
    }
}

// select a point (left eye), calculate its jet (called jet0).
// Then calculate the jets of the points on the same horizontal
// line. Then calculate the similarity, similarity_with_phase,
// estimated displacement between all these jets and jet0.
void test5()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);
    
    Mat image;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32F);

    const int eye_x = 48;
    const int eye_y = 59;
    const int lWidth = 30;
    const int rWidth = 40;
    const int len = lWidth + rWidth + 1;

    CalcJet<40> calcJet(image, kernels, 101, 101);

    Jet<40> eyeJet = calcJet.calcJet(eye_x, eye_y);

    float dx[len], dy[len];
    float simi[len];
    float simiph[len];
    for(int i = 0; i < len; i++){
        int x, y;
        Jet<40> jet;
        x = eye_x - lWidth + i;
        y = eye_y;

        jet = calcJet.calcJet(x, y);

        tie(dx[i], dy[i]) = displacementWithFocus(eyeJet, jet, 5);
        simi[i] = eyeJet.compare(jet);
        simiph[i] = eyeJet.compareWithPhase(jet, dx[i], dy[i]);
    }

    ofstream file("testjets.csv", ios::trunc);
    for(int i = 0; i < len; i++){
        file << i << ", " << simi[i] << ", " << simiph[i] << ", "
            << dx[i] << ", " << dy[i] << "\n";
    }    
    
}

// test addXXX() and replaceXXX() of Graph and BunchGraph
void test6()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);
    
    Mat image;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);

    Graph<40> graph;
    BunchGraph<40> bunch;

    Jet<40> jet;

    jet = calcJet.calcJet(10, 10);
    graph.addNode(jet);
    jet = calcJet.calcJet(20, 20);
    graph.addNode(jet);
    jet = calcJet.calcJet(30, 30);
    graph.addNode(jet);
    cout << "--------- graph1 ---------\n";
    cout << graph.toString() << "\n";
    bunch.addGraph(graph);
    cout << "--------- bunch1 ---------\n";
    cout << bunch.toString() << "\n";
    graph.clear();

    jet = calcJet.calcJet(10, 10);
    graph.addNode(jet);
    jet = calcJet.calcJet(20, 20);
    graph.addNode(jet);
    jet = calcJet.calcJet(30, 30);
    graph.addNode(jet);
    cout << "--------- graph2 ---------\n";
    cout << graph.toString() << "\n";
    bunch.addGraph(graph);
    cout << "--------- bunch2 ---------\n";
    cout << bunch.toString() << "\n";
    graph.clear();

    jet = calcJet.calcJet(10, 10);
    graph.addNode(jet);
    jet = calcJet.calcJet(30, 50);
    graph.addNode(jet);
    jet = calcJet.calcJet(50, 90);
    graph.addNode(jet);
    cout << "--------- graph3 ---------\n";
    cout << graph.toString() << "\n";
    bunch.addGraph(graph);
    cout << "--------- bunch3 ---------\n";
    cout << bunch.toString() << "\n";
    graph.clear();


    jet = calcJet.calcJet(10, 10);
    graph.addNode(jet);
    jet = calcJet.calcJet(30, 50);
    graph.addNode(jet);
    jet = calcJet.calcJet(50, 90);
    graph.addNode(jet);
    cout << "-------- original --------\n";
    cout << graph.toString() << "\n";
    graph.replaceNode(jet, 2);
    jet = calcJet.calcJet(0, 0);
    graph.replaceNode(jet, 1);
    cout << "-------- modified --------\n";
    cout << graph.toString() << "\n";
    graph.clear();

}

// test Points class
void test7()
{
    Points<int> points;
    points.addPoint({0,0});
    points.addPoint({{0,4}, {8,0}});
    //points.addPoint({8,0});
    points.addPoint({-10,10});
    points.modifyPoint({8,4}, 3);

    points.translate(-2, -2).scale(0, 0, 0.4, 0.4);

    cout << points.toString() << "\n";

    cout << points.toString() << "\n";

}

// test the comparation between graph and graphbunch
void test8() 
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };


    Graph<40> graph0, graph1, graph2;
    BunchGraph<40> bunch;

    int nPoints = points.size();

    showImage = rgbImage.clone();
    for(int i=0; i<points.size(); i++){
        auto point = points.get(i);
        graph0.addNode(calcJet.calcJet(point.x, point.y));

        cv::circle(showImage, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage);

    showImage = rgbImage.clone();
    points.translate(1, 0);
    for(int i=0; i<points.size(); i++){
        auto point = points.get(i);
        graph1.addNode(calcJet.calcJet(point.x, point.y));

        cv::circle(showImage, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage);

    showImage = rgbImage.clone();
    points.translate(-1, 3);
    for(int i=0; i<points.size(); i++){
        auto point = points.get(i);
        graph2.addNode(calcJet.calcJet(point.x, point.y));

        cv::circle(showImage, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage);




    float simi, simiph, sumdisp2;

    bunch.addGraph(graph0);

    simi = bunch.compare(graph0);
    tie(simiph, sumdisp2) = bunch.compareWithPhaseFocus(
        graph0, 5, displacementWithFocus
    );
    cout << "bunch: (graph0); graph: graph0\n";
    cout << "simi = " << simi << "\n";
    cout << "simiph = " << simiph << "\n";
    cout << "sumdisp2 = " << sumdisp2 << "\n\n";

    simi = bunch.compare(graph1);
    tie(simiph, sumdisp2) = bunch.compareWithPhaseFocus(
        graph1, 5, displacementWithFocus
    );
    cout << "bunch: (graph0); graph: graph1\n";
    cout << "simi = " << simi << "\n";
    cout << "simiph = " << simiph << "\n";
    cout << "sumdisp2 = " << sumdisp2 << "\n\n";

    simi = bunch.compare(graph2);
    tie(simiph, sumdisp2) = bunch.compareWithPhaseFocus(
        graph2, 5, displacementWithFocus
    );
    cout << "bunch: (graph0); graph: graph2\n";
    cout << "simi = " << simi << "\n";
    cout << "simiph = " << simiph << "\n";
    cout << "sumdisp2 = " << sumdisp2 << "\n\n";

    bunch.addGraph(graph1);

    simi = bunch.compare(graph0);
    tie(simiph, sumdisp2) = bunch.compareWithPhaseFocus(
        graph0, 5, displacementWithFocus
    );
    cout << "bunch: (graph0, graph1); graph: graph0\n";
    cout << "simi = " << simi << "\n";
    cout << "simiph = " << simiph << "\n";
    cout << "sumdisp2 = " << sumdisp2 << "\n\n";

    simi = bunch.compare(graph2);
    tie(simiph, sumdisp2) = bunch.compareWithPhaseFocus(
        graph2, 5, displacementWithFocus
    );
    cout << "bunch: (graph0, graph1); graph: graph2\n";
    cout << "simi = " << simi << "\n";
    cout << "simiph = " << simiph << "\n";
    cout << "sumdisp2 = " << sumdisp2 << "\n\n";

    bunch.clear();
    bunch.addGraph(graph2);
    bunch.addGraph(graph0);
    bunch.addGraph(graph1);

    simi = bunch.compare(graph0);
    tie(simiph, sumdisp2) = bunch.compareWithPhaseFocus(
        graph0, 5, displacementWithFocus
    );
    cout << "bunch: (graph2, graph0, graph1); graph: graph0\n";
    cout << "simi = " << simi << "\n";
    cout << "simiph = " << simiph << "\n";
    cout << "sumdisp2 = " << sumdisp2 << "\n\n";

}

// test step1
void test9()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    Mat image2, rgbImage2, showImage2;
    // image: 8UC1 (if test.png is 8-bit)
    image2 = imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage2 = imread("test2.png", CV_LOAD_IMAGE_COLOR);
    image2.convertTo(image2, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);
    CalcJet<40> calcJet2(image2, kernels, 101, 101);

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    Graph<40> graph0, resultGraph;
    Points<int> resultPoints;
    BunchGraph<40> bunch;

    graph0 = pointsToGraph(calcJet, points);
    bunch.addGraph(graph0);

    // display original points
    showImage = rgbImage.clone();
    for(int i=0; i<points.size(); i++){
        auto point = points.get(i);
        cv::circle(showImage, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage);

    // calculate the approximate face position in test1.png.
    tie(resultGraph, resultPoints) = step1(calcJet, bunch, points);

    // display result points 1
    showImage = rgbImage.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage);

    // calculate the approximate face position in test2.png.
    tie(resultGraph, resultPoints) = step1(calcJet2, bunch, points);

    // display result points 2
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

}

// test step2
void test10()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    Mat image2, rgbImage2, showImage2;
    // image: 8UC1 (if test.png is 8-bit)
    image2 = imread("test3.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage2 = imread("test3.png", CV_LOAD_IMAGE_COLOR);
    image2.convertTo(image2, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);
    CalcJet<40> calcJet2(image2, kernels, 101, 101);

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    Graph<40> graph0, resultGraph;
    Points<int> resultPoints;
    BunchGraph<40> bunch;

    graph0 = pointsToGraph(calcJet, points);
    bunch.addGraph(graph0);

    // step1: calculate the approximate face position in test2.png.
    tie(resultGraph, resultPoints) = step1(calcJet2, bunch, points);

    // display result points of step1
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step2: Refine position and size.
    tie(resultGraph, resultPoints) = 
        step2(calcJet2, bunch, resultPoints);

    // display result points of step2
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);
}


// test step3
void test11()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    Mat image2, rgbImage2, showImage2;
    // image: 8UC1 (if test.png is 8-bit)
    image2 = imread("test3.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage2 = imread("test3.png", CV_LOAD_IMAGE_COLOR);
    image2.convertTo(image2, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);
    CalcJet<40> calcJet2(image2, kernels, 101, 101);

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    Graph<40> graph0, resultGraph;
    Points<int> resultPoints;
    BunchGraph<40> bunch;

    graph0 = pointsToGraph(calcJet, points);
    bunch.addGraph(graph0);

    // step1: calculate the approximate face position in test2.png.
    tie(resultGraph, resultPoints) = step1(calcJet2, bunch, points);

    // display result points of step1
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step2: Refine position and size.
    tie(resultGraph, resultPoints) = 
        step2(calcJet2, bunch, resultPoints);

    // display result points of step2
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step3: Refine size and find aspect ratio.
    tie(resultGraph, resultPoints) =
        step3(calcJet2, bunch, resultPoints);

    // display result points of step3
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);


}


// test step4
void test12()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    Mat image2, rgbImage2, showImage2;
    // image: 8UC1 (if test.png is 8-bit)
    image2 = imread("test3.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage2 = imread("test3.png", CV_LOAD_IMAGE_COLOR);
    image2.convertTo(image2, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);
    CalcJet<40> calcJet2(image2, kernels, 101, 101);

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    Graph<40> graph0, resultGraph;
    Points<int> resultPoints;
    BunchGraph<40> bunch;

    graph0 = pointsToGraph(calcJet, points);
    bunch.addGraph(graph0);

    // step1: calculate the approximate face position in test2.png.
    tie(resultGraph, resultPoints) = step1(calcJet2, bunch, points);

    // display result points of step1
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step2: Refine position and size.
    tie(resultGraph, resultPoints) = 
        step2(calcJet2, bunch, resultPoints);

    // display result points of step2
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step3: Refine size and find aspect ratio.
    tie(resultGraph, resultPoints) =
        step3(calcJet2, bunch, resultPoints);

    // display result points of step3
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step4: 
    tie(resultGraph, resultPoints) = step4(calcJet2, bunch, resultGraph);

    // display result points of step4
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

}

// test modifyPointsOnImage()
void test13()
{
    Mat rgbImage;

    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    KeyPointsGui::modifyPoints(points, rgbImage);

}

// test iofiles
void test14()
{
	iofiles f;
	f.addpath("/home/liuqx/Downloads", "/home/liuqx/Downloads");
	f.addpath("/home/liuqx", "/home/liuqx/non-exist/", R"__(.*\.sh)__");

	bf::path ip, op;
	while (f.getnextfile(ip, op)) {
		cout << ip.native() << '\n' << op.native() << '\n';
	}
}

// test CalcJet serialization
void test15()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    Mat image2, rgbImage2, showImage2;
    // image: 8UC1 (if test.png is 8-bit)
    image2 = imread("test3.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage2 = imread("test3.png", CV_LOAD_IMAGE_COLOR);
    image2.convertTo(image2, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);
    CalcJet<40> calcJet2(image2, kernels, 101, 101);

    {
        ofstream ofs("1.jets", ios::trunc);
        ba::binary_oarchive oa(ofs);
        oa << calcJet;
    }

    {
        ofstream ofs("2.jets", ios::trunc);
        ba::binary_oarchive oa(ofs);
        oa << calcJet2;
    }

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    Graph<40> graph0, resultGraph;
    Points<int> resultPoints;
    BunchGraph<40> bunch;

    graph0 = pointsToGraph(calcJet, points);
    bunch.addGraph(graph0);

    // step1: calculate the approximate face position in test2.png.
    tie(resultGraph, resultPoints) = step1(calcJet2, bunch, points);

    // display result points of step1
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step2: Refine position and size.
    tie(resultGraph, resultPoints) = 
        step2(calcJet2, bunch, resultPoints);

    // display result points of step2
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step3: Refine size and find aspect ratio.
    tie(resultGraph, resultPoints) =
        step3(calcJet2, bunch, resultPoints);

    // display result points of step3
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step4: 
    tie(resultGraph, resultPoints) = step4(calcJet2, bunch, resultGraph);

    // display result points of step4
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

}

// test CalcJet deserialization
void test16()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Mat image, rgbImage, showImage;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage = imread("test.png", CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32F);

    Mat image2, rgbImage2, showImage2;
    // image: 8UC1 (if test.png is 8-bit)
    image2 = imread("test3.png", CV_LOAD_IMAGE_GRAYSCALE);
    rgbImage2 = imread("test3.png", CV_LOAD_IMAGE_COLOR);
    image2.convertTo(image2, CV_32F);

    CalcJet<40> calcJet;
    CalcJet<40> calcJet2;


    {
        ifstream ofs("1.jets");
        ba::binary_iarchive ia(ofs);
        ia >> calcJet;
    }

    {
        ifstream ofs("2.jets");
        ba::binary_iarchive ia(ofs);
        ia >> calcJet2;
    }

    Points<int> points{
        {49, 59}, {73, 57}, {60, 74}, {52, 88}, {71, 87},
        {32, 60}, {33, 77}, {37, 91}, {50, 108}, 
        {67, 108}, {81, 91}, {87, 76}, {92, 59},
        {59, 39}
    };

    Graph<40> graph0, resultGraph;
    Points<int> resultPoints;
    BunchGraph<40> bunch;

    graph0 = pointsToGraph(calcJet, points);
    bunch.addGraph(graph0);

    // step1: calculate the approximate face position in test2.png.
    tie(resultGraph, resultPoints) = step1(calcJet2, bunch, points);

    // display result points of step1
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step2: Refine position and size.
    tie(resultGraph, resultPoints) = 
        step2(calcJet2, bunch, resultPoints);

    // display result points of step2
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step3: Refine size and find aspect ratio.
    tie(resultGraph, resultPoints) =
        step3(calcJet2, bunch, resultPoints);

    // display result points of step3
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

    // step4: 
    tie(resultGraph, resultPoints) = step4(calcJet2, bunch, resultGraph);

    // display result points of step4
    showImage2 = rgbImage2.clone();
    for(int i=0; i<resultPoints.size(); i++){
        auto point = resultPoints.get(i);
        cv::circle(showImage2, {point.x, point.y}, 3, CV_RGB(255, 0, 0));
    }
    displayMat(showImage2);

}

// test serialization of Graph
void test17()
{
    Kernels<40> kernels;
    genGaborKernels(101, kernels);
    
    Mat image;
    // image: 8UC1 (if test.png is 8-bit)
    image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32F);

    CalcJet<40> calcJet(image, kernels, 101, 101);

    Graph<40> graph;

    Jet<40> jet;

    jet = calcJet.calcJet(10, 10);
    graph.addNode(jet);
    jet = calcJet.calcJet(20, 20);
    graph.addNode(jet);
    jet = calcJet.calcJet(30, 30);
    graph.addNode(jet);
    cout << "--------- graph1 ---------\n";
    cout << graph.toString() << "\n";

    ofstream ofs("1.graph", ios::trunc);
    ba::binary_oarchive oa(ofs);
    oa << graph;

}

// test deserialization of Graph
void test18()
{
    Graph<40> graph;

    //ifstream ofs("/home/liuqx/Downloads/face-recogn/test/graphs/s29-1.pgm.graph");
    //ifstream ofs("/home/liuqx/Downloads/face-recogn/test/graphs/s21-3.pgm.graph");
    ifstream ofs("1.graph");
    ba::binary_iarchive ia(ofs);
    ia >> graph;

    cout << "--------- graph1 ---------\n";
    cout << graph.toString() << "\n";

}

// test generating cache
void test19()
{
    const char *knownFacesDir = "/home/liuqx/Downloads/face-recogn/test/unknown-faces";
    const char *graphsDir = "/home/liuqx/Downloads/face-recogn/test/graphs";

    Kernels<40> kernels;
    genGaborKernels(101, kernels);
    
    iofiles iof;
    iof.addpath(knownFacesDir, graphsDir);

    bf::path ifilepath, ofilepath;
    while(iof.getnextfile(ifilepath, ofilepath)){
        if(extensionIs(ifilepath.native(), ".jets")){
            continue;
        }
        __getCalcJetWithCache(ifilepath.native(), kernels);
    }
    
}


// test generating graph 
void test20()
{
    const char *knownFacesDir = "/home/liuqx/Downloads/face-recogn/test/known-faces";
    const char *graphsDir = "/home/liuqx/Downloads/face-recogn/test/graphs";
    const char *modelsDir = "/home/liuqx/Downloads/face-recogn/test/models";
    const char *firstGraph = "/home/liuqx/Downloads/face-recogn/test/graphs/s29-1.pgm.graph";

    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Graph<40> graph;
    BunchGraph<40> bunch;
    Points<int> points;
    Points<int> startPoints;
    bool modified;

    iofiles iof;
    iof.addpath(knownFacesDir, graphsDir);

    bf::path mfilepath(modelsDir);

    bf::path ifilepath, ofilepath;
    while(iof.getnextfile(ifilepath, ofilepath)){
        if(extensionIs(ifilepath.native(), ".jets")){
            continue;
        }

        auto origfilename = ifilepath.filename();
        ofilepath /= origfilename;

        string ifilepathstr = ifilepath.native();
        string ofilepathstr = ofilepath.native() + ".graph";

        if(startPoints.empty()){
            tie(graph, startPoints, modified) = genGraph(
                ifilepathstr, kernels, bunch, startPoints
            );
        }
        else {
            tie(graph, points, modified) = genGraph(
                ifilepathstr, kernels, bunch, startPoints
            );
        }

        {
            std::ofstream graphfile(ofilepathstr, std::ios::trunc);
            boost::archive::binary_oarchive oa(graphfile);
            oa << graph;
        } // flush do disk immediately

        Log::info(std::string("Generated graph: '") + ofilepathstr + "'.");

        if(modified) {
            std::string mfilepathstr = (mfilepath/origfilename).native() + ".graph";
            try {
                bf::copy_file(
                    ofilepathstr, 
                    mfilepathstr,
                    bf::copy_option::overwrite_if_exists
                );
                Log::info(
                    std::string("Created file: '") +
                    mfilepathstr +
                    "'."
                );
            }
            catch(...) {
                Log::warning(
                    std::string("Failed to create file: '") + 
                    mfilepathstr +
                    "'."
                );
            }
        }

    }


} 

// test recognition
void test21()
{
    const char *knownFacesDir = "/home/liuqx/Downloads/face-recogn/test/known-faces";
    const char *unknownFacesDir = "/home/liuqx/Downloads/face-recogn/test/unknown-faces";
    const char *graphsDir = "/home/liuqx/Downloads/face-recogn/test/graphs";
    const char *modelsDir = "/home/liuqx/Downloads/face-recogn/test/models";
    const char *firstGraph = "/home/liuqx/Downloads/face-recogn/test/graphs/s29-1.pgm.graph";

    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    vector<tuple<Graph<40>,string>> knownGraphs;
    ofstream resultfile("results.csv", ios::trunc);

    Graph<40> graph;
    BunchGraph<40> bunch;
    Points<int> points;
    Points<int> startPoints;
    bool modified;

    // the .graph file to get starting points
    try {
        ifstream firstfile(firstGraph);
        boost::archive::binary_iarchive ia(firstfile);
        ia >> graph;

        startPoints = graphToPoints(graph);

        Log::info(
            std::string("Loaded starting points from: '") +
            firstGraph +
            "'."
        );
    }
    catch(...) {
        Log::warning(
            std::string("Failed to load starting points from: '") +
            firstGraph +
            "'. Exiting now."
        );
    }


    iofiles iof;
    // the output directory is not used here.
    iof.addpath(modelsDir, graphsDir);

    bf::path ifilepath, ofilepath;
    while(iof.getnextfile(ifilepath, ofilepath)){
        if(!extensionIs(ifilepath.native(), ".graph")){
            continue;
        }

        std::string mfilepathstr = ifilepath.native();

        try {
            std::ifstream mfile(mfilepathstr);
            boost::archive::binary_iarchive ia(mfile);
            ia >> graph;

            if(graph.getNodes().size() != startPoints.size()) {
                Log::warning(
                    std::string("File ignored due to a mismatch " 
                    "in the number of key points: '") +
                    mfilepathstr +
                    "'."
                );
                continue;
            }

            bunch.addGraph(graph);

            Log::info(
                std::string("Loaded model: '") +
                mfilepathstr +
                "'."
            );
        }
        catch(char) {
            Log::warning(
                std::string("Failed to load model: '") +
                mfilepathstr +
                "'."
            );
        }
    }

    if(bunch.empty()) {
        Log::error("No model loaded. Exiting now.");
        return;
    }

    iof.clear();
    // the output directory is not used here.
    iof.addpath(graphsDir, graphsDir);

    while(iof.getnextfile(ifilepath, ofilepath)){
        if(!extensionIs(ifilepath.native(), ".graph")){
            continue;
        }

        std::string gfilepathstr = ifilepath.native();
        try {
            std::ifstream gfile(gfilepathstr);
            boost::archive::binary_iarchive ia(gfile);
            ia >> graph;

            if(graph.getNodes().size() != startPoints.size()) {
                Log::warning(
                    std::string("File ignored due to a mismatch " 
                    "in the number of key points: '") +
                    gfilepathstr +
                    "'."
                );
                continue;
            }

            knownGraphs.push_back(make_tuple(graph, ifilepath.filename().string()));

            Log::info(
                std::string("Loaded known graph: '") +
                gfilepathstr +
                "'."
            );
        }
        catch(...) {
            Log::warning(
                std::string("Failed to load known graph: '") +
                gfilepathstr +
                "'."
            );
        }
    }

    if(knownGraphs.empty()) {
        Log::error("No known graph loaded. Exiting now.");
        return;
    }

    iof.clear();
    // the output directory is not used here.
    iof.addpath(unknownFacesDir, graphsDir);

    while(iof.getnextfile(ifilepath, ofilepath)){
        if(extensionIs(ifilepath.native(), ".jets")){
            continue;
        }

        std::string ufilepathstr = ifilepath.native();
        try {
            tie(graph, points, modified) = genGraph(
                ufilepathstr, kernels, bunch, startPoints, false
            );

            float maxSimi; 
            string resultName;
            maxSimi = -std::numeric_limits<float>::infinity(); 

            for(auto &i: knownGraphs){
                float simi;
                simi = get<0>(i).compare(graph);
                if(simi > maxSimi){
                    maxSimi = simi;
                    resultName = get<1>(i);
                }
            }

            resultfile << "\"" << ifilepath.filename().string() << "\", \""
                << resultName << "\"\n";

            Log::info(
                std::string("Finished recognizing image: '") +
                ufilepathstr +
                "'."
            );
        }
        catch(...) {
            Log::warning(
                std::string("Failed to recognize image: '") +
                ufilepathstr +
                "'."
            );
        }
    }



}
#endif