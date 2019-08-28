#pragma once

#include "jet.hpp"
#include "utils.h"
#include "graph.hpp"
#include "points.hpp"
#include "cvutils.h"

#include <tuple>
#include <limits>
#include <exception>

#include <opencv2/core.hpp>

#define PI 3.14159265358979323846F






// generate 40 Garbor kernels.
void genGaborKernels(
    int kernelSize,        // each kernel is a
                              // kernelSize*kernelSize matrix.
    Kernels<40> &result_kernels
);


// return value: dx, dy
std::tuple<float/*dx*/,float/*dy*/>
displacementWithFocus(
    const Jet<40> &jet1, 
    const Jet<40> &jet2, 
    int focus
);


// Calculate a graph, whose nodes' positions are specified by 'points'.
template<int N>
Graph<N> pointsToGraph(
    const CalcJet<N> &calcJet,
    const Points<int> &points
)
{
    Graph<N> ret;

    int nPoints = points.size();
    for(int i=0; i<nPoints; i++){
        Jet<N> jet;
        auto point = points.get(i);

        jet = calcJet.calcJet(point.x, point.y);
        ret.addNode(jet);
    }

    return ret;
}

template<int N>
Points<int> graphToPoints(const Graph<N> &graph)
{
    Points<int> ret;

    for(auto &i: graph.getNodes()) {
        ret.addPoint({i.x, i.y});
    }

    return ret;
}


// Find approximate face position.
// return value: Graph graph, Points points
template<int N>
std::tuple<Graph<N>,Points<int>> step1(
    const CalcJet<N> &calcJet,
    const BunchGraph<N> &bunch,
    const Points<int> &startPoints
)
{
    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    assert(!bunch.empty());
    assert(!startPoints.empty());

    // Calculates the size of image and the range of startPoints. 
    int graphMinX, graphMinY, graphMaxX, graphMaxY;
    std::tie(graphMinX, graphMinY, graphMaxX, graphMaxY) = startPoints.getMinMax();
    int graphWidth = graphMaxX - graphMinX + 1;
    int graphHeight = graphMaxY - graphMinY + 1;
    int srcWidth, srcHeight;
    std::tie(srcWidth, srcHeight) = calcJet.getSrcSize();

    // The range of startPoints must not be larger
    // than that of original image.
    if(graphWidth > srcWidth || graphHeight > srcHeight) {
        throw std::out_of_range(
            "Error in step1(): The range of startPoints is "
            "larger than that of source image."
        );
    }

    const int step = 4;
    int nHori = (srcWidth - graphWidth + 1) / step;
    int nVert = (srcHeight - graphHeight + 1) / step;

    Points<int> startpoints;
    Points<int> resultPoints;
    Graph<N> resultGraph;
    float maxSimi = -std::numeric_limits<float>::infinity();

    // Test similarity at each location of a square lattice
    // with a spacing of 'step' pixels.

    startpoints = startPoints;
    startpoints.translate(-graphMinX, -graphMinY);

    #pragma omp parallel for collapse(2) schedule(static)
    for(int ix=0; ix<nHori; ix++){
        for(int iy=0; iy<nVert; iy++){
            float simi;
            Points<int> points;
            Graph<N> graph;

            points = startpoints;
            points.translate(ix*step, iy*step);

            graph = pointsToGraph(calcJet, points);
            simi = bunch.compare(graph);

            #pragma omp critical
            {
                if(simi > maxSimi) {
                    maxSimi = simi;
                    resultPoints = points;
                    resultGraph = graph;
                }
            }
        }
    }

    // Repeat the scanning around the best fitting position
    // with a spacing of 1 pixel.

    startpoints = resultPoints;

    #pragma omp parallel for collapse(2) schedule(static)
    for(int ix=1-step; ix<step; ix++){
        for(int iy=1-step; iy<step; iy++){

            float simi;
            Points<int> points;
            Graph<N> graph;

            points = startpoints;
            points.translate(ix, iy);
            
            if( !points.isInRange(0, 0, srcWidth-1, srcHeight-1) ){
                continue;
            }

            graph = pointsToGraph(calcJet, points);
            simi = bunch.compare(graph);

            #pragma omp critical
            {
                if(simi > maxSimi) {
                    maxSimi = simi;
                    resultPoints = points;
                    resultGraph = graph;
                }
            }
        }
    }

    return std::make_tuple(resultGraph, resultPoints);
    
}



//// Original algorithm proposed in the paper.
//// Refine position and size.
//// The scaling information will be stored in bunch::xScale
//// and bunch::yScale.
//// return value: 
////     Graph graph, Points points
//template<int N>
//std::tuple<Graph<N>,Points<int>> step2(
    //const CalcJet<N> &calcJet,
    //BunchGraph<N> &bunch,
    //const Points<int> &step1Points
//)
//{
    //static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    //assert(!bunch.empty());
    //assert(!step1Points.empty());

    //int srcWidth, srcHeight;
    //std::tie(srcWidth, srcHeight) = calcJet.getSrcSize();

    //// The points are scaled first,
    //// then translated by -delta to delta.
    //const int scalesSize = 2;
    //const float scales[scalesSize] = {
        //1.18F, 1.0F/1.18F
    //};
    //const int delta = 4;
    
    //float minSumDisp2 = std::numeric_limits<float>::infinity();
    //Graph<N> resultGraph;
    //Points<int> resultPoints;
    //float resultScale = 1.0F;

    //#pragma omp parallel for collapse(2) schedule(static)
    //for(int ix=-delta; ix<=delta; ix++) {
        //for(int iy=-delta; iy<=delta; iy++) {
            //for(int j=0; j<scalesSize; j++) {
                //Points<int> points;
                //Graph<N> graph;
                //float sumDisp2;
    
                //points = step1Points;
                //points.scale(scales[j], scales[j]).translate(ix, iy);
                //if( !points.isInRange(0, 0, srcWidth-1, srcHeight-1) ){
                    //continue;
                //}

                //graph = pointsToGraph(calcJet, points);

                //sumDisp2 = std::get<1>(bunch.compareWithPhaseFocus(
                    //graph, 1, displacementWithFocus
                //));

                //#pragma omp critical
                //{
                    //if(sumDisp2 < minSumDisp2) {
                        //minSumDisp2 = sumDisp2;
                        //resultPoints = points;
                        //resultGraph = graph;
                        //resultScale = scales[j];
                    //}
                //}
            //}
        //}
    //}

    //bunch.xScale = resultScale;
    //bunch.yScale = resultScale;

    //return std::make_tuple(
        //resultGraph, 
        //resultPoints 
    //);

//}


//// Original algorithm proposed in the paper.
//// Refine size and find aspect ratio.
//// The scaling information will be stored in bunch::xScale
//// and bunch::yScale.
//// return value: 
////     Graph graph, Points points
//template<int N>
//std::tuple<Graph<N>,Points<int>> step3(
    //const CalcJet<N> &calcJet,
    //BunchGraph<N> &bunch,
    //const Points<int> &step2Points
//)
//{
    //static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    //assert(!bunch.empty());
    //assert(!step2Points.empty());

    //int srcWidth, srcHeight;
    //std::tie(srcWidth, srcHeight) = calcJet.getSrcSize();

    //// The x- and y-dimensions are both scaled.
    //// The scaled points and the original points will have the
    //// the same center position.
    //const float minScale = 0.8F;
    //const float maxScale = 1.2F;
    //const float step = 0.1F;
    ////const int delta = 4;
    //int nScale = round((maxScale - minScale) / step) + 1;
    
    //float minSumDisp2 = std::numeric_limits<float>::infinity();
    //Graph<N> resultGraph;
    //Points<int> resultPoints, startPoints;
    //float resultScaleX = 1.0F, resultScaleY = 1.0F;

    //startPoints = step2Points;
    //#pragma omp parallel for collapse(2) schedule(static)
    //for(int i=0; i<nScale; i++){
        //for(int j=0; j<nScale; j++){
            //Points<int> points;
            //Graph<N> graph;
            //float sumDisp2;
            //float scaleX, scaleY;

            //scaleX = minScale + (float)i*step;
            //scaleY = minScale + (float)j*step;
            //points = startPoints;
            //points.scale(scaleX, scaleY);
            //if( !points.isInRange(0, 0, srcWidth-1, srcHeight-1) ){
                //continue;
            //}

            //graph = pointsToGraph(calcJet, points);

            //sumDisp2 = std::get<1>(bunch.compareWithPhaseFocus(
                //graph, 5, displacementWithFocus
            //));

            //#pragma omp critical
            //{
                //if(sumDisp2 < minSumDisp2) {
                    //minSumDisp2 = sumDisp2;
                    //resultPoints = points;
                    //resultGraph = graph;
                    //resultScaleX = scaleX;
                    //resultScaleY = scaleY;
                //}
            //}
        //}
    //}

    //bunch.xScale *= resultScaleX;
    //bunch.yScale *= resultScaleY;

    //return std::make_tuple(
        //resultGraph,
        //resultPoints
    //);

//}



// My algorithm.
// Refine position and size.
// The scaling information will be stored in bunch::xScale
// and bunch::yScale.
// return value: 
//     Graph graph, Points points
template<int N>
std::tuple<Graph<N>,Points<int>> step2(
    const CalcJet<N> &calcJet,
    BunchGraph<N> &bunch,
    const Points<int> &step1Points
)
{
    // This step is merged into step3.

    return std::make_tuple(
        pointsToGraph(calcJet, step1Points), 
        step1Points 
    );

}


// My algorithm.
// Refine size and find aspect ratio and position.
// The scaling information will be stored in bunch::xScale
// and bunch::yScale.
// return value: 
//     Graph graph, Points points
template<int N>
std::tuple<Graph<N>,Points<int>> step3(
    const CalcJet<N> &calcJet,
    BunchGraph<N> &bunch,
    const Points<int> &step2Points
)
{
    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    assert(!bunch.empty());
    assert(!step2Points.empty());

    int srcWidth, srcHeight;
    std::tie(srcWidth, srcHeight) = calcJet.getSrcSize();

    // The x- and y-dimensions are scaled independently.
    // Then translated by -delta to delta.
    const float minScale = 0.8F;
    const float maxScale = 1.2F;
    const float scaleStep = 0.1F;
    //const float angles[] = {PI/6.0F, 0.0F, -PI/6.0F};
    const float angles[] = {0.0F};
    const int delta = 3;
    int nScale = round((maxScale - minScale) / scaleStep) + 1;
    
    //float minSumDisp2 = std::numeric_limits<float>::infinity();
    float maxSimi = -std::numeric_limits<float>::infinity();
    Graph<N> resultGraph;
    Points<int> resultPoints;
    float resultScaleX = 1.0F, resultScaleY = 1.0F;

    #pragma omp parallel for collapse(4) schedule(static)
    for(int ix=0; ix<nScale; ix++){
        for(int iy=0; iy<nScale; iy++){
            for(int jx=-delta; jx<=delta; jx++){
                for(int jy=-delta; jy<=delta; jy++){
                    for(float angle: angles){
                        Points<int> points;
                        Graph<N> graph;
                        //float sumDisp2;
                        float simi;
                        float scaleX, scaleY;

                        scaleX = minScale + (float)ix*scaleStep;
                        scaleY = minScale + (float)iy*scaleStep;
                        points = step2Points;
                        points
                            .scale(scaleX, scaleY)
                            .translate(jx, jy)
                            .rotate(angle);
                        if( !points.isInRange(0, 0, srcWidth-1, srcHeight-1) ){
                            continue;
                        }

                        graph = pointsToGraph(calcJet, points);

                        //sumDisp2 = std::get<1>(bunch.compareWithPhaseFocus(
                            //graph, 5, displacementWithFocus
                        //));
                        simi = bunch.compare(graph);

                        #pragma omp critical
                        {
                            //if(sumDisp2 < minSumDisp2) {
                            if(simi > maxSimi) {
                                //minSumDisp2 = sumDisp2;
                                maxSimi = simi;
                                resultPoints = points;
                                resultGraph = graph;
                                resultScaleX = scaleX;
                                resultScaleY = scaleY;
                            }
                        }
                    }
                }
            }
        }

    }

    bunch.xScale *= resultScaleX;
    bunch.yScale *= resultScaleY;

    return std::make_tuple(
        resultGraph,
        resultPoints
    );

}

// Local distortion.
template<int N>
std::tuple<Graph<N>,Points<int>> step4(
    const CalcJet<N> &calcJet,
    const BunchGraph<N> &bunch,
    const Graph<N> &step3Graph
)
{
    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    assert(!bunch.empty());
    assert(!step3Graph.empty());

    int srcWidth, srcHeight;
    std::tie(srcWidth, srcHeight) = calcJet.getSrcSize();

    // the position of each node is varied between -delta and delta.
    const int delta = 4;
    const float lambda = 2.0F;
    
    Points<int> resultPoints = graphToPoints(step3Graph);

    int nNodes = step3Graph.getNodes().size();
    #pragma omp parallel for schedule(static)
    for(int n=0; n<nNodes; n++) {
        Graph<N> graph;
        float maxSimi;
        int bestX, bestY;

        maxSimi = -std::numeric_limits<float>::infinity(); 
        graph = step3Graph;
        bestX = step3Graph.getNodes()[n].x;
        bestY = step3Graph.getNodes()[n].y;

        for(int ix=-delta; ix<=delta; ix++) {
            for(int iy=-delta; iy<=delta; iy++) {
                int x, y;
                float dispX, dispY, disp2;
                float simi;
                Jet<N> jet;
    
                x = graph.getNodes()[n].x + ix;
                y = graph.getNodes()[n].y + ix;

                if(x < 0 || x >= srcWidth || y < 0 || y >= srcHeight ){
                    continue;
                }

                jet = calcJet.calcJet(x, y);
                std::tie(dispX, dispY) = displacementWithFocus(
                    graph.getNodes()[n], jet, 5
                );
                disp2 = dispX*dispX + dispY*dispY;
                
                if(disp2 > 1.0F){
                    continue;
                }

                graph.replaceNode(jet, n);

                simi = std::get<0>(bunch.compareWithPhaseFocus(
                    graph, 5, displacementWithFocus, lambda
                ));

                if(simi > maxSimi) {
                    maxSimi = simi;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        #pragma omp critical
        {
            resultPoints.modifyPoint({bestX, bestY}, n);
        }
    }

    Graph<N> resultGraph = pointsToGraph(calcJet, resultPoints);

    return std::make_tuple(resultGraph, resultPoints);

}


#undef PI