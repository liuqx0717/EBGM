#pragma once

#include "jet.hpp"
#include "graph.hpp"
#include "gui.h"
#include "alg.h"
#include "utils.h"
#include "iofiles.h"

#include <exception>
#include <fstream>
#include <string>
#include <tuple>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


class Cfg {
public:
    static iofiles bunchFiles;
    static boost::filesystem::path bunchDirectory;
    static iofiles inputFiles;
    static iofiles knownGraphFiles;
    static std::string startGraphFile;
    static std::string recogResultFile;
    static bool noOverwrite;
};


// If both the image file and the cache file exists, use the cache.
// Else, generate new cache file. 
// You should pass image files ONLY!
template<int N> CalcJet<N> __getCalcJetWithCache(
    const std::string &imgfilename,
    const Kernels<N> &kernels
)
{
    CalcJet<N> ret;
    std::string err;

    cv::Mat image;
    // image: 8UC1 (if image file is 8-bit)
    image = cv::imread(imgfilename, cv::IMREAD_GRAYSCALE);
    if(image.data == NULL) {
        err = std::string("Failed to open image file: '") + imgfilename + "'.";
        Log::error(err);
        throw std::runtime_error(err);
    }

    std::string cachename = imgfilename + ".jets";
    if(fileexists(cachename)){
        try {
            ret.m_kx = kernels.kx;
            ret.m_ky = kernels.ky;

            std::ifstream cachefile(cachename);
            boost::archive::binary_iarchive ia(cachefile);
            ia >> ret;

            Log::info(std::string("Using cache: '") + cachename + "'.");

            return ret;
        }
        catch(...) {
            // if failed to read the cache file, regenerate it.
            Log::warning(
                std::string("Failed to read cache file: '") +
                cachename +
                "'. "
            );
        }
    }

    image.convertTo(image, CV_32F);
    ret.init(image, kernels, 101, 101);

    try {
        std::ofstream cachefile(cachename, std::ios::trunc);
        boost::archive::binary_oarchive oa(cachefile);
        oa << ret;

        Log::info(std::string("Generated cache: '") + cachename + "'.");
    }
    catch(...) {
        // if failed to generate the cache file
        Log::warning(
            std::string("Failed to generate cache file: '") +
            cachename +
            "'. "
        );
    }

    return ret;
}


// Get jet calculator for a file.
// You should pass image files ONLY!
template<int N>
CalcJet<N> getCalcJet(
    const std::string &imgfilename, 
    const Kernels<N> &kernels
)
{
    return __getCalcJetWithCache(imgfilename, kernels);
}




// Generate graph from a image with the help of BunchGraph.
// You should pass image files ONLY!
template<int N>
std::tuple<Graph<N>, Points<int>, bool> genGraph(
    const std::string &imgfilename,
    const Kernels<N> &kernels,
    BunchGraph<N> &bunch,     // If the points are modified by user, 
                              // the result graph will be added to 'bunch',
                              // and will return true.
    Points<int> startPoints,  // startPoints can may empty.
    bool displayGUI = true
)
{
    Graph<N> graph;
    Points<int> points;
    bool modified = false;

    CalcJet<N> calcJet = getCalcJet(imgfilename, kernels);
    cv::Mat rgbImg = cv::imread(imgfilename, cv::IMREAD_COLOR);

    if(startPoints.empty()){
        if(!displayGUI){
            std::string errstr = std::string("No starting points found for image '") +
                imgfilename +
                "'.";
            Log::error(
                errstr
            );
            throw std::runtime_error(errstr);
        }
        KeyPointsGui::modifyPoints(startPoints, rgbImg);
        while(startPoints.size() < 2) {
            Log::error(
                "The points you specified is too few. "
                "Please specify at least 2 points."
            );
            startPoints.clear();
            KeyPointsGui::modifyPoints(startPoints, rgbImg);
        }
    }

    if(bunch.empty()){
        graph = pointsToGraph(calcJet, startPoints);
        bunch.addGraph(graph);
        return std::make_tuple(graph, startPoints, true);
    }

    try {
        std::tie(graph, points) = step1(calcJet, bunch, startPoints);
        std::tie(graph, points) = step2(calcJet, bunch, points);
        std::tie(graph, points) = step3(calcJet, bunch, points);
        std::tie(graph, points) = step4(calcJet, bunch, graph);
    }
    catch(std::exception err){
        std::string errstr = 
            std::string("Error when processing '") +
            imgfilename +
            "': " +
            err.what();
        Log::error(errstr);
        throw std::runtime_error(errstr);
    }

    if(displayGUI) {
        Points<int> tmpPoints = points;
        while(modified = KeyPointsGui::modifyPoints(tmpPoints, rgbImg)) {
            if(tmpPoints.size() != startPoints.size()) {
                Log::error(
                    "The number of points you specified for this image "
                    "MUST be exactly the same as that of previous image."
                );
                tmpPoints = points;
                continue;
            }

            points = tmpPoints;
            graph = pointsToGraph(calcJet,points);
            bunch.addGraph(graph);
            break;
        }
    }

    return std::make_tuple(graph, points, modified);
}