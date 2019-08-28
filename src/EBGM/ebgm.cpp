#include "ebgm.h"
#include "kernels.h"
#include "cvutils.h"
#include "utils.h"
#include "alg.h"
#include "graph.hpp"
#include "jet.hpp"
#include "points.hpp"
#include "iofiles.h"
#include "tests.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <iomanip>
#include <limits>
#include <string>
#include <exception>

#include <opencv2/core.hpp>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace cv;
using namespace std;

// process command-line arguments for train mode
int main_train(int argc, char* argv[]);

// process command-line arguments for recog mode
int main_recog(int argc, char* argv[]);

// process common command-line arguments
void common_args(char *arg);

int train();
int recog();


iofiles Cfg::bunchFiles;
boost::filesystem::path Cfg::bunchDirectory;
iofiles Cfg::inputFiles;
iofiles Cfg::knownGraphFiles;
std::string Cfg::startGraphFile;
std::string Cfg::recogResultFile;
bool Cfg::noOverwrite = false;


const char *helptext =
R"__(Usage:

    ebgm train [<options>] -b <directory>
        -i <input> [-f <filter>] [<output>] 
        [-i <input> [-f <filter>] [<output>]]...

    ebgm recog [<options>] -b <directory>
        -k <file|directory> [-k <file|directory>]...
        -i <input> [-f <filter>] [-i <input> [-f <filter>]]...
        <filename>

Train Mode:

    Generate the graphs of known faces interactively with the help of
    bunch graph (if any). The graph whose key points were modified by
    the user will be added to the bunch graph automatically.

Recog Mode:

    First, get the graph of an unknown face with the help of bunch graph.
    Then, find the graph with the highest similarity among the known
    graphs. 

    -k <file|directory>
    Specify the graphs of known faces. Will process all the ".graph" file
    in the directory.

    <filename>
    The results will be stored in this file, in csv format.

Common Options:

    -b <directory>
    Specify the directory containing the graphs used to construct a bunch
    graph. Will process all the ".graph" file in this directory.
    If you specify multiple -b options, the last one will take effect.

    -s <filename>
    Specify a graph whose key points should serve as the start points.
    If not specified, this program will pick the first graph from the
    directory specified by -b.

    --no-overwrite
    Skip current input file if the corresponding output file exists.
    Zero-length files are considered to be non-exsistent.

<input>:

    Input image file name or directory name. If it is a directory, you can
    use a following -f option to get the file list filtered (.graph files
    will be ignored automatically). If it is a file, the following -f
    option will be ignored. Symbol links will be followed.

<filter>:

    A Perl regular expression. For example, to process ".jpg" files only,
    use ".+\.jpg". If you specify multiple -f options after the same -i,
    the last one will take effect.

<output>:

    <output> MUST be a directory in case <intput> is a directory. It may be
    a file or a directory otherwise. If you want to specify a non-existent
    directory, please use a trailing "\".

)__";


int main(int argc, char* argv[])
{
    if(argc >= 2){
        if(!strcmp(argv[1], "train")){
            return main_train(argc - 2, argv + 2);
        }

        if(!strcmp(argv[1], "recog")){
            return main_recog(argc - 2, argv + 2);
        }
    }

    clog << helptext;
    return 1;
}

void common_args(char *arg)
{
    static int state = 0;
    string errmsg;

    switch (state)
    {
    case 0:       // initial state
        if(!strcmp(arg, "-s")){
            Cfg::startGraphFile.clear();
            state = 1;
            break;
        }
        else if(!strcmp(arg, "-b")){
            Cfg::bunchFiles.clear();
            state = 2;
            break;
        }
        else if(!strcmp(arg, "--no-overwrite")){
            Cfg::noOverwrite = true;
            state = 0;
            break;
        }
        errmsg = string("Unrecognized parameter: '") + arg + "'.";
        Log::error(errmsg);
        throw(runtime_error(errmsg));
        break;

    case 1:        // after -s
        if(fileexists(arg)){
            Cfg::startGraphFile = arg;
            state = 0;
            break;
        }
        errmsg = string("File does not exist: '") + arg + "'.";
        Log::error(errmsg);
        throw(runtime_error(errmsg));
        break;
    
    case 2:        // after -b
        try{
            if(boost::filesystem::create_directory(arg)){
                Log::info(string("Created directory: '") + arg + "'.");
            }
        }
        catch(...){
        }

        if(Cfg::bunchFiles.addpath(arg, arg, R"__(.+\.graph)__", true)){
            Cfg::bunchDirectory = arg;
            state = 0;
            break;
        }
        errmsg = string("The path of -b must be a directory: '") + arg + "'.";
        Log::error(errmsg);
        throw(runtime_error(errmsg));
        break;

    default:
        errmsg = "Unknown state in common_args().";
        Log::error(errmsg);
        throw(runtime_error(errmsg));
        break;
    }


}

int main_train(int argc, char* argv[])
{
    int state = 0;
    string errmsg;
    string regexErr;
    const char *lastInput = "";
    const char *lastFilter = "";

    try{
        for(int i=0; i<argc; i++){
            char *arg = argv[i];

            switch (state)
            {
            case 0:       // initial state
                if(!strcmp(arg, "-i")){
                    state = 1;
                    break;
                }

                common_args(arg);
                break;

            case 1:       // after -i
                lastInput = arg;
                lastFilter = "";
                state = 2;
                break;
            
            case 2:       // after -i <input>
                if(!strcmp(arg, "-f")){
                    state = 3;
                    break;
                }
                else if(!strcmp(arg, "-i")) {
                    try{
                        auto oDirectory = getDirectory(lastInput);
                        if(Cfg::inputFiles.addpath(lastInput, oDirectory, lastFilter)){
                            state = 1;
                            break;
                        }
                    }
                    catch(...){

                    }

                    errmsg = string(
                        "<input> must be an existing file or directory: '") +
                        lastInput + "'.";
                    Log::error(errmsg);
                    throw(runtime_error(errmsg));
                }
                
                if(Cfg::inputFiles.addpath(lastInput, arg, lastFilter)){
                    state = 0;
                    break;
                }
                errmsg = string("<output> is invalid: '") + arg + "'.";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;

            case 3:           // after -f
                if(checkregex(arg, regexErr)){
                    lastFilter = arg;
                    state = 2;
                    break;
                }
                errmsg = string("Syntax error in '") + arg + "': " + regexErr;
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;

            default:
                errmsg = "Unknown state in main_train().";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;
            }
        }

        if(state == 2){
            bool ret = false;
            try{
                auto oDirectory = getDirectory(lastInput);
                ret = Cfg::inputFiles.addpath(lastInput, oDirectory, lastFilter);
            }
            catch(...){
            }

            if(!ret){
                errmsg = string(
                    "<input> must be an existing file or directory: '") +
                    lastInput + "'.";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
            }
        }
        
        if(Cfg::bunchFiles.empty()){
            errmsg = "Missing -b option.";
            Log::error(errmsg);
            throw(runtime_error(errmsg));
        }
        if(Cfg::inputFiles.empty()){
            errmsg = "Missing -i option.";
            Log::error(errmsg);
            throw(runtime_error(errmsg));
        }
       
    }
    catch(...){
        Log::error("Invalid Parameters. Exiting now.");
        return 1;
    }

    return train();
}

int main_recog(int argc, char* argv[])
{
    int state = 0;
    string errmsg;
    string regexErr;
    const char *lastInput = "";
    const char *lastFilter = "";

    try{
        for(int i=0; i<argc; i++){
            char *arg = argv[i];

            switch (state)
            {
            case 0:       // initial state
                if(!strcmp(arg, "-i")){
                    state = 1;
                    break;
                }
                else if(!strcmp(arg, "-k")){
                    state = 5;
                    break;
                }

                common_args(arg);
                break;

            case 1:       // after -i
                lastInput = arg;
                lastFilter = "";
                state = 2;
                break;
            
            case 2:       // after -i <input>
                if(!strcmp(arg, "-f")){
                    state = 3;
                    break;
                }
                else if(!strcmp(arg, "-i")) {
                    // opath not used.
                    if(Cfg::inputFiles.addpath(lastInput, lastInput, lastFilter)){
                        state = 1;
                        break;
                    }

                    errmsg = string(
                        "<input> must be an existing file or directory: '") +
                        lastInput + "'.";
                    Log::error(errmsg);
                    throw(runtime_error(errmsg));
                }

                // opath not used.
                if(Cfg::inputFiles.addpath(lastInput, lastInput, lastFilter)){
                    Cfg::recogResultFile = arg;
                    state = 4;
                    break;
                }
                    
                errmsg = string(
                    "<input> must be an existing file or directory: '") +
                    lastInput + "'.";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;

            case 3:           // after -f
                if(checkregex(arg, regexErr)){
                    lastFilter = arg;
                    state = 2;
                    break;
                }
                errmsg = string("Syntax error in '") + arg + "': " + regexErr;
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;

            case 4:           // after <filename>
                errmsg = string("Unexpected parameter: '") + arg + "'.";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;

            case 5:           // after -k
                // opath is not used
                if(Cfg::knownGraphFiles.addpath(arg, arg, R"__(.+\.graph)__")){
                    state = 0;
                    break;
                }

                errmsg = string(
                    "the path of -k must be an existing file or directory: '") +
                    arg + "'.";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;

            default:
                errmsg = "Unknown state in main_recog().";
                Log::error(errmsg);
                throw(runtime_error(errmsg));
                break;
            }
        }

        if(Cfg::bunchFiles.empty()){
            errmsg = "Missing -b option.";
            Log::error(errmsg);
            throw(runtime_error(errmsg));
        }
        if(Cfg::inputFiles.empty()){
            errmsg = "Missing -i option.";
            Log::error(errmsg);
            throw(runtime_error(errmsg));
        }
        if(Cfg::knownGraphFiles.empty()){
            errmsg = "Missing -k option.";
            Log::error(errmsg);
            throw(runtime_error(errmsg));
        }
        if(state != 4){
            errmsg = "Missing the last parameter <filename>.";
            Log::error(errmsg);
            throw(runtime_error(errmsg));
        }
    }
    catch(...){
        Log::error("Invalid Parameters. Exiting now.");
        return 1;
    }

    return recog();

}


int train()
{
    using namespace boost::filesystem;

    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Graph<40> graph;
    BunchGraph<40> bunch;
    Points<int> points;
    Points<int> startPoints;
    bool modified;

    path ifilepath, ofilepath;

    // load start points.
    if(!Cfg::startGraphFile.empty()){
        try{
            std::ifstream sgf(Cfg::startGraphFile);
            boost::archive::binary_iarchive ia(sgf);
            ia >> graph;
            startPoints = graphToPoints(graph);
            Log::info(
                std::string("Loaded start points from: '") +
                Cfg::startGraphFile + "'."
            );
        }
        catch(...){
            Log::warning(string("Failed to load start points from file: '") +
                Cfg::startGraphFile + "'."
            );
        }
    }

    // load bunch graphs.
    while(Cfg::bunchFiles.getnextfile(ifilepath,ofilepath)){
        string ifilepathstr = ifilepath.string();

        try{
            std::ifstream bf(ifilepathstr);
            boost::archive::binary_iarchive ia(bf);
            ia >> graph;

            if(startPoints.empty()){
                bunch.addGraph(graph);
                Log::info(
                    std::string("Loaded bunch graph: '") +
                    ifilepathstr + "'."
                );

                startPoints = graphToPoints(graph);
                Log::info(
                    std::string("Loaded start points from: '") +
                    ifilepathstr + "'."
                );
            }
            else {
                if(graph.getNodes().size() != startPoints.size()) {
                    Log::warning(
                        std::string("Bunch graph ignored due to a mismatch " 
                        "in the number of key points: '") +
                        ifilepathstr +
                        "'."
                    );
                    continue;
                }

                bunch.addGraph(graph);
                Log::info(
                    std::string("Loaded bunch graph: '") +
                    ifilepathstr + 
                    "'." 
                );
            }
        }
        catch(...) {
            Log::warning(
                std::string("Failed to load bunch graph: '") +
                ifilepathstr +
                "'."
            );
        }
    }

    // process input files (images to train)
    while(Cfg::inputFiles.getnextfile(ifilepath, ofilepath)){
        if(
            extensionIs(ifilepath.native(), ".jets") ||
            extensionIs(ifilepath.native(), ".graph")
        ){
            continue;
        }

        auto origfilename = ifilepath.filename();
        string ifilepathstr;
        string ofilepathstr;

        try{
            ifilepathstr = ifilepath.native();
            file_type type = status(ofilepath).type();
            if(type == directory_file){
                ofilepath /= origfilename;
                ofilepathstr = ofilepath.native() + ".graph";
            }
            else {
                ofilepathstr = ofilepath.native();
            }

            if(Cfg::noOverwrite && fileexists(ofilepathstr)){
                Log::info(
                    string("File skipped due to --no-overwrite: '") +
                    ifilepathstr +
                    "'."
                );
                continue;
            }

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
                std::string bfilepathstr = 
                    (Cfg::bunchDirectory/origfilename).native() + ".graph";
                try {
                    copy_file(
                        ofilepathstr, 
                        bfilepathstr,
                        copy_option::overwrite_if_exists
                    );
                    Log::info(
                        std::string("Added to bunch graphs: '") +
                        bfilepathstr +
                        "'."
                    );
                }
                catch(...) {
                    Log::warning(
                        std::string("Failed to create file: '") + 
                        bfilepathstr +
                        "'."
                    );
                }
            }
        }
        catch(...){
            Log::warning(
                std::string("Failed to process file: '") + 
                ifilepathstr +
                "'."
            );
        }
    }

    return 0;
}

int recog()
{
    using namespace boost::filesystem;

    Kernels<40> kernels;
    genGaborKernels(101, kernels);

    Graph<40> graph;
    BunchGraph<40> bunch;
    Points<int> points;
    Points<int> startPoints;
    bool modified;
    vector<tuple<Graph<40>,string>> knownGraphs;
    std::ofstream resultfile;

    path ifilepath, ofilepath;

    // open output file
    try {
        if(Cfg::noOverwrite && fileexists(Cfg::recogResultFile)){
            Log::info(
                string("Exiting now due to --no-overwrite: '") +
                Cfg::recogResultFile +
                "'."
            );
            return 0;
        }

        resultfile.open(Cfg::recogResultFile, ios::trunc);
    }
    catch(...){
        Log::error(
            string("Cannot open file '") +
            Cfg::recogResultFile +
            "'. Exiting now."
        );
        return 1;
    }

    // load start points.
    if(!Cfg::startGraphFile.empty()){
        try{
            std::ifstream sgf(Cfg::startGraphFile);
            boost::archive::binary_iarchive ia(sgf);
            ia >> graph;
            startPoints = graphToPoints(graph);
            Log::info(
                std::string("Loaded start points from: '") +
                Cfg::startGraphFile + "'."
            );
        }
        catch(...){
            Log::warning(string("Failed to load start points from file: '") +
                Cfg::startGraphFile + "'."
            );
        }
    }

    // load bunch graphs.
    while(Cfg::bunchFiles.getnextfile(ifilepath,ofilepath)){
        string ifilepathstr = ifilepath.string();

        try{
            std::ifstream bf(ifilepathstr);
            boost::archive::binary_iarchive ia(bf);
            ia >> graph;

            if(startPoints.empty()){
                bunch.addGraph(graph);
                Log::info(
                    std::string("Loaded bunch graph: '") +
                    ifilepathstr + "'."
                );

                startPoints = graphToPoints(graph);
                Log::info(
                    std::string("Loaded start points from: '") +
                    ifilepathstr + "'."
                );
            }
            else {
                if(graph.getNodes().size() != startPoints.size()) {
                    Log::warning(
                        std::string("Bunch graph ignored due to a mismatch " 
                        "in the number of key points: '") +
                        ifilepathstr +
                        "'."
                    );
                    continue;
                }

                bunch.addGraph(graph);
                Log::info(
                    std::string("Loaded bunch graph: '") +
                    ifilepathstr + "'."
                );
            }
        }
        catch(...) {
            Log::warning(
                std::string("Failed to load bunch graph: '") +
                ifilepathstr +
                "'."
            );
        }
    }
    if(bunch.empty()) {
        Log::error("No bunch graph loaded. Exiting now.");
        return 1;
    }

    // load known graphs
    while(Cfg::knownGraphFiles.getnextfile(ifilepath, ofilepath)){

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
        return 1;
    }

    // process input files (images to recognize)
    while(Cfg::inputFiles.getnextfile(ifilepath, ofilepath)){
        if(
            extensionIs(ifilepath.native(), ".jets") ||
            extensionIs(ifilepath.native(), ".graph")
        ){
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

    return 0;
}