# EBGM
An implementation of **EBGM** (Elastic Bunch Graph Matching) algorithm using C++. Aims at **one-sample-per-person problem** in face recognition. Reached an accuracy of **82%** on *ORL dataset*.
<br>
<br>
The source code contains both the **original** version and the **improved** version of EBGM algorithm. Comment/Uncomment either version in `alg.cpp` to switch.
<br>
<br>
**Reference**: *Face Recognition by Elastic Bunch Graph Matching*<br>
**Demo**: [demo.mp4](/demo.mp4)
<br><br>

## Features
1. **Cross-Platform**
   * Uses CMake as the build system.
   * All of the third-party libraries (OpenCV, OpenMP, Boost, Eigen) are cross-platform.
   * Avoids using platform-specific APIs.
2. **Modular**<br>
![structure](/structure.svg)
   * The upper modules can call the lower modules.
   * `utils` and `cvutils`: Some utilities (e.g. determine the extension of a file).
   * `points`: transforming a group of points (e.g. translation, stretching, and rotation).
   * `kernels`: generating single Gabor kernel, plus doing convolution operation.
   * `jets`: definition of struct Jet, plus algorithms for generating jets, comparing jets and displacement estimation of jets.
   * `graph`: definition of struct Graph and GraphBunch, plus the similarity algorithms for graphs and graph bunches.
   * `alg`: Implementation of EBGM algorithm by putting the lower modules together.
   * `iofiles`: for file enumeration.
   * `gui`: for viewing and modifying the key points on a image.
   * `ebgm`: the topmost module, processing the command-line arguments and performing face recognition.
   * `test`: test codes for all other modules.
3. **High performance**
   * Uses vectorization techniques.
   * Uses multithreading techniques.
4. **Robust**
   * Uses detailed logs (Info, Warning, Error).
   * Has a complete error handling mechanism (asserts and exceptions). In the face of various errors, it will print a detailed message, then proceed without crash.
<br><br>

## Invocation
After compiling the project, you will get a command-line program: `ebgm` (or `ebgm.exe`...).
```Usage:

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
    directory, please use a trailing "/" or "\".
```
<br><br>

## Compilation
This program has the following dependencies:
* OpenCV
* OpenMP
* Boost
* Eigen

### Linux
1. Install OpenCV, Boost and Eigen (development version) using your package manager.
2. Run the script `linux-build-debug.sh` or `linux-build-release.sh`.
3. `cd` into the directory `build-debug` or `build-release`, run `make -j4`.

### Others
1. Install CMake.
2. Download and compile OpenCV, Boost and Eigen.
3. Create a folder to store the generated build files (e.g. `build-debug`).
4. Use `cmake-gui` to configure the project. Modify the corresponding CMake cache variables (`OPENCV_LIB_PATH`, `OPENCV_INCLUDE_PATH`, `BOOST_LIB_PATH`, `BOOST_INCLUDE_PATH`, `EIGEN_INCLUDE_PATH`).
5. Compile the project using your favorite compilers.