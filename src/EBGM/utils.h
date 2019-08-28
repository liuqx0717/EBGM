#pragma once

#include <tuple>
#include <type_traits>
#include <string>
#include <ostream>

#include <boost/filesystem.hpp>

#define PI 3.14159265358979323846

// Convert real and imaginary parts of complex number
// to magnitude and phase. Phase is within [-0.5pi, 1.5pi).
template<typename T>
std::tuple<T/*Magnitude*/, T/*Phase*/>
complex2mag(T re, T im)
{
    T mag = sqrt(re*re + im*im);
    T phase = atan(im/re);
    
    if(re < 0.0) {
        phase += PI;
    }

    return std::make_tuple(mag, phase);

}

// Map an angle to the range [0, 2pi) radians.
// If you want to map to [a, a+2pi), use: wrapAngle(angle-a) + a.
template<typename T>
T wrapAngle(T angle)
{
    const T twoPi = 2.0 * PI;
    return angle - twoPi*floor(angle/twoPi);
}

class Log {
public:
    enum class MsgType {
        INFO,
        WARNING,
        ERROR
    };

    static std::ostream *os;


    static void log(const std::string &str, MsgType type);

    static void info(const std::string &str)
    {
        log(str, MsgType::INFO);
    }
    static void warning(const std::string &str)
    {
        log(str, MsgType::WARNING);
    }
    static void error(const std::string &str)
    {
        log(str, MsgType::ERROR);
    }
};


// if path doesn't exist, return false
// if path is a zero-length file, return false
// if path is a directory, return false
bool fileexists(const boost::filesystem::path &path);

bool extensionIs(const std::string &filename, const std::string &ext);

// if path is a file, return its parent directory.
// if path is a directory, return itself.
// if neither, throw a runtime_error.
boost::filesystem::path getDirectory(const boost::filesystem::path &path);

// check the syntax of a regular expression
bool checkregex(const char *regex, std::string &errordescription) noexcept;


#undef PI