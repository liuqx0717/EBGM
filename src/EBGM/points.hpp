#pragma once

#include <cassert>
#include <vector>
#include <tuple>
#include <initializer_list>
#include <cmath>
#include <type_traits>


#ifndef NDEBUG
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#endif

// Stores a set of points (their coordinates) and performs
// transformations on these points.
// T: original type for the coordinates. This class will
// always use float for internal operations.
template<typename T>
class Points {
public:
    using Point = struct {T x; T y;};

private:
    // internal type for storing and transforming
    using _Point = struct {float x; float y;};

    std::vector<_Point> m_points;

    // function for converting float to T
    // (int)1.999F = 1, (int)round(1.999F) = 2
    template<typename T2 = T>
    typename std::enable_if<std::is_integral<T2>::value, T2>::type
    convFunc(float f) const
    {
        return (T2)round(f);
    }

    template<typename T2 = T>
    typename std::enable_if<!std::is_integral<T2>::value, T2>::type
    convFunc(float f) const
    {
        return (T2)f;
    }

    float m_maxX, m_maxY;
    float m_minX, m_minY; 

    void recalcMinMax()
    {
        for(auto &i: m_points){
            if(i.x > m_maxX){
                m_maxX = i.x;
            }
            else if(i.x < m_minX){
                m_minX = i.x;
            }

            if(i.y > m_maxY){
                m_maxY = i.y;
            }
            else if(i.y < m_minY){
                m_minY = i.y;
            }
        }
    }

public:
    Points() noexcept {}

    Points(std::initializer_list<Point> points) noexcept
    {
        addPoint(points);
    }

    int size() const 
    {
        return m_points.size();
    }

    bool empty() const
    {
        return m_points.empty();
    }

    void clear() {
        m_points.clear();
    }

    void addPoint(Point point)
    {
        _Point _point{(float)point.x, (float)point.y};

        // initialization
        if(m_points.empty()){
            m_maxX = m_minX = _point.x;
            m_maxY = m_minY = _point.y;
        }

        m_points.push_back(_point);

        if(point.x > m_maxX){
            m_maxX = point.x;
        }
        else if(point.x < m_minX){
            m_minX = point.x;
        }

        if(point.y > m_maxY){
            m_maxY = point.y;
        }
        else if(point.y < m_minY){
            m_minY = point.y;
        }

    }

    void addPoint(std::initializer_list<Point> points)
    {
        for(auto i: points){
            addPoint(i);
        }
    }

    // This function will cause the iteration through all the points
    // to re-calculate the data range.
    void modifyPoint(Point point, int index)
    {
        assert(index < m_points.size() && index >=0);

        _Point _point{(float)point.x, (float)point.y};

        m_points[index] = _point;

        recalcMinMax();
    }

    Point get(int i) const
    {
        assert(i < m_points.size() && i >=0);

        const _Point &_point = m_points[i];
        
        Point ret;
        ret.x = convFunc(_point.x);
        ret.y = convFunc(_point.y);

        return ret;
    }

    // minX, minY, maxX, maxY
    std::tuple<T/*minX*/, T/*minY*/, T/*maxX*/, T/*maxY*/>
    getMinMax() const
    {
        assert(!m_points.empty());

        return std::make_tuple(
            convFunc(m_minX), 
            convFunc(m_minY), 
            convFunc(m_maxX), 
            convFunc(m_maxY)
        );
    }

    // whether all points in this class fall into the specified range
    bool isInRange(T minX, T minY, T maxX, T maxY) const
    {
        if(
            convFunc(m_minX) >=  minX &&
            convFunc(m_minY) >=  minY &&
            convFunc(m_maxX) <=  maxX &&
            convFunc(m_maxY) <=  maxY 
        ) {
            return true;
        }
        else {
            return false;
        }
    }

    // return a reference to itself
    Points &translate(float tx, float ty)
    {
        for(auto &i: m_points){
            i.x += tx;
            i.y += ty;
        }

        m_maxX += tx;
        m_minX += tx;
        m_maxY += ty;
        m_minY += ty;
        
        return *this;
    }

    // return a reference to itself
    Points &scale(
        float centerX,
        float centerY, 
        float scaleX, 
        float scaleY
    )
    {
        for(auto &i: m_points){
            i.x = centerX + (i.x - centerX)*scaleX;
            i.y = centerY + (i.y - centerY)*scaleY;
        }

        m_maxX = centerX + (m_maxX - centerX)*scaleX;
        m_minX = centerX + (m_minX - centerX)*scaleX;
        m_maxY = centerY + (m_maxY - centerY)*scaleY;
        m_minY = centerY + (m_minY - centerY)*scaleY;

        return *this;
    }

    // center: (minX + maxX)/2, (minY + maxY)/2
    // return a reference to itself
    Points &scale(float scaleX, float scaleY)
    {
        return scale(
            (m_minX + m_maxX) / 2.0F,
            (m_minY + m_maxY) / 2.0F,
            scaleX,
            scaleY
        );
    }


    // angle: in radian
    // return a reference to itself
    Points &rotate(float centerX, float centerY, float angle)
    {
        for(auto &i: m_points){
            float deltaX = i.x - centerX;
            float deltaY = i.y - centerY;
            i.x = centerX + deltaX*cos(angle) - deltaY*sin(angle);
            i.y = centerY + deltaX*sin(angle) + deltaY*cos(angle);
        }

        recalcMinMax();

        return *this;
    }

    // angle: in radian
    // center: (minX + maxX)/2, (minY + maxY)/2
    // return a reference to itself
    Points &rotate(float angle)
    {
        return rotate(
            (m_minX + m_maxX) / 2.0F,
            (m_minY + m_maxY) / 2.0F,
            angle
        );
    }




#ifndef NDEBUG
    std::string toString()
    {
        std::ostringstream buff, tmp;
        std::string ret;

        buff << std::left << std::setw(20) << "internal format";
        buff << std::left << std::setw(20) << "original format" << "\n";
        for(auto &i: m_points){
            // clear tmp
            tmp.str(std::string());
            tmp << "(";
            tmp << std::setprecision(2) << i.x;
            tmp << ", ";
            tmp << std::setprecision(2) << i.y;
            tmp << ")";
            buff << std::left << std::setw(20) << tmp.str();

            // clear tmp
            tmp.str(std::string());
            tmp << "(";
            tmp << std::setprecision(2) << convFunc(i.x);
            tmp << ", ";
            tmp << std::setprecision(2) << convFunc(i.y);
            tmp << ")";
            buff << std::left << std::setw(20) << tmp.str() << "\n";
        }

        tmp.str(std::string());
        tmp << "minX: " << std::setprecision(2) << m_minX;
        buff << std::left << std::setw(20) << tmp.str();

        tmp.str(std::string());
        tmp << "maxX: " << std::setprecision(2) << m_maxX;
        buff << std::left << std::setw(20) << tmp.str() << "\n";

        tmp.str(std::string());
        tmp << "minY: " << std::setprecision(2) << m_minY;
        buff << std::left << std::setw(20) << tmp.str();

        tmp.str(std::string());
        tmp << "maxY: " << std::setprecision(2) << m_maxY;
        buff << std::left << std::setw(20) << tmp.str() << "\n";

        ret = buff.str();
        return ret;
    }



#endif

};