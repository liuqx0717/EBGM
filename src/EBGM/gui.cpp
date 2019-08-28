#include "gui.h"
#include "points.hpp"
#include "cvutils.h"


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include <iostream>


using namespace std;
using namespace cv;

cv::Mat KeyPointsGui::m_origImage, KeyPointsGui::m_showImage;
Points<int> *KeyPointsGui::m_points;
bool KeyPointsGui::m_modified;
KeyPointsGui::State KeyPointsGui::m_state;
int KeyPointsGui::m_selectedIndex;

// The result will be stored in 'points'.
// img: you'd better use a matrix with color (3-channel).
// Return value: whether the points have been modified by user.
bool KeyPointsGui::modifyPoints(Points<int> &points, const cv::Mat &img)
{
    namedWindow("Key Points", WINDOW_NORMAL);
    setMouseCallback("Key Points", &mouseCallback);

    m_origImage = img;
    m_showImage = m_origImage.clone();
    m_points = &points;
    m_modified = false;
    

    int nPoint = m_points->size();
    for(int i=0; i<nPoint; i++){
        markPoint(i);
    }

    show();
    waitKey(0);

    return m_modified;
}

void KeyPointsGui::show()
{
    imshow("Key Points", m_showImage);
}

void KeyPointsGui::mouseCallback(
    int event, int x, int y, int flags, void *userdata
)
{
    int index;

    switch (event)
    {
        case EVENT_LBUTTONDOWN:
            //cout << x << ", " << y << "\n";
            switch (m_state)
            {
                case State::NORMAL:
                    if(flags & EVENT_FLAG_CTRLKEY) {
                        m_points->addPoint({x, y});
                        m_modified = true;
                        markPoint(m_points->size() - 1);
                        show();
                    }
                    else {
                        index = atWhichPoint(x, y);
                        if(index == -1){
                            break;
                        }
                        unmarkPoint(index);
                        m_selectedIndex = index;
                        m_state = State::SELECTED;
                        show();
                    }
                    break;
                case State::SELECTED:
                    m_points->modifyPoint({x, y}, m_selectedIndex);
                    m_modified = true;
                    markPoint(m_selectedIndex);
                    m_state = State::NORMAL;
                    show();
                    break;
                default:
                    break;
            }
            break;
        case EVENT_RBUTTONDOWN:
            switch (m_state)
            {
                case State::SELECTED:
                    markPoint(m_selectedIndex);
                    m_state = State::NORMAL;
                    show();
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }
}



void KeyPointsGui::markPoint(int index)
{
    auto point = m_points->get(index);
    cv::circle(m_showImage, {point.x, point.y}, 2, CV_RGB(255, 0, 0));
}

void KeyPointsGui::unmarkPoint(int index)
{
    int nPoints = m_points->size();

    m_showImage = m_origImage.clone();

    for(int i=0; i<index; i++){
        markPoint(i);
    }
    for(int i=index+1; i<nPoints; i++){
        markPoint(i);
    }
}

// Will return -1 if the mouse is not at any of the points.
int KeyPointsGui::atWhichPoint(int x, int y)
{
    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

    // the max distance to the point
    const float tolerance = 4.0F;

    float minDist2 = std::numeric_limits<float>::infinity();
    int minDistIndex;

    int nPoint = m_points->size();
    for(int i=0; i<nPoint; i++) {
        float dist2, deltaX, deltaY;
        Points<int>::Point point;

        point = m_points->get(i);
        deltaX = (float)x - point.x;
        deltaY = (float)y - point.y;
        dist2 = deltaX*deltaX + deltaY*deltaY;

        if(dist2 < minDist2) {
            minDist2 = dist2;
            minDistIndex = i;
        }
    }

    if(minDist2 <= tolerance*tolerance) {
        return minDistIndex;
    }
    else {
        return -1;
    }
}
