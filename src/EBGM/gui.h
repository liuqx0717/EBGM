#pragma once

#include "points.hpp"

#include <opencv2/core.hpp>


class KeyPointsGui {
private:
    enum class State {
        NORMAL,
        SELECTED    // a point has been selected
    };

    static cv::Mat m_origImage, m_showImage;
    static Points<int> *m_points;
    static bool m_modified;
    static State m_state;
    static int m_selectedIndex;

    static void mouseCallback(int event, int x, int y, int flags, void *userdata);
    static void markPoint(int index);
    static void unmarkPoint(int index);
    // Call this function after modifying m_showImage.
    static void show();
    // Will return -1 if the mouse is not at any of the points.
    static int atWhichPoint(int x, int y);

public:
    // The result will be stored in 'points'.
    // Return value: whether the points have been modified by user.
    static bool modifyPoints(Points<int> &points, const cv::Mat &img);

};