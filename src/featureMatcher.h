#ifndef FEATURE_MATCHER_H_
#define FEATURE_MATCHER_H_

#include <vector>

#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

class FeatureMatcher {
 public:
  void matchByBruteForce(
    const cv::Mat &descriptors1, const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Mat &descriptors2, const std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches, cv::Size image_size, bool useGMS = true);


};

#endif