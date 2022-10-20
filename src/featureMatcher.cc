#include "featureMatcher.h"

void FeatureMatcher::matchByBruteForce(
  const cv::Mat &descriptors1, const std::vector<cv::KeyPoint> &keypoints1,
  const cv::Mat &descriptors2, const std::vector<cv::KeyPoint> &keypoints2,
  std::vector<cv::DMatch> &matches, cv::Size image_size, bool useGMS) {
  if(descriptors1.type() == CV_8U) {
      cv::BFMatcher matcher(cv::NORM_HAMMING, true);
      matcher.match(descriptors1, descriptors2, matches);
      // std::cout << "matches size: " << matches.size() << std::endl;

      if (useGMS) {
        std::vector<cv::DMatch> gms_matches;
        cv::xfeatures2d::matchGMS(image_size, image_size, keypoints1,
        keypoints2, matches, gms_matches);
        matches = gms_matches;
      }
  }
}