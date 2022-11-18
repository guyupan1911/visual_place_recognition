/*
 -------------------------------------------------------------------------------
 * Frame implementation file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#include "frame.h"

DEFINE_int32(image_row, 2160, "image_row");
DEFINE_int32(image_col, 2160, "image_col");

Frame::Frame(uint64_t frame_id, const std::string& path2im, cv::Vec3d pose,
std::string type, int feature_nums): id_(frame_id), pose_(pose), path_to_image(path2im) {
  cv::Mat image = cv::imread(path2im, cv::IMREAD_GRAYSCALE);
  if (image.empty())
    std::cout << "wrong path to image : " << path2im << std::endl;
  // cv::equalizeHist(image, image_);
  image_ = image.clone();
  // mask_ = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
  // detect feature and compute descriptor
  cv::Ptr<cv::Feature2D> feature_detector;
  if (type == "ORB")
    feature_detector = cv::ORB::create(feature_nums);
  else if (type == "SURF")
    feature_detector = cv::xfeatures2d::SURF::create(200, 4, 3, true);
  else if (type == "SIFT")
    feature_detector = cv::SIFT::create(feature_nums);
  else if (type == "ORB2") {
    ORB_SLAM2::ORBextractor feature_detector(feature_nums, 1.2, 8, 40, 14);
    feature_detector(image_, mask_, keypoints_, descriptors_);
    // cv::drawKeypoints(image_, keypoints_, image_);
    // cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    // cv::namedWindow("image_with_keypoints", cv::WINDOW_NORMAL);
    // cv::imshow("image_with_keypoints", image_);
    // cv::waitKey(5);
    map_points_.resize(keypoints_.size(), nullptr);
    return;
  }

  feature_detector->detectAndCompute(image_, mask_, keypoints_, descriptors_);
  // cv::drawKeypoints(image_, keypoints_, image_);
  // cv::namedWindow("image_with_keypoints", cv::WINDOW_NORMAL);
  // cv::imshow("image_with_keypoints", image_);
  // cv::waitKey(5);
  // cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
}

void Frame::updateConnections() {
  std::vector<std::shared_ptr<MapPoint>> mpts;
  std::map<std::shared_ptr<Frame>, int> Fcounter;

  mpts = map_points_;
  for (auto it = mpts.begin(); it != mpts.end(); ++it) {
    std::shared_ptr<MapPoint> mpt = *it;
    if (mpt == nullptr)
      continue;

    std::map<std::shared_ptr<Frame>, double> observations =
      mpt->getObservations();

    for (auto mit = observations.begin(); mit != observations.end(); ++mit) {
      if (mit->first->getId() == id_)
        continue;
      Fcounter[mit->first]++;
    }
  }

  connections_ = Fcounter;

  // print connection information
  std::cout << "--------connection information-------- " << std::endl
  << "Frame " << id_ << " connected with: " << std::endl << std::endl;
  for (const auto &elem : connections_) {
    std::cout << "Frame: " << elem.first->getId() << " with " << elem.second 
    << " points" << std::endl;
  }
}

int MapPoint::nextId = 0;

MapPoint::MapPoint() {
  id = nextId++;
}