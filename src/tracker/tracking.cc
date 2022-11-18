#include "tracking.h"


Tracking::Tracking() {
  feature_matcher_ = std::make_shared<FeatureMatcher>(FeatureMatcher());
}

void Tracking::track(const std::string &path_to_image, const cv::Vec3d pose,
  uint64_t frame_id) {
    current_frame_ = std::make_shared<Frame>(Frame(frame_id, path_to_image,
      pose, "ORB2", 2000));
    if (last_frame_ == nullptr) {
      last_frame_ = current_frame_;
      vFrames_.push_back(last_frame_);
      return;
    }

    // feature matching 
    std::vector<cv::DMatch> matches;
    feature_matcher_->matchByBruteForce(
      last_frame_->getDes(), last_frame_->getKpts(),
      current_frame_->getDes(), current_frame_->getKpts(),
      matches, last_frame_->getImage().size(), FeatureMatcher::GMS);

    std::cout << "matches: " << matches.size() << std::endl;

    //draw matches
    cv::Mat im_match;
    cv::drawMatches(last_frame_->getImage(), last_frame_->getKpts(),
      current_frame_->getImage(), current_frame_->getKpts(), matches, im_match);

    cv::namedWindow("im match", cv::WINDOW_NORMAL);
    cv::imshow("im match", im_match);
    cv::waitKey(5);

    // create new map points and update connections
    // 1. create new map points;
    for (const auto elem : matches) {
      if(last_frame_->map_points_[elem.queryIdx] == nullptr) {
        std::shared_ptr<MapPoint> mpt(new MapPoint());
        mpt->add_Observations(last_frame_);
        mpt->add_Observations(current_frame_);
        last_frame_->map_points_[elem.queryIdx] = mpt;
        current_frame_->map_points_[elem.trainIdx] = mpt;
      }
      else {
        std::shared_ptr<MapPoint> mpt = last_frame_->map_points_[elem.queryIdx];
        mpt->add_Observations(current_frame_);
        current_frame_->map_points_[elem.trainIdx] = mpt;
      }      
    }

    last_frame_->updateConnections();
    current_frame_->updateConnections();

    vFrames_.push_back(current_frame_);
    last_frame_ = current_frame_;
}

void Tracking::save(const std::string filename) {

  std::ofstream fout(filename);

  for(auto& elem : vFrames_) {
    uint64_t id = elem->getId();
    fout.write((char*)&id, sizeof(id));
    int n = elem->connections_.size();
    fout.write((char*)&n, sizeof(n));
    for(auto elem2 : elem->connections_) {
      uint64_t connected_id = elem2.first->getId();
      fout.write((char*)&connected_id, sizeof(connected_id));
    }
  }
}
