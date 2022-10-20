#include <stdint.h>

#include "DBoW3/DBoW3.h"
#include "opencv2/xfeatures2d.hpp"


#include "frame.h"
#include "io.h"

void query(int image_step, std::string feature_type, int feature_nums,
          int nums_candidate) {
  // create database
  std::map<uint64_t, std::pair<std::string, cv::Vec3d>> train_images =
    readImagesAndGNSS("../dataset/train/recording_2021-02-25_13-39-06_images");

  std::map<int64_t, Frame> train_frames, test_frames;  // map image id to frame
  std::map<DBoW3::EntryId, int64_t> entry_id_to_frame_id;

  std::vector<cv::Mat> vDes_train, vDes_test;

  // detect feature points
  int i = 0;
  for (auto& elem : train_images) {
    if (i % image_step != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second, feature_type,
      feature_nums);
    train_frames.insert(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  std::cout << "Creating a small database..." << std::endl;

  // load the vocabulary from disk
  std::string path_to_voc = feature_type + ".yml.gz";
  DBoW3::Vocabulary voc(path_to_voc);

  DBoW3::Database db(voc, false, 0);  // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (auto& elem : train_frames) {
    DBoW3::EntryId eid = db.add(elem.second.getDes());
    entry_id_to_frame_id[eid] = elem.first;
  }

  std::cout << "... done!" << std::endl;

  // query

  std::map<uint64_t, std::pair<std::string, cv::Vec3d>> test_images =
    readImagesAndGNSS("../dataset/test/recording_2020-12-22_12-04-35_images");

  int i2 = 0;
  for (auto& elem : test_images) {
    if (i2++ % 5 !=0)
        continue;
    Frame frame(elem.first, elem.second.first, elem.second.second, feature_type,
                feature_nums);
    test_frames.insert(std::make_pair(elem.first, frame));
    vDes_test.push_back(frame.getDes());
  }

  int success = 0;
  int failure = 0;
  DBoW3::QueryResults ret;
  for (auto& elem : test_frames) {
    db.query(elem.second.getDes(), ret, nums_candidate);

    cv::Vec3d pose_test = elem.second.getPose();
    cv::Mat image_test = elem.second.getImage();
    std::vector<cv::KeyPoint> keypoints_test = elem.second.getKpts();
    cv::Mat descriptors_test = elem.second.getDes();


    for (int i=0; i < ret.size(); i++) {
      uint64_t frame_id = entry_id_to_frame_id[ret[i].Id];
      cv::Vec3d pose_train = train_frames.find(frame_id)->second.getPose();
      cv::Mat image_train = train_frames.find(frame_id)->second.getImage();
      std::vector<cv::KeyPoint> keypoints_train =
        train_frames.find(frame_id)->second.getKpts();
      cv::Mat descriptors_train = train_frames.find(frame_id)->second.getDes();

      cv::Vec3d dis = pose_train - pose_test;
      cv::Vec2d txy(dis[0], dis[1]);
      cv::Vec2d tz(dis[2]);

      if (1) {
        std::cout << "candidate: " << i << " score: " << ret[i].Score <<
        std::endl;
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches, gms_matches;
        matcher.match(descriptors_test, descriptors_train, matches);
        cv::xfeatures2d::matchGMS(image_test.size(), image_train.size(),
          keypoints_test, keypoints_train, matches, gms_matches);
        cv::Mat image_match;
        cv::drawMatches(image_test, keypoints_test,
                        image_train, keypoints_train,
                        gms_matches, image_match);
        cv::imshow("image_match", image_match);
        cv::waitKey(0);
      }


      if (cv::norm(txy, cv::NORM_L2) < 5.0 && cv::norm(tz, cv::NORM_L2) < 1.0) {
        success++;
        break;
      }

      if (i == ret.size()-1)
        failure++;
    }
  }
  std::cout << "success: " << success << " failure: " << failure << std::endl;
}


int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr <<
    "usage: ./query image_step feature_type feature_nums nums_candidate"
    << std::endl;
  }

  int image_step = std::stoi(argv[1]);
  int feature_nums = std::stoi(argv[3]);
  int nums_candidate = std::stoi(argv[4]);
  std::string feature_type = argv[2];

  query(image_step, feature_type, feature_nums, nums_candidate);
}
