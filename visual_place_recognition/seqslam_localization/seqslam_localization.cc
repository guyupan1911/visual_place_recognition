#include "visual_place_recognition/seqslam_localization/seqslam_localization.h"
#include "visual_place_recognition/common/converter.h"
#include "visual_place_recognition/common/proto/map.pb.h"

#include <fstream>

void SeqSlamLocalization::mapping(const std::vector<Frame> &frames,
  const std::string &filename) {
  std::cout << "train frame size: " << frames.size() << std::endl;
  
  double trajectory_distance = 0;
  cv::Vec3d last_frame_pose;
  cv::Vec3d last_pose;
  bool first = true;
  Map::Map serializableMap;
  for (const auto &elem : frames) {
    if (first) {
      first = false;
      Map::Frame seriablized_frame = Converter::FrameToFrameMsg(elem);
      serializableMap.add_frames()->CopyFrom(seriablized_frame);
      last_pose = elem.gnss_pose_;
      last_frame_pose = elem.gnss_pose_;
      continue;
    }

    double ds = cv::norm(elem.gnss_pose_-last_frame_pose);
    if( ds < 0.3) {
      trajectory_distance += cv::norm(elem.gnss_pose_-last_pose);
      last_pose = elem.gnss_pose_;
      continue;
    }

    Map::Frame seriablized_frame = Converter::FrameToFrameMsg(elem);
    serializableMap.add_frames()->CopyFrom(seriablized_frame);
    trajectory_distance += cv::norm(elem.gnss_pose_-last_pose);
    last_pose = elem.gnss_pose_;
    last_frame_pose = elem.gnss_pose_;
  }

  std::cout << "Map size: " << serializableMap.frames_size() << std::endl;
  std::cout << "trajectory distance: " << trajectory_distance << std::endl;

  std::ofstream fout;
  fout.open(filename, std::ios::out | std::ios::binary);
  if(!serializableMap.SerializeToOstream(&fout)) {
    std::cout << "fail to serialize" << std::endl;
  }
  else {
    std::cout << "serialize map to file" << std::endl;
  }
}

void SeqSlamLocalization::fetchMap(const std::string &url_to_map_1, const std::string &url_to_map_2) {
  std::fstream fin;
  fin.open(url_to_map_1, std::ios::in | std::ios::binary);

  Map::Map serialized_map;
  if (!serialized_map.ParseFromIstream(&fin)) {
    std::cout << "fail to read from file" << std::endl;
  }

  vFrames_[0].clear();

  for (int i = 0; i < serialized_map.frames_size(); ++i) {
    vFrames_[0].push_back(Converter::FrameMsgToFrame(serialized_map.frames(i)));
    if (0) {
      cv::namedWindow("resized_image", cv::WINDOW_NORMAL);
      cv::imshow("resized_image", vFrames_[0][i].resized_image_);
      cv::waitKey(5);
    }
  }

  std::cout << "train_Frames size: " << vFrames_[0].size() << std::endl;

  std::fstream fin_test;
  fin_test.open(url_to_map_2, std::ios::in | std::ios::binary);

  Map::Map serialized_map_test;
  if (!serialized_map_test.ParseFromIstream(&fin_test)) {
    std::cout << "fail to read from file" << std::endl;
  }

  vFrames_[1].clear();

  for (int i = 0; i < serialized_map_test.frames_size(); ++i) {
    vFrames_[1].push_back(Converter::FrameMsgToFrame(serialized_map_test.frames(i)));
    if (0) {
      cv::namedWindow("resized_image", cv::WINDOW_NORMAL);
      cv::imshow("resized_image", vFrames_[0][i].resized_image_);
      cv::waitKey(5);
    }
  }

  std::cout << "test_Frames size: " << vFrames_[1].size() << std::endl;
}

void SeqSlamLocalization::calculate_difference_matrix() {
  int m = vFrames_[0].size(); // train data size
  int n = vFrames_[1].size(); // test_data size
  difference_matrix_ = cv::Mat::zeros(m,n,CV_32FC1);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cv::Mat abs_difference;
      cv::subtract(vFrames_[0][i].resized_image_, vFrames_[1][j].resized_image_,
        abs_difference);
      abs_difference = cv::abs(abs_difference);
      double sad = cv::sum(abs_difference)[0];
      float normalized_sad = sad / m;
      difference_matrix_.at<float>(i,j) = normalized_sad;
    }
  }

  std::cout << "difference_matrix_: " << difference_matrix_ << std::endl;

  cv::normalize(difference_matrix_, D_, 0, 255, cv::NORM_MINMAX);
  D_.convertTo(D_, CV_8UC1);
  cv::cvtColor(D_, D_, cv::COLOR_GRAY2BGR);

  if (1) {
    cv::namedWindow("D", cv::WINDOW_NORMAL);
    cv::imshow("D", D_);
    cv::waitKey(0);
  }
  cv::imwrite("../map/D.png", D_);
}

void SeqSlamLocalization::drawMatch(int trainId, int testId) {
  
  cv::Vec3d train_pose = vFrames_[0][trainId].gnss_pose_;
  cv::Vec3d test_pose = vFrames_[1][testId].gnss_pose_;

  double distance = cv::norm(train_pose-test_pose);
  // std::cout << "distance: " << distance << std::endl;
  
  if (distance < 10.0)
    success++;
  else
    failure++;

  cv::Mat train_image = cv::imread(vFrames_[0][trainId].url_to_raw_image_, cv::IMREAD_COLOR);
  cv::Mat test_image = cv::imread(vFrames_[1][testId].url_to_raw_image_, cv::IMREAD_COLOR);
  cv::Mat match_result;
  cv::hconcat(test_image, train_image, match_result);

 std::stringstream ss;
  ss << " distance: " << distance << std::endl;
  cv::putText(match_result, ss.str(), cv::Point(5,350), cv::FONT_HERSHEY_DUPLEX,0.8, cv::Scalar(0,255,0));

  cv::circle(DD_, cv::Point(testId, trainId), 1, cv::Scalar(0,0,255), -1);

  cv::namedWindow("match_result", cv::WINDOW_NORMAL);
  cv::namedWindow("DD", cv::WINDOW_NORMAL);

  cv::imshow("match_result", match_result);
  cv::imshow("DD", DD_);
  cv::waitKey(5);
}

void SeqSlamLocalization::findMatchByDifferenceMatrix(int testId) {
  int best_match = -1;
  double best_score = 1e10;
  int second_best_match = -1;
  double second_best_score = 1e10;

  for (int i = 0; i < difference_matrix_.rows; ++i) {
    float score = difference_matrix_.at<float>(i, testId);
    if (score < best_score) {
      second_best_match = best_match;
      second_best_score = best_score;
      best_match = i;
      best_score = score;
    }
    else if(score < second_best_score) {
      second_best_match = i;
      second_best_score = score;
    }
  }

  drawMatch(best_match, testId);
}

cv::Mat SeqSlamLocalization::calculateTrajectories(int length) {
  float v_min = 0.6;
  float v_max = 1.8;
  float step = 0.2;
  // cv::Mat seqIndex = cv::Mat::zeros(int((v_max-v_min)/step), length, CV_8SC1);
  cv::Mat seqIndex;

  for (float v = v_min; v < v_max; v += step) {
    cv::Mat one_trajectory = cv::Mat::zeros(1, length, CV_8SC1);
    for (int i = 0; i < length; ++i) {
      one_trajectory.at<char>(0,i) = -1 * i * v;
    }
    seqIndex.push_back(one_trajectory);
  }

  return seqIndex;
}

void SeqSlamLocalization::findMatchByTrajectory(int testId) {
  int best_match = -1;
  double best_score = 1e10;
  int second_best_match = -1;
  double second_best_score = 1e10;
  
  int length = 20;
  cv::Mat SeqIndex = calculateTrajectories(length);

  for (int trainId = 0; trainId < difference_matrix_.rows; ++trainId) {
    double min_score = 1e10;
    for(int n = 0; n < SeqIndex.rows; ++n) {
      cv::Mat trajectory = SeqIndex.row(n);
      // sum score
      double sum_score = 0;
      for(int k = 0; k < trajectory.cols; ++k) {
        int q_id = testId - k;
        int t_id = trajectory.at<char>(0,k) + trainId;
        if(q_id < 0 || t_id < 0)
          continue;
        sum_score += enhanced_difference_matrix_.at<float>(t_id, q_id);
      }
      if (sum_score < min_score)
        min_score = sum_score;
    }
    if (min_score < best_score) {
      second_best_score = best_score;
      second_best_match = best_match;
      best_match = trainId;
      best_score = min_score;
    }
    else if (min_score < second_best_score) {
      second_best_score = min_score;
      second_best_match = best_match;
    }
  }

  drawMatch(best_match, testId);
}

void SeqSlamLocalization::enhanced_difference_matrix() {
  int width = 20;
  enhanced_difference_matrix_ = difference_matrix_.clone();
  for (int i = 0; i < difference_matrix_.rows; ++i) {
    int a = i-width/2 > 0 ? i-width/2 : 0;
    int b = i+width/2 < difference_matrix_.rows? i+width/2 : difference_matrix_.rows;

    for (int j = 0; j < difference_matrix_.cols; ++j) {
      float value = difference_matrix_.at<float>(i,j);
      cv::Mat v = difference_matrix_.rowRange(a,b).col(j).clone();
      cv::Mat mean, stddev;
      cv::meanStdDev(v, mean, stddev);
      float mean_value = mean.at<double>(0,0);
      float stddev_value = stddev.at<double>(0,0);
      float enhanced_value = (value - mean_value) / stddev_value;
      enhanced_difference_matrix_.at<float>(i,j) = enhanced_value;
    }
  } 
  cv::normalize(enhanced_difference_matrix_, DD_, 0, 255, cv::NORM_MINMAX);
  DD_.convertTo(DD_, CV_8UC1);
  cv::cvtColor(DD_, DD_, cv::COLOR_GRAY2BGR);

  if (1) {
    cv::namedWindow("DD", cv::WINDOW_NORMAL);
    cv::imshow("DD", DD_);
    cv::waitKey(0);
  }
  cv::imwrite("../map/DD.png", DD_);
}

void SeqSlamLocalization::run() {
  calculate_difference_matrix();
  enhanced_difference_matrix();
  for(int i=0; i < enhanced_difference_matrix_.cols; ++i) {
    findMatchByTrajectory(i);
    // findMatchByDifferenceMatrix(i);
  }
  std::stringstream ss;
  ss << "ratio: " << float(success) / float(success+failure)
     << " length: " << int(15) << std::endl;
  cv::putText(DD_, ss.str(), cv::Point(5,50), cv::FONT_HERSHEY_DUPLEX,0.5, cv::Scalar(0,255,0));
  cv::imwrite("../map/DD_result.png", DD_);
}