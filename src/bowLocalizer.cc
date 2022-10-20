#include "bowLocalizer.h"

BowLocalizer::BowLocalizer() {
  load();
}

void BowLocalizer::load() {
  // load database
  std::cout << "load database..." << std::endl;
  pDataBase_ = new DBoW3::Database("db.yml.gz");
  readEntry2Frame("entry2frame.bin");

  // load train data
  std::string path_to_dir =
    "../dataset/train/recording_2021-02-25_13-39-06_images";
  trainData_ = DataIO(path_to_dir);
  std::cout << "done" << std::endl;

  relocalize_result_ = cv::Mat::zeros(420, 2*800, CV_8UC3);
  result_bar_ = cv::Mat::zeros(10, 1199, CV_8UC3);

  video_writer_ = cv::VideoWriter("result.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
    relocalize_result_.size(), true);

  if (!video_writer_.isOpened())
    std::cout << "not open" << std::endl;
}

void BowLocalizer::readEntry2Frame(std::string filename) {
  std::ifstream fin(filename);
  unsigned int size;
  fin.read((char*)&size, sizeof(size));
  for (int i = 0; i < size; ++i) {
    DBoW3::EntryId entry_id;
    uint64_t frame_id;
    fin.read((char*)&entry_id, sizeof(DBoW3::EntryId));
    fin.read((char*)&frame_id, sizeof(uint64_t));
    databaseID_to_FrameID_.insert(std::make_pair(entry_id, frame_id));
  }
  fin.close();
}

void BowLocalizer::relocalize(const MetaData& data) {
  // load config file
  cv::FileStorage fs_read("config.yaml", cv::FileStorage::READ);
  std::string feature_type;
  int feature_size;
  fs_read["feature_type"] >> feature_type;
  fs_read["feature_size"] >> feature_size;
  
  cv::Vec3d pose{data.gnss_pose[0], data.gnss_pose[1], data.gnss_pose[2]};
  Frame frame(data.image_id, data.path_to_image, pose, feature_type,
    feature_size);
  cv::Mat image_test = frame.getImage();
  cv::cvtColor(image_test, image_test, cv::COLOR_GRAY2BGR);
  cv::Mat descriptor_test = frame.getDes();
  std::vector<cv::KeyPoint> keypoints_test = frame.getKpts();

  DBoW3::QueryResults results;
  pDataBase_->query(frame.getDes(), results, 8);

  static int success = 0;
  static int fail = 0;
  std::cout << "success: " << success << " fail: " << fail << std::endl;

  for (int i=0; i < results.size(); i++) {
    uint64_t frameId = databaseID_to_FrameID_[results[i].Id];
    MetaData temp = trainData_.findByFrameId(frameId);
    cv::Vec3d pose_train{temp.gnss_pose[0], temp.gnss_pose[1],
      temp.gnss_pose[2]};

    Frame frame_train(temp.image_id, temp.path_to_image, pose, feature_type,
      feature_size);

    cv::Mat image_train = frame_train.getImage();
    cv::cvtColor(image_train, image_train, cv::COLOR_GRAY2BGR);

    cv::Mat descriptor_train = frame_train.getDes();
    std::vector<cv::KeyPoint> keypoints_train = frame_train.getKpts();

    cv::Vec3d dis = pose_train - pose;
    cv::Vec2d txy(dis[0], dis[1]);
    cv::Vec2d tz(dis[2]);

    // feature matching
    FeatureMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.matchByBruteForce(descriptor_test, keypoints_test, descriptor_train, 
      keypoints_train, matches, image_test.size(), true);

    // std::cout << "matches size: " << matches.size() << std::endl;


    if ( (cv::norm(txy, cv::NORM_L2) < 10.0 && cv::norm(tz, cv::NORM_L2) < 3.0 && (matches.size() > 18 || i==0) )) {
    // if ( matches.size() > 18 ) {
    
      success++;
      // std::cout << "image_train channels: " << image_train.channels()
      //           << "relocalize_result_ channels: " << relocalize_result_.channels() << std::endl;
      // cv::imshow("image_test", frame.getImage());
      // cv::imshow("image_train", image_train);
      // cv::waitKey(5);
      image_test.copyTo(relocalize_result_.colRange(0, 800).rowRange(0,400));
      image_train.copyTo(relocalize_result_.colRange(800, 1600).rowRange(0,400));

      // draw matches
      // FeatureMatcher matcher;
      // std::vector<cv::DMatch> matches;
      // matcher.matchByBruteForce(descriptor_test, keypoints_test, descriptor_train, 
      //   keypoints_train, matches, image_test.size(), true);

      // std::cout << "matches size: " << matches.size() << std::endl;

      for (auto elem : matches) {
        cv::Point p_test = keypoints_test[elem.queryIdx].pt;
        cv::Point p_train = keypoints_train[elem.trainIdx].pt;
        p_train.x += 800;

        cv::line(relocalize_result_, p_test, p_train, cv::Scalar(0,255,0));
      }

      for (int row = 400; row < 420; ++row) {
        for (int col = success+fail-1; col < success+fail+1; ++col) {
          relocalize_result_.at<cv::Vec3b>(row,col) = cv::Vec3b(0, 255, 0);
        }
      }

      cv::imshow("relocalize result", relocalize_result_);
      cv::imwrite("relocalize_result_.png", relocalize_result_);
      cv::waitKey(0);
      video_writer_.write(relocalize_result_);
      break;
    }

    if (i == results.size()-1) {
      fail++;
      image_test.copyTo(relocalize_result_.colRange(0,800).rowRange(0,400));

      // best score
      uint64_t best_frameId = databaseID_to_FrameID_[results[0].Id];
      MetaData temp = trainData_.findByFrameId(best_frameId);

      // cv::Mat best_score = cv::imread(best_temp.path_to_image, cv::IMREAD_UNCHANGED);
      // cv::imshow("best_score", best_score);

      // MetaData temp = trainData_.find_closest(data.gnss_pose);
      cv::Vec3d pose_train{temp.gnss_pose[0], temp.gnss_pose[1],
        temp.gnss_pose[2]};

      Frame frame_train(temp.image_id, temp.path_to_image, pose, feature_type,
        feature_size);

      cv::Mat image_train = frame_train.getImage();
      cv::cvtColor(image_train, image_train, cv::COLOR_GRAY2BGR);

      cv::Mat descriptor_train = frame_train.getDes();
      std::vector<cv::KeyPoint> keypoints_train = frame_train.getKpts();

      image_train.copyTo(relocalize_result_.colRange(800,1600).rowRange(0,400));

      // draw matches
      // FeatureMatcher matcher;
      // std::vector<cv::DMatch> matches;
      // matcher.matchByBruteForce(descriptor_test, keypoints_test, descriptor_train, 
      //   keypoints_train, matches, image_test.size(), true);

      // std::cout << "matches size: " << matches.size() << std::endl;

      // for (auto elem : matches) {
      //   cv::Point p_test = keypoints_test[elem.queryIdx].pt;
      //   cv::Point p_train = keypoints_train[elem.trainIdx].pt;
      //   p_train.x += 800;

      //   cv::line(relocalize_result_, p_test, p_train, cv::Scalar(0,255,0));
      // }

      for (int row = 400; row < 420; ++row) {
        for (int col = success+fail-1; col < success+fail+1; ++col) {
          relocalize_result_.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
        }
      }

      cv::imshow("relocalize result", relocalize_result_);
      cv::waitKey(0);
      video_writer_.write(relocalize_result_);
    }
  }
}
