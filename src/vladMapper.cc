#include "vladMapper.h"

VladMapper::VladMapper() {
  // load train data
  // std::string path_to_dir =
  //   "../dataset/train/recording_2021-02-25_13-39-06_images";
  trainData_.ReadAutoX(FLAGS_autox_train_images_dir);

  relocalize_result_ = cv::Mat::zeros(420, 2*800, CV_8UC3);

  video_writer_ = cv::VideoWriter("result_vlad.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
    relocalize_result_.size(), true);

  if (!video_writer_.isOpened())
    std::cout << "not open" << std::endl;
}

void VladMapper::mapping(const std::vector<MetaData>& mapping_data) {
  std::vector<cv::Mat> descriptors;
  for (auto& elem : mapping_data) {
    cv::Vec3d pose{elem.gnss_pose[0], elem.gnss_pose[1], elem.gnss_pose[2]};
    Frame frame(elem.image_id, elem.path_to_image, pose, "ORB2", 2000);
    descriptors.push_back(frame.getDes());
  }

  std::cout << "create vocabulary" << std::endl;
  vlad::Vocabulary voc;
  voc.create(descriptors);
  std::cout << "done" << std::endl;

  pdb_ = new vlad::DataBase("vlad_voc.yml.gz");

  for (int i=0; i < descriptors.size(); ++i) {
    unsigned int entryID = pdb_->add(descriptors[i]);
    entryId_to_frameID.insert(std::make_pair(entryID, mapping_data[i].image_id));
  }

  saveEntry2Frame("vlad_entry2frame.bin");
}

void VladMapper::saveEntry2Frame(std::string filename) {
  std::ofstream fout("entry2frame.bin");
  unsigned int size = entryId_to_frameID.size();
  fout.write((char*)&size, sizeof(size));
  for (auto& elem : entryId_to_frameID) {
    fout.write((char*)&elem.first, sizeof(DBoW3::EntryId));
    fout.write((char*)&elem.second, sizeof(uint64_t));
  }
  fout.close();
}

void VladMapper::relocalize(const MetaData& data) {
  cv::Vec3d pose{data.gnss_pose[0], data.gnss_pose[1], data.gnss_pose[2]};
  Frame frame(data.image_id, data.path_to_image, pose, "ORB2", 2000);
  std::vector<unsigned int> results = pdb_->query(frame.getDes(), 8);

  cv::Mat image_test = frame.getImage();
  cv::cvtColor(image_test, image_test, cv::COLOR_GRAY2BGR);

  cv::Mat descriptor_test = frame.getDes();
  std::vector<cv::KeyPoint> keypoints_test = frame.getKpts();

  static int success = 0;
  static int fail = 0;
  std::cout << "success: " << success << " fail: " << fail << std::endl;

  for (int i=0; i < results.size(); i++) {
    uint64_t frameId = entryId_to_frameID[results[i]];
    MetaData temp = trainData_.findByFrameId(frameId);
    cv::Vec3d pose_train{temp.gnss_pose[0], temp.gnss_pose[1],
      temp.gnss_pose[2]};
    Frame frame_train(temp.image_id, temp.path_to_image, pose, "ORB2",
      2000);
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
      keypoints_train, matches, image_test.size(), false);

    if (cv::norm(txy, cv::NORM_L2) < 10.0 && cv::norm(tz, cv::NORM_L2) < 3.0 && (matches.size()>18 || i==0)) {
      success++;

      image_test.copyTo(relocalize_result_.colRange(0, 800).rowRange(0,400));
      image_train.copyTo(relocalize_result_.colRange(800, 1600).rowRange(0,400));

      for (int row = 400; row < 420; ++row) {
        for (int col = success+fail-1; col < success+fail+1; ++col) {
          relocalize_result_.at<cv::Vec3b>(row,col) = cv::Vec3b(0, 255, 0);
        }
      }

      cv::imshow("relocalize result", relocalize_result_);
      cv::waitKey(5);
      video_writer_.write(relocalize_result_);
      break;
    }

    if (i == results.size()-1) {
      fail++;
      image_test.copyTo(relocalize_result_.colRange(0,800).rowRange(0,400));
      cv::Mat black = cv::Mat::zeros(400, 800, CV_8UC3);
      black.copyTo(relocalize_result_.colRange(800,1600).rowRange(0,400));

      for (int row = 400; row < 420; ++row) {
        for (int col = success+fail-1; col < success+fail+1; ++col) {
          relocalize_result_.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
        }
      }

      cv::imshow("relocalize result", relocalize_result_);
      cv::waitKey(5);
      video_writer_.write(relocalize_result_);
    }
  }
}
