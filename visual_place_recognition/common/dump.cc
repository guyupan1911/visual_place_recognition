#include "visual_place_recognition/common/dump.h"

DEFINE_string(autox_train_images_dir,
  "/home/yuxuanhuang/projects/mapping-experimental/data/bag/train_images",
  "path to autox train images");
DEFINE_string(autox_test_images_dir,
  "/home/yuxuanhuang/projects/mapping-experimental/data/bag/test_images",
  "path to autox test images");

DEFINE_string(seasons_test_images_dir,
  "//home/yuxuanhuang/projects/visual_place_recognition/dataset/train/recording_2021-02-25_13-39-06_images",
  "path to seasons train images");
DEFINE_string(seasons_train_images_dir,
  "/home/yuxuanhuang/projects/visual_place_recognition/dataset/test/recording_2020-12-22_12-04-35_images",
  "path to seasons test images");

void dump4Seasons(const std::string &path, std::vector<Frame> &vFrames) {
  vFrames.clear();
  
  std::string path2GNSS = path + "/GNSSPoses_small.txt";
  std::ifstream fin(path2GNSS.c_str());
  std::string line;

  while (getline(fin, line)) {
    if (line[0] == '#')
      continue;

    std::stringstream ss(line);

    std::vector<std::string> vWords;
    std::string word;
    while (getline(ss, word, ','))
      vWords.push_back(word);

    uint64_t frame_id;
    double tx, ty, tz;

    frame_id = stoull(vWords[0]);
    tx = stod(vWords[1]);
    ty = stod(vWords[2]);
    tz = stod(vWords[3]);
    cv::Vec3d gnss_pose = {tx, ty, tz};

    std::string path2image = path + "/undistorted_images/cam0/" + std::to_string(frame_id) + ".png";
    cv::Mat raw_image = cv::imread(path2image, cv::IMREAD_GRAYSCALE);
    cv::Mat resized_image;
    cv::resize(raw_image, resized_image, cv::Size(80,40), cv::INTER_LINEAR);
    if(1) {
      cv::namedWindow("resized_image", cv::WINDOW_NORMAL);
      cv::imshow("resized_image", resized_image);
      cv::waitKey(5);
    }

    cv::Mat normalized_image = patchNormalization(resized_image);
    // cv::Mat normalized_image = normalization(resized_image);

    if(1) {
      cv::namedWindow("normalized_image", cv::WINDOW_NORMAL);
      cv::imshow("normalized_image", normalized_image);
      cv::waitKey(5);
    }
    uint64_t id = vFrames.size();
    vFrames.push_back(Frame(path2image, id, normalized_image, gnss_pose));
}
}

void dumpAutoX(const std::string &path, std::vector<Frame> &vFrames) {
  vFrames.clear();
  std::string path_to_file = path + "/image_and_pose.txt";
  std::ifstream fin(path_to_file, std::ios_base::in);

  std::string line;
  while (getline(fin, line)) {
    std::stringstream ss(line);
    uint64_t frame_id;
    double x, y, z;
    ss >> frame_id >> x >> y >> z;
    std::string path2image = path + "/" + std::to_string(frame_id) + ".png";
    uint64_t id = vFrames.size();
    cv::Vec3d gnss_pose = {x, y, z};
    cv::Mat raw_image = cv::imread(path2image, cv::IMREAD_GRAYSCALE);
    cv::Mat resized_image;
    cv::resize(raw_image, resized_image, cv::Size(20,40), cv::INTER_LINEAR);
    if(1) {
      cv::namedWindow("resized_image", cv::WINDOW_NORMAL);
      cv::imshow("resized_image", resized_image);
      cv::waitKey(5);
    }

    cv::Mat normalized_image = patchNormalization(resized_image);
    if(1) {
      cv::namedWindow("normalized_image", cv::WINDOW_NORMAL);
      cv::imshow("normalized_image", normalized_image);
      cv::waitKey(5);
    }


    vFrames.push_back(Frame(path2image, id, normalized_image, gnss_pose));
    
    if(0) {
      std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;
    }
  }

  // std::cout << "total: " << vFrames.size() << " Frames" << std::endl;
}