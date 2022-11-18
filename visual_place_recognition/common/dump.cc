#include "visual_place_recognition/common/dump.h"

DEFINE_string(autox_train_images_dir,
  "/home/yuxuanhuang/projects/mapping-experimental/data/bag/train_images",
  "path to autox train images");
DEFINE_string(autox_test_images_dir,
  "/home/yuxuanhuang/projects/mapping-experimental/data/bag/test_images",
  "path to autox test images");

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
    cv::Mat raw_image = cv::imread(path2image, cv::IMREAD_UNCHANGED);
    uint64_t id = vFrames.size();
    cv::Vec3d gnss_pose = {x, y, z};
    vFrames.push_back(Frame(id, raw_image, gnss_pose));
    
    if(1) {
      std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;
      cv::namedWindow("raw_image", cv::WINDOW_NORMAL);
      cv::imshow("raw_image", raw_image);
      cv::waitKey(5); 
    }
  }

  std::cout << "total: " << vFrames.size() << " Frames" << std::endl;
}