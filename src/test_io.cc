#include "io.h"

int main() {
  std::string path_to_dir_train =
    "../dataset/train/recording_2021-02-25_13-39-06_images";
    std::string path_to_dir_test =
  "../dataset/test/recording_2020-12-22_12-04-35_images";

  DataIO train_data(path_to_dir_train);
  DataIO test_data(path_to_dir_test);

  for (auto& elem : test_data.get_all_data()) {
    cv::Mat test_image = cv::imread(elem.path_to_image);
    MetaData cloest_train_data = train_data.find_closest(elem.gnss_pose);
    cv::Mat train_image = cv::imread(cloest_train_data.path_to_image);
    std::cout << "test gnss pose: " << elem.gnss_pose << " " << std::endl <<
      "train gnss pose: " << cloest_train_data.gnss_pose << std::endl;
    cv::imshow("train_image", train_image);
    cv::imshow("test_image", test_image);
    cv::waitKey(5);
  }
}
