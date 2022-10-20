#include <string>

#include "opencv4/opencv2/opencv.hpp"

#include "vladMapper.h"
#include "io.h"
#include "frame.h"

int main() {
  // read images
  std::string path_to_dir_train =
    "../dataset/train/recording_2021-02-25_13-39-06_images";
  DataIO train_data(path_to_dir_train);

  std::string path_to_dir_test =
    "../dataset/test/recording_2020-12-22_12-04-35_images";
  DataIO test_data(path_to_dir_test);

  VladMapper mapper;
  mapper.mapping(train_data.get_add_data());

  for (const auto& elem : test_data.get_add_data()) {
    mapper.relocalize(elem);
  }
}
