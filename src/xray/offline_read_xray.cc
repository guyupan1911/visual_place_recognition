#include <string>
#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "modules/calibration/common/utils/utils.h"
#include "modules/xray/format/xray_reader.h"
#include "modules/drivers/proto/sensor_image.pb.h"
#include "modules/common/adapters/adapter_gflags.h"


int main() {
  std::string xray_path = "/xurban/data/bag/2022-10-13-15-59-45.xray";
  std::vector<std::string> channels = {
    "/xurban/sensor/camera/center_0_n_6mm/image/compressed"
  };
  auto xray_reader = std::make_unique<autox::xray::XRayReader>(xray_path, channels);

  int i = 0;
  while (true) {

    auto message = xray_reader->ReadNextSerialized();
    auto compressed_image = message->Instantiate<autox::drivers::CompressedImage>();
    
    std::cout << "read image " << i  << " width: " << compressed_image->width()
      << " height: " << compressed_image->height() << std::endl;

    cv::Mat image_mat_;
    double timestamp;
    autox::calibration::DriverImageToCVMat(*compressed_image, &image_mat_, &timestamp);

    // cv::imshow("image", image_mat_);
    // cv::waitKey(5);

    std::string path_to_image = "/xurban/data/bag/images/top_view/" + std::to_string(i)
      + ".png";
    cv::imwrite(path_to_image, image_mat_);
    i++;
  }
}