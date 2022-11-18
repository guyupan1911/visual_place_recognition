#include <fstream>

#include "opencv4/opencv2/opencv.hpp"

#include "msg/cvMat.pb.h"

void write() {
  cv::Mat image = cv::imread("0.png",cv::IMREAD_UNCHANGED);
  std::fstream output("image.bin", std::ios::out | std::ios::trunc | std::ios::binary);

  opencv::cvMat serializableMat;

  serializableMat.set_rows(image.rows);
  serializableMat.set_cols(image.cols);
  serializableMat.set_elt_type(image.type());
  serializableMat.set_elt_size((int)image.elemSize());

  size_t dataSize = image.rows * image.cols * image.elemSize();
  serializableMat.set_mat_data(image.data, dataSize);
  
  if(serializableMat.SerializeToOstream(&output))
    std::cout << "encode done" << std::endl;
  output.close();
  
}

void read() {
  std::fstream input("image.bin", std::ios::in | std::ios::binary);

  cv::Mat im;
  opencv::cvMat serializableM;
  serializableM.ParseFromIstream(&input);
  im.create(serializableM.rows(), serializableM.cols(), serializableM.elt_type());
  
  size_t dataSize = serializableM.rows() * serializableM.cols() * serializableM.elt_size();

  std::copy(reinterpret_cast<uchar*>(const_cast<char*>(serializableM.mat_data().data())),
    reinterpret_cast<uchar*>(const_cast<char*>(serializableM.mat_data().data())+dataSize),
    im.data);

  cv::namedWindow("image", cv::WINDOW_NORMAL);
  cv::imshow("image", im);
  cv::waitKey(0);
}

int main() {
  write();
  read();
}