#include "Vocabulary.h"

namespace vlad {

void Vocabulary::create(const std::vector<cv::Mat> &descriptors) {
  DBoW3::Vocabulary voc(clusterCount, 1, DBoW3::TF_IDF, DBoW3::L1_NORM);
  std::cout << "Creating a vlad vocabulary..." << std::endl;
  voc.create(descriptors);
  std::cout << "... done!" << " words size: " << voc.size() << std::endl;

  voc.save("vlad_voc.yml.gz");
}


DataBase::DataBase(std::string voc_filename) {
  pVoc_ = new DBoW3::Vocabulary(voc_filename);
}

unsigned int DataBase::add(const cv::Mat &descriptor) {
  cv::Mat vlad_vec = calculate_vlad_vector(descriptor);
  unsigned int entryId  = vlad_vectors_.size();
  vlad_vectors_.push_back(vlad_vec);
  return entryId;
}

cv::Mat DataBase::calculate_vlad_vector(const cv::Mat& descriptor) {
  if (descriptor.type() == CV_32FC1) {
    std::vector<cv::Mat> vladMatrix;
    for (int i=0; i < clusterCount; ++i) {
      cv::Mat Z = cv::Mat::zeros(1, 128, CV_32FC1);
      vladMatrix.push_back(Z);
    }

    for (int i=0; i < descriptor.rows; ++i) {
      cv::Mat A = descriptor(cv::Rect(0, i, 128, 1));
      unsigned int wordID = pVoc_->transform(A);
      cv::Mat central = pVoc_->getWord(wordID);

      cv::Mat tmpA, tmpCen;
      A.convertTo(tmpA, CV_32FC1);
      central.convertTo(tmpCen, CV_32FC1);
      vladMatrix[wordID] += (tmpA - tmpCen);
    }

    cv::Mat vlad = cv::Mat::zeros(1, 0, CV_32FC1);
    for (int j = 0; j < clusterCount; ++j) {
      cv::hconcat(vladMatrix[j], vlad, vlad);
    }

    cv::Mat vladNorm;
    vlad.copyTo(vladNorm);

    cv::normalize(vlad, vladNorm, 1, 0, cv::NORM_L2);

    return vlad;
  }
  else if (descriptor.type() == CV_8U) {
    std::vector<cv::Mat> vladBitMatrix;
    std::vector<cv::Mat> vladMatrix;
    for (int j = 0; j < clusterCount; ++j) {
      cv::Mat Z = cv::Mat::zeros(1, 32 * sizeof(uint8_t) * 8, CV_32FC1);
      cv::Mat X = cv::Mat::zeros(1, 32 * sizeof(uint8_t) * 8, CV_8U);
      vladBitMatrix.push_back(Z);
      vladMatrix.push_back(X);
    }

    int sum = 0;
		for (int j = 0; j < descriptor.rows; ++j) {
			//遍历每张图片的所有特征
			cv::Mat A = descriptor(cv::Rect(0, j, 32, 1));
			unsigned int wi = pVoc_->transform(A);
			cv::Mat central = pVoc_->getWord(wi);
	
			int bitCur = 0;
			for (int k = 0; k < A.cols; ++k) { // 遍历描述子的每个元素
				//取一张图片的一个特征向量和最近的聚类中心向量
				uint8_t a = A.at<uint8_t>(0, k); 
				uint8_t cen = central.at<uint8_t>(0, k);
				uint8_t tmp = a ^ cen; //遍历向量中每个元素并进行异或操作用于统计不同位数的个数
				uint8_t mask = 0x01;
				for (int l = 0; l < 8; ++l) {
					if (tmp & mask) {
						vladBitMatrix[wi].at<float>(0, bitCur)++;  //如果特征向量与聚类中心不同，则vlad位矩阵该位置+1
						bitCur++;
						sum++;  //统计一张图片中所有不同位数的个数
					}
					else {
						bitCur++;
					}
					mask = mask << 1;
				}
			}


		}
		float thr = sum / (256 * 32 * 8.0f);  //用总的不同位数的个数除以所有位数计算一个阈值thr

		for (int j = 0; j < 256; ++j) {
			int bitCur = 0;
			for (int k = 0; k < 32; ++k) {
				//uint8_t mask = 0x01;
				for (int l = 0; l < 8; l++) {
					if (vladBitMatrix[j].at<uint8_t>(0, bitCur) > thr) {
						//如果vlad位矩阵中的不同位数累积结果大于阈值thr，则改位记为1，否则记为0
						vladMatrix[j].at<uint8_t>(0, bitCur) = 1;

					}
					//mask = mask << 1;
					bitCur++;
				}
			}
		}

		//将得到的vlad矩阵扩展成为向量
		cv::Mat vlad = cv::Mat::zeros(1, 0, CV_8U);
		for (int j = 0; j < 256; ++j) {
			cv::hconcat(vladMatrix[j], vlad, vlad);
		}
		//cout << vlad << endl;

		return vlad;
  }

}

std::vector<unsigned int> DataBase::query(const cv::Mat &descriptor, int nums) {
  cv::Mat vlad_vec = calculate_vlad_vector(descriptor);

	if (descriptor.type() == CV_32F) {
		std::vector<dist> dists;
		for (int i = 0; i < vlad_vectors_.size(); i++) {
			double dis = euclidean_distance(vlad_vec, vlad_vectors_[i]);
			dist temp;
			temp.dis = dis;
			temp.index = i;
			dists.push_back(temp);
		}

		sort(dists.begin(), dists.end(), vecCmp);

		std::vector<unsigned int> ret;
		for(int i=0; i<8; i++) {
			ret.push_back(dists[i].index);
		}

		return ret;
	}
	else if (descriptor.type() == CV_8U) {
		std::vector<distHamming> dists;
		for (int i = 0; i < vlad_vectors_.size(); i++) {
			int dis = hammingDistance(vlad_vec, vlad_vectors_[i]);
			distHamming temp;
			temp.dis = dis;
			temp.index = i;
			dists.push_back(temp);
		}

		sort(dists.begin(), dists.end(), vecCmpHamming);

		std::vector<unsigned int> ret;
		for(int i=0; i<8; i++) {
			ret.push_back(dists[i].index);
		}

		return ret;
	}

}

double DataBase::euclidean_distance(cv::Mat baseImg, cv::Mat targetImg)
{
	/*
	*计算两个向量的欧氏距离
	*@param baseImg 一个向量
	*@param targetImg 一个向量
	*@return 两个向量的欧氏距离
	*/
	double sumDescriptor = 0;
	for (int i = 0; i < baseImg.cols; i++)
	{
		double numBase = abs(baseImg.at<float>(0, i));
		double numTarget = abs(targetImg.at<float>(0, i));
		sumDescriptor += pow(numBase - numTarget, 2);
	}
	double simility = sqrt(sumDescriptor);
	return simility;
}

int DataBase::hammingDistance(cv::Mat baseImg, cv::Mat targetImg)
{
	/*
	*计算两个向量的汉明距离
	*@param baseImg 一个向量
	*@param targetImg 一个向量
	*@return 两个向量的汉明距离
	*/
	int sumDescriptor = 0;
	for (int i = 0; i < baseImg.cols; i++)
	{
		uint8_t numBase = baseImg.at<uint8_t>(0, i);
		uint8_t numTarget = targetImg.at<uint8_t>(0, i);
		//sumDescriptor += pow(numBase - numTarget, 2);
		if (numBase != numTarget) sumDescriptor++;
	}

	return sumDescriptor;
}

}  // namespace vlad
