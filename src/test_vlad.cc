#include "io.h"
#include "DBoW3/DBoW3.h"
#include "frame.h"

using namespace DBoW3;
using namespace std;

const int COLUMNOFCODEBOOK = 256;
const int DESSIZE = 128;

struct dist {
	int dis;
	int site;
  bool result = false;
};

bool vecCmp(const dist &a, const dist &b) {
	     return a.dis < b.dis;
}

int hammingDistance(cv::Mat baseImg, cv::Mat targetImg)
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

double euclidean_distance(cv::Mat baseImg, cv::Mat targetImg)
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

void calVLADHamming(const vector<cv::Mat> &features, DBoW3::Vocabulary &codebook, vector<cv::Mat> &vladBase)
{
	/*
	*以bit为单位使用汉明距离计算残差
	*@param features 数据集中所有图片的所有特征
	*@param codebook 生成好的码本
	*@param vladBase 用于返回得到的所有图的vlad向量
	*/

	WordId wi;
	for (int i = 0; i < features.size(); ++i) {
		//遍历每张图片

		//初始化vlad矩阵和vlad位矩阵分别用于返回最终的vlad矩阵和保存每bit的统计结果
		vector< cv::Mat > vladBitMatrix;
		vector <cv::Mat > vladMatrix;
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::Mat Z = cv::Mat::zeros(1, DESSIZE * sizeof(uint8_t) * 8, CV_32FC1);
			cv::Mat X = cv::Mat::zeros(1, DESSIZE * sizeof(uint8_t) * 8, CV_8U);
			vladBitMatrix.push_back(Z);
			vladMatrix.push_back(X);
		}

		int sum = 0;
		for (int j = 0; j < features[i].rows; ++j) {
			//遍历每张图片的所有特征


			cv::Mat A = features[i](cv::Rect(0, j, DESSIZE, 1));
			wi = codebook.transform(A);
			cv::Mat central = codebook.getWord(wi);
		
			
			int bitCur = 0;
			for (int k = 0; k < A.cols; ++k) {
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
		float thr = sum / (COLUMNOFCODEBOOK * DESSIZE * 8.0f);  //用总的不同位数的个数除以所有位数计算一个阈值thr

		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			int bitCur = 0;
			for (int k = 0; k < DESSIZE; ++k) {
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
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::hconcat(vladMatrix[j], vlad, vlad);
		}
		//cout << vlad << endl;

		vladBase.push_back(vlad);

		//cout << vladNorm << endl;
	}


}

void calVLAD(const vector<cv::Mat> &features, DBoW3::Vocabulary &codebook, vector<cv::Mat> &vladBase) 
{
	/*
	*使用直接做差的方式计算残差
	*@param features 数据集中所有图片的所有特征
	*@param codebook 生成好的码本
	*@param vladBase 用于返回得到的所有图的vlad向量
	*/
	WordId wi;
	for (int i = 0; i < features.size(); ++i) {
		//遍历每张图片的特征


		//初始化每张图片的vlad矩阵
		vector< cv::Mat > vladMatrix;
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::Mat Z = cv::Mat::zeros(1, DESSIZE, CV_32FC1);
			vladMatrix.push_back(Z);
		}

		for (int j = 0; j < features[i].rows; ++j) {
			//遍历一张图片的所有特征

			cv::Mat A = features[i](cv::Rect(0, j, DESSIZE, 1));
			wi = codebook.transform(A);   //寻找一个特征向量最近的聚类中心的id
			//cout << wi << " ";
			cv::Mat central = codebook.getWord(wi);  //取最近的聚类中心向量
			cv::Mat tmpA, tmpCen;

			//将该特征向量和找到的最近聚类中心向量转换为float型用于后续计算和归一化
			A.convertTo(tmpA, CV_32FC1);
			central.convertTo(tmpCen, CV_32FC1);

			vladMatrix[wi] += (tmpA - tmpCen);
		}

		cv::Mat vlad = cv::Mat::zeros(1, 0, CV_32FC1);
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::hconcat(vladMatrix[j], vlad, vlad);  //将vlad矩阵展开成向量
		}
		//cout << vlad << endl;

		cv::Mat vladNorm;
		vlad.copyTo(vladNorm);

		cv::normalize(vlad, vladNorm, 1, 0, cv::NORM_L2);  //对得到的vlad向量进行归一化

		vladBase.push_back(vladNorm);

		//cout << vladNorm << endl;
	}


}

int main() {
  // read train data and test data
  std::map<uint64_t, std::pair<std::string, cv::Vec3d>> train_images =
    readImagesAndGNSS("../dataset/train/recording_2021-02-25_13-39-06_images");

  std::map<uint64_t, std::pair<std::string, cv::Vec3d>> test_images =
    readImagesAndGNSS("../dataset/test/recording_2020-12-22_12-04-35_images");

  std::vector<std::pair<int64_t, Frame>> train_frames, test_frames;
  std::vector<cv::Mat> vDes_train, vDes_test;

  int i = 0;
  for (auto& elem : train_images) {
    if (i % 20 != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second,
      "SIFT", 800);
    train_frames.push_back(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  // calculate clusters
  const int k = COLUMNOFCODEBOOK;
  const int L = 1;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;

  DBoW3::Vocabulary voc(k, L, weight, score);
  voc.create(vDes_train);
  std::cout << "create voc" << std::endl;
  // calculate vlad vectors of training images
  i = 0;
  train_frames.clear();
  vDes_train.clear();
  for (auto& elem : train_images) {
    if (i++ % 5 != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second,
      "SIFT", 800);
    train_frames.push_back(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  vector<cv::Mat> vladBase_train;
  // calVLADHamming(vDes_train, voc, vladBase_train);
  calVLAD(vDes_train, voc, vladBase_train);
  std::cout << "vladBase_train size: " << vladBase_train.size() << std::endl;

  // calculate vlad vectors of querying images
  i = 0;
  for (auto& elem : test_images) {
    if (i++ % 5 !=0)
        continue;
    Frame frame(elem.first, elem.second.first, elem.second.second, "SIFT",
                800);
    test_frames.push_back(std::make_pair(elem.first, frame));
    vDes_test.push_back(frame.getDes());
  }

  vector<cv::Mat> vladBase_test;
  // calVLADHamming(vDes_test, voc, vladBase_test);
  calVLAD(vDes_test, voc, vladBase_test);
  std::cout << "vladBase_test size: " << vladBase_test.size() << std::endl;

  // querying
  int success = 0;
  for (int i = 0; i < vladBase_test.size(); ++i) {
    vector<struct dist> disVec; //该向量用于对测试结果进行存储和排序
    cv::Vec3d pose_test = test_frames[i].second.getPose();
    cv::Mat image_test = test_frames[i].second.getImage();
    for (int j = 0; j < vladBase_train.size(); ++j) {
      
      // int dis = hammingDistance(vladBase_train[j], vladBase_test[i]); //计算图片集中每张图片与待查询图片的vlad向量的汉明距离
      int dis = euclidean_distance(vladBase_train[j], vladBase_test[i]); //计算图片集中每张图片与待查询图片的vlad向量的汉明距离
      
      dist tmp;
      tmp.dis = dis;
      tmp.site = j;
      disVec.push_back(tmp);
    }

    sort(disVec.begin(), disVec.end(), vecCmp);

    for(int k=0; k < 8; k++)
    {
      cv::Mat image_train = train_frames[disVec[k].site].second.getImage();
      cv::Vec3d pose_train = train_frames[disVec[k].site].second.getPose();

      cv::Vec3d dis = pose_train - pose_test;
      cv::Vec2d txy(dis[0], dis[1]);
      cv::Vec2d tz(dis[2]);

      if (cv::norm(txy, cv::NORM_L2) < 5.0 && cv::norm(tz, cv::NORM_L2) < 1.0) {
        // cv::imshow("image_test", image_test);
        // cv::imshow("image_train", image_train);
        // cv::waitKey(0);
        success++;
        break;
      }
    }

  }
  std::cout << "success: " << success << " failure: " <<
  vladBase_test.size() - success << std::endl;

}
