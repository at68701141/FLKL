#ifndef BASEMODEL_H_
#define BASEMODEL_H_

#include "lang.hpp"
#include <opencv2/opencv.hpp>

class BaseModel
{
public:
	virtual int forward(const cv::Mat& raw_image, std::map<std::string, lang::FloatStream>& output) = 0;
	virtual int forward(const cv::Mat& raw_image, std::map<std::string, std::vector<float>>& output) = 0;
};

#endif