#ifndef FACEFEATURE_H_
#define FACEFEATURE_H_

#include "BaseModel.hpp"

class FaceFeature
{
public:
	FaceFeature(BaseModel& model_) : model(model_) {}

	int GetFeature(cv::Mat& raw_image, lang::FloatStream& output) {
		std::map<std::string, lang::FloatStream> m{};
		model.forward(raw_image, m);
		output = m.begin()->second;
		return 0;
	}
	int GetFeature(cv::Mat& raw_image, std::vector<float>& output) {
		std::map<std::string, std::vector<float>> m{};
		model.forward(raw_image, m);
		output = std::move(m.begin()->second);
		return 0;
	}

private:
	BaseModel& model;
};

#endif // !FACEFEATURE_H_
