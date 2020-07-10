#ifndef FACELANDMARK_H_
#define FACELANDMARK_H_

#include "BaseModel.hpp"

class FaceLandmark
{
public:
	FaceLandmark(BaseModel& model_) : model(model_) {}

	int GetLandmark(const cv::Mat& raw_image, lang::FloatStream& output) {
		std::map<std::string, lang::FloatStream> m{};
		model.forward(raw_image, m);
		//output = m["batchnorm0"];
		//output = m["embeddings"];
		output = m.begin()->second;
		return 0;
	}
	int GetLandmark(const cv::Mat& raw_image, std::vector<float>& output) {
		std::map<std::string, std::vector<float>> m{};
		model.forward(raw_image, m);
		//output = m["batchnorm0"];
		//output = m["embeddings"];
//		output = m.begin()->second;
		output = std::move(m.begin()->second);
		return 0;
	}

private:
	BaseModel& model;
};

#endif // !FACELANDMARK_H_