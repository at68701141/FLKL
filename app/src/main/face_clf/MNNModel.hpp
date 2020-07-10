#ifndef MNNMODEL_H_
#define MNNMODEL_H_

#include <chrono>
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include "BaseModel.hpp"

class MNNModel : public BaseModel {
	int in_w;
	int in_h;
	int num_thread;
	float mean_vals[3];
	float norm_vals[3];

	std::shared_ptr<MNN::Interpreter> ultraface_interpreter;
	MNN::Session* ultraface_session = nullptr;
	MNN::Tensor* input_tensor = nullptr;
public:
	MNNModel(const std::string& mnn_path,
		int input_width, int input_length, int num_thread_ = 4, 
		std::vector<float> mean_vals_ = { 0, 0, 0 }, 
		std::vector<float> norm_vals_ = { 1.0f / 255, 1.0f / 255, 1.0f / 255 }) {
		num_thread = num_thread_;
		in_w = input_width;
		in_h = input_length;
		memcpy(mean_vals,mean_vals_.data(),sizeof(mean_vals));
		memcpy(norm_vals, norm_vals_.data(), sizeof(norm_vals));

		ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
		MNN::ScheduleConfig config;
		config.numThread = num_thread;
		MNN::BackendConfig backendConfig;
		backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
		config.backendConfig = &backendConfig;
		ultraface_session = ultraface_interpreter->createSession(config);
		input_tensor = ultraface_interpreter->getSessionInput(ultraface_session, nullptr);
	}
	MNNModel(const void* mnn_model_buf, int buf_len,
			 int input_width, int input_length, int num_thread_ = 4,
			 std::vector<float> mean_vals_ = { 0, 0, 0 },
			 std::vector<float> norm_vals_ = { 1.0f / 255, 1.0f / 255, 1.0f / 255 }) {
		num_thread = num_thread_;
		in_w = input_width;
		in_h = input_length;
		memcpy(mean_vals,mean_vals_.data(),sizeof(mean_vals));
		memcpy(norm_vals, norm_vals_.data(), sizeof(norm_vals));

		ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(mnn_model_buf,buf_len));
		MNN::ScheduleConfig config;
		config.numThread = num_thread;
//        config.type = MNN_FORWARD_VULKAN;
		config.type = MNN_FORWARD_CPU;
		MNN::BackendConfig backendConfig;
		backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
		config.backendConfig = &backendConfig;
		ultraface_session = ultraface_interpreter->createSession(config);
		input_tensor = ultraface_interpreter->getSessionInput(ultraface_session, nullptr);
	}

	~MNNModel() {
		ultraface_interpreter->releaseModel();
		ultraface_interpreter->releaseSession(ultraface_session);
	}

	virtual int forward(const cv::Mat& raw_image, std::map<std::string, lang::FloatStream>& output) {
		cv::Mat image;
		cv::resize(raw_image, image, cv::Size(in_w, in_h));
		ultraface_interpreter->resizeTensor(input_tensor, { 1, 3, in_h, in_w });
		ultraface_interpreter->resizeSession(ultraface_session);
		std::shared_ptr<MNN::CV::ImageProcess> pretreat(
			MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
				norm_vals, 3));
		pretreat->convert(image.data, in_w, in_h, (int)image.step[0], input_tensor);

		auto start = std::chrono::steady_clock::now();

		// run network
		ultraface_interpreter->runSession(ultraface_session);
		std::map<std::string, MNN::Tensor*> tensor = ultraface_interpreter->getSessionOutputAll(ultraface_session);

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "inference time:" << elapsed.count() << " s" << std::endl;

		// get output data
		//MNN::Tensor* tensor_scores = tensor["cls_branch_concat"];
		//MNN::Tensor* tensor_boxes = tensor["loc_branch_concat"];
		for (auto itr = tensor.begin(); itr != tensor.end(); ++itr) {
			//std::cout << '\t' << itr->first
			//	<< '\t' << itr->second << '\n';
			//if (!itr->first || !itr->second) {
			//	return 1;
			//}
			lang::FloatStream fs{ itr->second->host<float>(),itr->second->elementSize() };
			output.emplace(std::make_pair(itr->first, fs));
		}
		//if (!tensor_scores || !tensor_boxes) {
		//	return 1;
		//}
		/*cls = { tensor_scores->host<float>(), tensor_scores->elementSize()};
		loc = { tensor_boxes->host<float>(), tensor_boxes->elementSize()};*/
		return 0;
	}
	virtual int forward(const cv::Mat& raw_image, std::map<std::string, std::vector<float>>& output) {
		cv::Mat image;
		cv::resize(raw_image, image, cv::Size(in_w, in_h));
		ultraface_interpreter->resizeTensor(input_tensor, { 1, 3, in_h, in_w });
		ultraface_interpreter->resizeSession(ultraface_session);
		std::shared_ptr<MNN::CV::ImageProcess> pretreat(
				MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
											  norm_vals, 3));
		pretreat->convert(image.data, in_w, in_h, (int)image.step[0], input_tensor);

		auto start = std::chrono::steady_clock::now();

		// run network
		ultraface_interpreter->runSession(ultraface_session);
		std::map<std::string, MNN::Tensor*> tensor = ultraface_interpreter->getSessionOutputAll(ultraface_session);

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "inference time:" << elapsed.count() << " s" << std::endl;

		// get output data
		//MNN::Tensor* tensor_scores = tensor["cls_branch_concat"];
		//MNN::Tensor* tensor_boxes = tensor["loc_branch_concat"];
		for (auto itr = tensor.begin(); itr != tensor.end(); ++itr) {
			//std::cout << '\t' << itr->first
			//	<< '\t' << itr->second << '\n';
			//if (!itr->first || !itr->second) {
			//	return 1;
			//}
			auto nchwTensor = new MNN::Tensor(itr->second, MNN::Tensor::CAFFE);
			itr->second->copyToHostTensor(nchwTensor);
			std::vector<float> vec(nchwTensor->elementSize());
			memcpy(vec.data(),nchwTensor->host<float>(),nchwTensor->size());
			delete nchwTensor;
			output.emplace(std::make_pair(itr->first, std::move(vec)));
		}
		//if (!tensor_scores || !tensor_boxes) {
		//	return 1;
		//}
		/*cls = { tensor_scores->host<float>(), tensor_scores->elementSize()};
		loc = { tensor_boxes->host<float>(), tensor_boxes->elementSize()};*/
		return 0;
	}
};

#endif