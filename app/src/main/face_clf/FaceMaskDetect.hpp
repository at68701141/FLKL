#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace std;
using namespace cv;

#define RUN_TYPE_IMAGE		0
#define RUN_TYPE_CAMARA		1

#define USE_RUN_TYPE RUN_TYPE_CAMARA

vector<vector<float>> generate_anchors(const vector<float>& ratios, const vector<int>& scales, vector<float>& anchor_base)
{
	vector<vector<float>> anchors;
	for (int idx = 0; idx < scales.size(); idx++) {
		vector<float> bbox_coords;
		int s = scales[idx];
		vector<float> cxys;
		vector<float> center_tiled;
		for (int i = 0; i < s; i++) {
			float x = (0.5 + i) / s;
			cxys.push_back(x);
		}

		for (int i = 0; i < s; i++) {
			float x = (0.5 + i) / s;
			for (int j = 0; j < s; j++) {
				for (int k = 0; k < 8; k++) {
					center_tiled.push_back(cxys[j]);
					center_tiled.push_back(x);
					//printf("%f %f ", cxys[j], x);
				}
				//printf("\n");
			}
			//printf("\n");
		}

		vector<float> anchor_width_heights;
		for (int i = 0; i < anchor_base.size(); i++) {
			float scale = anchor_base[i] * pow(2, idx);
			anchor_width_heights.push_back(-scale / 2.0);
			anchor_width_heights.push_back(-scale / 2.0);
			anchor_width_heights.push_back(scale / 2.0);
			anchor_width_heights.push_back(scale / 2.0);
			//printf("%f %f %f %f\n", -scale / 2.0, -scale / 2.0, scale / 2.0, scale / 2.0);
		}

		for (int i = 0; i < anchor_base.size(); i++) {
			float s1 = anchor_base[0] * pow(2, idx);
			float ratio = ratios[i + 1];
			float w = s1 * sqrt(ratio);
			float h = s1 / sqrt(ratio);
			anchor_width_heights.push_back(-w / 2.0);
			anchor_width_heights.push_back(-h / 2.0);
			anchor_width_heights.push_back(w / 2.0);
			anchor_width_heights.push_back(h / 2.0);
			//printf("s1:%f, ratio:%f w:%f h:%f\n", s1, ratio, w, h);
			//printf("%f %f %f %f\n", -w / 2.0, -h / 2.0, w / 2.0, h / 2.0);
		}

		int index = 0;
		//printf("\n");
		for (float& a : center_tiled) {
			float c = a + anchor_width_heights[(index++) % anchor_width_heights.size()];
			bbox_coords.push_back(c);
			//printf("%f ", c);
		}

		//printf("bbox_coords.size():%d\n", bbox_coords.size());
		int anchors_size = bbox_coords.size() / 4;
		for (int i = 0; i < anchors_size; i++) {
			vector<float> f;
			for (int j = 0; j < 4; j++) {
				f.push_back(bbox_coords[i * 4 + j]);
			}
			anchors.push_back(f);
		}
	}

	return anchors;
}

vector<cv::Rect2f> decode_bbox(vector<vector<float>> & anchors, float* raw)
{
	vector<cv::Rect2f> rects;
	float v[4] = { 0.1, 0.1, 0.2, 0.2 };

	int i = 0;
	for (vector<float>& k : anchors) {
		float acx = (k[0] + k[2]) / 2;
		float acy = (k[1] + k[3]) / 2;
		float cw = (k[2] - k[0]);
		float ch = (k[3] - k[1]);

		float r0 = raw[i++] * v[i % 4];
		float r1 = raw[i++] * v[i % 4];
		float r2 = raw[i++] * v[i % 4];
		float r3 = raw[i++] * v[i % 4];

		float centet_x = r0 * cw + acx;
		float centet_y = r1 * ch + acy;

		float w = exp(r2) * cw;
		float h = exp(r3) * ch;
		float x = centet_x - w / 2;
		float y = centet_y - h / 2;
		rects.push_back(cv::Rect2f(x, y, w, h));
	}

	return rects;
}

typedef struct FaceInfo {
	Rect2f rect;
	float score;
	int id;
} FaceInfo;

bool increase(const FaceInfo & a, const FaceInfo & b) {
	return a.score > b.score;
}

std::vector<int> do_nms(std::vector<FaceInfo> & bboxes, float thresh, char methodType) {
	std::vector<int> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), increase);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}

		bboxes_nms.push_back(bboxes[select_idx].id);
		mask_merged[select_idx] = 1;

		Rect2f& select_bbox = bboxes[select_idx].rect;
		float area1 = (select_bbox.width + 1) * (select_bbox.height + 1);

		select_idx++;
#pragma omp parallel for num_threads(8)
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			Rect2f & bbox_i = bboxes[i].rect;
			float x = std::max<float>(select_bbox.x, bbox_i.x);
			float y = std::max<float>(select_bbox.y, bbox_i.y);
			float w = std::min<float>(select_bbox.width + select_bbox.x, bbox_i.x + bbox_i.width) - x + 1;
			float h = std::min<float>(select_bbox.height + select_bbox.y, bbox_i.y + bbox_i.height) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = (bbox_i.width + 1) * (bbox_i.height + 1);
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (area_intersect / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (area_intersect / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}

vector<int> single_class_non_max_suppression(vector<cv::Rect2f> & rects, float* confidences, int c_len, vector<int> & classes, vector <float> & bbox_max_scores)
{
	vector<int> keep_idxs;

	float conf_thresh = 0.2;
	float iou_thresh = 0.5;
	int keep_top_k = -1;
	if (rects.size() <= 0) {
		return keep_idxs;
	}

	for (int i = 0; i < c_len; i += 2) {
		float max = confidences[i];
		int classess = 0;
		if (max < confidences[i + 1]) {
			max = confidences[i + 1];
			classess = 1;
		}
		classes.push_back(classess);
		bbox_max_scores.push_back(max);
	}

	vector <FaceInfo>infos;
	for (int i = 0; i < bbox_max_scores.size(); i++) {
		if (bbox_max_scores[i] > conf_thresh) {
			FaceInfo info;
			info.rect = rects[i];
			info.score = bbox_max_scores[i];
			info.id = i;
			infos.push_back(info);
		}
	}

	keep_idxs = do_nms(infos, 0.7, 'u');
	return keep_idxs;
}

#include "Interpreter.hpp"

#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
//#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
typedef struct _FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;

} _FaceInfo;

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

//using namespace std;

class UltraFace {
public:
	UltraFace(const std::string& mnn_path,
		int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.3,
		int topk_ = -1) {
		num_thread = num_thread_;
		score_threshold = score_threshold_;
		iou_threshold = iou_threshold_;
		in_w = input_width;
		in_h = input_length;
		w_h_list = { in_w, in_h };

		for (auto size : w_h_list) {
			std::vector<float> fm_item;
			for (float stride : strides) {
				fm_item.push_back(ceil(size / stride));
			}
			featuremap_size.push_back(fm_item);
		}

		for (auto size : w_h_list) {
			shrinkage_size.push_back(strides);
		}
		/* generate prior anchors */
		for (int index = 0; index < num_featuremap; index++) {
			float scale_w = in_w / shrinkage_size[0][index];
			float scale_h = in_h / shrinkage_size[1][index];
			for (int j = 0; j < featuremap_size[1][index]; j++) {
				for (int i = 0; i < featuremap_size[0][index]; i++) {
					float x_center = (i + 0.5) / scale_w;
					float y_center = (j + 0.5) / scale_h;

					for (float k : min_boxes[index]) {
						float w = k / in_w;
						float h = k / in_h;
						priors.push_back({ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
					}
				}
			}
		}
		/* generate prior anchors finished */

		num_anchors = priors.size();

		ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
		MNN::ScheduleConfig config;
		config.numThread = num_thread;
		MNN::BackendConfig backendConfig;
		backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
		config.backendConfig = &backendConfig;

		ultraface_session = ultraface_interpreter->createSession(config);

		input_tensor = ultraface_interpreter->getSessionInput(ultraface_session, nullptr);
	}

	~UltraFace() {
		ultraface_interpreter->releaseModel();
		ultraface_interpreter->releaseSession(ultraface_session);
	}

	int detect(cv::Mat & raw_image, std::vector<_FaceInfo> & face_list) {
		if (raw_image.empty()) {
			std::cout << "image is empty ,please check!" << std::endl;
			return -1;
		}

		image_h = raw_image.rows;
		image_w = raw_image.cols;
		cv::Mat image;
		cv::resize(raw_image, image, cv::Size(in_w, in_h));

		ultraface_interpreter->resizeTensor(input_tensor, { 1, 3, in_h, in_w });
		ultraface_interpreter->resizeSession(ultraface_session);
		std::shared_ptr<MNN::CV::ImageProcess> pretreat(
			MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
				norm_vals, 3));
		pretreat->convert(image.data, in_w, in_h, image.step[0], input_tensor);

		auto start = chrono::steady_clock::now();


		// run network
		ultraface_interpreter->runSession(ultraface_session);

		// get output data

		/*string scores = "scores";
		string boxes = "boxes";*/
		//string scores = "scores";
		//string boxes = "bboxes";
  //      MNN::Tensor *tensor_scores = ultraface_interpreter->getSessionOutput(ultraface_session, scores.c_str());
  //      MNN::Tensor *tensor_boxes = ultraface_interpreter->getSessionOutput(ultraface_session, boxes.c_str());
		auto tensor = ultraface_interpreter->getSessionOutputAll(ultraface_session);
		printf("tensor.size():%u\n", tensor.size());
		for (auto itr = tensor.begin(); itr != tensor.end(); ++itr) {
			cout << '\t' << itr->first
				<< '\t' << itr->second << '\n';
		}
		MNN::Tensor* tensor_scores = tensor["cls_branch_concat_1/concat"];
		MNN::Tensor* tensor_boxes = tensor["loc_branch_concat_1/concat"];

		MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());

		tensor_scores->copyToHostTensor(&tensor_scores_host);

		MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());

		tensor_boxes->copyToHostTensor(&tensor_boxes_host);

		std::vector<FaceInfo> bbox_collection;


		auto end = chrono::steady_clock::now();
		chrono::duration<double> elapsed = end - start;
		cout << "inference time:" << elapsed.count() << " s" << endl;

		//generateBBox(bbox_collection, tensor_scores, tensor_boxes);
		//nms(bbox_collection, face_list);

		vector<float> ratios{1.0,0.62,0.42};
		vector<int> scales{33,17,9,5,3};
		vector<float> anchor_base{0.04,0.056};
		auto anchors = generate_anchors(ratios, scales, anchor_base);
		float* bboxes = (float*)tensor_boxes_host.host<float>();
		auto decode_rects = decode_bbox(anchors, bboxes);
		vector<int> classes;
		vector <float>scores;
		float* confidences = (float*)tensor_scores_host.host<float>();
		vector<int> keep_idxs = single_class_non_max_suppression(decode_rects, confidences, tensor_scores_host.elementSize(), classes, scores);
		for (auto &i : keep_idxs)
		{
			_FaceInfo faceinfo{};
			faceinfo.x1 = decode_rects[i].x * image_w;
			faceinfo.y1 = decode_rects[i].y * image_h;
			faceinfo.x2 = (decode_rects[i].x + decode_rects[i].width) * image_w;
			faceinfo.y2 = (decode_rects[i].y + decode_rects[i].height) * image_h;
			face_list.emplace_back(std::move(decode_rects[i]));
		}
		return 0;
	}

private:
	void generateBBox(std::vector<_FaceInfo> & bbox_collection, MNN::Tensor * scores, MNN::Tensor * boxes) {
		for (int i = 0; i < num_anchors; i++) {
			if (scores->host<float>()[i * 2 + 1] > score_threshold) {
				_FaceInfo rects;
				float x_center = boxes->host<float>()[i * 4] * center_variance * priors[i][2] + priors[i][0];
				float y_center = boxes->host<float>()[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
				float w = exp(boxes->host<float>()[i * 4 + 2] * size_variance) * priors[i][2];
				float h = exp(boxes->host<float>()[i * 4 + 3] * size_variance) * priors[i][3];

				rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
				rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
				rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
				rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
				rects.score = clip(scores->host<float>()[i * 2 + 1], 1);
				bbox_collection.push_back(rects);
			}
		}
	}

	void nms(std::vector<_FaceInfo> & input, std::vector<_FaceInfo> & output, int type = blending_nms) {
		std::sort(input.begin(), input.end(), [](const _FaceInfo & a, const _FaceInfo & b) { return a.score > b.score; });

		int box_num = input.size();

		std::vector<int> merged(box_num, 0);

		for (int i = 0; i < box_num; i++) {
			if (merged[i])
				continue;
			std::vector<_FaceInfo> buf;

			buf.push_back(input[i]);
			merged[i] = 1;

			float h0 = input[i].y2 - input[i].y1 + 1;
			float w0 = input[i].x2 - input[i].x1 + 1;

			float area0 = h0 * w0;

			for (int j = i + 1; j < box_num; j++) {
				if (merged[j])
					continue;

				float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
				float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

				float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
				float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

				float inner_h = inner_y1 - inner_y0 + 1;
				float inner_w = inner_x1 - inner_x0 + 1;

				if (inner_h <= 0 || inner_w <= 0)
					continue;

				float inner_area = inner_h * inner_w;

				float h1 = input[j].y2 - input[j].y1 + 1;
				float w1 = input[j].x2 - input[j].x1 + 1;

				float area1 = h1 * w1;

				float score;

				score = inner_area / (area0 + area1 - inner_area);

				if (score > iou_threshold) {
					merged[j] = 1;
					buf.push_back(input[j]);
				}
			}
			switch (type) {
			case hard_nms: {
				output.push_back(buf[0]);
				break;
			}
			case blending_nms: {
				float total = 0;
				for (int i = 0; i < buf.size(); i++) {
					total += exp(buf[i].score);
				}
				_FaceInfo rects;
				memset(&rects, 0, sizeof(rects));
				for (int i = 0; i < buf.size(); i++) {
					float rate = exp(buf[i].score) / total;
					rects.x1 += buf[i].x1 * rate;
					rects.y1 += buf[i].y1 * rate;
					rects.x2 += buf[i].x2 * rate;
					rects.y2 += buf[i].y2 * rate;
					rects.score += buf[i].score * rate;
				}
				output.push_back(rects);
				break;
			}
			default: {
				printf("wrong type of nms.");
				exit(-1);
			}
			}
		}
	}

private:

	std::shared_ptr<MNN::Interpreter> ultraface_interpreter;
	MNN::Session* ultraface_session = nullptr;
	MNN::Tensor* input_tensor = nullptr;

	int num_thread;
	int image_w;
	int image_h;

	int in_w;
	int in_h;
	int num_anchors;

	float score_threshold;
	float iou_threshold;


	const float mean_vals[3] = { 127, 127, 127 };
	const float norm_vals[3] = { 1.0 / 128, 1.0 / 128, 1.0 / 128 };

	const float center_variance = 0.1;
	const float size_variance = 0.2;
	const std::vector<std::vector<float>> min_boxes = {
			{10.0f,  16.0f,  24.0f},
			{32.0f,  48.0f},
			{64.0f,  96.0f},
			{128.0f, 192.0f, 256.0f} };
	const std::vector<float> strides = { 8.0, 16.0, 32.0, 64.0 };
	std::vector<std::vector<float>> featuremap_size;
	std::vector<std::vector<float>> shrinkage_size;
	std::vector<int> w_h_list;

	std::vector<std::vector<float>> priors = {};
};

int __main()
{
	vector<float> ratios;
	ratios.push_back(1.0);
	ratios.push_back(0.62);
	ratios.push_back(0.42);

	vector<int> scales;
	scales.push_back(33);
	scales.push_back(17);
	scales.push_back(9);
	scales.push_back(5);
	scales.push_back(3);

	vector<float> anchor_base;
	anchor_base.push_back(0.04);
	anchor_base.push_back(0.056);

	vector<vector<float>> anchors = generate_anchors(ratios, scales, anchor_base);

	dnn::Net PNet_ = cv::dnn::readNetFromCaffe("face_mask_detection.prototxt", "face_mask_detection.caffemodel");

#if (USE_RUN_TYPE == RUN_TYPE_IMAGE)
	Mat img = imread("11.jpg");
#endif

#if (USE_RUN_TYPE == RUN_TYPE_CAMARA)
	VideoCapture vc(0);
	Mat img;
	while (1)
	{
		vc >> img;
		if (!img.data) {
			break;
		}
#endif

		Mat rgb_img;
		cvtColor(img, rgb_img, COLOR_BGR2RGB);
		cv::Mat inputBlob = cv::dnn::blobFromImage(rgb_img, 1 / 255.0, cv::Size(260, 260), cv::Scalar(0, 0, 0), false);
		PNet_.setInput(inputBlob, "data");
		const std::vector< cv::String >  targets_node{ "loc_branch_concat","cls_branch_concat" };
		std::vector< cv::Mat > targets_blobs;
		PNet_.forward(targets_blobs, targets_node);
		cv::Mat y_bboxes = targets_blobs[0];
		cv::Mat y_score = targets_blobs[1];
		float* bboxes = (float*)y_bboxes.data;
		float* confidences = (float*)y_score.data;

		vector<cv::Rect2f> decode_rects = decode_bbox(anchors, bboxes);
		vector<int> classes;
		vector <float>scores;
		vector<int> keep_idxs = single_class_non_max_suppression(decode_rects, confidences, y_score.total(), classes, scores);

		for (int i : keep_idxs) {
			Rect2f& r = decode_rects[i];
			char str[32];
			cv::Scalar str_coclr;
			if (classes[i] == 0) {
				sprintf(str, "mask");
				str_coclr = cv::Scalar(0, 255, 255);
			}
			else {
				sprintf(str, "numask");
				str_coclr = cv::Scalar(0, 0, 255);
			}
			int x = r.x * img.cols;
			int y = r.y * img.rows;
			int w = r.width * img.cols;
			int h = r.height * img.rows;

			cv::putText(img, str, cv::Point(x, y), 1, 1.4, str_coclr, 2, 8, 0);
			sprintf(str, "%0.2f%%", scores[i] * 100);
			cv::putText(img, str, cv::Point(x, y + 14), 1, 1.0, cv::Scalar(255, 255, 255), 1, 8, 0);

			cv::rectangle(img, Rect(x, y, w, h), cv::Scalar(0, 255, 255), 1, 8);
		}

		imshow("img", img);

#if (USE_RUN_TYPE == RUN_TYPE_IMAGE)
		waitKey(0);
#endif

#if (USE_RUN_TYPE == RUN_TYPE_CAMARA)
		waitKey(1);
	}
#endif

	return 0;
}