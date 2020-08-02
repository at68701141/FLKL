#ifndef FACEDETECT_H_
#define FACEDETECT_H_

#include "BaseModel.hpp"

class FaceDetect
{
public:
	typedef struct FaceInfo {
		float x1;
		float y1;
		float x2;
		float y2;
		float score;

	} FaceInfo;

	FaceDetect(BaseModel& model_,float score_threshold_ = 0.7, float iou_threshold_ = 0.3) : model(model_), score_threshold(score_threshold_), iou_threshold(iou_threshold_) {
		const std::vector<float> ratios{ 1.0f,0.62f,0.42f };
		//const vector<int> scales{ 33,17,9,5,3 };
		const std::vector<int> scales{ 45,23,12,6,4 };
		const std::vector<float> anchor_base{ 0.04f,0.056f };
		anchors = generate_anchors(ratios, scales, anchor_base);
	}
	FaceDetect(BaseModel* model_,float score_threshold_ = 0.7, float iou_threshold_ = 0.3) : model(*model_), score_threshold(score_threshold_), iou_threshold(iou_threshold_) {
		const std::vector<float> ratios{ 1.0f,0.62f,0.42f };
		//const vector<int> scales{ 33,17,9,5,3 };
		const std::vector<int> scales{ 45,23,12,6,4 };
		const std::vector<float> anchor_base{ 0.04f,0.056f };
		anchors = generate_anchors(ratios, scales, anchor_base);
	}

	int FindFaceLocations(cv::Mat& raw_image, std::vector<FaceInfo>& face_list) {
		std::map<std::string, std::vector<float>> m{};
		model.forward(raw_image, m);
#if 0
//		__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG","cls_branch_concat:");
//		char *buf = new char[310000];
//		int num = 0;
//		for (num = 0; num < m["cls_branch_concat"].size(); ++num) {
////			__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG","%f ",m["cls_branch_concat"][i]);
////			__android_log_buf_write(0, ANDROID_LOG_DEBUG, "LOG_TAG",std::to_string(m["cls_branch_concat"][i]).c_str());
//			sprintf(buf+7*num,"%1.4f ",m["cls_branch_concat"][num]);
//		}
//		buf[7*num] = '\0';
//		__android_log_write(ANDROID_LOG_DEBUG, "LOG_TAG",buf);
//		__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG","\n");
//		__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG","loc_branch_concat:");
//		for (num = 0; num < m["loc_branch_concat"].size(); ++num) {
////			__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG","%f ",m["loc_branch_concat"][i]);
//			sprintf(buf+7*num,"%1.4f ",m["loc_branch_concat"][num]);
//		}
//		buf[7*num] = '\0';
//		__android_log_write(ANDROID_LOG_DEBUG, "LOG_TAG",buf);
//		__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG","\n");
//		delete[] buf;
#endif
		lang::FloatStream cls = {m["cls_branch_concat"].data(),(int)m["cls_branch_concat"].size()};
		lang::FloatStream loc = {m["loc_branch_concat"].data(),(int)m["loc_branch_concat"].size()};

		auto decode_rects = decode_bbox(anchors, loc.ptr);
		std::vector<int> classes;
		std::vector <float>scores;
		std::vector<int> keep_idxs = single_class_non_max_suppression(decode_rects, cls.ptr, cls.len, classes, scores, score_threshold, iou_threshold);
		for (auto& i : keep_idxs)
		{
			std::cout << "x:" << decode_rects[i].x << \
				" y:" << decode_rects[i].y << \
				" width:" << decode_rects[i].width << \
				" height:" << decode_rects[i].height << std::endl;
			const auto image_h = raw_image.rows;
			const auto image_w = raw_image.cols;
			FaceInfo faceinfo{};
			faceinfo.x1 = decode_rects[i].x * image_w;
			faceinfo.y1 = decode_rects[i].y * image_h;
			faceinfo.x2 = (decode_rects[i].x + decode_rects[i].width) * image_w;
			faceinfo.y2 = (decode_rects[i].y + decode_rects[i].height) * image_h;
			faceinfo.score = scores[i];
			face_list.emplace_back(std::move(faceinfo));
		}
		return 0;
	}

private:
	static std::vector<std::vector<float>> generate_anchors(const std::vector<float>& ratios, const std::vector<int>& scales, const std::vector<float>& anchor_base)
	{
		std::vector<std::vector<float>> anchors;
		for (int idx = 0; idx < scales.size(); idx++) {
			std::vector<float> bbox_coords;
			int s = scales[idx];
			std::vector<float> cxys;
			std::vector<float> center_tiled;
			for (int i = 0; i < s; i++) {
				float x = (0.5f + i) / s;
				cxys.push_back(x);
			}

			for (int i = 0; i < s; i++) {
				float x = (0.5f + i) / s;
				for (int j = 0; j < s; j++) {
					for (int k = 0; k < 8; k++) {
						center_tiled.push_back(cxys[j]);
						center_tiled.push_back(x);
					}
				}
			}

			std::vector<float> anchor_width_heights;
			for (int i = 0; i < anchor_base.size(); i++) {
				float scale = anchor_base[i] * (float)pow(2, idx);
				anchor_width_heights.push_back(-scale / 2.0f);
				anchor_width_heights.push_back(-scale / 2.0f);
				anchor_width_heights.push_back(scale / 2.0f);
				anchor_width_heights.push_back(scale / 2.0f);
			}

			for (int i = 0; i < anchor_base.size(); i++) {
				float s1 = anchor_base[0] * (float)pow(2, idx);
				float ratio = ratios[i + 1];
				float w = s1 * sqrt(ratio);
				float h = s1 / sqrt(ratio);
				anchor_width_heights.push_back(-w / 2.0f);
				anchor_width_heights.push_back(-h / 2.0f);
				anchor_width_heights.push_back(w / 2.0f);
				anchor_width_heights.push_back(h / 2.0f);
			}

			int index = 0;
			for (float& a : center_tiled) {
				float c = a + anchor_width_heights[(index++) % anchor_width_heights.size()];
				bbox_coords.push_back(c);
			}

			int anchors_size = (int)bbox_coords.size() / 4;
			for (int i = 0; i < anchors_size; i++) {
				std::vector<float> f;
				for (int j = 0; j < 4; j++) {
					f.push_back(bbox_coords[i * 4 + j]);
				}
				anchors.push_back(f);
			}
		}

		return anchors;
	}

	static std::vector<cv::Rect2f> decode_bbox(std::vector<std::vector<float>> & anchors, float* raw)
	{
		std::vector<cv::Rect2f> rects;
		const float v[4] = { 0.1f, 0.1f, 0.2f, 0.2f };

		int i = 0;
		for (std::vector<float>& k : anchors) {
			float acx = (k[0] + k[2]) / 2;
			float acy = (k[1] + k[3]) / 2;
			float cw = (k[2] - k[0]);
			float ch = (k[3] - k[1]);

			float r0 = raw[i] * v[i % 4];
			++i;
			float r1 = raw[i] * v[i % 4];
			++i;
			float r2 = raw[i] * v[i % 4];
            ++i;
			float r3 = raw[i] * v[i % 4];
            ++i;

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

	typedef struct _FaceInfo {
		cv::Rect2f rect;
		float score;
		int id;
	} _FaceInfo;

	static bool increase(const _FaceInfo & a, const _FaceInfo & b) {
		return a.score > b.score;
	}

	std::vector<int> do_nms(std::vector<_FaceInfo> & bboxes, float thresh, char methodType) {
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

			cv::Rect2f& select_bbox = bboxes[select_idx].rect;
			const float area1 = (select_bbox.width + 1e-3f) * (select_bbox.height + 1e-3f);

			select_idx++;
#pragma omp parallel for num_threads(8)
			for (int32_t i = select_idx; i < num_bbox; i++) {
				if (mask_merged[i] == 1)
					continue;

				cv::Rect2f & bbox_i = bboxes[i].rect;
				float x = std::max<float>(select_bbox.x, bbox_i.x);
				float y = std::max<float>(select_bbox.y, bbox_i.y);
				float w = std::min<float>(select_bbox.width + select_bbox.x, bbox_i.x + bbox_i.width) - x + 1e-3f;
				float h = std::min<float>(select_bbox.height + select_bbox.y, bbox_i.y + bbox_i.height) - y + 1e-3f;
				if (w <= 0 || h <= 0)
					continue;

				const float area2 = (bbox_i.width + 1e-3f) * (bbox_i.height + 1e-3f);
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

	std::vector<int> single_class_non_max_suppression(std::vector<cv::Rect2f> & rects, float* confidences, int c_len, std::vector<int> & classes, std::vector <float> & bbox_max_scores, float conf_thresh = 0.5, float iou_thresh = 0.4)
	{
		std::vector<int> keep_idxs;

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

		std::vector <_FaceInfo>infos;
		for (int i = 0; i < bbox_max_scores.size(); i++) {
			if (bbox_max_scores[i] > conf_thresh) {
				_FaceInfo info;
				info.rect = rects[i];
				info.score = bbox_max_scores[i];
				info.id = i;
				infos.push_back(info);
			}
		}

		keep_idxs = do_nms(infos, iou_thresh, 'u');
		return keep_idxs;
	}

	std::vector<std::vector<float>> anchors;

	float score_threshold;
	float iou_threshold;

	BaseModel& model;
};

#endif // !FACEDETECT_H_
