#ifndef FACERECOGNITION_H_
#define FACERECOGNITION_H_

#include "FaceDetect.hpp"
#include "FaceLandmark.hpp"
#include "FaceFeature.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <dlib/matrix.h>
#include <dlib/opencv.h>
//#include <dlib/gui_widgets.h>

class FaceRecognition
{
public:
	FaceRecognition(BaseModel& landmark_model, BaseModel& feature_model) :
		_landmark_model(landmark_model), _feature_model(feature_model) {

	}
	FaceRecognition(BaseModel* landmark_model, BaseModel* feature_model) :
		_landmark_model(*landmark_model), _feature_model(*feature_model) {

	}
	~FaceRecognition() {};

	std::vector<float> encode(const cv::Mat& raw_image, const FaceDetect::FaceInfo& face_info) {
//		lang::FloatStream landmark{};
		std::vector<float> landmark;
		_landmark_model.GetLandmark(raw_image({ MAX(0,MIN((int)face_info.y1,raw_image.rows)),MAX(0,MIN((int)face_info.y2,raw_image.rows)) }, { MAX(0,MIN((int)face_info.x1,raw_image.cols)),MAX(0,MIN((int)face_info.x2,raw_image.cols)) }), landmark);
		cv::Mat face_chip;
		DlibGetFaceChip(raw_image, face_info, {landmark.data(),(int)landmark.size()}, face_chip);

//		lang::FloatStream fs{};
		std::vector<float> fs;
		_feature_model.GetFeature(face_chip, fs);
//		return { fs.ptr, fs.ptr + fs.len };
		return fs;
	}

	static int DlibGetFaceChip(const cv::Mat &cvimg, const FaceDetect::FaceInfo& face_info, const lang::FloatStream& landmark, cv::Mat &out) {
		dlib::cv_image<dlib::bgr_pixel> img(cvimg);
		std::vector<dlib::point> parts(landmark.len / 2);
		const long w = (long)(face_info.x2 - face_info.x1);
		const long h = (long)(face_info.y2 - face_info.y1);
		for (size_t i = 0; i < parts.size(); i++)
		{
			parts[i] = { (long)face_info.x1 + (long)(landmark[i * 2] * w),(long)face_info.y1 + (long)(landmark[i * 2 + 1] * h) };
		}
		dlib::full_object_detection shape{ { (long)face_info.x1,(long)face_info.y1,(long)face_info.x2,(long)face_info.y2 },parts };

		dlib::matrix<dlib::bgr_pixel> face_chip;
		extract_image_chip(img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
		out = cv::Mat(face_chip.nr(), face_chip.nc(), CV_8UC3);
		for (size_t i = 0; i < face_chip.nr(); i++)
		{
			unsigned char *ptr = (unsigned char*)(out.data);
			for (size_t j = 0; j < face_chip.nc(); j++)
			{
				ptr[i * out.step + j * 3 + 0] = face_chip(i, j).blue;
				ptr[i * out.step + j * 3 + 1] = face_chip(i, j).green;
				ptr[i * out.step + j * 3 + 2] = face_chip(i, j).red;
			}
		}

#if DEBUG
		dlib::image_window win(face_chip);
		win.wait_until_closed();

		auto ld_img = cvimg;
		for (size_t i = 0; i < parts.size(); i++)
		{
			cv::circle(ld_img, { parts[i].x(),parts[i].y() }, 4, { 0, 255, 0 });
			//auto font = cv::FONT_HERSHEY_SIMPLEX;
			//char buf[32];
			//itoa(i, buf, 10);
			//cv::putText(ld_img, std::string(buf), { parts[i].x(),parts[i].y() }, font, 0.8, (0, 0, 255), 1, cv::LINE_AA);
		}
		//cv::resize(ld_img, ld_img, { 150,150 });
		cv::imshow("landmarks", ld_img);
#endif	//DEBUG

		return 0;
	}

	static double calculSimilar(const std::vector<float>& v1, const std::vector<float>& v2, int distance_metric = 0)
	{
		//if (v1.size() != v2.size() || !v1.size())
		//	return 0;
		assert(v1.size() == v2.size() && v1.size());
		double ret = 0.0, mod1 = 0.0, mod2 = 0.0, dist = 0.0, diff = 0.0;

		if (distance_metric == 0) {         // Euclidian distance
			for (std::vector<double>::size_type i = 0; i != v1.size(); ++i) {
				diff = v1[i] - v2[i];
				dist += (diff * diff);
			}
			dist = sqrt(dist);
		}
		else {                              // Distance based on cosine similarity
			for (std::vector<double>::size_type i = 0; i != v1.size(); ++i) {
				ret += v1[i] * v2[i];
				mod1 += v1[i] * v1[i];
				mod2 += v2[i] * v2[i];
			}
			dist = ret / (sqrt(mod1) * sqrt(mod2));
		}
		return dist;
	}

	static float EuclideanDistance(std::vector<float>& vec1, std::vector<float>& vec2) {
		assert(vec1.size() == vec2.size() && vec1.size());
		float ret = 0.0;
		for (int i{}; i < vec1.size(); ++i) {
			float dist = (vec1[i]++) - (vec2[i]++);
			ret += dist * dist;
		}
		return ret > 0.0f ? sqrt(ret) : 0.0f;
	}

private:
	FaceFeature _feature_model;
	FaceLandmark _landmark_model;
};

#endif // FACERECOGNITION_H_
