#ifndef _JNI_H_
#define _JNI_H_

#include <jni.h>
#include <android/asset_manager_jni.h>
#include <vector>
#include <map>
#include <string>
#include "FaceDetect.hpp"
#include "FaceRecognition.hpp"
#include "MNNModel.hpp"
#include <android/log.h>
#include <chrono>

//JNIEXPORT jint JNICALL Java_com_example_flkl
//        (JNIEnv *, jobject, jint, jint);

lang::Placement<MNNModel> mnn_model;
//FaceDetect face_detect{ mnn_model,0.65f,0.4f};
lang::Placement<FaceDetect> face_detect;

//MNNModel mnn_feature_model{ "H:/MyData/AI/face_recognize/ultra_mobile_face_clf/mnn/feature_model/tf_mobilefacenet.mnn",
//          112, 112, 4, {127.5f, 127.5f, 127.5f}, {1/127.5f, 1/127.5f, 1/127.5f} };
lang::Placement<MNNModel> mnn_feature_model;

//MNNModel mnn_landmark_model{ "H:/MyData/AI/face_recognize/ultra_mobile_face_clf/mnn/landmark_model/torch_landmark.mnn",
//                             56, 56, 4,{ 127.5f, 127.5f, 127.5f },{ 1/127.5f, 1 / 127.5f, 1 / 127.5f } };
lang::Placement<MNNModel> mnn_landmark_model;

//FaceRecognition faec_clf(mnn_landmark_model, mnn_feature_model);
lang::Placement<FaceRecognition> faec_clf;

std::map<std::string,std::vector<float>> recog_faces;

static int LoadMNNModel(lang::Placement<MNNModel>& model,AAssetManager* mgr, const char* model_name, int input_width, int input_length, int num_thread_ = 4,
        std::vector<float> mean_vals_ = { 0, 0, 0 },
        std::vector<float> norm_vals_ = { 1.0f / 255, 1.0f / 255, 1.0f / 255 }) {
    AAsset* asset = AAssetManager_open(mgr, model_name, AASSET_MODE_BUFFER);
    off_t size = AAsset_getLength(asset);
    char* buffer = (char*) malloc (sizeof(char)*size);
    AAsset_read (asset,buffer,size);
    AAsset_close(asset);
//    model = MNNModel{buffer,(int)size,input_width, input_length, num_thread_,mean_vals_,norm_vals_};
    model.Initialize(buffer,(int)size,input_width, input_length, num_thread_,mean_vals_,norm_vals_);
    free(buffer);
    return 0;
}

extern "C" {
JNIEXPORT jobjectArray
JNICALL
Java_com_example_flkl_FaceClf_clf(JNIEnv *env, jobject thiz, jlong raw_image_addr) {
    // TODO: implement clf()

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::vector<FaceDetect::FaceInfo> face_info;
    cv::Mat &raw_image = *(cv::Mat*)raw_image_addr;
    cv::Mat c3u8_image(raw_image.rows,raw_image.cols,CV_8UC3);
    raw_image.convertTo(c3u8_image, CV_8UC3);
    cv::cvtColor(c3u8_image,c3u8_image,cv::COLOR_BGR2RGB);
//    for (int i = 0; i < c3u8_image.rows; ++i) {
//        auto *ptr = c3u8_image.data;
//        for (int j = 0; j < c3u8_image.cols; ++j) {
//            ptr[i*c3u8_image.cols+j*3+0] = 1;
//            ptr[i*c3u8_image.cols+j*3+1] = 1;
//            ptr[i*c3u8_image.cols+j*3+2] = 1;
//        }
//    }
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start);
    LOGD("image deal used %f ms\n",time_span.count() * 1000);
    start = std::chrono::steady_clock::now();
    face_detect->FindFaceLocations(c3u8_image, face_info);
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start);
    LOGD("FindFaceLocations used %f ms\n",time_span.count() * 1000);

    struct rect {
        uint32_t x1;
        uint32_t y1;
        uint32_t x2;
        uint32_t y2;
    };
    uint32_t unknown_num{};
    std::map<std::string,rect> face_rects;
    for (const auto& face : face_info) {
        start = std::chrono::steady_clock::now();
        std::vector<float> facechipmobilefeature = faec_clf->encode(c3u8_image, face);
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start);
        LOGD("encode used %f ms\n",time_span.count() * 1000);
        LOGD( "feature:%f %f %f %f\n",facechipmobilefeature[0], facechipmobilefeature[1],facechipmobilefeature[2],facechipmobilefeature[3]);
        bool is_find = {};
        for (const auto& feature : recog_faces) {
            start = std::chrono::steady_clock::now();
            auto face_dist = faec_clf->calculSimilar(feature.second, facechipmobilefeature);
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start);
            LOGD("calculSimilar used %f ms\n",time_span.count() * 1000);
            printf("face_dist:%f\n", face_dist);
            if (face_dist < 1.0) {
//                face_rects.emplace(feature.first,{face.x1,face.y1,face.x2,face.y2});
                face_rects.insert(std::pair<std::string,rect>(feature.first,{(uint32_t)(face.x1+0.5),(uint32_t)(face.y1+0.5),(uint32_t)(face.x2+0.5),(uint32_t)(face.y2+0.5)}));
                is_find = true;
                break;
            }
        }
        if (!is_find) {
            face_rects.insert(std::pair<std::string,rect>(std::string("?")+std::to_string(unknown_num),{(uint32_t)(face.x1+0.5),(uint32_t)(face.y1+0.5),(uint32_t)(face.x2+0.5),(uint32_t)(face.y2+0.5)}));
            unknown_num += 1;
        }
    }

    const auto size = face_rects.size();
    jclass face_info_cls = env->FindClass( "com/example/flkl/FaceInfo");
//    jclass rect_cls = env->FindClass( "org/opencv/core/Rect");
////    result = (*env)->NewObject(env, intArrCls,NULL);
////    jmethodID cnstrctr = (*env)->GetMethodID(env, cls, "<init>", "(Ljava/lang/String;[B)V");
//    jmethodID rect_construct = env->GetMethodID(rect_cls, "<init>", "(IIII)V");
//    jmethodID faceinfo_construct = env->GetMethodID(face_info_cls, "<init>",
//                                                    "(Ljava/lang/String;Lorg/opencv/core/Rect;)V");
    jobjectArray result = env->NewObjectArray( size, face_info_cls,
                                    NULL);

    uint32_t i{};
    for (auto const& face_rect : face_rects)
    {
        jclass rect_cls = env->FindClass( "org/opencv/core/Rect");
        jmethodID rect_construct = env->GetMethodID(rect_cls, "<init>", "(IIII)V");
        jmethodID faceinfo_construct = env->GetMethodID(face_info_cls, "<init>",
                                                        "(Ljava/lang/String;Lorg/opencv/core/Rect;)V");

        jint x,y,width,height;
        x = face_rect.second.x1;
        y = face_rect.second.y1;
        width = face_rect.second.x2 - face_rect.second.x1;
        height = face_rect.second.y2 - face_rect.second.y1;
        auto rect_tmp = env->NewObject(rect_cls, rect_construct,x,y,width,height);
        auto str_tmp = env->NewStringUTF(face_rect.first.c_str());
        auto faceinfo_tmp = env->NewObject(face_info_cls, faceinfo_construct, str_tmp, rect_tmp);
        env->SetObjectArrayElement( result, i++, faceinfo_tmp);
//        (*env)->DeleteLocalRef(env, iarr);
    }

    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_example_flkl_FaceClf_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
    // TODO: implement Init()
//    mnn_model.Initialize();
//    mnn_feature_model.Initialize();
//    mnn_landmark_model.Initialize();
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    LoadMNNModel(mnn_model,mgr,"torch_model360.mnn",360,360,4);
    LoadMNNModel(mnn_feature_model,mgr,"tf_mobilefacenet.mnn", 112, 112, 4, {127.5f, 127.5f, 127.5f}, {1/127.5f, 1/127.5f, 1/127.5f});
    LoadMNNModel(mnn_landmark_model,mgr,"torch_landmark.mnn", 56, 56, 4,{ 127.5f, 127.5f, 127.5f },{ 1/127.5f, 1 / 127.5f, 1 / 127.5f });
    faec_clf.Initialize(mnn_landmark_model.operator->(), mnn_feature_model.operator->());
    face_detect.Initialize(mnn_model.operator->(),0.65f,0.4f);
    return {1};
}

JNIEXPORT jboolean JNICALL
Java_com_example_flkl_FaceClf_AddFace(JNIEnv *env, jobject thiz, jstring name,jlong raw_image_addr) {
    // TODO: implement AddFace()
    const char *_name = env->GetStringUTFChars( name, NULL);
    if (NULL == _name) {
        return {0};
    }

    std::vector<FaceDetect::FaceInfo> face_info;
    cv::Mat &raw_image = *(cv::Mat*)raw_image_addr;
    cv::Mat c3u8_image(raw_image.rows,raw_image.cols,CV_8UC3);
    raw_image.convertTo(c3u8_image, CV_8UC3);
    cv::cvtColor(c3u8_image,c3u8_image,cv::COLOR_BGR2RGB);

    face_detect->FindFaceLocations(c3u8_image, face_info);
    std::vector<float> facechipmobilefeature = faec_clf->encode(c3u8_image, face_info[0]);
    if ( recog_faces.find(_name) != recog_faces.end() ) {
        recog_faces.erase(_name);
    }
    recog_faces.emplace(std::move(std::string(_name)),std::move(facechipmobilefeature));
    return {1};
}

}

#endif