#include <jni.h>

extern "C" JNIEXPORT jstring JNICALL
whatever(
        JNIEnv *env,
        jobject /* this */){
    return env->NewStringUTF("");
};