cmake_minimum_required(VERSION 3.4.1)

include_directories(
        H:\\opencv-4.3.0-android-sdk\\OpenCV-android-sdk\\sdk\\native\\jni\\include
        include
        dlib_include
        )

link_directories(H:/AndroidStudioProjects/FLKL/app/src/main/jniLibs/arm64-v8a)

add_definitions(-DANDROID_LOGCAT)

add_library( # Sets the name of the library.
        face_clf

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        jni.cpp)


find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library(android-lib android )

target_link_libraries( # Specifies the target library.
        face_clf
        MNN
        opencv_java4
        dlib
        ${android-lib}
#        -Wl,--whole-archive ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/libMNN.a -Wl,--no-whole-archive
        ${log-lib}
        )

#        # Links the target library to the log library
#        # included in the NDK.
#        ${log-lib})