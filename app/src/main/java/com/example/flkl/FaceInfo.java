package com.example.flkl;

import org.opencv.core.Rect;

public class FaceInfo {
    public String clf;
    public Rect rect;

    FaceInfo(String clf_,Rect rect_) {
        clf = clf_;
        rect = rect_;
    }
//    static FaceInfo face = new FaceInfo("123",new Rect());
}
