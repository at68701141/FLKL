package com.example.flkl;
import android.content.res.AssetManager;

public class FaceClf {
    static {
        System.loadLibrary("face_clf");
//        System.loadLibrary("MNN");
//        System.loadLibrary("MNN_CL");
//        System.loadLibrary("MNN_Arm82");
    }

    FaceClf(AssetManager mgr) {
        Init(mgr);
    }

    native FaceInfo[] clf(long  raw_image_addr);

    public native boolean Init(AssetManager mgr);

    public native boolean AddFace(String name,long raw_image_addr);
}
