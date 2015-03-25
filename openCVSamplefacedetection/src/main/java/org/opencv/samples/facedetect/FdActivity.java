package org.opencv.samples.facedetect;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;


    private int learn_frames = 0;
    private Mat teplateR;
    private Mat teplateL;
    int method = 0;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;

    private Mat mRgba;
    private Mat mGray;
    // matrix for zooming
    private Mat mZoomWindow;
    private Mat mZoomWindow2;

    private File mCascadeFile;
    private CascadeClassifier mJavaDetectorFace;
    private CascadeClassifier mJavaDetectorEye;
    private CascadeClassifier mJavaDetectorMouth;
    private CascadeClassifier mJavaDetectorNose;


    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    double xCenter = -1;
    double yCenter = -1;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {


                    try {
                        // load cascade file from application resources
                        File cascadeDir = loadFace();
                        File cascadeDirER = loadEyes();
                        File cascadeDirERM = loadMouth();
                        File cascadeDirERN = loadNose();

                        cascadeDir.delete();
                        cascadeDirER.delete();
                        cascadeDirERM.delete();
                        cascadeDirERN.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
//                    mOpenCvCameraView.setCameraIndex(0);
//                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.enableView();


        }
    };

    private File loadMouth() throws IOException {
        InputStream iserM = getResources().openRawResource(
                R.raw.haarcascade_mcs_mouth);
        File cascadeDirERM = getDir("cascadeERM",
                Context.MODE_PRIVATE);
        File cascadeFileERM = new File(cascadeDirERM,
                "haarcascade_mcs_mouth.xml");
        FileOutputStream oserM = new FileOutputStream(cascadeFileERM);

        byte[] bufferERM = new byte[4096];
        int bytesReadERM;
        while ((bytesReadERM = iserM.read(bufferERM)) != -1) {
            oserM.write(bufferERM, 0, bytesReadERM);
        }
        iserM.close();
        oserM.close();

        mJavaDetectorMouth = new CascadeClassifier(
                cascadeFileERM.getAbsolutePath());
        if (mJavaDetectorMouth.empty()) {
            Log.e(TAG, "Failed to load cascade classifier");
            mJavaDetectorMouth = null;
        } else
            Log.i(TAG, "Loaded cascade classifier from "
                    + mCascadeFile.getAbsolutePath());
        return cascadeDirERM;
    }

    private File loadNose() throws IOException {
        InputStream iserM = getResources().openRawResource(
                R.raw.haarcascade_mcs_nose);
        File cascadeDirERM = getDir("cascadeERN",
                Context.MODE_PRIVATE);
        File cascadeFileERM = new File(cascadeDirERM,
                "haarcascade_mcs_nose.xml");
        FileOutputStream oserM = new FileOutputStream(cascadeFileERM);

        byte[] bufferERM = new byte[4096];
        int bytesReadERM;
        while ((bytesReadERM = iserM.read(bufferERM)) != -1) {
            oserM.write(bufferERM, 0, bytesReadERM);
        }
        iserM.close();
        oserM.close();

        mJavaDetectorNose = new CascadeClassifier(
                cascadeFileERM.getAbsolutePath());
        if (mJavaDetectorNose.empty()) {
            Log.e(TAG, "Failed to load cascade classifier");
            mJavaDetectorNose = null;
        } else
            Log.i(TAG, "Loaded cascade classifier from "
                    + cascadeFileERM.getAbsolutePath());
        return cascadeDirERM;
    }

    private File loadEyes() throws IOException {
        InputStream iser = getResources().openRawResource(
                R.raw.haarcascade_lefteye_2splits);
        File cascadeDirER = getDir("cascadeER",
                Context.MODE_PRIVATE);
        File cascadeFileER = new File(cascadeDirER,
                "haarcascade_eye_right.xml");
        FileOutputStream oser = new FileOutputStream(cascadeFileER);

        byte[] bufferER = new byte[4096];
        int bytesReadER;
        while ((bytesReadER = iser.read(bufferER)) != -1) {
            oser.write(bufferER, 0, bytesReadER);
        }
        iser.close();
        oser.close();

        mJavaDetectorEye = new CascadeClassifier(
                cascadeFileER.getAbsolutePath());
        if (mJavaDetectorEye.empty()) {
            Log.e(TAG, "Failed to load cascade classifier");
            mJavaDetectorEye = null;
        } else
            Log.i(TAG, "Loaded cascade classifier from "
                    + cascadeDirER.getAbsolutePath());
        return cascadeDirER;
    }

    private File loadFace() throws IOException {
        InputStream is = getResources().openRawResource(
                R.raw.lbpcascade_frontalface);
        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
        mCascadeFile = new File(cascadeDir,
                "lbpcascade_frontalface.xml");
        FileOutputStream os = new FileOutputStream(mCascadeFile);

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        is.close();
        os.close();

        mJavaDetectorFace = new CascadeClassifier(
                mCascadeFile.getAbsolutePath());
        if (mJavaDetectorFace.empty()) {
            Log.e(TAG, "Failed to load cascade classifier");
            mJavaDetectorFace = null;
        } else
            Log.i(TAG, "Loaded cascade classifier from "
                    + mCascadeFile.getAbsolutePath());
        return cascadeDir;
    }

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,
                mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        if(mZoomWindow != null){
            mZoomWindow.release();
            mZoomWindow2.release();
        }
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        if (mZoomWindow == null || mZoomWindow2 == null)
            CreateAuxiliaryMats();



//
        MatOfRect faces = new MatOfRect();

        if (mJavaDetectorFace != null)
            mJavaDetectorFace.detectMultiScale(mGray, faces, 1.1, 2,
                    2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                    new Size());

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
//            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),
//                    FACE_RECT_COLOR, 3);
            xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
            yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            Rect r = facesArray[i];

            Core.circle(mRgba, center, r.width/2, new Scalar(255, 0, 0, 255), 3);



//            // compute the eye area
//            Rect eyearea = new Rect(r.x + r.width / 8,
//                    (int) (r.y + (r.height / 4.5)), r.width - 2 * r.width / 8,
//                    (int) (r.height / 3.0));
//            // split it
//
//            Rect eyearea_left = new Rect(r.x + r.width / 16
//                    + (r.width - 2 * r.width / 16) / 2,
//                    (int) (r.y + (r.height / 4.5)),
//                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
//            Rect eyearea_right = new Rect(r.x + r.width / 16,
//                    (int) (r.y + (r.height / 4.5)),
//                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));

            // draw the area - mGray is working grayscale mat, if you want to
            // see area in rgb preview, change mGray to mRgba
//            Core.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
//                    new Scalar(255, 0, 0, 255), 2);
//            Core.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
//                    new Scalar(255, 0, 0, 255), 2);

//            Core.circle(mRgba, new Point(r.x + r.width / 16 + (r.width - 2 * r.width / 16) / 2 + (r.width - 2 * r.width / 16) / 4, r.y + (r.height / 4.5) + r.height / 6.0), (int) (r.height / 7.0),
//                    new Scalar(255, 0, 255, 255), 2);
//            Core.circle(mRgba, new Point(r.x + r.width / 16 + (r.width - 2 * r.width / 16) / 4, r.y + (r.height / 4.5) + r.height / 6.0), (int) (r.height / 7.0),
//                    new Scalar(255, 0, 255, 255), 2);


//            if (learn_frames < 5) {
//                teplateR = get_template(mJavaDetectorEye, eyearea_right, 24);
//                teplateL = get_template(mJavaDetectorEye, eyearea_left, 24);
//                learn_frames++;
//            } else {
//                // Learning finished, use the new templates for template
//                // matching
//                match_eye(eyearea_right, teplateR, method);
//                match_eye(eyearea_left, teplateL, method);
//
//            }


//            // cut eye areas and put them to zoom windows
//            Imgproc.resize(mRgba.submat(eyearea_left), mZoomWindow2,
//                    mZoomWindow2.size());
//            Imgproc.resize(mRgba.submat(eyearea_right), mZoomWindow,
//                    mZoomWindow.size());

            detectEyes();
            detectMouth();
            detectNose();
        }
        return mRgba;
    }

    private void detectEyes() {
        MatOfRect eyes = new MatOfRect();

        if (mJavaDetectorEye != null)
            mJavaDetectorEye.detectMultiScale(mGray, eyes, 1.1, 2,
                    2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                    new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int j = 0; j < eyesArray.length; j++) {
//            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),
//                    FACE_RECT_COLOR, 3);
            xCenter = (eyesArray[j].x + eyesArray[j].width + eyesArray[j].x) / 2;
            yCenter = (eyesArray[j].y + eyesArray[j].y + eyesArray[j].height) / 2;
            Point center1 = new Point(xCenter, yCenter);


            Core.putText(mRgba, "[" + center1.x + "," + center1.y + "]",
                    new Point(center1.x + 20, center1.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            Rect r1 = eyesArray[j];

            Core.circle(mRgba, center1, r1.width/2, new Scalar(0, 0, 255, 255), 3);
        }
    }

    private void detectMouth() {
        MatOfRect mouth = new MatOfRect();

        if (mJavaDetectorMouth != null)
            mJavaDetectorMouth.detectMultiScale(mGray, mouth, 1.1, 2,
                    2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                    new Size());

        Rect[] mouthArray = mouth.toArray();
        for (int i = 0; mouthArray.length > 0 && i < 1; i++) {
            xCenter = (mouthArray[i].x + mouthArray[i].width + mouthArray[i].x) / 2;
            yCenter = (mouthArray[i].y + mouthArray[i].y + mouthArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            Rect r1 = mouthArray[i];

            Core.rectangle(mRgba, mouthArray[i].tl(), mouthArray[i].br(), new Scalar(0, 255, 0, 255), 3);

//            Core.circle(mRgba, center, r1.width/2, new Scalar(0, 255, 0, 255), 3);
        }
    }

    private void detectNose() {
        MatOfRect mouth = new MatOfRect();

        if (mJavaDetectorNose != null)
            mJavaDetectorNose.detectMultiScale(mGray, mouth, 1.1, 2,
                    2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                    new Size());

        Rect[] mouthArray = mouth.toArray();
        for (int i = 0; i < mouthArray.length; i++) {
            xCenter = (mouthArray[i].x + mouthArray[i].width + mouthArray[i].x) / 2;
            yCenter = (mouthArray[i].y + mouthArray[i].y + mouthArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            Rect r1 = mouthArray[i];

            Core.rectangle(mRgba, mouthArray[i].tl(), mouthArray[i].br(), new Scalar(128, 128, 128, 255), 3);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }


    private void CreateAuxiliaryMats() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

        if (mZoomWindow == null) {
            mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2
                    + cols / 10, cols);
            mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 10, cols / 2
                    + cols / 10, cols);
        }

    }

    private void match_eye(Rect area, Mat mTemplate, int type) {
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return ;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (type) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
                matchLoc.y + mTemplate.rows() + area.y);

        Core.circle(mRgba, matchLoc_tx, 30, new Scalar(255, 255, 0,
                255));
        Rect rec = new Rect(matchLoc_tx,matchLoc_ty);


    }

    private Mat get_template(CascadeClassifier clasificator, Rect area, int size) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
                new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.4), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);


            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Core.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y
                    - size / 2, size, size);
            Core.rectangle(mRgba, eye_template.tl(), eye_template.br(),
                    new Scalar(255, 0, 0, 255), 2);
            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    public void onRecreateClick(View v)
    {
        learn_frames = 0;
    }
}