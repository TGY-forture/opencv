package com.example.opencv;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.imgproc.Imgproc.calcHist;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.rectangle;
import static org.opencv.imgproc.Imgproc.resize;


public class MainActivity extends AppCompatActivity implements View.OnClickListener, CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase cameraView;
    private CascadeClassifier classifier;  //级联分类器
    private Mat mGray;                     //图像数据
    private Mat mRgba;
    private int mAbsoluteFaceSize = 0;
    private Mat previewface = new Mat();   //预览图像中的人脸
    private Mat existimage = new Mat();    //待对比图片中的人脸
    private ImageView left;   //左侧视图
    private ImageView right;  //右侧视图
    private Mat t1 =  new Mat();
    private Mat t2 = new Mat();
    List<String> s = new ArrayList<>();


    static {
        System.loadLibrary("opencv_java4");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) { //启动时调用
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);   //放置视图
        //找到五个按钮
        Button openBtn = findViewById(R.id.open);//找到"打开"按钮
        Button faceDet = findViewById(R.id.face_detect);
        Button faceRec = findViewById(R.id.face_rec);
        Button closeBtn = findViewById(R.id.close);

        left = findViewById(R.id.img1);
        right = findViewById(R.id.img2);

        openBtn.setOnClickListener(this);//绑定点击事件
        faceDet.setOnClickListener(this);
        faceRec.setOnClickListener(this);
        closeBtn.setOnClickListener(this);
                //29                     //23
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) { //检查相机权限
            requestPermissions( new String[]{Manifest.permission.CAMERA}, 203);
        }

        cameraView = findViewById(R.id.camera_view); //找到预览视图
        cameraView.setCvCameraViewListener(this); // 设置相机监听
        cameraView.disableFpsMeter();  //不显示帧率

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Toast tip = Toast.makeText(getApplicationContext(), "相机未授权!", Toast.LENGTH_LONG);
            tip.show();
        } else {
            Log.d(TAG, "Permissions granted");
            cameraView.setCameraPermissionGranted();  //已授权，允许预览
        }

    }

    // 初始化人脸级联分类器，必须先初始化
    private void initClassifier() {
        try {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt); //OpenCV的人脸模型文件： lbpcascade_frontalface_improved
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // 加载 正脸分类器
            classifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //OpenCV库加载并初始化成功后的回调函数
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            // TODO Auto-generated method stub
            switch (status){
                case BaseLoaderCallback.SUCCESS:
                    Toast toast = Toast.makeText(getApplicationContext(), "Load opencv success!", Toast.LENGTH_SHORT);
                    toast.show();
                    break;
                default:
                    super.onManagerConnected(status);
                    Toast toast1 = Toast.makeText(getApplicationContext(), "Load opencv failed!", Toast.LENGTH_LONG);
                    toast1.show();
                    break;
            }
        }
    };

    @Override
    public void onResume() { //这里会执行两次，暂未解决
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        //获取外部存储"读"权限
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            //没有授权，编写申请权限代码
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
        } else {
            Log.d(TAG, "requestMyPermissions: 有读SD权限");
        }
    }

    @Override
    public void onClick(View v) {//按钮点击事件
        switch (v.getId()) {
            case R.id.open:
                cameraView.enableView();
                break;
            case R.id.face_detect:
                initClassifier();
                break;
            case R.id.face_rec:
                faceRecognize();
                break;
            case R.id.close:
                cameraView.disableView();
                break;
            default:break;
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        // 翻转矩阵以适配前后置摄像头
        Core.flip(mRgba, mRgba, 1);
        Core.flip(mGray, mGray, 1);
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows(); //720X720
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);  //144
            }
        }
        MatOfRect faces = new MatOfRect();
        if (classifier != null)
            classifier.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();//检测到的人脸数组
        Scalar faceRectColor = new Scalar(0, 255, 0);
        for (Rect faceRect : facesArray) {
            rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);
            previewface = new Mat(mGray, faceRect); //获得检测到的人脸矩形区域图像
        }
        return mRgba;
    }

    public void faceRecognize() {
        Mat reimg = new Mat();
        resize(previewface, reimg, new Size(400, 400));
        int size = s.size();
        String basepath = "/storage/emulated/0/Atgy/";
        double[] resarray = new double[11];
        for (int i=0;i<size;i++) {
            Mat readimg = Imgcodecs.imread(basepath+s.get(i));
            Mat unilbp1 = getUniformPatternLBP(readimg,3,8);
            unilbp1.convertTo(t1, CvType.CV_8UC1);
            Mat hist_1 = getLBPH(t1,59,8,8,true);
            //预览的人脸
            Mat unilbp2 = getUniformPatternLBP(reimg,3,8);
            unilbp2.convertTo(t2, CvType.CV_8UC1);
            Mat hist_2 = getLBPH(t2,59,8,8,true);
            double res = Imgproc.compareHist(hist_1, hist_2, Imgproc.CV_COMP_CORREL);
            resarray[i] = res;
            //Log.i(TAG, "faceRecognize: " + res);
        }
        double max=0;
        int ek = 0;
        for (int j=0;j<resarray.length;j++) {
            double tmpmax = 0;
            for (int k = 0;k<resarray.length;k++) {
                if (resarray[k] > resarray[j]) {
                    tmpmax = resarray[k];
                    ek = k;
                }
            }
            if (tmpmax > max) max = tmpmax;
        }
        Mat show = Imgcodecs.imread(basepath+s.get(ek));
        Bitmap tmpbitmap = Bitmap.createBitmap(show.width(),show.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(show,tmpbitmap);
        right.setImageBitmap(tmpbitmap);

        TextView restext = findViewById(R.id.text_view);
        restext.setText("最大相似度为："+max);
    }

    public void fun(View v) {
        Mat stdfaceimg = new Mat();
        resize(previewface, stdfaceimg, new Size(400, 400));
        Bitmap bitimg1 = Bitmap.createBitmap(400,400, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(stdfaceimg, bitimg1);
        left.setImageBitmap(bitimg1);
        if (s.isEmpty()) {
            File file = new File("/storage/emulated/0/Atgy");
            File[] files=file.listFiles();
        if (files == null){
            Log.e("error","空目录");
        }
            for(int i =0;i<files.length;i++){
                s.add(files[i].getName());
                //System.out.println(files[i].getName());
            }
        }
        //获取外部存储"写"权限
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            //没有授权，编写申请权限代码
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 100);
        } else {
            Log.d(TAG, "requestMyPermissions: 有写SD权限");
        }
    }

    public Mat getLbpHistogram(Mat face){
        int width = face.width();
        int height = face.height();
        Mat lbpface = new Mat(width, height, CvType.CV_32SC3);
        //initialize the lbp histogram
//        int[] lbpHistogram = new int[256];
//        Arrays.fill(lbpHistogram,0);
        for(int i = 1; i<width-1; i++){
            for(int j = 1;j<height-1; j++){
                int l = 0;
                if(face.get(i-1,j-1)[0]>face.get(i, j)[0]) l+=1<<7;
                if(face.get(i-1,j)[0]>face.get(i, j)[0]) l+=1<<6;
                if(face.get(i-1,j+1)[0]>face.get(i, j)[0]) l+=1<<5;
                if(face.get(i,j+1)[0]>face.get(i, j)[0]) l+=1<<4;
                if(face.get(i+1,j+1)[0]>face.get(i, j)[0]) l+=1<<3;
                if(face.get(i+1,j)[0]>face.get(i, j)[0]) l+=1<<2;
                if(face.get(i+1,j-1)[0]>face.get(i, j)[0]) l+=1<<1;
                if(face.get(i,j-1)[0]>face.get(i, j)[0]) l+=1;
                //fill the lbp image
                lbpface.put(i, j, new int[]{l,l,l});
                //calc the lbp histogram
//                lbpHistogram[l]++;
            }
        }
//        return lbpHistogram;
        return lbpface;
    }

    public Mat getUniformPatternLBP(Mat src,int radius,int neighbors) {
        Mat dst = Mat.zeros(src.rows()-2*radius,src.cols()-2*radius,CvType.CV_8UC1);
        //LBP特征值对应图像灰度编码表，直接默认采样点为8位
        int temp = 1;
        float[] table = new float[256];
        Arrays.fill(table,0);
        for(int i=0;i<256;i++)
        {
            if(getHopTimes(i)<3)
            {
                table[i] = temp;
                temp++;
            }
        }
        //是否进行UniformPattern编码的标志
        boolean flag = false;
        //计算LBP特征图
        for(int k=0;k<neighbors;k++) {
            if(k==neighbors-1)
            {
                flag = true;
            }
            //计算采样点对于中心点坐标的偏移量rx，ry
            double rx = (radius * Math.cos(2.0 * Math.PI * k / neighbors));
            double ry = -(radius * Math.sin(2.0 * Math.PI * k / neighbors));
            //为双线性插值做准备
            //对采样点偏移量分别进行上下取整
            int x1 = (int)Math.floor(rx);
            int x2 = (int)Math.ceil(rx);
            int y1 = (int)Math.floor(ry);
            int y2 = (int)Math.ceil(ry);
            //将坐标偏移量映射到0-1之间
            double tx = rx - x1;
            double ty = ry - y1;
            //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
            double w1 = (1-tx) * (1-ty);
            double w2 =    tx  * (1-ty);
            double w3 = (1-tx) *    ty;
            double w4 =    tx  *    ty;
            //循环处理每个像素
            for(int i=radius;i<src.rows()-radius;i++) {
                for(int j=radius;j<src.cols()-radius;j++) {
                    //获得中心像素点的灰度值
                    int center = (int)src.get(i,j)[0];
                    //根据双线性插值公式计算第k个采样点的灰度值
                    double neighbor = src.get(i+x1, j+y1)[0] * w1 + src.get(i+x1,j+y2)[0] *w2+ src.get(i+x2,j+y1)[0] * w3 +src.get(i+x2,j+y2)[0] *w4;
                    //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                    if (neighbor>center) {
                        dst.put(i-radius,j-radius,(int)dst.get(i-radius,j-radius)[0] | (1 << (neighbors-k-1)));
                    } else {
                        dst.put(i-radius,j-radius,(int)dst.get(i-radius,j-radius)[0] | (0 << (neighbors-k-1)));
                    }
                    //进行LBP特征的UniformPattern编码
                    if(flag)
                    {
                        dst.put(i-radius,j-radius,table[(int)dst.get(i-radius,j-radius)[0]]);
                    }
                }
            }
        }
        return dst;
    }

    public Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, boolean normlized) {
        int width = src.cols() / grid_x; //每一块的列数
        int height = src.rows() / grid_y; //每一块的行数
        //定义LBPH的行和列，grid_x*grid_y表示将图像分割成这么些块，numPatterns表示LBP值的模式种类
        Mat result = Mat.zeros(grid_x * grid_y,numPatterns, CvType.CV_32FC1);
        if(src.empty())
        {
            return result.reshape(1,1);
        }
        int resultRowIndex = 0;
        //对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
        for(int i=0;i<grid_x;i++)
        {
            for(int j=0;j<grid_y;j++)
            {
                //图像分块
                Mat src_cell = new Mat(src, new Range(i*height,(i+1)*height), new Range(j*width,(j+1)*width));
                //计算直方图
                Mat hist_cell = getLocalRegionLBPH(src_cell,0,(numPatterns-1),normlized);
                //将直方图放到result中
                Mat rowResult = result.row(resultRowIndex);
                hist_cell.reshape(1,1).convertTo(rowResult,CvType.CV_32FC1);
                resultRowIndex++;
            }
        }
        return result;
    }

    //计算一个LBP特征图像块的直方图
    public Mat getLocalRegionLBPH(Mat src,int minValue,int maxValue,boolean normalized) {
        //定义存储直方图的矩阵
        Mat result = new Mat();
        //直方图每一维变化范围
        MatOfFloat ranges = new MatOfFloat(0f, 59f);
        //直方图大小， 越大匹配越精确 (越慢)
        MatOfInt histSize = new MatOfInt(59);
        calcHist(Arrays.asList(src),new MatOfInt(0),new Mat(),result,histSize,ranges);
//        Mat out = new Mat();
        //归一化
        if(normalized)
        {
            Core.normalize(result,result,1,0,Core.NORM_L1);
//            result /= src.total();
        }
        //结果表示成只有1行的矩阵
        return result.reshape(1,1);
    }

    //计算跳变次数
    public int getHopTimes(int n) {
        int count = 0;
        String s = Integer.toBinaryString(n);
        //判断一下：如果转化为二进制为0或者1或者不满8位，要在数前补0
        int bit = 8-s.length();
        if(s.length()<8){
            for(int j=0; j<bit; j++){
                s = "0"+s;
            }
        }
        char[] arr = s.toCharArray();
        for(int i=1;i<8;i++)
        {
            if(arr[i] != arr[i-1])
            {
                count++;
            }
        }
        return count;
    }

    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
}