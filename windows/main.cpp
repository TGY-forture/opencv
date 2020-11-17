#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/predict_collector.hpp>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
namespace fs = boost::filesystem;

using namespace cv;
using namespace cv::face;
using namespace std;


CascadeClassifier face_detector, eye_detector;
string face_path = "D:\\myocv\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
string eye_path = "D:\\myocv\\install\\etc\\haarcascades\\haarcascade_eye.xml";

//通过摄像头拍摄人脸收集数据
void photograph(string save_path)
{
	VideoCapture cap(0); // open camera
	int image_number = 0;
	if (!face_detector.load(face_path))
	{
		std::cout << "Can't load face xml!" << endl;
		exit(0);
	}
	if (!eye_detector.load(eye_path))
	{
		std::cout << "Can't load eye xml!" << endl;
		exit(0);
	}
	if (!cap.isOpened())
	{
		std::cout << "Can't open Camera!" << std::endl;
		exit(0);
	}
	Mat frame;
	Mat gray;
	Mat use_face;
	vector<Rect> faces;
	vector<Rect> eyes;
	while (cap.read(frame))
	{
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		//flip(gray, gray, 1);

		vector<Rect> faces;

		face_detector.detectMultiScale(gray, faces, 1.2, 3, 0, Size(30, 30));//检测人脸

		for (size_t t = 0; t < faces.size(); t++)
		{
			Rect roi;
			roi.x = faces[static_cast<int>(t)].x;
			roi.y = faces[static_cast<int>(t)].y;
			roi.width = faces[static_cast<int>(t)].width;
			roi.height = faces[static_cast<int>(t)].height / 2;
			Mat faceROI = frame(roi);
			//当检测到只有一张脸时，检测眼睛
			if (faces.size() == 1)
			{
				eye_detector.detectMultiScale(faceROI, eyes, 1.2, 3, 0, Size(20, 20));//检测眼睛
				for (size_t k = 0; k < eyes.size(); k++)
				{
					Rect rect;
					rect.x = faces[static_cast<int>(t)].x + eyes[k].x;
					rect.y = faces[static_cast<int>(t)].y + eyes[k].y;
					rect.width = eyes[k].width;
					rect.height = eyes[k].height;
					rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);

					//只有都检测到两只眼睛时开始拍照
					if (eyes.size() == 2)
					{
						char key = waitKey(100);
						switch (key)
						{
							//按下p键时开始拍照
							case 'p':
							{
								Mat face_roi = gray(faces[0]);
								//图像序号加1
								image_number++;
								//将检测到的人脸大小改为92*112与官司数据一样
								resize(face_roi, use_face, Size(92, 112));

								string filename = save_path + format("%d.jpg", image_number);
								//存放到传入的目录
								imwrite(filename, use_face);
								imshow(filename, use_face);
								waitKey(1000);
								//销毁指定的窗口
								destroyWindow(filename);
								break;
							}
						}

						//按esc键退出拍照
						if (key == 27)
						{
							cap.release();
							exit(0);
						}
					}
				}
				rectangle(frame, faces[t], Scalar(255, 0, 0), 2, 8, 0);
			}
		}
		imshow("camera", frame);
		waitKey(33);
	}
}

//生成txt清单文件
void buildList(string face_iamge_path)
{
	string bow_path = face_iamge_path + string("faceList.txt");
	ifstream read_file(bow_path);
	ofstream ous(bow_path);
	fs::path face_path(face_iamge_path);
	if (!fs::exists(face_path))
	{
		std::cout << "NO File Need to Train!" << std::endl;
		exit(0);
	}
	fs::directory_iterator begin_iter(face_iamge_path);
	fs::directory_iterator end_iter;
	int i = 0;
	//递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
	for (; begin_iter != end_iter; ++begin_iter)
	{
		i++;
		if (fs::is_directory(begin_iter->path()))  //判断当前文件夹是不是一个目录，一个人脸数据集即为一个目录
		{
			fs::directory_iterator begin(begin_iter->path());
			fs::directory_iterator end;
			for (; begin != end; ++begin)
			{
				string face_name = begin->path().string() + ";" + to_string(i);
				//cout << face_name << endl;
				ous << face_name << endl;
			}
		}
	}
	ous.close();
}

//训练样本生成模型
void trainFacesTxt(string faces_list, string save_model)
{
	//打开人脸列表文件
	ifstream face_file(faces_list, ifstream::in);
	if (!face_file)
	{
		std::cout << "无法打开训练集的列表文件！" << endl;
		return;
	}
	string line, path, class_label;
	vector<Mat> faces;
	vector<int> labels;
	char separator = ';';

	while (getline(face_file, line))
	{
		stringstream liness(line);
		//读一行
		getline(liness, path, separator);
		getline(liness, class_label);
		if (!path.empty() && !class_label.empty())
		{
			//把图像压入容器
			faces.push_back(imread(path, 0));
			//把标签压入容器
			labels.push_back(atoi(class_label.c_str()));
		}
	}
	//判断是否为空
	if (faces.size() < 1 || labels.size() < 1)
	{
		std::cout << "初始化训练集....." << std::endl;
		return;
	}
	int height = faces[0].rows;
	int width = faces[0].cols;
	std::cout << "训练集的图像的高：" << height << "训练集的图像的宽：" << width << std::endl;

	Mat test_sample = faces[faces.size() - 1];
	int test_label = labels[labels.size() - 1];
	faces.pop_back();
	labels.pop_back();

	//判断图像类型
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (faces.at(i).type() != CV_8UC1)
		{
			std::cerr << "图像的类型必须为CV_8UC1!" << endl;
			return;
		}
	}

	//检测尺寸等于正样本尺寸第一张的尺寸
	Size positive_image_size = faces[0].size();
	cout << "正样本的尺寸是:" << positive_image_size << endl;
	//遍历所有样品，检测尺寸是否相同
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (positive_image_size != faces[i].size())
		{
			std::cerr << "所有的样本的尺寸大小不一，请重新调整好样本大小！" << endl;
			return;
		}
	}

	//创建一个人脸识别的类
	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	//开始训练
	model->train(faces, labels);
	// recognition face
	int predicted_label = model->predict(test_sample);
	std::cout << "样本标签类型为：" << test_label << "预测的样本标签为：" << predicted_label << endl;
	//保存训练好的模型
	model->write(save_model);
	std::cout << "训练完成！" << endl;
}

//从摄像头识别人脸
void testFace(int cap_index, string model_path)
{
	//加载一个人脸识别器
	//Ptr<BasicFaceRecognizer>model = FisherFaceRecognizer::create();
	//Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
	Ptr<BasicFaceRecognizer> model_test = EigenFaceRecognizer::create();

	//opencv3.3要用read，要不然会出错
	model_test->read(model_path);
	//加载一个人脸检测分类器
	CascadeClassifier faceDetector;
	faceDetector.load(face_path);

	//检测传入的摄像头
	VideoCapture capture(cap_index);
	if (!capture.isOpened())
	{
		std::cerr << "无法打开当前摄像头！" << endl;
		return;
	}
	Mat frame;
	namedWindow("faceRecognition", CV_WINDOW_AUTOSIZE);
	vector<Rect> faces;
	Mat dst;
	Mat test_sample;
	string name;

	while (capture.read(frame))
	{
		//镜像
		flip(frame, frame, 1);
		//检测人脸
		faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(80, 100), Size(380, 400));

		for (int i = 0; i < faces.size(); i++)
		{
			Mat roi = frame(faces[i]);
			cvtColor(roi, dst, COLOR_BGR2GRAY);
			resize(dst, test_sample, Size(92, 112));
			int label = 0;
			label = model_test->predict(test_sample);
			//输出检测到的人脸标签
			cout << label << endl;
			//画出人脸
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
			switch (label)
			{
				case 2:
					name = "tgy";
					break;
				default:
					name = "Unknown";
					break;
			}
			putText(frame, name, faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);
		}

		imshow("faceRecognition", frame);
		char c = waitKey(10);
		if (c == 27)
		{
			break;
		}
	}
}

int main(void) {
	testFace(0,"E:\\vsimg\\TGY.xml");
	return 0;
}