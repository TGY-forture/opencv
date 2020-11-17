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

//ͨ������ͷ���������ռ�����
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

		face_detector.detectMultiScale(gray, faces, 1.2, 3, 0, Size(30, 30));//�������

		for (size_t t = 0; t < faces.size(); t++)
		{
			Rect roi;
			roi.x = faces[static_cast<int>(t)].x;
			roi.y = faces[static_cast<int>(t)].y;
			roi.width = faces[static_cast<int>(t)].width;
			roi.height = faces[static_cast<int>(t)].height / 2;
			Mat faceROI = frame(roi);
			//����⵽ֻ��һ����ʱ������۾�
			if (faces.size() == 1)
			{
				eye_detector.detectMultiScale(faceROI, eyes, 1.2, 3, 0, Size(20, 20));//����۾�
				for (size_t k = 0; k < eyes.size(); k++)
				{
					Rect rect;
					rect.x = faces[static_cast<int>(t)].x + eyes[k].x;
					rect.y = faces[static_cast<int>(t)].y + eyes[k].y;
					rect.width = eyes[k].width;
					rect.height = eyes[k].height;
					rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);

					//ֻ�ж���⵽��ֻ�۾�ʱ��ʼ����
					if (eyes.size() == 2)
					{
						char key = waitKey(100);
						switch (key)
						{
							//����p��ʱ��ʼ����
							case 'p':
							{
								Mat face_roi = gray(faces[0]);
								//ͼ����ż�1
								image_number++;
								//����⵽��������С��Ϊ92*112���˾����һ��
								resize(face_roi, use_face, Size(92, 112));

								string filename = save_path + format("%d.jpg", image_number);
								//��ŵ������Ŀ¼
								imwrite(filename, use_face);
								imshow(filename, use_face);
								waitKey(1000);
								//����ָ���Ĵ���
								destroyWindow(filename);
								break;
							}
						}

						//��esc���˳�����
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

//����txt�嵥�ļ�
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
	//�ݹ����rescursive ֱ�Ӷ���������������iΪ������㣨�в�������end_iter�����յ�
	for (; begin_iter != end_iter; ++begin_iter)
	{
		i++;
		if (fs::is_directory(begin_iter->path()))  //�жϵ�ǰ�ļ����ǲ���һ��Ŀ¼��һ���������ݼ���Ϊһ��Ŀ¼
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

//ѵ����������ģ��
void trainFacesTxt(string faces_list, string save_model)
{
	//�������б��ļ�
	ifstream face_file(faces_list, ifstream::in);
	if (!face_file)
	{
		std::cout << "�޷���ѵ�������б��ļ���" << endl;
		return;
	}
	string line, path, class_label;
	vector<Mat> faces;
	vector<int> labels;
	char separator = ';';

	while (getline(face_file, line))
	{
		stringstream liness(line);
		//��һ��
		getline(liness, path, separator);
		getline(liness, class_label);
		if (!path.empty() && !class_label.empty())
		{
			//��ͼ��ѹ������
			faces.push_back(imread(path, 0));
			//�ѱ�ǩѹ������
			labels.push_back(atoi(class_label.c_str()));
		}
	}
	//�ж��Ƿ�Ϊ��
	if (faces.size() < 1 || labels.size() < 1)
	{
		std::cout << "��ʼ��ѵ����....." << std::endl;
		return;
	}
	int height = faces[0].rows;
	int width = faces[0].cols;
	std::cout << "ѵ������ͼ��ĸߣ�" << height << "ѵ������ͼ��Ŀ�" << width << std::endl;

	Mat test_sample = faces[faces.size() - 1];
	int test_label = labels[labels.size() - 1];
	faces.pop_back();
	labels.pop_back();

	//�ж�ͼ������
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (faces.at(i).type() != CV_8UC1)
		{
			std::cerr << "ͼ������ͱ���ΪCV_8UC1!" << endl;
			return;
		}
	}

	//���ߴ�����������ߴ��һ�ŵĳߴ�
	Size positive_image_size = faces[0].size();
	cout << "�������ĳߴ���:" << positive_image_size << endl;
	//����������Ʒ�����ߴ��Ƿ���ͬ
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (positive_image_size != faces[i].size())
		{
			std::cerr << "���е������ĳߴ��С��һ�������µ�����������С��" << endl;
			return;
		}
	}

	//����һ������ʶ�����
	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	//��ʼѵ��
	model->train(faces, labels);
	// recognition face
	int predicted_label = model->predict(test_sample);
	std::cout << "������ǩ����Ϊ��" << test_label << "Ԥ���������ǩΪ��" << predicted_label << endl;
	//����ѵ���õ�ģ��
	model->write(save_model);
	std::cout << "ѵ����ɣ�" << endl;
}

//������ͷʶ������
void testFace(int cap_index, string model_path)
{
	//����һ������ʶ����
	//Ptr<BasicFaceRecognizer>model = FisherFaceRecognizer::create();
	//Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
	Ptr<BasicFaceRecognizer> model_test = EigenFaceRecognizer::create();

	//opencv3.3Ҫ��read��Ҫ��Ȼ�����
	model_test->read(model_path);
	//����һ��������������
	CascadeClassifier faceDetector;
	faceDetector.load(face_path);

	//��⴫�������ͷ
	VideoCapture capture(cap_index);
	if (!capture.isOpened())
	{
		std::cerr << "�޷��򿪵�ǰ����ͷ��" << endl;
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
		//����
		flip(frame, frame, 1);
		//�������
		faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(80, 100), Size(380, 400));

		for (int i = 0; i < faces.size(); i++)
		{
			Mat roi = frame(faces[i]);
			cvtColor(roi, dst, COLOR_BGR2GRAY);
			resize(dst, test_sample, Size(92, 112));
			int label = 0;
			label = model_test->predict(test_sample);
			//�����⵽��������ǩ
			cout << label << endl;
			//��������
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