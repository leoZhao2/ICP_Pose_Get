//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main(int argc, char** argv)
//{
//
//
//	//��ȡͼ��
//	Mat img_1 = imread("1.png", CV_LOAD_IMAGE_COLOR);
//	Mat img_2 = imread("2.png", CV_LOAD_IMAGE_COLOR);
//
//	//��ʼ��
//	vector<KeyPoint> keypoints_1, keypoints_2;//�ؼ���,ָ��������ͼ�����λ��
//	Mat descriptors_1, descriptors_2;//������,ͨ��������
//	Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
//
//	//��һ�������OrientFAST�ǵ�λ��
//	orb->detect(img_1, keypoints_1);
//	orb->detect(img_2, keypoints_2);
//
//	//��2�������ݽǵ�λ�ü���BRIEF������
//	orb->compute(img_1, keypoints_1, descriptors_1);
//	orb->compute(img_2, keypoints_2, descriptors_2);
//
//	Mat outimg1;
//	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	imwrite("1.png", outimg1);
//	Mat outimg2;
//	drawKeypoints(img_2, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	imwrite("2.png", outimg2);
//
//	//��3����������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ��Hamming����
//	vector<DMatch> matches;
//	//����ƥ��ķ���������ƥ��
//	BFMatcher matcher(NORM_HAMMING);
//	matcher.match(descriptors_1, descriptors_2, matches);
//	//    for(auto it=matches.begin();it!=matches.end();++it)
//	//    {
//	//        cout<<*it<<" ";
//	//    }
//	//    cout<<endl;
//
//		//��4����ƥ����ɸѡdistance��min_dist
//
//	double min_dist = 10000, max_dist = 0;
//
//	//�ҳ�����ƥ��֮�����С����������룬�������Ƶĺ�����Ƶĺ�����Ƶ������֮��ľ���
//	for (int i = 0; i < descriptors_1.rows; ++i)
//	{
//		double dist = matches[i].distance;
//		//        cout<<dist<<endl;
//		if (dist < min_dist) min_dist = dist;
//		if (dist > max_dist) max_dist = dist;
//	}
//
//	printf("--Max dist:%f\n", max_dist);
//	printf("--Min dist:%f\n", min_dist);
//
//	//��������֮��ľ��������������С����ʱ������Ϊƥ������
//	//����ʱ����С�����ǳ�С������һ������ֵ��Ϊ����
//	vector<DMatch> good_matches;
//	for (int i = 0; i < descriptors_1.rows; ++i)
//	{
//		if (matches[i].distance <= max(2 * min_dist, 30.0))
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	//��5��������ƥ����
//	Mat img_match;
//	Mat img_goodmatch;
//	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
//	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
//	imwrite("match.png", img_match);
//	imwrite("goodmatch.png", img_goodmatch);
//	waitKey(0);
//
//	return 0;
//}
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp> //cv2eigen
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
	const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1,
	std::vector<KeyPoint>& keypoints_2,
	std::vector< DMatch >& matches);

// ��������ת�����һ������
Point2d pixel2cam(const Point2d& p, const Mat& K);

void pose_estimation_3d3d(
	const vector<Point3f>& pts1,
	const vector<Point3f>& pts2,
	Mat& R, Mat& t
);

//matrx 2 qual
Eigen::Quaterniond rotationMatrix2Quaterniond(Eigen::Matrix3d R)
{
	Eigen::Quaterniond q = Eigen::Quaterniond(R);
	q.normalize();
	cout << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
	return q;
}

int main(int argc, char** argv)
{
	//-- ��ȡͼ��
	Mat img_1 = imread("1.png", CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread("2.png", CV_LOAD_IMAGE_COLOR);

	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "һ���ҵ���" << matches.size() << "��ƥ���" << endl;

	// ����3D��
	Mat depth1 = imread("1.pgm", CV_LOAD_IMAGE_UNCHANGED);       // ���ͼΪ16λ�޷���������ͨ��ͼ��
	Mat depth2 = imread("2.pgm", CV_LOAD_IMAGE_UNCHANGED);       // ���ͼΪ16λ�޷���������ͨ��ͼ��
	Mat K = (Mat_<double>(3, 3) << 682.421509, 0, 633.947449, 0, 682.421509, 404.559906, 0, 0, 1);
	vector<Point3f> pts1, pts2;

	for (DMatch m : matches)
	{
		ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
		ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
		if (d1 == 0 || d2 == 0)   // bad depth
			continue;
		Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
		Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
		float dd1 = float(d1) / 5000.0;
		float dd2 = float(d2) / 5000.0;
		pts1.push_back(Point3f(p1.x*dd1, p1.y*dd1, dd1));
		pts2.push_back(Point3f(p2.x*dd2, p2.y*dd2, dd2));
	}

	cout << "ƥ���: " << pts1.size() << endl;
	Mat R, t;
	pose_estimation_3d3d(pts1, pts2, R, t);
	cout << "ICP SVD���:\n " << endl;
	cout << "��ת����\n" << endl;
	cout << R << endl;
	cout << "ƽ�ƾ���\n" << endl;
	cout << t << endl;
	cout << "\nƽ������+��ת��Ԫ����\n" << endl;
	Eigen::Matrix3d m;
	cv2eigen(R, m);
	cout << t.at<double>(0, 0) << " " << t.at<double>(0, 1) << " " << t.at<double>(0, 2) << endl;
	rotationMatrix2Quaterniond(m);
	
	cin.get();
}

void find_feature_matches(const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1,
	std::vector<KeyPoint>& keypoints_2,
	std::vector< DMatch >& matches)
{
	//-- ��ʼ��
	Mat descriptors_1, descriptors_2;
	// used in OpenCV3
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	// use this if you are in OpenCV2
	// Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	//-- ��һ��:��� Oriented FAST �ǵ�λ��
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	//-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
	vector<DMatch> match;
	// BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, match);

	//-- ���Ĳ�:ƥ����ɸѡ
	double min_dist = 10000, max_dist = 0;

	//�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (match[i].distance <= max(2 * min_dist, 30.0))
		{
			matches.push_back(match[i]);
		}
	}
}

Point2d pixel2cam(const Point2d& p, const Mat& K)
{
	return Point2d
	(
		(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
	);
}

void pose_estimation_3d3d(
	const vector<Point3f>& pts1,
	const vector<Point3f>& pts2,
	Mat& R, Mat& t
)
{
	Point3f p1, p2;     // center of mass
	int N = pts1.size();
	for (int i = 0; i < N; i++)
	{
		p1 += pts1[i];
		p2 += pts2[i];
	}
	p1 = Point3f(Vec3f(p1) / N);
	p2 = Point3f(Vec3f(p2) / N);
	vector<Point3f>     q1(N), q2(N); // remove the center
	for (int i = 0; i < N; i++)
	{
		q1[i] = pts1[i] - p1;
		q2[i] = pts2[i] - p2;
	}

	// compute q1*q2^T
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i < N; i++)
	{
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
	}
	cout << "W=" << W << endl;

	// SVD on W
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();

	if (U.determinant() * V.determinant() < 0)
	{
		for (int x = 0; x < 3; ++x)
		{
			U(x, 2) *= -1;
		}
	}

	cout << "U=" << U << endl;
	cout << "V=" << V << endl;

	Eigen::Matrix3d R_ = U * (V.transpose());
	Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

	// convert to cv::Mat
	R = (Mat_<double>(3, 3) <<
		R_(0, 0), R_(0, 1), R_(0, 2),
		R_(1, 0), R_(1, 1), R_(1, 2),
		R_(2, 0), R_(2, 1), R_(2, 2)
		);
	t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}