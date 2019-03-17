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
//	//读取图像
//	Mat img_1 = imread("1.png", CV_LOAD_IMAGE_COLOR);
//	Mat img_2 = imread("2.png", CV_LOAD_IMAGE_COLOR);
//
//	//初始化
//	vector<KeyPoint> keypoints_1, keypoints_2;//关键点,指特征点在图像里的位置
//	Mat descriptors_1, descriptors_2;//描述子,通常是向量
//	Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
//
//	//第一步：检测OrientFAST角点位置
//	orb->detect(img_1, keypoints_1);
//	orb->detect(img_2, keypoints_2);
//
//	//第2步：根据角点位置计算BRIEF描述子
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
//	//第3步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
//	vector<DMatch> matches;
//	//特征匹配的方法：暴力匹配
//	BFMatcher matcher(NORM_HAMMING);
//	matcher.match(descriptors_1, descriptors_2, matches);
//	//    for(auto it=matches.begin();it!=matches.end();++it)
//	//    {
//	//        cout<<*it<<" ";
//	//    }
//	//    cout<<endl;
//
//		//第4步：匹配点对筛选distance是min_dist
//
//	double min_dist = 10000, max_dist = 0;
//
//	//找出所有匹配之间的最小距离和最大距离，即最相似的和最不相似的和最不相似的两组点之间的距离
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
//	//当描述子之间的距离大于两倍的最小距离时，即认为匹配有误
//	//但有时候最小距离会非常小，设置一个经验值作为下限
//	vector<DMatch> good_matches;
//	for (int i = 0; i < descriptors_1.rows; ++i)
//	{
//		if (matches[i].distance <= max(2 * min_dist, 30.0))
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	//第5步：绘制匹配结果
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

// 像素坐标转相机归一化坐标
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
	//-- 读取图像
	Mat img_1 = imread("1.png", CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread("2.png", CV_LOAD_IMAGE_COLOR);

	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "一共找到了" << matches.size() << "组匹配点" << endl;

	// 建立3D点
	Mat depth1 = imread("1.pgm", CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
	Mat depth2 = imread("2.pgm", CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
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

	cout << "匹配点: " << pts1.size() << endl;
	Mat R, t;
	pose_estimation_3d3d(pts1, pts2, R, t);
	cout << "ICP SVD结果:\n " << endl;
	cout << "旋转矩阵：\n" << endl;
	cout << R << endl;
	cout << "平移矩阵：\n" << endl;
	cout << t << endl;
	cout << "\n平移向量+旋转四元数：\n" << endl;
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
	//-- 初始化
	Mat descriptors_1, descriptors_2;
	// used in OpenCV3
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	// use this if you are in OpenCV2
	// Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	//-- 第一步:检测 Oriented FAST 角点位置
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> match;
	// BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, match);

	//-- 第四步:匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
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