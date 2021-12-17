/**
* Main file for boats detection
* @file detector.cpp
* @author Alessandro Canevaro
* @version 23/07/21
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <../include/MLUtils.h>
#include <../include/DetectionUtils.h>

//Files directories
//#define IMAGE_FILE "../data/09.png"
//#define BB_FILE "../data/TEST_DATASET/venice_labels_txt/09.txt"

#define EIGENVECTORS_FILE "../data/dataset_eigenvectors.csv"
#define NORMALIZATION_FILE "../data/dataset_normalization.csv"
#define MODEL_FILE "../data/MLP64.xml"

using namespace std;

/** Read and draw the true bounding boxes
* @param image Input/Output image
* @param txt_dir Input filename containing the coordinates of the boxes
* @return Array of bounding boxes coordinates for further processing
*/
vector<vector<float>> get_real_bb(cv::Mat& image, cv::String txt_dir);

int main(int argc, char* argv[]) {

	//main parameters
	double r = 50; //50
	int win_size = 100; //100
	float prune_thr = 100; //100
	float sup_thr1 = 0.2; //0.2
	float sup_thr2 = 0.5; //0.5
	float bb_thr1 = 0.05; //0.05
	float bb_thr2 = 0.2; //0.2
	
	cv::String image_dir;
	if(argc == 2){
   	   	image_dir = argv[1];
	}

	cv::Mat img = cv::imread(image_dir);
	if (img.empty()) {
		cout << "Could not read the image: " << image_dir << endl;
	}
	//cv::imshow("original", img);

	cout << "Detecting..." << endl;
	
	FeatureImage fimg = FeatureImage(img);

	Preprocessing data = Preprocessing(fimg.getDescriptors());
	data.eigenvectorsFromCSV(EIGENVECTORS_FILE);
	data.PCA();
	data.normParamsFromCSV(NORMALIZATION_FILE);
	data.normalize();
	
	cv::Mat predictions;
	auto mlp = cv::ml::ANN_MLP::load(cv::String(MODEL_FILE));
	mlp->predict(data.getDescriptors(), predictions);

	//cv::Mat sift_img = img.clone();
	//for (int i = 0; i < fimg.getKeyPointsPos().size(); i++) {
	//	int color = 255 / (1 + 1) * (predictions.at<float>(i) + 1);
	//	vector<cv::KeyPoint> kp;
	//	kp.push_back(fimg.getKeyPoints()[i]);
	//	cv::drawKeypoints(sift_img, kp, sift_img, cv::Scalar(0, color, 255 - color), cv::DrawMatchesFlags::DEFAULT);
	//}
	//cv::imshow("Feature classification", sift_img);

	Detector detector = Detector(predictions, fimg.getKeyPointsPos(), img.size());

	detector.computeProbMap(r);

	cv::Mat segmentation = img.clone();
	cv::Mat basins;

	detector.computeMeanShiftClustering(segmentation, basins, win_size, prune_thr, sup_thr1, sup_thr2);

	//cv::imshow("Basins", basins);
	//cv::imshow("Segmentation", segmentation);

	detector.computeBoundingBox(bb_thr1, bb_thr2);

	cv::Mat bb = img.clone();
	detector.drawBoundingBoxes(bb);

	//vector<vector<float>> true_bb_list = get_real_bb(bb, BB_FILE);
	//detector.computeIoU(true_bb_list, bb);

	cv::imshow("Bounding boxes", bb);

	cout << "All done." << endl;
	cv::waitKey(0);
	return 0;
}

vector<vector<float>> get_real_bb(cv::Mat &image, cv::String txt_dir) {
	vector<vector<float>> true_bb_list;
	string line;
	ifstream file(txt_dir);
	if (file.is_open()) {
		while (getline(file, line)) {
			size_t pos = 0;
			vector<float> bb; //xmin, xmax, ymin, ymax
			line.erase(0, line.find(":") + 1);
			while ((pos = line.find(";")) != string::npos) {
				bb.push_back(stod(line.substr(0, pos)));
				line.erase(0, pos + 1);
			}
			cv::rectangle(image, cv::Point(bb[0], bb[2]), cv::Point(bb[1], bb[3]), cv::Scalar(0, 255, 0), 2, 8, 0);
			vector<float> sorted = { bb[0], bb[2], bb[1], bb[3] };
			true_bb_list.push_back(sorted);
		}
		file.close();
	}
	else { cout << "Unable to open file: " << txt_dir << endl; };
	return true_bb_list;
}
