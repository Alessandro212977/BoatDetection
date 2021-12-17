/**
* Utilities for training the model
* @file MLUtils.cpp
* @author Alessandro Canevaro
* @version 23/07/21
*/

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <../include/MLUtils.h>

using namespace std;


FeatureImage::FeatureImage(cv::Mat image) {
	img = image;
	cv::Mat mask;
	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
	siftPtr->detectAndCompute(img, mask, kp_vec, desc_mat);
}

cv::Mat FeatureImage::getImage() {
	return img;
}

cv::Mat FeatureImage::getDescriptors() {
	return desc_mat;
}

vector<cv::KeyPoint> FeatureImage::getKeyPoints() {
	return kp_vec;
}

vector<cv::Point> FeatureImage::getKeyPointsPos() {
	vector<cv::Point>  keypoints_pos;
	for (const auto& kp : kp_vec) {
		keypoints_pos.push_back(cv::Point(kp.pt.x, kp.pt.y));
	}
	return keypoints_pos;
}


FilesToFeatures::FilesToFeatures(vector<cv::String> img_filenames, vector<cv::String> txt_filenames) {
	img_files = img_filenames;
	txt_files = txt_filenames;
}

FilesToFeatures::FilesToFeatures(cv::String filename) {
	cv::Mat_<float> data = CSVToMat(filename);
	descriptors = data.colRange(0, data.size().width - 1); //all except last
	labels = data.col(data.size().width - 1); //last columns
}

void FilesToFeatures::matchFilenames() {
	int j = 0; int i = 0;
	vector<cv::String> matched_img_filenames, matched_txt_filenames;
	while (true) {
		if (matched_txt_filenames.size() == txt_files.size()) { break; } //assuming to have more images then txt
		if (stoi(img_files[i].substr(37, 4)) == stoi(txt_files[j].substr(41, 4))) { //to be changed so that it start from the end
			matched_img_filenames.push_back(img_files[i]);
			matched_txt_filenames.push_back(txt_files[j]);
			i++; j++;
		}
		else {
			if (stoi(img_files[i].substr(37, 4)) > stoi(txt_files[j].substr(41, 4))) {j++;}
			else {i++;}
		}
	}
	img_files = matched_img_filenames;
	txt_files = matched_txt_filenames;
}

void FilesToFeatures::shuffleFilenames() {
	vector<int> seeds;
	for (int i = 0; i < img_files.size(); i++) {
		seeds.push_back(i);
	}
	cv::randShuffle(seeds);

	vector<cv::String> shuffled_img, shuffled_txt;
	for (int i = 0; i < img_files.size(); i++) {
		shuffled_img.push_back(img_files[seeds[i]]);
		shuffled_txt.push_back(txt_files[seeds[i]]);
	}
	img_files = shuffled_img;
	txt_files = shuffled_txt;
}

void FilesToFeatures::reduceFilenames(int n) {
	vector<cv::String> img_subset, txt_subset;
	for (int i = 0; i < n; i++) { //n shoud be smaller then num img
		img_subset.push_back(img_files[i]);
		txt_subset.push_back(txt_files[i]);
	}
	img_files = img_subset;
	txt_files = txt_subset;
}

void FilesToFeatures::generateDataset() {
	cv::Mat desc_tmp, labels_tmp;
	//cout << "####################" << endl;
	for (int i = 0; i < img_files.size(); i++) {
		cv::Mat img = cv::imread(img_files[i]);
		if (img.empty()) {
			cout << "Could not read the image: " << img_files[i] << endl;
			return;
		}
		FeatureImage f_img = FeatureImage(img);
		desc_tmp.push_back(f_img.getDescriptors());
		labels_tmp.push_back(generateTrueLabels(txt_files[i], f_img.getKeyPointsPos()));

		//if (i % (img_files.size() / 20) == 0) { cout << "#"; }
	}
	cout << endl;
	descriptors = desc_tmp.clone();
	labels = labels_tmp.clone();
}

void FilesToFeatures::saveDataset(cv::String filename) {
	cv::Mat matrix;
	cv::hconcat(descriptors, labels, matrix);
	MatToCSV(matrix, filename);
}

cv::Mat FilesToFeatures::getDescriptors() {
	return descriptors;
}

cv::Mat FilesToFeatures::getLabels() {
	return labels;
}

cv::Mat FilesToFeatures::generateTrueLabels(cv::String txt_filename, vector<cv::Point> kp_pos) {
	cv::Mat_<float> labels_mat = cv::Mat_<float>::zeros(cv::Size(1, kp_pos.size())) - 1.0;
	string line;
	ifstream file(txt_filename);
	if (file.is_open()) {
		while (getline(file, line)) {
			size_t pos = 0;
			vector<float> bb; //xmin, xmax, ymin, ymax
			line.erase(0, line.find(":") + 1);
			while ((pos = line.find(";")) != string::npos) {
				bb.push_back(stod(line.substr(0, pos)));
				line.erase(0, pos + 1);
			}
			for (int i = 0; i < kp_pos.size(); i++) {
				bool x_cond = kp_pos[i].x >= bb[0] && kp_pos[i].x <= bb[1]; //xmin <= x <= xmax
				bool y_cond = kp_pos[i].y >= bb[2] && kp_pos[i].y <= bb[3]; //ymin <= y <= ymax
				if (x_cond && y_cond) {labels_mat.row(i) = 1.0;}
			}
		}
		file.close();
	}
	else { cout << "Unable to open file: " << txt_filename << endl; };
	return labels_mat.clone();
}


Preprocessing::Preprocessing(cv::Mat descriptors, cv::Mat labels) {
	desc_mat = descriptors;
	lab_mat = labels;
}

Preprocessing::Preprocessing(cv::Mat descriptors) {
	desc_mat = descriptors;
}

void Preprocessing::equalize() {
	float positive_labels = cv::countNonZero(lab_mat + 1); //count num of 1
	float negative_labels = cv::countNonZero(lab_mat - 1); //count num of -1

	float boats_labels_prob = (float)min(positive_labels, negative_labels) / positive_labels;
	float other_labels_prob = (float)min(positive_labels, negative_labels) / negative_labels;

	cv::Mat new_desc, new_labels;
	for (int i = 0; i < lab_mat.size().height; i++) {
		if (lab_mat.at<float>(i) > 0) {
			if ((float)rand() / RAND_MAX <= boats_labels_prob) {
				new_desc.push_back(desc_mat.row(i));
				new_labels.push_back(lab_mat.row(i));
			}
		}
		else {
			if ((float)rand() / RAND_MAX <= other_labels_prob) {
				new_desc.push_back(desc_mat.row(i));
				new_labels.push_back(lab_mat.row(i));
			}
		}
	}
	desc_mat = new_desc.clone();
	lab_mat = new_labels.clone();
}

void Preprocessing::reduce(float percentage) {
	cv::Mat new_desc, new_labels;
	for (int i = 0; i < lab_mat.size().height; i++) {
		if ((float)rand() / RAND_MAX <= percentage) {
			new_desc.push_back(desc_mat.row(i));
			new_labels.push_back(lab_mat.row(i));
		}
	}
	desc_mat = new_desc.clone();
	lab_mat = new_labels.clone();
}

void Preprocessing::shuffle() {
	vector<int> seeds;
	for (int i = 0; i < lab_mat.size().height; i++) {
		seeds.push_back(i);
	}
	cv::randShuffle(seeds);

	cv::Mat new_desc, new_labels;
	for (int i = 0; i < lab_mat.size().height; i++) {
		new_desc.push_back(desc_mat.row(seeds[i]));
		new_labels.push_back(lab_mat.row(seeds[i]));
	}
	desc_mat = new_desc.clone();
	lab_mat = new_labels.clone();
}

void Preprocessing::initNormParams() {
	cv::Scalar mean_i, stddev_i;
	for (int i = 0; i < desc_mat.size().width; i++) {
		cv::meanStdDev(desc_mat.col(i), mean_i, stddev_i);
		mean_mat.push_back((float)mean_i[0]);
		stddev_mat.push_back((float)stddev_i[0]);
	}
}

void Preprocessing::normParamsToCSV(cv::String filename) {
	cv::Mat matrix;
	cv::hconcat(mean_mat, stddev_mat, matrix);
	MatToCSV(matrix, filename);
}

void Preprocessing::normParamsFromCSV(cv::String filename) {
	cv::Mat matrix = CSVToMat(filename);
	mean_mat = matrix.col(0);
	stddev_mat = matrix.col(1);
}

void Preprocessing::normalize() {
	for (int i = 0; i < desc_mat.size().width; i++) {
		desc_mat.col(i) = desc_mat.col(i) - mean_mat.at<float>(i);
		desc_mat.col(i) = desc_mat.col(i) / stddev_mat.at<float>(i);
	}
}

void Preprocessing::initPCA(int num_dim) {
	cv::PCA pca(desc_mat, cv::Mat(), cv::PCA::DATA_AS_ROW, num_dim);
	eigenvectors = pca.eigenvectors.clone();
}

void Preprocessing::eigenvectorsToCSV(cv::String filename) {
	MatToCSV(eigenvectors, filename);
}

void Preprocessing::eigenvectorsFromCSV(cv::String filename) {
	eigenvectors = CSVToMat(filename);
}

void Preprocessing::PCA() {
	desc_mat = desc_mat * eigenvectors.t();
}

void Preprocessing::saveDataset(cv::String filename) {
	cv::Mat matrix;
	cv::hconcat(desc_mat, lab_mat, matrix);
	MatToCSV(matrix, filename);
}

cv::Mat Preprocessing::getDescriptors() {
	return desc_mat;
}

cv::Mat Preprocessing::getLabels() {
	return lab_mat;
}


void MatToCSV(cv::Mat matrix, cv::String filename, cv::String delimiter) {
	ofstream file;
	file.open(filename, ios::trunc);
	if (file.is_open()) {
		for (int i = 0; i < matrix.size().height; i++) {
			file << matrix.at<float>(i, 0); //write first element of the row
			for (int j = 1; j < matrix.size().width; j++) {
				file << delimiter << matrix.at<float>(i, j);
			}
			file << endl;
		}
		file.close();
	}
	else { cout << "Unable to open file: " << filename << endl; };
}

cv::Mat CSVToMat(cv::String filename, cv::String delimiter) {
	cv::Mat matrix;
	string line;
	ifstream file(filename);
	if (file.is_open()) {
		while (getline(file, line)) {
			size_t pos = 0;
			cv::Mat_<float> row;
			while ((pos = line.find(delimiter)) != string::npos) {
				row.push_back(stof(line.substr(0, pos)));
				line.erase(0, pos + 1);
			}
			row.push_back(stof(line)); //last element of the line
			matrix.push_back(row.t());
		}
		file.close();
	}
	else { cout << "Unable to open file: " << filename << endl; };
	return matrix;
}