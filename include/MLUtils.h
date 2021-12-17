/**
* Utilities for training the model
* @file MLUtils.h
* @author Alessandro Canevaro
* @version 23/07/21
*/

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

/**
* Class for handling an image and its features
*/
class FeatureImage {

public:

	/** Default constructor (extract features keypoints and descriptors)
	* @param image Input image
	*/
	FeatureImage(cv::Mat image);

	/** Method for getting the input image
	* @return Return the input image
	*/
	cv::Mat getImage();

	/** Method for getting the features descriptors [nx128] mat
	* @return Return features descriptors
	*/
	cv::Mat getDescriptors();

	/** Method for getting the features keypoints
	* @return Return features keypoints
	*/
	vector<cv::KeyPoint> getKeyPoints();

	/** Method for getting the features keypoints position (list of points)
	* @return Return features keypoints position
	*/
	vector<cv::Point> getKeyPointsPos();

protected:

	cv::Mat img, desc_mat;
	vector<cv::KeyPoint> kp_vec;
};

/**
* Class for generating a features dataset starting from images 
* and txt files containing the ground thruth
*/
class FilesToFeatures {

public:

	/** Default constructor
	* @param img_filenames Vector of image filenames
	* @param txt_filenames Vector of txt files for deriving the true labels
	*/
	FilesToFeatures(vector<cv::String> img_filenames, vector<cv::String> txt_filenames);

	/** Secondary constructor for reading a dataset from file
	* @param filename Filename of the dataset
	*/
	FilesToFeatures(cv::String filename);

	/** Checks if an image has its corresponding txt files, 
	*   otherwise discards the image (and vice-versa)
	*/
	void matchFilenames();

	/** Shuffle the list of image filenames, 
	*   preserving the same order also for the txt filenames.
	*/
	void shuffleFilenames();

	/** Keeps the first n images/txt
	* @param n Number of elements to retain
	*/
	void reduceFilenames(int n);

	/** Extract features and true labels from the filenames lists
	*/
	void generateDataset();

	/** Save a nx129 matrix to the given filename.
	*   the first 128 columns are the descriptors 
	*   while the last column are the labels.
	* @param filename Filename where to save the dataset
	*/
	void saveDataset(cv::String filename);

	/** Get the descriptors matrix
	* @return Return a nx128 matrix containing the descriptors
	*/
	cv::Mat getDescriptors();

	/** Get the labels vector
	* @return Return a nx1 columns vector containing the labels
	*/
	cv::Mat getLabels();

protected:

	/** Classifies each keypoints as boat/non-boat based on the true bounding box
	* @param txt_filename Filename containing the positions of each bounding box
	* @param kp_pos list of keypoints positions to be classified
	* @return Return a nx1 columns vector containing the labels
	*/
	cv::Mat generateTrueLabels(cv::String txt_filename, vector<cv::Point> kp_pos);
	
	vector<cv::String> img_files, txt_files;
	cv::Mat descriptors, labels;
};

/**
* Class for process a descriptors dataset
*/
class Preprocessing {

public:

	/** Default constructor (used in the training process)
	* @param descriptors nx128 matrix of descriptors
	* @param labels nx1 vector of labels
	*/
	Preprocessing(cv::Mat descriptors, cv::Mat labels);

	/** Secondary constructor (used in the prediction process)
	* @param descriptors nx128 matrix of descriptors
	*/
	Preprocessing(cv::Mat descriptors);

	/** Discards some samples such that the number of samples with the 
	*   "boat" label are approximately equals to the ones with "non-boat" label
	*/
	void equalize();

	/** Randomly discards some samples such that the number of final samples 
	*   is equal to the number of initial samples reduced by the given  percentage
	* @param percentage Reduce dataset by the given percentage
	*/
	void reduce(float percentage);

	/** Shuffle the samples of the dataset (and the labels too)
	*/
	void shuffle();
	
	/** Computes the mean and standard deviation for each column of the descriptors matrix
	*/
	void initNormParams();

	/** Saves the computed mean and stddev into a file
	* @param filename Filename where to save the data
	*/
	void normParamsToCSV(cv::String filename);

	/** Reads the mean and stddev from a file
	* @param filename Filename from where read the data
	*/
	void normParamsFromCSV(cv::String filename);

	/** Normalize the descriptors matrix with the precomputed mean and stddev 
	* either from initNormParams or normParamsFromCSV
	*/
	void normalize();

	/** Computes the eigenvectors for reducing the number of dimensions
	* @param num_dim Number of dimensions to retain
	*/
	void initPCA(int num_dim);

	/** Saves the computed eigenvectors into a file
	* @param filename Filename where to save the data
	*/
	void eigenvectorsToCSV(cv::String filename);

	/** Reads the eigenvectors from a file
	* @param filename Filename from where read the data
	*/
	void eigenvectorsFromCSV(cv::String filename);

	/** Reduce the number of columns of the descriptors matrix according 
	* to the eigenvectos computed either from initPCA or eigenvectorsFromCSV
	*/
	void PCA();

	/** Saves the dataset (descriptors+labels) into a file
	* @param filename Filename where to save the data
	*/
	void saveDataset(cv::String filename);

	/** Get the descriptors matrix
	* @return Return the matrix containing the descriptors
	*/
	cv::Mat getDescriptors();

	/** Get the labels vector
	* @return Return a nx1 columns vector containing the labels
	*/
	cv::Mat getLabels();

protected:
	cv::Mat desc_mat, lab_mat;
	cv::Mat eigenvectors;
	cv::Mat mean_mat, stddev_mat;
};

/** Saves a cv::Mat into a file
* @param matrix Input matrix
* @param filename Filename where to save the data
* @param delimiter Delimiter between values
*/
void MatToCSV(cv::Mat matrix, cv::String filename, cv::String delimiter = cv::String(","));

/** Reads a cv::Mat from a file
* @param filename Filename where to read the data
* @param delimiter Delimiter between values
* @return Output matrix
*/
cv::Mat CSVToMat(cv::String filename, cv::String delimiter = cv::String(","));