/**
* Main file for training the classifier
* @file training.cpp
* @author Alessandro Canevaro
* @version 23/07/21
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <../include/MLUtils.h>

//Training parameters
#define NUM_IMG 3000 //total number of images for training
#define BATCH_SIZE 250 //Number batches
#define PCA_DIM 64 //dimension of the reduced dataset (max 128)
#define ITERATIONS 1000 //iterations per batch
#define PRECISION 0.0001 //termination condition
//Directories and filenames
#define TRAIN_IMG_DIR "../data/TRAINING_DATASET/IMAGES/"
#define TRAIN_TXT_DIR "../data/TRAINING_DATASET/LABELS_TXT/"
#define TRAIN_IMG_NAME "image*.png"
#define TRAIN_TXT_NAME "image*.txt"
#define EIGENVECTORS_FILENAME "../data/training_files/dataset" + to_string(NUM_IMG) + "_eigenvectors.csv"
#define NORMALIZATION_FILENAME "../data/training_files/dataset" + to_string(NUM_IMG) + "_normalization.csv"
#define TRAINING_DATASET_FILENAME "../data/training_files/dataset" + to_string(NUM_IMG) + ".csv"
#define MODEL_FILENAME "../data/Model3000.xml"

using namespace std;

/** Read images, extract features, preprocess, save to files
* @param descriptors Input/Output descriptors matrix
* @param labels Input/Output labels vector
*/
void generate_features_dataset(cv::Mat &descriptors, cv::Mat &labels);

/** Creates the classifier
* @return Pointer to the classifier
*/
cv::Ptr<cv::ml::ANN_MLP> create_mlp();

/** Do the training process
* @param descriptors Input descriptors matrix
* @param labels Input labels vector
*/
void train(cv::Mat descriptors, cv::Mat labels);

/** Computes the error between predictions and ground thruth
* @param labels Input labels vector
* @param predictions Input predictions vector
*/
float getMetrics(cv::Mat labels, cv::Mat predictions);

int main(int argc, char* argv[]) {
	//generates the dataset
	cv::Mat_<float> descriptors, labels;
	generate_features_dataset(descriptors, labels); 

	//do the training
	train(descriptors, labels);

	cout << endl << "All done." << endl;
	cv::waitKey(0);
	return 0;
}

void generate_features_dataset(cv::Mat& descriptors, cv::Mat& labels) {
	//get all images and txts
	cout << "Reading input files" << endl;
	vector<cv::String> img_filenames, txt_filenames;
	cv::utils::fs::glob(TRAIN_IMG_DIR, TRAIN_IMG_NAME, img_filenames);
	cv::utils::fs::glob(TRAIN_TXT_DIR, TRAIN_TXT_NAME, txt_filenames);

	FilesToFeatures FtF = FilesToFeatures(img_filenames, txt_filenames);
	FtF.matchFilenames();
	FtF.shuffleFilenames();
	FtF.reduceFilenames(NUM_IMG);

	cout << "extracting features..." << endl;
	FtF.generateDataset();

	cout << "Preprocessing..." << endl;
	Preprocessing dataset = Preprocessing(FtF.getDescriptors(), FtF.getLabels());
	cout << "PCA..." << endl;
	dataset.initPCA(PCA_DIM);
	dataset.eigenvectorsToCSV(EIGENVECTORS_FILENAME);
	dataset.PCA();
	cout << "Shuffling" << endl;
	dataset.shuffle();
	cout << "Normalizing..." << endl;
	dataset.initNormParams();
	dataset.normParamsToCSV(NORMALIZATION_FILENAME);
	dataset.normalize();
	cout << "Equalizing..." << endl;
	dataset.equalize();
	cout << "Saving dataset..." << endl;
	dataset.saveDataset(TRAINING_DATASET_FILENAME);

	descriptors = dataset.getDescriptors();
	labels = dataset.getLabels();
	cout << "Done..." << endl << endl;
}

cv::Ptr<cv::ml::ANN_MLP> create_mlp() {
	//Set the architecture
	cv::Mat_<int> layers(3, 1);
	layers(0) = PCA_DIM;		// input
	layers(1) = 42;			    // hidden
	layers(2) = 1;				// output,

	auto mlp = cv::ml::ANN_MLP::create();
	mlp->setLayerSizes(layers);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, ITERATIONS, PRECISION));
	mlp->setTrainMethod(cv::ml::ANN_MLP::RPROP);
	return mlp;
}

void train(cv::Mat descriptors, cv::Mat labels) {
	auto train_dataset = cv::ml::TrainData::create(descriptors, cv::ml::ROW_SAMPLE, labels);

	int n_samples = train_dataset->getNSamples();
	if (n_samples == 0) {
		cout << "No samples found. Exiting..." << endl;
		exit(-1);
	}
	else { cout << "Loaded " << n_samples << " samples. "; }

	train_dataset->setTrainTestSplitRatio(0.9, false);
	cout << train_dataset->getNTrainSamples() << " train samples, and ";
	cout << train_dataset->getNTestSamples() << " test samples" << endl;

	int batch_samples = train_dataset->getNTrainSamples() / BATCH_SIZE;

	auto mlp = cv::ml::ANN_MLP::load(MODEL_FILENAME);// create_mlp();

	int training_flags = cv::ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE + 
						 cv::ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE + cv::ml::ANN_MLP::TrainFlags::UPDATE_WEIGHTS;

	for (int i = 0; i < BATCH_SIZE; i++) {
		cout << "Iteration: " << i + 1 << "/" << BATCH_SIZE << endl;
		cv::Mat_<float> batch_descriptors = train_dataset->getTrainSamples().rowRange(i * batch_samples, (i + 1) * batch_samples).clone();
		cv::Mat_<float> batch_labels = train_dataset->getTrainResponses().rowRange(i * batch_samples, (i + 1) * batch_samples).clone();
		auto batch_train_dataset = cv::ml::TrainData::create(batch_descriptors, cv::ml::ROW_SAMPLE, batch_labels);

		int n_samples = batch_train_dataset->getNSamples();
		if (n_samples == 0) {
			cout << "No samples found. Exiting..." << endl;
			exit(-1);
		}
		cout << "Batch size: " << n_samples << " samples. " << endl;

		auto t1 = chrono::high_resolution_clock::now();
		mlp->train(batch_train_dataset, training_flags);
		auto t2 = chrono::high_resolution_clock::now();

		training_flags = cv::ml::ANN_MLP::TrainFlags::UPDATE_WEIGHTS + 
						 cv::ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE + 
						 cv::ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE;

		cv::Mat test_predictions;
		mlp->predict(train_dataset->getTestSamples(), test_predictions);
		float test_error = getMetrics(train_dataset->getTestResponses(), test_predictions);
		cout << "Test set error: " << test_error * 100 << "%";

		cv::Mat train_predictions;
		mlp->predict(train_dataset->getTrainSamples(), train_predictions);
		float train_error = getMetrics(train_dataset->getTrainResponses(), train_predictions);
		cout << "; Train set error: " << train_error * 100 << "%" << endl;

		chrono::duration<double, milli> ms = t2 - t1;
		cout << "Update complete in: " << ms.count() << " ms" << endl << endl;
	}

	mlp->save(cv::String(MODEL_FILENAME));
	cout << "Model saved." << endl;
}

float getMetrics(cv::Mat labels, cv::Mat predictions) {
	float eps = 0.1;
	float error = 0;
	cv::Mat quantized_pred;
	for (int i = 0; i < labels.size().height; i++) {
		if (predictions.at<float>(i) >= 0) { //0
			quantized_pred.push_back(1);
		}
		else { quantized_pred.push_back(-1); } //- 1
	}
	for (int i = 0; i < labels.size().height; i++) {
		if (fabs(labels.at<float>(i) - quantized_pred.at<int>(i)) > eps) {
			error++;
		}
	}
	return error / labels.size().height;
}