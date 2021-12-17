/**
* Utilities for detecting the boat
* @file DetectionUtils.h
* @author Alessandro Canevaro
* @version 23/07/21
*/

#include <opencv2/opencv.hpp>

using namespace std;

/**
* Main class for detecting boats
* Generates the probaility map, cluster with mean shift
* and generate bounding box 
*/
class Detector {

public:

	/** Default constructor
	* @param predictions Column vector containing the prediction for each feature
	* @param keypoints_pos Vector of the position of the keypoints
	* @param img_size Size of the image
	*/
	Detector(cv::Mat predictions, vector<cv::Point> keypoints_pos, cv::Size image_size);

	/** Generates the 2D probability density function
	* @param radious Radious of the circle where features are considered
	* @param step Gap between evaluated points
	*/
	void computeProbMap(double radious = 50, int step = 4);

	/** Computes clusters using mean shift
	* @param image Image to be passed at MeanShift.drawClusters
	* @param windows_size Size of the circle where the CoM is computed
	* @param prune_thr Threshold for pruning modes
	* @param sup1_thr Threshold for suppressing modes before pruning
	* @param sup2_thr Threshold for suppressing modes after pruning
	*/
	void computeMeanShiftClustering(cv::Mat& image, cv::Mat& basins, int window_size = 100, float prune_thr = 100,
									float sup1_thr = 0.2, float sup2_thr = 0.5);

	/** Computes the bounding (1 for each cluster)
	* @param thr1 Passed to the first BoundingBox.optimize()
	* @param thr2 Passed to the second BoundingBox.optimize()
	* @param offset1 Passed to the first BoundingBox.optimize()
	* @param offset2 Passed to the second BoundingBox.optimize()
	*/
	void computeBoundingBox(float thr1 = 0.05, float thr2 = 0.2, 
							int offset1 = 10, int offset2 = 10);

	/** Draws the founded bounding boxes
	* @param image Input/Output image
	* @param color Color of the bounding boxes
	*/
	void drawBoundingBoxes(cv::Mat& image, cv::Scalar color = cv::Scalar(0, 0, 255));

	/** Draws the IoU metric on the bounding boxes
	* @param bb_corner_list True bounding boxes
	* @param image Input/Output image
	*/
	void computeIoU(vector<vector<float>> bb_corner_list, cv::Mat& image);
	
protected:

	/** Computes the IoU metric between two rectangles
	* @param corners1 xmin, ymin, xmax, ymax for the first box
	* @param corners2 xmin, ymin, xmax, ymax for the second box
	* @return IoU for the two rectangles
	*/
	float IoU(vector<float> corners1, vector<float> corners2);

	/** Compute the distance between two points
	* @param p1 First point
	* @param p2 Second point
	* @return Distance between p1 and p2
	*/
	float distance(cv::Point p1, cv::Point p2);

	int win_size;
	cv::Mat pred, prob_map;
	cv::Size img_size;
	vector<cv::Point> kp_pos;
	vector<cv::Mat> clusters_masks;
	vector<cv::Point> clusters_centers;
	vector<vector<float>> boundingboxes;
};


/**
* Class that implement the mean shift algorithm
*
* It finds modes in a 2-dim probability density function
* In addition it cluster togheter all initial points that
* end in the same mode.
*/
class MeanShift {

public:

	/** Default constructor
	* @param density_map The starting 2-D probability density function (a CV_32FC1 mat normalized in [0, 1])
	*/
	MeanShift(cv::Mat density_map);

	/** Main method to run the algorithm
	* @param window_size Considered area when compputing the center of mass.
	* @param stopping_toll Regulates when to stop (Are we still moving?)
	* @param peak_radious Dimension of what is considered the peak
	*/
	void compute(int window_size = 100, float stopping_toll = 0.1, float peak_radious = 10);

	/** Merge clusters that are close togheter
	* @param threshold Only cluster whose distance is less than the threshold to each other are volid to be merged
	*/
	void prune(float threshold = 100);

	/** Remove clusters whose peak average value is under a threshold
	* @param percentage Threshold to determine if a cluster is removed or not
	*/
	void suppress(float percentage);

	/** Returns a vector of masks covering each cluster
	* @return Vector of CV_8UC1 matrices 
	*/
	vector<cv::Mat> getClustersMasks();

	/** Returns a vector containing the points of the center of each cluster
	* @return Centers of clusters
	*/
	vector<cv::Point> getClustersCenters();

	/** Draws the clusters countours and centers on the given image.
	*   Also write on the image a number representing the strongness of the cluster
	*   0 = weak cluster = low average peak probability
	* @param image Input/Output image
	*/
	void drawClusters(cv::Mat& image);

	void drawPaths(cv::Mat& image);

protected:

	/** Reorder start_to_end_map and centers_list such that
	*	the weak clusters are placed at the beginning 
	*   and the strong ones at the end
	*/
	void sortNameMap();

	/** Compute the distance between two points
	* @param p1 First point
	* @param p2 Second point
	* @return Distance between p1 and p2
	*/
	float distance(cv::Point p1, cv::Point p2);

	/** Compute the center of mass in a cricle of given radious
	* @param center Center of the circle
	* @param radious Radious of the circle
	* @param step Gap between evaluated points
	* @return The point representing the center of mass
	*/
	cv::Point2f getCenterOfMass(cv::Point2f center, int radious, int step = 2);

	/** Compute the average of the probability density function 
	*   in a circle with given center
	* @param x X coordinate of the center of the circle
	* @param y Y coordinate of the center of the circle
	* @param radious Dimension of what is considered the peak
	* @param step Gap between evaluated points
	* @return The average density in the circle
	*/
	float peakAverage(float x, float y, int radious = 10, int step = 2);

	//density = input probability density function to be clustered
	//start_to_end_map = each row contains an initial point 
	//					 and its relative final position
	//start_to_name_map = each row contains the name of a cluster 
	//					  and one initial points which belong to that cluster
	//centers_list = list of the center of each cluster
	int win_size;
	cv::Mat density, start_to_end_map, start_to_name_map;
	vector<cv::Point> centers_list;
	vector<vector<cv::Point2f>> paths;
};


/**
* Class for handling a rectangular bounding box
* 
* To describe the rectangle the coordinates of the two oppisite corners are used
* (xmin, ymin), (xmax, ymax)
* The optimize method implements the algorithm to move the boundingbox towards a
* peak of the given density.
*/
class BoundingBox {

public:

	/** Default constructor
	* @param x_min Top left corner x component
	* @param y_min Top left corner y component
	* @param x_max Bottom right corner x component
	* @param y_max Bottom right corner y component
	*/
	BoundingBox(float x_min, float y_min, float x_max, float y_max);

	/** Implement the optimization algorithm that find the rectangle that
	*	best suits the given density (normalized in [0, 1])
	* @param density target density (should be a CV_32FC1 matrix with values between 0 and 1)
	* @param threshold threshold for expanding/shrinking the bounding box
	* @param offset Determines the amount of expansion/compression of the rectangle
	* @param max_iter Maximum iterations before stopping
	*/
	void optimize(cv::Mat density, float threshold = 0.2, int offset = 10, int max_iter = 300);

	/** Return the current position of the bounding box
	* @return Current xmin, ymin, xmax, ymax
	*/
	vector<float> getCorners();

protected:

	/** Implementation of the expansion step of the optimization algorithm.
	* @param threshold threshold for expanding the bounding box
	* @param offset Determines the amount of expansion of the rectangle
	*/
	void expand(float threshold, int offset);

	/** Implementation of the shrinking step of the optimization algorithm.
	* @param threshold threshold for shrinking the bounding box
	* @param offset Determines the amount of compression of the rectangle
	*/
	void shrink(float threshold, int offset);

	/** Computes the average of the density in the given rectangle.
	*   It uses the integral image to speed-up the process.
	* @param x1 top left corner x component
	* @param y1 top left corner y component
	* @param x2 bottom right corner x component
	* @param y2 bottom right corner y component
	*/
	float average(float x1, float y1, float x2, float y2);


	float xmin, ymin, xmax, ymax; //current position
	cv::Mat integral_density;	  //integral image of the density
};