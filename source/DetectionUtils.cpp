/**
* Utilities for detecting the boat
* @file DetectionUtils.cpp
* @author Alessandro Canevaro
* @version 23/07/21
*/

#include <opencv2/opencv.hpp>
#include <../include/DetectionUtils.h>

using namespace std;

Detector::Detector(cv::Mat predictions, vector<cv::Point> keypoints_pos, cv::Size image_size){
	pred = predictions;
	kp_pos = keypoints_pos;
	img_size = image_size;
}

void Detector::computeProbMap(double radious, int step) {
	cv::Mat_<float> distance_prob_map = cv::Mat::zeros(img_size, CV_32F);
	cv::Mat_<float> density_prob_map = cv::Mat::zeros(img_size, CV_32F);

	for (int i = 0; i < img_size.height; i+=step) {
		for (int j = 0; j < img_size.width; j+=step) {
			float prob = 0;
			float tot_dist = 0;
			float kp_count = 0;
			for (int k = 0; k < kp_pos.size(); k++) {
				float dist = distance(cv::Point2d(j, i), kp_pos[k]);
				if (dist < radious) {
					prob += pred.at<float>(k) * (radious-dist);
					tot_dist += (radious - dist);
					kp_count++;
				}
			}
			if (prob < 0) { prob = 0; }
			cv::rectangle(distance_prob_map, cv::Point(j, i), cv::Point(j + step, i + step),
						  cv::Scalar(prob/tot_dist), -1);
			cv::rectangle(density_prob_map, cv::Point(j, i), cv::Point(j + step, i + step),
						  cv::Scalar(kp_count), -1);
		}
	}
	cv::normalize(distance_prob_map, distance_prob_map, 0, 1, cv::NORM_MINMAX, CV_32F);
	cv::normalize(density_prob_map, density_prob_map, 0, 1, cv::NORM_MINMAX, CV_32F);
	//cv::imshow("prior map", density_prob_map);
	//cv::imshow("measure map", distance_prob_map);
	prob_map = distance_prob_map.mul(density_prob_map);
	cv::patchNaNs(prob_map, 0);
	cv::GaussianBlur(prob_map, prob_map, cv::Size(17, 17), 7);
	cv::normalize(prob_map, prob_map, 0, 1, cv::NORM_MINMAX, CV_32F);
	//cv::imshow("Probability map", prob_map);
}

void Detector::computeMeanShiftClustering(cv::Mat& image, cv::Mat& basins, int window_size, float prune_thr, float sup1_thr, float sup2_thr) {
	MeanShift ms = MeanShift(prob_map);
	ms.compute(window_size);

	ms.suppress(sup1_thr);
	ms.prune(prune_thr);
	ms.suppress(sup2_thr);

	ms.drawPaths(basins);

	clusters_masks = ms.getClustersMasks();
	clusters_centers = ms.getClustersCenters();

	ms.drawClusters(image);
}

void Detector::computeBoundingBox(float thr1, float thr2, int offset1, int offset2) {
	vector<vector<float>> bb_list;
	for (int i = 0; i < clusters_centers.size(); i++) {
		cv::Mat density;
		prob_map.copyTo(density, clusters_masks[i]);
		density.convertTo(density, CV_32F);
		cv::normalize(density, density, 0, 1, cv::NORM_MINMAX, CV_32F);

		BoundingBox bb = BoundingBox((float)0, (float)0, (float)density.cols-1, (float)density.rows - 1);

		bb.optimize(density, thr1, offset1);
		bb.optimize(density, thr2, offset2);

		bb_list.push_back(bb.getCorners());
	}
	boundingboxes = bb_list;
}

void Detector::drawBoundingBoxes(cv::Mat& image, cv::Scalar color) {
	for (int i = 0; i < boundingboxes.size(); i++) {
		cv::rectangle(image, cv::Point2f(boundingboxes[i][0], boundingboxes[i][1]),
							 cv::Point2f(boundingboxes[i][2], boundingboxes[i][3]), 
							 color, 2);
	}
}

void Detector::computeIoU(vector<vector<float>> bb_corner_list, cv::Mat &image) {
	float avg = 0;
	int count = 0;
	for (int i = 0; i < bb_corner_list.size(); i++) {
		float best = 0;
		for (int j = 0; j < boundingboxes.size(); j++) {
			float tmp = IoU(bb_corner_list[i], boundingboxes[j]);
			if (tmp > best) { best = tmp; }
		}
		cv::rectangle(image, cv::Point(bb_corner_list[i][0], bb_corner_list[i][1]), 
							 cv::Point(bb_corner_list[i][0]+50, bb_corner_list[i][1]+20), 
							 cv::Scalar(0, 255, 0), -1);
		cv::putText(image, to_string((int)round(best*100))+"%", cv::Point(bb_corner_list[i][0] + 5, bb_corner_list[i][1] + 15),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		cout << "box: " << i << " IoU: " << best << endl;
		avg += best;
		count++;
	}
	cout << "avg: " << avg / count << endl;
}

float Detector::IoU(vector<float> corners1, vector<float> corners2) {
	float xmininter = corners1[0]; float ymininter = corners1[1];
	float xmaxinter = corners1[2]; float ymaxinter = corners1[3];
	float xmin2 = corners2[0]; float ymin2 = corners2[1];
	float xmax2 = corners2[2]; float ymax2 = corners2[3];

	if (xmininter > xmax2 || xmaxinter<xmin2 || ymininter>ymax2 || ymaxinter < ymin2) { return 0; }

	if (xmin2 > xmininter) { xmininter = xmin2; }
	if (ymin2 > ymininter) { ymininter = ymin2; }
	if (xmax2 < xmaxinter) { xmaxinter = xmax2; }
	if (ymax2 < ymaxinter) { ymaxinter = ymax2; }
	float inter_area = abs(xmaxinter - xmininter) * abs(ymaxinter - ymininter);
	float union_area = (corners1[2] - corners1[0]) * (corners1[3] - corners1[1]) + (xmax2 - xmin2) * (ymax2 - ymin2) - inter_area;
	return inter_area / union_area;
}

float Detector::distance(cv::Point p1, cv::Point p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}


MeanShift::MeanShift(cv::Mat density_map) {
	density = density_map;
}

void MeanShift::compute(int window_size, float stopping_toll, float peak_radious) {
	win_size = window_size;
	//start here
	cv::Point2f CoM;
	for (int i = 0; i < density.rows; i += win_size / 2) {
		for (int j = 0; j < density.cols; j += win_size / 2) {
			cv::Point2f center = cv::Point2f(j, i);
			vector<cv::Point2f> path_tmp = { center };
			for (int k = 0; k < 300; k++) {
				CoM = getCenterOfMass(center, win_size);
				if (abs(CoM.x - center.x) < stopping_toll && abs(CoM.y - center.y) < stopping_toll) { break; }
				center.x = CoM.x;
				center.y = CoM.y;
				path_tmp.push_back(center);
			}
			vector<float> tmp = { (float)j, (float)i, center.x, center.y };
			start_to_end_map.push_back(cv::Mat(1, 4, CV_32F, tmp.data()));
			paths.push_back(path_tmp);
		}
	}

	//find all initial points that lead to the same peak
	for (int i = 0; i < start_to_end_map.rows; i++) {
		cv::Point peak_loc = cv::Point(start_to_end_map.at<float>(i, 2), start_to_end_map.at<float>(i, 3));
		cv::Point start_loc = cv::Point(start_to_end_map.at<float>(i, 0), start_to_end_map.at<float>(i, 1));
		if (start_loc.x == peak_loc.x && start_loc.y == peak_loc.y) { continue; }
		int j;
		for (j = 0; j < centers_list.size(); j++) {
			if (distance(centers_list[j], peak_loc) < peak_radious) {
				vector<int> tmp = { j, start_loc.x, start_loc.y };
				start_to_name_map.push_back(cv::Mat(1, 3, CV_32S, tmp.data()));
				break;
			}
		}
		if (j == centers_list.size()) {
			vector<int> tmp = { j, start_loc.x, start_loc.y };
			start_to_name_map.push_back(cv::Mat(1, 3, CV_32S, tmp.data()));
			centers_list.push_back(peak_loc);
		}
	}
	sortNameMap();
}

void MeanShift::sortNameMap() {
	int new_index = 0;
	int min_peak_idx;
	cv::Mat new_start_to_name_map;
	vector<cv::Point> new_centers_list;
	int count = 0;
	while (count < centers_list.size()) {
		float min_peak = 1;
		//find minimum peak
		for (int j = 0; j < centers_list.size(); j++) {
			if (centers_list[j].x == -1) { continue; }
			if (peakAverage(centers_list[j].x, centers_list[j].y) < min_peak) {
				min_peak = peakAverage(centers_list[j].x, centers_list[j].y);
				min_peak_idx = j;
			}
		}
		//insert minimum peak in new_start_to_name_map with updated index
		for (int j = 0; j < start_to_name_map.rows; j++) {
			if (min_peak_idx == start_to_name_map.at<int>(j, 0)) {
				vector<int> tmp = { new_index, start_to_name_map.at<int>(j, 1), start_to_name_map.at<int>(j, 2) };
				new_start_to_name_map.push_back(cv::Mat(1, 3, CV_32S, tmp.data()));
			}
		}
		new_index++;
		//update new_centers_list
		new_centers_list.push_back(centers_list[min_peak_idx]);
		centers_list[min_peak_idx] = cv::Point(-1, -1);
		count++;
	}
	start_to_name_map = new_start_to_name_map.clone();
	centers_list = new_centers_list;
}

void MeanShift::suppress(float percentage) {
	vector<cv::Point> new_centers;
	cv::Mat new_cluster;
	for (int i = 0; i < centers_list.size(); i++) {
		if (peakAverage(centers_list[i].x, centers_list[i].y) > percentage) {
			//cout << "keeeping" << endl;
			for (int j = 0; j < start_to_name_map.rows; j++) {
				if (start_to_name_map.at<int>(j, 0) == i) {
					vector<int> tmp = { (int)new_centers.size(), start_to_name_map.at<int>(j, 1), start_to_name_map.at<int>(j, 2) };
					new_cluster.push_back(cv::Mat(1, 3, CV_32S, tmp.data()));
				}
			}
			new_centers.push_back(centers_list[i]);
		}
	}
	centers_list = new_centers;
	start_to_name_map = new_cluster.clone();
}

float MeanShift::peakAverage(float x, float y, int radious, int step) {
	float avg = 0;
	float k = 0;
	for (int i = y - radious; i < y + radious; i += step) {
		for (int j = x - radious; j < x + radious; j += step) {
			float dist = distance(cv::Point(j, i), cv::Point(x, y));
			if (dist < radious) {
				if (i < 0 || i >= density.rows || j < 0 || j >= density.cols) { continue; }
				avg += density.at<float>(i, j);
				k++;
			}
		}
	}
	return avg / k;
}

void MeanShift::prune(float threshold) {
	vector<cv::Point> new_centers;
	for (int i = 0; i < centers_list.size(); i++) {
		float minimum = 100000;
		int min_idx = 100000;
		//find possible candidate
		for (int j = 0; j < start_to_name_map.rows; j++) {
			if (start_to_name_map.at<int>(j, 0) != i) {
				float dist = distance(centers_list[i], cv::Point(start_to_name_map.at<int>(j, 1), start_to_name_map.at<int>(j, 2)));
				if (dist < minimum) {
					minimum = dist;
					min_idx = start_to_name_map.at<int>(j, 0);
				}
			}
		}
		//if is good, merge the 2 cluster togheter
		if (minimum < threshold) {
			//cout << "accepted" << endl;
			for (int j = 0; j < start_to_name_map.rows; j++) {
				if (start_to_name_map.at<int>(j, 0) == i) {
					start_to_name_map.at<int>(j, 0) = min_idx;
				}
			}
			for (int j = 0; j < start_to_name_map.rows; j++) {
				if (start_to_name_map.at<int>(j, 0) > i) {
					start_to_name_map.at<int>(j, 0) -= 1;
				}
			}
			centers_list.erase(centers_list.begin()+i);
			i -= 1;
		}
	}
}

vector<cv::Mat> MeanShift::getClustersMasks() {
	vector<cv::Mat> masks_vec;
	for (int i = 0; i < centers_list.size(); i++) {
		cv::Mat mask = cv::Mat::zeros(density.size(), CV_8U);
		for (int j = 0; j < start_to_name_map.rows; j++) {
			if (start_to_name_map.at<int>(j, 0) == i) {
				cv::Point p1 = cv::Point(start_to_name_map.at<int>(j, 1) - win_size / 4,
										 start_to_name_map.at<int>(j, 2) - win_size / 4);
				cv::Point p2 = cv::Point(start_to_name_map.at<int>(j, 1) + win_size / 4,
										 start_to_name_map.at<int>(j, 2) + win_size / 4);
				cv::rectangle(mask, p1, p2, cv::Scalar(255), -1);
			}
		}
		masks_vec.push_back(mask);
	}
	return masks_vec;
}

vector<cv::Point> MeanShift::getClustersCenters() {
	return centers_list;
}

void MeanShift::drawClusters(cv::Mat& image) {
	vector<cv::Mat> masks = getClustersMasks();
	cv::RNG rng(12345);
	for (int i = 0; i < masks.size(); i++) {
		vector<vector<cv::Point>> countour;
		cv::findContours(masks[i], countour, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::drawContours(image, countour, -1, color, 3);
		cv::circle(image, centers_list[i], 10, color, 2);
		//cout << i << endl;
		cv::putText(image, to_string(i), centers_list[i]+cv::Point(20, 0),
					cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	}
}

void MeanShift::drawPaths(cv::Mat& image) {
	cv::Mat cluster_img = cv::Mat::zeros(density.size(), CV_8UC3);
	cv::Mat tmp;
	density.convertTo(tmp, CV_8UC1, 255.0);
	for (int i = 0; i < cluster_img.rows; i++) {
		for (int j = 0; j < cluster_img.cols; j++) {
			cluster_img.at<cv::Vec3b>(i, j)[0] = tmp.at<cv::uint8_t>(i, j);
			cluster_img.at<cv::Vec3b>(i, j)[1] = tmp.at<cv::uint8_t>(i, j);
			cluster_img.at<cv::Vec3b>(i, j)[2] = tmp.at<cv::uint8_t>(i, j);
		}
	}
	for (int i = 0; i < paths.size(); i++) {
		for (int j = 1; j < paths[i].size(); j++) {
			cv::line(cluster_img, paths[i][j - 1], paths[i][j], cv::Scalar(255, 255, 0), 1, 8, 0); //plot lines
		}
	}
	image = cluster_img.clone();
}

float MeanShift::distance(cv::Point p1, cv::Point p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

cv::Point2f MeanShift::getCenterOfMass(cv::Point2f center, int radious, int step) {
	float CoM_x = 0;
	float CoM_y = 0;
	float tot_density = 0;
	for (int i = center.y - radious; i < center.y + radious; i += step) {
		for (int j = center.x - radious; j < center.x + radious; j += step) {
			float dist = distance(cv::Point(j, i), cv::Point(center.x, center.y));
			if (dist < radious) {
				if (i < 0 || i >= density.rows || j < 0 || j >= density.cols) { continue; }
				CoM_x += (j - center.x) * density.at<float>(i, j);
				CoM_y += (i - center.y) * density.at<float>(i, j);
				tot_density += density.at<float>(i, j);
			}
		}
	}
	if (tot_density == 0) { return center; }
	return cv::Point2f(CoM_x / tot_density, CoM_y / tot_density) + center;
}


BoundingBox::BoundingBox(float x_min, float y_min, float x_max, float y_max) {
	xmin = x_min; ymin = y_min; xmax = x_max; ymax = y_max;
}

void BoundingBox::optimize(cv::Mat density, float threshold, int offset, int max_iter) {
	//Compute the integral image
	cv::integral(density, integral_density, CV_32F);

	float xmin_old, ymin_old, xmax_old, ymax_old;
	for (int i = 0; i < max_iter; i++) {
		xmin_old = xmin; ymin_old = ymin; 
		xmax_old = xmax; ymax_old = ymax;

		expand(threshold, offset);
		shrink(threshold, offset);

		//bounding box degenerated into a point -> stop
		if (xmin > xmax || ymin > ymax) {
			cout << "Convergence failed at iter: " << i << endl;
			break;
		}

		//no more moevment -> stop
		if (xmin_old == xmin && ymin_old == ymin &&
			xmax_old == xmax && ymax_old == ymax) {
			break;
		}
	}
}

void BoundingBox::expand(float threshold, int offset) {
	//expand LEFT side
	if (xmin - offset < 0) { xmin = offset; }
	if (average(xmin - offset, ymin, xmin, ymax) >= threshold) { xmin -= offset; } 
	//expand RIGHT side
	if (xmax + offset > integral_density.cols - 1) { xmax = integral_density.cols - 1 - offset; }
	if (average(xmax, ymin, xmax + offset, ymax) >= threshold) { xmax += offset; } 
	//expand UP side
	if (ymin - offset < 0) { ymin = offset; }
	if (average(xmin, ymin - offset, xmax, ymin) >= threshold) { ymin -= offset; } 
	//expand DOWN side
	if (ymax + offset > integral_density.rows - 1) { ymax = integral_density.rows - 1 - offset; }
	if (average(xmin, ymax, xmax, ymax + offset) >= threshold) { ymax += offset; } 
}

void BoundingBox::shrink(float threshold, int offset) {
	//shrink LEFT side
	if (average(xmin, ymin, xmin + offset, ymax) < threshold) { xmin += offset; }
	//shrink RIGHT side
	if (average(xmax - offset, ymin, xmax, ymax) < threshold) { xmax -= offset; } 
	//shrink UP side
	if (average(xmin, ymin, xmax, ymin + offset) < threshold) { ymin += offset; } 
	//shrink DOWN side
	if (average(xmin, ymax - offset, xmax, ymax) < threshold) { ymax -= offset; }
}

float BoundingBox::average(float x1, float y1, float x2, float y2) {
	//Total number of pixels in the rectangle
	float num_pix = abs(x2 - x1) * abs(y2 - y1);

	float rect_avg = integral_density.at<float>(y2, x2);
	rect_avg -=      integral_density.at<float>(y1, x2);
	rect_avg -=		 integral_density.at<float>(y2, x1);
	rect_avg +=		 integral_density.at<float>(y1, x1);

	return rect_avg / num_pix;
}

vector<float> BoundingBox::getCorners() {
	//Return corners in a single vector
	vector<float> corners = { xmin, ymin, xmax, ymax };
	return corners;
}