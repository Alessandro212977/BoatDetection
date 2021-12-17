# BoatDetection
1) Project structure:
Boat_detection/
	-> CMakeLists.txt
	-> source/
		-> detector.cpp (main file for detection)
		-> training.cpp (main file for training the neural network)
		-> DetectionUtils.cpp (lib for detection)
		-> MLUtils.cpp (lib for the machine learning part and a few other things)
	-> include/
		-> DetectionUtils.h
		-> MLUtils.h
	-> data/
		-> MLP64.xml (trained neural network)
		-> dataset_eigenvectors.csv (used for PCA)
		-> dataset_normalization.csv (used for normalizing the inputs)

2) Compiling instructions:
The main file for detection is detector.cpp, and it is currently configured for single image detection. 
Therefore, after the cmake/make commands, it can be executed with .\detector x/y/z/my_image.jpg
(i.e. it requires a line argument containing the to-be-detected image)
This file requires access to the data/ folder for the neural network files, 
hence please keep the current directories structures.
As output, it shows the input image with the predicted bounding boxes (in red).
For displaing the intermidiate images (described in the reports) uncomment the cv::imshow() inside the detector.cpp file.

3) For training the neural network, the training.cpp file is used.
To execute it, the dataset with the ground truth files is required.
