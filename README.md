# Face_Detection_by_Adaboost_and_Realboost
The whole project used the Boosting techniques, especially the Adaboost and the Realboost algorithms to do a Basic-level Facial Detection job.

## Data
The original trainig data using here are 16 by 16 image patches with faces and non-faces. There are 11838 image patches with faces and 45356 image patches without faces. To transfer these image patches as numerically readable data in the algorithm, we need to use the Haar Filters. The Haar Filters is a design to calculate the difference of the pixel numbers in a specific rectangle in a image. In general there will be four different kinds of Haar Filters and apply all these filters design on our training image patches, we would have 10032 static filters in total. Each filter would give an activation value (response value) for each image patch and we can assign a label 1 to the face image patch activation values or a label -1 to the nonface image patch activation values. As a result, we could form the weak classifiers for the Boosting.

## Test Image
The test image actually could be any kind of real life images with people's faces on it. However, beacause here we only do the face detection process on the V-channel of the image (grey images), you need to extract the V-channel of the test image for the detection. The algorithms would divide the images into many 16 by 16 patches and determine if the patch can be determined as a face or not. A NMS (non maximum suppression) has been used here so that there will be no duplicated square patches shown for exact one face.

## Hard Negative Mining
Hard negative mining is a tech to make the test image perform detections better. It means that we want to take two images for the test images. One image with all people show the faces and another one all people stay in the same position but turn around with only the back but no faces. The non-face image will be used as the last step of the training that we would do the face detection on this image first. We know that there should be no faces so that all the patches detected as faces can be add back to the training set with label -1. It would help increase the training set and help reduce the noise in the real-life image because Haar filters are very sensitive to the boundary-shaped thing (with grey color intensity in a very small patch). After the hard negative mining, we could do the test again and the result is supposed to be much better.

## Adaboost fo Face detection

## Transfer to the Realboost
