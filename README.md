# Face_Detection_by_Adaboost_and_Realboost
The whole project used the Boosting techniques, especially the Adaboost and the Realboost algorithms to do a Basic-level Facial Detection job.

## Data
The original trainig data using here are 16 by 16 image patches with faces and non-faces. There are 11838 image patches with faces and 45356 image patches without faces. To transfer these image patches as numerically readable data in the algorithm, we need to use the Haar Filters. The Haar Filters is a design to calculate the difference of the pixel numbers in a specific rectangle in a image. In general there will be four different kinds of Haar Filters and apply all these filters design on our training image patches, we would have 10032 static filters in total. Each filter would give an activation value (response value) for each image patch and we can assign a label 1 to the face image patch activation values or a label -1 to the nonface image patch activation values. As a result, we could form the weak classifiers for the Boosting.

## Test Image
The test image actually could be any kind of real life images with people's faces on it. However, beacause here we only do the face detection process on the V-channel of the image (grey images), you need to extract the V-channel of the test image for the detection. The algorithms would divide the images into many 16 by 16 patches and determine if the patch can be determined as a face or not. A NMS (non maximum suppression) has been used here so that there will be no duplicated square patches shown for exact one face.

## Hard Negative Mining
Hard negative mining is a tech to make the test image perform detections better. It means that we want to take two images for the test images. One image with all people show the faces and another one all people stay in the same position but turn around with only the back but no faces. The non-face image will be used as the last step of the training that we would do the face detection on this image first. We know that there should be no faces so that all the patches detected as faces can be add back to the training set with label -1. It would help increase the training set and help reduce the noise in the real-life image because Haar filters are very sensitive to the boundary-shaped thing (with grey color intensity in a very small patch). After the hard negative mining, we could do the test again and the result is supposed to be much better.

## Adaboost fo Face detection
Adaboost use a recursion mechanism to select and add on the weak classifiers to form a stronger classifier. For each step, a very simple classification has been done in a way that the activation values of a filter are seperated to form two groups by a threshold. We would choose the threshold that gives the smallest classification error rate. Because of the number of filters is large, it would be impossible to try the thresold from the minimum value to the maximum value of each filter one by one. The default set here is to choose 25 random values from all the activation values in one filter each epoch and the one with the smallest classification error rate will be chosen as the weak classifier used for the strong classifier. In another word, the whole pool of weak classifiers has a size of 10032x25x2. The reason there is a "x2" is that each classification there will be two kinds of group labeling, we want to find the one with the smallest error rate.

## Transfer to the Realboost
The main difference between Realboost and the Adaboost is that the Realboost does not set a simple threshold to seperate the groups for weak classifiers. However, it divided the activation values into numbers of bins and in each ranges of bins, we will count the number of members in different groups. Here we do not use Realboost entirely but just use the selected filters and update the parameters for the Realboost weak classifiers in each round. It is open to modify the code to do a full Realboost process. However, it would just costs much more computing power.

## Result and the report

The top 20 Haar filters are chosen by sort the weights descend.

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/Top%2020%20Haar%20Filters.png" width="400px"</img> 
</div>

The Training Errors of the strong classifiers:

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/The_Training_Error_of_Strong.png" width="400px"</img> 
</div>

The training errors of the top 1000 weak classifiers:

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/Weak%20Classifier%20Errors.png" width="400px"</img> 
</div>

As what we expected, more weak classifers chosen, closer the error rate goes to 0.5. And the rest in the pool would performs worse and worse.

The histogram when T=10:

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/histogram_9.png" width="400px"</img> 
</div>

The histogram when T=50:

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/histogram_49.png" width="400px"</img> 
</div>

The histogram when T=100:

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/histogram_99.png" width="400px"</img> 
</div>

From T=10 to T=50, there is a very clear progress that two distributions are formed. From T=50 to T=100, the distributions become smoother and the means of them just get farther to each other.

The ROC plot:

<div align="center">
        <img src="https://github.com/nji3/Face_Detection_by_Adaboost_and_Realboost/blob/master/readme_img/ROC%20Curve.png" width="400px"</img> 
</div>

The ROC plot shows the classification result just goes better and better as we increase the number of chosen weak classifiers.
