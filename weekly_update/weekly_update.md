# MSIA 400 Grand Teton (Project Build Change) weekly Update
## Nov 21 2019 WEEK 6
This week we used several measures to address the problem of feature extraction functions not working properly on the real images:

1. We resize the real images since most of them have large sizes. The extra sizes make testing slow don't add too much useful information for feature extraction. 

2. We tried SKImage and found that it detects line better than OpenCV since it has less constraints on how we threshold the image. We found the real images usually have many noises, and when used OpenCV for line detection, the lines are usually detached into dots that are not connected. 
[line_detection_new.py](https://github.com/MSIA/MSiA400_GrandTeton/blob/master/line_detection_new.py)

3. We used Otsu's thresholding for all images.

4. We are in the process of writing script to turn the real images into the style of the generated images using Neural Style Transfer, which could potentially make the feature extraction functions work more properly.

NST IS A GREAT IDEA. IMPLEMENT FEATURE EXTRACTION IN PARALLEL; OUR CLUSTER HAS MANY CORES THAT WILL SIGNIFICANTLY SPEED UP FEATURE EXTRACTION. 

## Nov 14 2019 WEEK 5
This week we implemented more features to check whether the accuracy of model improves, and tested how our model performs on real images.

Last week we only had 3 features, namely: number of floors, number of openings and the fraction width of openings for each building.

THESE CANNOT BE FEATURES UNLESS YOU HAVE AN ALGORITHM THAT EXTRACTS THEM FROM IMAGES. 
KEEP IN MIND THAT YOUR INPUT IS ONLY AN IMAGE. 

This week we defined 5 more functions to perform feature extraction. Now we have the following 8 features:
1. Number of floors in each building
2. Number of openings (windows/doors etc.)
3. The average fraction width of openings for each building (excluding vertically overlapping openings)
4. The average fraction width of openings for each building (average of fraction width per floor)
5. The average fraction height of openings for each building (excluding horizontally overlapping openings)
6. The sum of heights of openings for each building divided by the total height of each building
7. The image width in number of pixels
8. The image height in number of pixels

I DON'T GET IT. YOU CANNOT USE THESE FEATURES UNLESS YOU HAVE AN ALGORITHM THAT EXTRACTS THEM FROM IMAGES AUTOMATICALLY. 

We tried to improve the model performance using the following steps:
1. Implementing class weights for class imbalance (3370 Go and 5001 NoGo)
2. Normalizing the features to a mean of 0 and standard deviation of 1
3. Tuning the logistic regression penalty hyperparameters across 'L1' and 'L2' loss penalties and across different values of C using GridSearchCV and 5 fold cross-validation.

We fit our logistic regreesion with all 8 features we have, and didn't see a significant improvement in our model accuracy (still at 79.66%).

TRY LIGHTGBM AND XGBOOST. 

We think there might be a need to change the model now (maybe SVM or Random Forest).

To have a detailed look at the code and results, please look at the notebook:
https://github.com/MSIA/MSiA400_GrandTeton/blob/master/LogisticModel_FeatureClass.ipynb

Then we tested this model on real images, and found the results are not consistent with our expectation. For example, the count_level() fails to detect non-horizonal lines which represent the number of floors, and the count_openinings() returns more contours comparing with the actual image. Please look at this notebook for our testing on real image:
https://github.com/MSIA/MSiA400_GrandTeton/blob/master/test_real_image.ipynb

In the next week, we will try implement haar features, HOG, or other approaches to get better results.

THESE ARE 'ALLOWABLE' FEATURES. 

## Nov 6 2019 WEEK 4

This week we extracted feautures that could be easily detected through OpenCV packages and show the most significance in previous logistic regression model, from the generated images. Based on these features, we fit a logistic regression model to test the model's performance through predicting the GO/NoGO results.

In the notebook - [LogisticModel_ThreeFeature.ipynb](https://github.com/MSIA/MSiA400_GrandTeton/blob/master/LogisticModel_ThreeFeature.ipynb) We defined several functions to perform feature extraction:

1. count_openings() to extract the contours of quadrilaterals which represent the number of openings (e.g. windows, doors);
2. fraction_width() to calculate the average percentage of openings on each floor;
3. count_level() to detect the horizonal lines which represent the number of levels (floors);

YOU SHOULD ALSO CONSIDER HAAR FEATURES, HOG (Histogram of Oriented Gradients).

Then we fit our logistic regression model with these three features above, split train and test size by 7:3, and reached accuracy of 0.8.

In the following week, we will continue to work on extracting more features from the generated images and improve our current model by comparing the accuracy.

HOW DOES YOUR MODEL PERFORM ON REAL WORLD IMAGES? 




## Nov 1 2019 WEEK 3

This week, we discussed how we are going to extract feautures from the training images. We found it extremely hard to extract all the characteristics (more than 40 as listed in the csv file) from the generated images. Hence, we decided to approach this challenge by choosing the features that can be easily detected through OpenCV packages and show the most significance in the logistic regression model we built through the csv file. 

I'M CONFUSED HERE; YOU SHOULD EXTRACT FEATURES DIRECTLY FROM IMAGES AND THUS I DO NOT SEE THE ROLE OF THE CSV FILE.

We first defined two functions (count_contours(), count_squares()) to extract the contours of squares from those generated images which can represent the number of windows(or other openings). But we encountered a challenge that not only the window/door squares have been detected but also the outer square (squares on the edges). We will work on this problem next week. 

THE NUMBER OF SQUARES, ETC SHOULD BE CONSTRUCTED FROM IMAGES (AND NOT FROM CSV; JUST MAKING IT CLEAR)

In the following week, we will primarily work on methods to extract more features from the images, try to fit them in the logistic regression model and test the model's performance by comparing the accuracy of predicting the GO/NoGO results.

YOU SHOULD HAVE FIRST RESULTS THIS WEEK

In addtion, we are going to schedule a meeting with Adam on some techinial problems we have encountered throughout the first three weeks.

CODE: NOT COMMENTED ENOUGH; STILL CODE OUTSIDE OF ANY FUNCTION AND CLASS




## Oct 24 2019 WEEK 2

This week we were able to setup the remote desktop environement and installed the required packages (OpenCV, etc.) useful for us.

We have created two ipynb notebooks this week. The same code can also be found in corresponding py script files.

In the first notebook - [explore_images.ipynb](https://nbviewer.jupyter.org/github/MSIA/MSiA400_GrandTeton/blob/master/explore_images.ipynb) we explored the images inside the Go/NoGo folders and did basic tasks like reading images through OpenCV python package, interpreting them as numpy ndarrays and understood their different color channels.

In the second notebook - [link_images_csv.ipynb](https://nbviewer.jupyter.org/github/MSIA/MSiA400_GrandTeton/blob/master/link_images_csv.ipynb) we explored the file: "Building_Set_Balanced_10000_Generated_9_MAY_2019.csv" which has data about the generated images inside the Go/NoGo folders. We were able to link each image file name with the specific row inside the csv file which corresponds to the same image and also verified the same using the labels.

In the third notebook - [logistic.py](https://github.com/MSIA/MSiA400_GrandTeton/blob/master/logistic.py) we built logistic regression classifier based on features contained in csv file, and further checked the accuracy of this classifier on test set.

In the coming week, we will attempt the follwing tasks:

1. For each generated image, try to generate more features for model fitting

WHAT FEATURES DO YOU HAVE IN MIND? DON'T CREATE FEATURES BY HAVING THE SIMULATED IMAGES IN MIND, BUT RATHER THE TRUE IMAGES. 

2. Once we finalize our feature vector we will move forward to the next step of developing a classifier that can predict Go/NoGo from the input feature vector of any image.




## Oct 17 2019 WEEK 1

First group meeting on Oct 15 2019, we set up our github team repository and private slack channel. 

This week our team looked at the project description and laid out a tentative schedule regarding how we are going to approach the dataset and deliver the project. We decided to analyze the datasets in two approaches. We want to first use classical machine learning algorithms in OpenCV to classify the image files, and then take advantage of deep learning methods to potentially build a more powerful classifier. 

DON'T USE DEEP LEARNING; USE TRADITIONAL MACHINE LEARNING. 

We encountered several problems about the project deliverables:

- How should we incorporate the dataset   "Balanced_Set_Balanced_10000_Generated_9_May_2019.csv” for the OpenCV analysis?

- What approach should we use to generate a list of file names for the  entries in “Balanced_Set_Balanced_10000_Generated_9_May_2019.csv”

IF THESE QUESTIONS ARE FOR ME, THEN PLEASE ASK BORCHULUUN SINCE SHE IS ON TOP WHEN IT COMES TO DATA. IF THESE ARE RHETORICAL QUESTIONS, I DON'T UNDERSTAND THEIR PURPOSE. 
