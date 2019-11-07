# MSIA 400 Grand Teton (Project Build Change) weekly Update

## Nov 6 2019 WEEK 4

This week we extracted feautures that can be easily detected through OpenCV packages and show the most significance in previous logistic regression model.



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
