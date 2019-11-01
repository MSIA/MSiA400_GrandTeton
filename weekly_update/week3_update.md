# MSIA 400 Grand Teton (Project Build Change) weekly Update

## Nov 1 2019 WEEK 3

This week, we have discussed how we are going to extract feautures from the training images. We found it extremely hard to extract all the characteristics (more than 40 as listed in the csv file) from those generated images. Hence, we decided to approach this challenge by choosing features that can be easily detected through OpenCV packages and also show significance in the logistic regression model. 

We firstly tried to use two functions (count_contours(), count_squares()) extracting contours of squares from those generated images which can represent the number of windows(or doors). But we have encountered a challenge that not only the window/door squares have been detected but also the outer square. We will work on that in the next week. 

In the following week, we will primarily work on methods to extract more featurese from the images, try to fit them in the regression model and test the model's performance by comparing the accuracy of predicting the GO/NoGO results.

In addtion, we are going to schedule a meeting with Adam on some techinial problems we have encountered throughout the first three weeks.
