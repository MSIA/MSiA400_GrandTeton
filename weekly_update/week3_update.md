# MSIA 400 Grand Teton (Project Build Change) weekly Update

## Nov 1 2019 WEEK 3

This week, we discussed how we are going to extract feautures from the training images. We found it extremely hard to extract all the characteristics (more than 40 as listed in the csv file) from the generated images. Hence, we decided to approach this challenge by choosing the features that can be easily detected through OpenCV packages and show the most significance in the logistic regression model we built through the csv file. 

I'M CONFUSED HERE; YOU SHOULD EXTRACT FEATURES DIRECTLY FROM IMAGES AND THUS I DO NOT SEE THE ROLE OF THE CSV FILE.

We first defined two functions (count_contours(), count_squares()) to extract the contours of squares from those generated images which can represent the number of windows(or other openings). But we encountered a challenge that not only the window/door squares have been detected but also the outer square (squares on the edges). We will work on this problem next week. 

THE NUMBER OF SQUARES, ETC SHOULD BE CONSTRUCTED FROM IMAGES (AND NOT FROM CSV; JUST MAKING IT CLEAR)

In the following week, we will primarily work on methods to extract more features from the images, try to fit them in the logistic regression model and test the model's performance by comparing the accuracy of predicting the GO/NoGO results.

YOU SHOULD HAVE FIRST RESULTS THIS WEEK

In addtion, we are going to schedule a meeting with Adam on some techinial problems we have encountered throughout the first three weeks.

CODE: NOT COMMENTED ENOUGH; STILL CODE OUTSIDE OF ANY FUNCTION AND CLASS
