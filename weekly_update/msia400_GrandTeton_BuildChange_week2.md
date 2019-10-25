## MSIA 400 Grand Teton (Project Build Change) Weekly Update
## Oct 24 2019 WEEK 2

This week we were able to setup the remote desktop environement and installed the required packages (OpenCV, etc.) useful for us.

We have created two ipynb notebooks this week. The same code can also be found in corresponding py script files.

In the first notebook - [explore_images.ipynb](https://nbviewer.jupyter.org/github/MSIA/MSiA400_GrandTeton/blob/master/explore_images.ipynb) we explored the images inside the Go/NoGo folders and did basic tasks like reading images through OpenCV python package, interpreting them as numpy ndarrays and understood their different color channels.

In the second notebook - [link_images_csv.ipynb](https://nbviewer.jupyter.org/github/MSIA/MSiA400_GrandTeton/blob/master/link_images_csv.ipynb) we explored the file: "Building_Set_Balanced_10000_Generated_9_MAY_2019.csv" which has data about the generated images inside the Go/NoGo folders. We were able to link each image file name with the specific row inside the csv file which corresponds to the same image and also verified the same using the labels.

In the third notebook - [logistic.py](https://github.com/MSIA/MSiA400_GrandTeton/blob/master/logistic.py) we built logistic regression classifier based on features contained in csv file, and further checked the accuracy of this classifier on test set.

We now face the following questions:
1. For each generated image, are the features listed inside the csv file enough or do we need to generate more features using OpenCV and other classical computer vision tools? 

Once we finalize our feature vector we will move forward to the next step of developing a classifier that can predict Go/NoGo from the input feature vector of any image.

2. In order to test our model performance on real world data, we will need labelled real world images. Currently the real images provided to us are not labelled. So are we supposed to evaluate our model based on other techniques?