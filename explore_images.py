#!/usr/bin/env python
# coding: utf-8

# ### OpenCV is an open source computer vision package.
# ### It can be imported into python environments as 'cv2'.

# In[1]:


import cv2


# In[2]:


import os


# In[3]:


import matplotlib.pyplot as plt
import numpy as np


# ### The data is stored inside the 'team' directory

# In[4]:


path = '../../team/courses/MSiA400/GrandTeton'


# In[5]:


path_real = path + '/Photos-20191020T020916Z-001/Photos/'
path_generated_go = path + '/GO_noGO Data Set_Images/TestGo/'
path_generated_nogo = path + '/GO_noGO Data Set_Images/TestNoGo/'


# In[6]:


## A list of files inside the 'path_generated_go' directory:
files_go = list(os.walk(path_generated_go))[0][2]
files_go[:10]


# In[7]:


len(files_go)


# ### cv2.imread function reads an image as a numpy ndarray

# In[8]:


img = cv2.imread(path_generated_go+files_go[0])
print(img)


# In[9]:


type(img)


# In[10]:


img.shape


# #### Images are 3 dimensional arrays. First two dimensions denote the pixel resolution (206x531). The last dimension is for colour depth. In cases where the image is a 2D array then it is generally a Black and White image. 

# In[11]:


img.dtype


# In[12]:


"""
cv2.imshow('image', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


# In[13]:


## we can plot the image using matplotlib but the colors are inverted so reverting back using cv2.COLOR_BGR2RGB function.
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[14]:


# 0 means black
black_img = np.zeros((206, 531, 3), dtype = 'uint8')
plt.imshow(cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[15]:


# 255 means white
white_img = 255*np.ones((206, 531, 3), dtype = 'uint8')
plt.imshow(cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[16]:


## BGR channels

# First channel Blue
blue_img = np.zeros((206, 531, 3), dtype = 'uint8')
blue_img[:,:,0] = 255*np.ones((206, 531), dtype = 'uint8')

plt.imshow(cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[17]:


# Second channel Green
green_img = np.zeros((206, 531, 3), dtype = 'uint8')
green_img[:,:,1] = 255*np.ones((206, 531), dtype = 'uint8')

plt.imshow(cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[18]:


# Third channel Red
red_img = np.zeros((206, 531, 3), dtype = 'uint8')
red_img[:,:,2] = 255*np.ones((206, 531), dtype = 'uint8')

plt.imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[19]:


plt.imshow(cv2.cvtColor(img[:,:,0], cv2.COLOR_BGR2RGB))
plt.show()


# In[20]:


plt.imshow(cv2.cvtColor(img[:,:,1], cv2.COLOR_BGR2RGB))
plt.show()


# In[21]:


plt.imshow(cv2.cvtColor(img[:,:,2], cv2.COLOR_BGR2RGB))
plt.show()


# In[22]:


## combined effect
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:





# In[ ]:




