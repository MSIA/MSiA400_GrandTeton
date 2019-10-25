#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[2]:


path = '../../team/courses/MSiA400/GrandTeton'


# In[3]:


path_real = path + '/Photos-20191020T020916Z-001/Photos/'
path_generated_go = path + '/GO_noGO Data Set_Images/TestGo/'
path_generated_nogo = path + '/GO_noGO Data Set_Images/TestNoGo/'


# In[4]:


files = list(os.walk(path))[0][2]
files


# ### Reading the csv file

# ##### The file 'Building_Set_Balanced_10000_Generated_9_MAY_2019.csv' has data about the images inside the Go and NoGo folders. Each of the images inside the Go and NoGo folders has a specific row correspoiding to it inside the csv file. The rows and images are linked in the following way:
# The index of the csv is embedded in file name. For example, the index 1007 corresponds to the file named 'Img1007.png'.

# In[5]:


df = pd.read_csv(path+'/Building_Set_Balanced_10000_Generated_9_MAY_2019.csv')
df


# In[6]:


files_go = list(os.walk(path_generated_go))[0][2]
files_go[:10]


# In[7]:


files_nogo = list(os.walk(path_generated_nogo))[0][2]
files_nogo[:10]


# ### Extracting the indices out of the file names.

# In[8]:


files_go_idx = []
for file in files_go:
    if "Img" in file:
        files_go_idx.append(int(file.split("Img")[1].split(".")[0]))
files_go_idx[:10]


# In[9]:


len(files_go_idx)


# In[10]:


files_nogo_idx = []
for file in files_nogo:
    if "Img" in file:
        files_nogo_idx.append(int(file.split("Img")[1].split(".")[0]))
files_nogo_idx[:10]


# In[11]:


len(files_nogo_idx)


# In[12]:


print(sorted(files_go_idx)[:100])


# In[13]:


print(max(files_go_idx))


# In[14]:


print(sorted(files_nogo_idx)[:100])


# In[15]:


print(max(files_nogo_idx))


# In[16]:


df_go = []
for i in sorted(files_go_idx):
    df_go.append(df.iloc[i-1])
df_go = pd.DataFrame(df_go) 
df_go


# In[17]:


df_nogo = []
for i in sorted(files_nogo_idx):
    df_nogo.append(df.iloc[i-1])
df_nogo = pd.DataFrame(df_nogo) 
df_nogo


# ## Testing whether the labels are consistent

# In[18]:


count = 0
mismatch_count = 0
for i in range(3370):
    if df_go.iloc[i]['Go'] == True:
        count+=1
    if df_go.iloc[i]['Go'] == False:
        mismatch_count+=1
        print(df_go.iloc[i])


# In[19]:


print(count)
print(mismatch_count)


# In[20]:


count = 0
mismatch_count = 0
for i in range(5001):
    if df_nogo.iloc[i]['Go'] == False:
        count+=1
    if df_nogo.iloc[i]['Go'] == True:
        mismatch_count+=1
        print(df_nogo.iloc[i])


# In[21]:


print(count)
print(mismatch_count)


# #### Observation: the file "Img8327.png" which is located in the NoGo folder is actually labelled as a Go file in the csv.
# #### Since the label is not clear, we will omit this file out.

# In[22]:


df_nogo = df_nogo[0:5000]
df_nogo


# ### Printing a list of the file names with the attached labels.

# In[23]:


file_label = []
for file in files_go:
    file_label.append((file,"Go"))


# In[24]:


for file in files_nogo:
    if file != "Img8237.png":
        file_label.append((file,"NoGo"))


# In[25]:


print(file_label)


# In[ ]:




