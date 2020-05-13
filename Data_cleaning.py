#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import glob
from PIL import Image


# In[2]:


pwd


# ## Process the images

# ## Method 1:   No Augmentation
# 
# *source:* https://kapernikov.com/tutorial-image-classification-with-scikit-learn/ 
# 
# **Steps**
# 
# 1. Import all picture
# 2. Resize to the largest picture
# 3. Color to grayscale
# 4. Crop 
# 5. HOG transform

# In[8]:


covid_list = os.listdir('CT_COVID')
nocovid_list = os.listdir('CT_NonCOVID')


# **Correct file type, Resize and Import**

# In[6]:


# Resize the original images to be the same as the largest image (ONE TIME PROCESSING)
#for filename in glob.glob('CT_COVID/*.png'):  
#    im1 = Image.open(filename)
#    prefix = filename.split(".png")[0]
#    imResize=im1.resize((1671,1225)) #largest width and height
#    imResize.save(prefix+'.png')  

# Resize and convert "jpg" to "png" and “save-as” (ONE TIME PROCESSING)
#for filename in glob.glob('CT_NonCOVID/*.jpg'):  
#    im1 = Image.open(filename)
#    prefix = filename.split(".jpg")[0]
#    im1.save(prefix+'.png')  
    
#for filename in glob.glob('CT_NonCOVID/*.png'):  
#    im1 = Image.open(filename)
#    prefix = filename.split(".png")[0]
#    imResize=im1.resize((1050,830)) #largest width and height
#    imResize.save(prefix+'.png')      


# **RGB to Grayscale, Crop, HOG transform**

# In[ ]:


# read in covid CT
covid_list = []
covid_processed = []
for filename in glob.glob('CT_COVID/*.png'): 
    im=plt.imread(filename)
    covid_list.append(im)

min_width=200
min_length=400

for i in range(len(covid_list)):
    #RGB to grayscale
    if len(covid_list[i].shape)==3:
        grey = rgb2gray(covid_list[i])
    else:
        grey=covid_list[i]
    c = cropR(grey,min_length,min_width)  #no cropping is done since all pictures are resized
    g = grad(c,min_length-2,min_width-2,False)
    h = hog(g[0],g[1],12,12,9)
    covid_processed.append(h)
covid_processed = np.array(covid_processed)
covid_processed.shape

    
# Find the max dimension for covid pictures to crop later: used for resizing 
#a = np.empty(349, dtype=object) 
#for i in range(len(covid_list)):
#     a[i] = covid_list[i].shape[0]
        
#b = np.empty(349, dtype=object) 
#for i in range(len(covid_list)):
#     b[i] = covid_list[i].shape[1]
        
#max_length = max(a)
#max_width = max(b)
#print("the max length for covid pictures is",max_length)
#print("the max width for covid pictures is",max_width)


# In[30]:


# read in non-covid CT
nocovid_list = []
nocovid_processed = []
for filename in glob.glob('CT_NonCOVID/*.png'): 
    im=plt.imread(filename,0)
    nocovid_list.append(im)

min_width=200
min_length=400


for i in range(len(nocovid_list)):
    #RGB to grayscale
    if len(nocovid_list[i].shape)==3:
        grey = rgb2gray(nocovid_list[i])
    else:
        grey=nocovid_list[i]
    c = cropR(grey,min_length,min_width)  #no cropping is done since all pictures are resized
    g = grad(c,min_length-2,min_width-2,False)
    h = hog(g[0],g[1],12,12,9)
    nocovid_processed.append(h)
nocovid_processed = np.array(nocovid_processed)
nocovid_processed.shape


# Find the max dimension for non-covid picturesto crop later: used for resizing
#c = np.empty(397, dtype=object) 
#for i in range(len(nocovid_list)):
#     c[i] = nocovid_list[i].shape[0]
        
#d = np.empty(397, dtype=object) 
#for i in range(len(nocovid_list)):
#     d[i] = nocovid_list[i].shape[1]
        
#max_length_non_covid = max(c)
#max_width_non_covid = max(d)
#print("the max length for non-covid pictures is",max_length_non_covid)
#print("the max width for non-covid pictures is",max_width_non_covid)


# **Combine images with patients ID**

# In[31]:


covid_info = pd.read_csv('COVID-CT-MetaInfo.csv')
nocovid_info = pd.read_csv('NonCOVID-CT-MetaInfo.csv')


# In[32]:


covid_data = covid_info.iloc[:,2:5]     # extract covid patient id & covariate information


# In[33]:


nocovid_data = nocovid_info.iloc[:,4] # extract non-covid patient id


# In[ ]:


covid_comp_data = np.column_stack((covid_processed,covid_data,np.full((len(covid_data), 2), 'Na')))
# covid_comp_data = pd.DataFrame(covid_comp_data)
# covid_comp_data.head()


# In[35]:


nocovid_comp_data = np.column_stack((nocovid_processed,nocovid_data,np.full((len(nocovid_data), 2), 'Na')))
# nocovid_comp_data = pd.DataFrame(nocovid_comp_data)
# nocovid_comp_data.head()


# In[36]:


Y = np.concatenate((np.repeat(1,covid_processed.shape[0]),np.repeat(-1,nocovid_processed.shape[0]))) # create label


# In[21]:


mydata = np.row_stack((covid_comp_data,nocovid_comp_data))
mydata = np.column_stack((mydata,Y))
mydata = pd.DataFrame(mydata)


# ### Check for batch effect

# In[36]:


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

pixels = np.row_stack((covid_processed,nocovid_processed))
kmeans_kpp_1 = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0).fit(pixels)
kmeans_kpp_labels_1 = kmeans_kpp_1.labels_
kmeans_kpp_labels_1

from numpy import savetxt
savetxt('/Users/chloehe/Desktop/kmeans_kpp_labels_1.csv', kmeans_kpp_labels_1, delimiter=',')


# In[38]:


size=mydata.shape[1]


mydata.rename(columns={mydata.columns[size-4]: "patient_id", 
                       mydata.columns[size-3]: "age", 
                       mydata.columns[size-2]: "gender", 
                       mydata.columns[size-1]: "label" }, inplace = True)
mydata.shape


# In[39]:


mydata.head()


# 
# **Keep all data**

# In[40]:


# change directory to your own
mydata.to_csv('mydata_all.csv', index = False)


# **Only randomly keep one image if multiple images for single patient**

# In[41]:


# randomly select one observation if this patient has more than one pictures taken
blocks = [data.sample(n=1) for _,data in mydata.groupby(['patient_id', 'label'])]
mydata = pd.concat(blocks)
mydata.head()
mydata.shape


# In[42]:


# change directory to your own
mydata.to_csv('mydata_unique.csv', index = False)


# In[2]:


## Possible Data Augmentation


# ## Method 2:   Data Augmentation
# 
# **Source:** https://github.com/mdbloice/Augmentor
# 
# **Need to install Augmentor package manually with pip**
# 
# Data Augmentation only on the training set, so we will have 4 folders
# 
# (1) COVID_train (label: np.repeat(1,COVID_train.shape[0]))
# 
# (2) nonCOVID_train
# 
# (3) COVID_test
# 
# (4) nonCOVID_test

# ### First split folders (one time implementation)

# In[4]:


import os
import shutil
import numpy as np

#pip install Augmentor
import Augmentor

#sourceP = '/Users/chloehe/Desktop/images/' + "/train/CT_COVID"    # this one had COVID_CT, will later be train COVID_CT
#destP = '/Users/chloehe/Desktop/images/' + "/test/CT_COVID"       # this WILL have test COVID_CT
#sourceN = '/Users/chloehe/Desktop/images/' + "/train/CT_NonCOVID" # this one had non_COVID_CT, will later be train non_COVID_CT
#destN = '/Users/chloehe/Desktop/images/' + "/test/CT_nonCOVID"    # this WILL have test non_COVID_CT

sourceP ="augment/train/CT_COVID"    # this one had COVID_CT, will later be train COVID_CT
destP = "augment/test/CT_COVID"       # this WILL have test COVID_CT
sourceN = "augment/train/CT_NonCOVID" # this one had non_COVID_CT, will later be train non_COVID_CT
destN = "augment/test/CT_nonCOVID"    # this WILL have test non_COVID_CT


filesP = os.listdir(sourceP)
filesN = os.listdir(sourceN)       

#for f in filesP:
#    if np.random.rand(1) < 0.2:
#        shutil.move(sourceP + '/'+ f, destP + '/'+ f)

#for i in filesN:
#    if np.random.rand(1) < 0.2:
#        shutil.move(sourceN + '/'+ i, destN + '/'+ i)

train_covid=len(os.listdir(sourceN))
train_nocovid=len(os.listdir(sourceP))
test_covid=len(os.listdir(destN))
test_nocovid=len(os.listdir(destP))

print(len(os.listdir(sourceN))) # train non-covid
print(len(os.listdir(sourceP))) # train covid
print(len(os.listdir(destN)))   # test non-covid
print(len(os.listdir(destP)))   # test covid     


# ### Augment image and output (one time implementation)
# 
# **rotate(80%)**
# 
# **slip from left to right (50%)**
# 
# **zoom 80% of the area (50%)**
# 
# **flip top to bottom (50%)**

# In[5]:


#augment training data for covid 
#source: https://github.com/mdbloice/Augmentor

#give three replicates
#for i in range(3):
#    p = Augmentor.Pipeline("augment/train/CT_COVID") 
#    p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)
#    p.flip_left_right(probability=0.5)
#    p.zoom_random(probability=0.5, percentage_area=0.8)
#    p.flip_top_bottom(probability=0.5)
#    p.process()


# In[7]:


#augment training data for nocovid 

#give three replicates
for i in range(3):
    p = Augmentor.Pipeline("augment/train/CT_NonCOVID") 
    p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.process()


# ### HOG procedure 

# In[4]:


#convert to PNG and resize (one time implementation) 
for filename in glob.glob('augment/train/CT_COVID/*.png'):  
   im1 = Image.open(filename)
   suffix = filename.split("CT_COVID/")[1]
   imResize=im1.resize((1671,1225)) #largest width and height
   imResize.save('augment/train/CT_COVID/'+suffix+'.png')  

for filename in glob.glob('augment/train/CT_NonCOVID/*.png'):  
   im1 = Image.open(filename)
   suffix = filename.split("CT_NonCOVID/")[1]
   imResize=im1.resize((1050,830)) #largest width and height
   imResize.save('augment/train/CT_NonCOVID/'+suffix+'.png')  

    # Resize and convert "jpg" to "png" and “save-as” (ONE TIME PROCESSING)
for filename in glob.glob('augment/test/CT_COVID/*.png'):  
   im1 = Image.open(filename)
   suffix = filename.split("CT_COVID/")[1]
   imResize=im1.resize((1671,1225)) #largest width and height
   imResize.save('augment/test/CT_COVID/'+suffix)     

for filename in glob.glob('augment/test/CT_NonCOVID/*.png'):  
   im1 = Image.open(filename)
   suffix = filename.split("CT_NonCOVID/")[1]
   imResize=im1.resize((1050,830)) #largest width and height
   imResize.save('augment/test/CT_NonCOVID/'+suffix)      


# In[ ]:


# read in covid CT train
covid_train_list = []
covid_train_processed = []
for filename in glob.glob('augment/train/CT_COVID/*.png'): 
    im=plt.imread(filename)
    covid_train_list.append(im)


# In[8]:


covid_train_label = np.repeat(1,covid_train_processed.shape[0])


# In[9]:


# read in non-covid CT train
nocovid_train_list = []
nocovid_train_processed = []
for filename in glob.glob('augment/train/CT_NonCOVID/*.png'): 
    im=plt.imread(filename,0)
    nocovid_train_list.append(im)

min_width=200
min_length=400


for i in range(len(nocovid_train_list)):
    #RGB to grayscale
    if len(nocovid_train_list[i].shape)==3:
        grey = rgb2gray(nocovid_train_list[i])
    else:
        grey=nocovid_train_list[i]
    c = cropR(grey,min_length,min_width)  #no cropping is done since all pictures are resized
    g = grad(c,min_length-2,min_width-2,False)
    h = hog(g[0],g[1],12,12,9)
    nocovid_train_processed.append(h)
nocovid_train_processed = np.array(nocovid_train_processed)
nocovid_train_processed.shape


# In[10]:


nocovid_train_label = np.repeat(-1,nocovid_train_processed.shape[0])


# In[11]:


# read in covid CT test
covid_test_list = []
covid_test_processed = []
for filename in glob.glob('test/CT_COVID/*.png'): 
    im=plt.imread(filename)
    covid_test_list.append(im)

min_width=200
min_length=400

for i in range(len(covid_test_list)):
    #RGB to grayscale
    if len(covid_test_list[i].shape)==3:
        grey = rgb2gray(covid_test_list[i])
    else:
        grey=covid_test_list[i]
    c = cropR(grey,min_length,min_width)  #no cropping is done since all pictures are resized
    g = grad(c,min_length-2,min_width-2,False)
    h = hog(g[0],g[1],12,12,9)
    covid_test_processed.append(h)
covid_test_processed = np.array(covid_test_processed)
covid_test_processed.shape


# In[12]:


covid_test_label = np.repeat(1,covid_test_processed.shape[0])


# In[13]:


# read in non-covid CT test
nocovid_test_list = []
nocovid_test_processed = []
for filename in glob.glob('test/CT_NonCOVID/*.png'): 
    im=plt.imread(filename,0)
    nocovid_test_list.append(im)

min_width=200
min_length=400


for i in range(len(nocovid_test_list)):
    #RGB to grayscale
    if len(nocovid_test_list[i].shape)==3:
        grey = rgb2gray(nocovid_test_list[i])
    else:
        grey=nocovid_test_list[i]
    c = cropR(grey,min_length,min_width)  #no cropping is done since all pictures are resized
    g = grad(c,min_length-2,min_width-2,False)
    h = hog(g[0],g[1],12,12,9)
    nocovid_test_processed.append(h)
nocovid_test_processed = np.array(nocovid_test_processed)
nocovid_test_processed.shape


# In[1]:


nocovid_test_label = np.repeat(-1,nocovid_test_processed.shape[0])


# ### Export data

# In[16]:


## Train and Test
X_train = np.row_stack((covid_train_processed,nocovid_train_processed))
X_test = np.row_stack((covid_test_processed,nocovid_test_processed))

y_train = np.concatenate((covid_train_label,nocovid_train_label))
y_test = np.concatenate((covid_test_label,nocovid_test_label))

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train.to_csv('/X_train.csv', index = False)
X_test.to_csv('/X_test.csv', index = False)

from numpy import savetxt

savetxt('/y_train.csv', y_train, delimiter=',')
savetxt('/y_test.csv', y_test, delimiter=',')


# ## The following are debug coding

# In[27]:


#debug import (there was valueError, suspecting some images were originally non-PNG file)
covid_pix=[]
error_indicator=[]
error_pix=[]
i=0
for im in covid_list:
    i+=1
    try:
        covid_pix.append(plt.imread('CT_COVID/'+im))
    except:
        error_indicator.append(i)
        error_pix.append(im)
        pass
    
error_pix_table=pd.DataFrame(np.array(error_indicator), np.array(error_pix))    


# In[24]:


#no error importing in non-cases
nocovid_pix=[]
error_indicator=[]
error_pix=[]
i=0
for im in noncovid:
    i+=1
    try:
        nocovid_pix.append(plt.imread('CT_NonCOVID/'+im))
    except:
        error_indicator.append(i)
        error_pix.append(im)
        pass


# In[29]:


for i in range(len(covid_pix)):
    print(covid_pix[i].shape)

