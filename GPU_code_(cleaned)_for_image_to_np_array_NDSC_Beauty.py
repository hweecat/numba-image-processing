#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hweecat/numba-image-processing/blob/master/GPU_code_(cleaned)_for_image_to_np_array_NDSC_Beauty.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import io
import pandas as pd
import numpy as np
from PIL import Image

from google.colab import files
from google.colab import auth

import sys
import time
from retrying import retry  # retry handling
from numba import jit   # JIT processing of numpy arrays
from concurrent.futures import ProcessPoolExecutor # parallel processing

import pickle

uploaded = files.upload()

beauty_train = pd.read_csv(io.BytesIO(uploaded['beauty_data_info_train_competition.csv']))
beauty_train = pd.read_csv('beauty_data_info_train_competition.csv')

def add_jpg(imagepath):
  if '.jpg' in imagepath:
    return imagepath
  else:
    return imagepath + '.jpg'
  
def remove_jpg(imagepath):
  if '.jpg' in imagepath:
    return imagepath.replace('.jpg','')
  else:
    return imagepath

beauty_train['image_path'] = beauty_train['image_path'].apply(add_jpg)

# Authentication with GCP:
auth.authenticate_user()

# First, we need to set our project. Replace the assignment below
# with your project ID.
project_id = 'shopee-ndsc-cindyandfriends'
get_ipython().system('gcloud config set project {project_id}')

bucket_name = 'shopee-cindyandfriends'

# To create directory to (temporarily) store image files.
get_ipython().system('mkdir /content/beauty_image')

# Extract image from each image path, convert each image into matrix and reshape the matrix. Append matrix (1 x N) into list of lists.
# Let's mass-import the images:
def define_imagepath(index):
  '''Function to define image paths for each index'''
  imagepath = beauty_train.at[index, 'image_path']
  return imagepath

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def gcp_imageimport(index):
  '''Import image from GCP using image path'''
  # Create the service client.
  from googleapiclient.discovery import build
  gcs_service = build('storage', 'v1')

  from apiclient.http import MediaIoBaseDownload
  colab_imagepath = '/content/' + define_imagepath(index)

  with open(colab_imagepath, 'wb') as f:
    request = gcs_service.objects().get_media(bucket=bucket_name,
                                              object=define_imagepath(index))
    media = MediaIoBaseDownload(f, request)

    done = False
    while not done:
      _, done = media.next_chunk()
      

# Now, let's convert each image into array form, and add each image array to a list!
# Since there's limited space on Google Colab, remove image from directory after placing the array into the list.
@jit
def image_resizereshape(index):
  '''Resize and reshape image'''
  im = Image.open(define_imagepath(index))
  im = im.convert("RGB")
  im_resized = np.array(im.resize((64,64)))
  im_resizedreshaped = im_resized.reshape(64*64*3)
  
  #return im_resized
  return im_resizedreshaped

# Processing in batches + using list comprehensions:
def image_proc(image, start, end):    
  gcp_imageimport(image)
  im_resizedreshaped = image_resizereshape(image)
  if (image + 1) % 100 == 0 or (image == N - 1):
    sys.stdout.write('{0:.3f}% completed. '.format((image - start + 1)*100.0/(end - start)) + 'CPU Time elapsed: {} seconds. '.format(time.clock() - start_cpu_time) + 'Wall Time elapsed: {} seconds. \n'.format(time.time() - start_wall_time))
    time.sleep(1)
  return im_resizedreshaped

def arraypartition_calc(start):
  end = start + batch_size
  if end > N:
     end = N  
  partition_list = [image_proc(image, start, end) for image in range(start, end)]
  return partition_list


# Initialise range (start, end) of process, batch_size, to create partitions of batches for parallel GPU computation:
#N = len(beauty_train['image_path'])
N = 140000
start = 105000
batch_size = 1000
partition, mod = divmod(N, batch_size)
# partition_count = 0

# imagearray_list = [None] * partition

if mod:
  partition_index = [i*batch_size for i in range(start//batch_size,partition + 1)]
else:
  partition_index = [i*batch_size for i in range(start//batch_size,partition)]
  
partition_index


# If running out of disk space, clear folder:
get_ipython().system('rm -rf /content/beauty_image/*')


# Parallel processing:
start_cpu_time = time.clock()
start_wall_time = time.time()

with ProcessPoolExecutor() as executor:
  future = executor.map(arraypartition_calc, partition_index)

imgarray_np = np.array([x for x in future])

# Pickle dump numpy array:
filename = 'beauty_train_' + '4' + '.pkl'

with open(filename, 'wb') as f:
  pickle.dump(imgarray_np, f)


# Note on batches:
# 
# *   beauty_train_1.pkl is from index 0 to 34999 (N = 35000)
# *   beauty_train_2.pkl is from index 35000 to 69999 (N = 70000)
# *   beauty_train_3.pkl is from index 70000 to 104999 (N = 105000)
# *   beauty_train_4.pkl is from index 105000 to 139999 (N = 140000)
# *   beauty_train_5.pkl is from index 140000 to 174999 (N = 175000)
# *   beauty_train_6.pkl is from index 175000 to 209999 (N = 210000)
# *   beauty_train_7.pkl is from index 210000 to 244999 (N = 245000)
# *   beauty_train_8.pkl is from index 245000 to 286582 (N = 286583)
# 
# 