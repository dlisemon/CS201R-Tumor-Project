import nibabel as nib
import numpy as np
from skimage.transform import resize
import random
import os
import logging
import threading
import math
import pickle


logging.basicConfig(encoding='utf-8', level=logging.INFO)
lock = threading.Lock()
dataset = []

def thread_function(start_index, end_index, file_paths, labels, desired_shape):
    logging.info("Entered thread")
   
    for index in range(start_index, end_index + 1):
        volume = load_nifti_file(file_paths[index])
        volume = normalize(volume)
        volume = resize_volume(volume, desired_shape)
        slices = split_into_slices(volume)
        with lock:
            dataset.extend([(slice, labels[index]) for slice in slices])
    logging.info("FINISHED")
   

def dataset_threading(files, threads, file_paths, labels, desired_shape):

    chunk = math.floor(files/threads)
    threads_ready = []
    for i in range(threads):
       start_index = i * chunk
       end_index = (i + 1) * chunk if i < threads - 1 else files

       thread = threading.Thread(target=thread_function, args=(start_index, end_index, file_paths, labels, desired_shape))
       threads_ready.append(thread)
    for t in threads_ready:
       t.start()
    for t in threads_ready:
       t.join()
    return dataset

def load_nifti_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(volume, desired_shape):
    # Resize width, height and depth
    resized_volume = resize(volume, desired_shape, mode='constant', preserve_range=True)
    return resized_volume

def split_into_slices(volume, slice_axis=2):
    slices = []
    for i in range(volume.shape[slice_axis]):
        if slice_axis == 0:
            slice = volume[i, :, :]
        elif slice_axis == 1:
            slice = volume[:, i, :]
        else:
            slice = volume[:, :, i]
        slices.append(slice)
    return slices

def create_dataset(file_paths, labels, desired_shape):
    dataset = []
    for file_path, label in zip(file_paths, labels):
        volume = load_nifti_file(file_path)
        volume = normalize(volume)
        volume = resize_volume(volume, desired_shape)
        slices = split_into_slices(volume)
        dataset.extend([(slice, label) for slice in slices])
    return dataset

def batch_shuffle(dataset, batch_size):
    random.shuffle(dataset)
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

def getPathLabel(filePathName):
  for filename in os.listdir(filePathName):
    totalpath = filePathName + '/' + filename
    if filename[0] == ".":
      continue
    file_paths.append(totalpath)
    if "control" in filePathName:
      labels.append(0)
    else:
      labels.append(1)

#add files here
file_paths = []  # List of NIfTI file paths
labels = []      # Corresponding labels

filePathsTumor = "./tumor"
filePathsControl = "./control"



getPathLabel(filePathsControl)
getPathLabel(filePathsTumor)
logging.info("Files loaded")

# Desired shape for the CNN input
desired_shape = (256, 256, 1)

# Preprocess the data
images_dataset = dataset_threading(len(file_paths), 16, file_paths, labels, desired_shape)
logging.info("Dataset created")

with open('dataset.pkl', 'wb') as file:
    pickle.dump(images_dataset, file)
