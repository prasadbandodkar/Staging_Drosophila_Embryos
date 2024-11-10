#
# This is a class to handle the files on the disc. The idea is to load a file from the id.csv file, and 
# use it to obtain synthetic images that are interpolated from two adjacent images. This class will also
# handle regression ids
#

import os, sys

import pandas as pd
import numpy as np
import cv2 as cv

# from Python.tfimage import TFImage
from Python.cvimage import CVImage
from Python.torchimage import TorchImage

import torch
from torch.utils.data import Dataset
import random

class SData:
    def __init__(self, path, test = [], val = [], ignore = []):
        self.data_path   = path
        self.train_list  = []
        self.val_list    = []
        self.test_list   = []
        self.ignore_list = []
        
        self.train_data  = {}
        self.val_data    = {}
        self.test_data   = {}
        
        self.train_test_val_data(test, val, ignore)


    def train_test_val_dir(self, test, val, ignore = []):
        '''
        test and val are lists of numerical digits that correspond to the testing and validation files
        '''
        folders = os.listdir(self.data_path)
        for folder in folders:
            if not os.path.isdir(os.path.join(self.data_path, folder)):
                continue                        # Skip files
            number_before_underscore = folder.split('_')[0]
            if not number_before_underscore.isdigit():
                continue                        # Skip if the character before the underscore is not a number
            number_before_underscore = int(number_before_underscore)
            if number_before_underscore in test:
                self.test_list.append(folder)
            elif number_before_underscore in val:
                self.val_list.append(folder)
            elif number_before_underscore in ignore:
                self.ignore_list.append(folder)
            else:
                self.train_list.append(folder)      # default is train
    
    
    def get_folder_data(self, folders):
        """
        Reads the 'id.csv' file from each specified folder, sorts it by the second column, 
        and appends the data path to the file name in the first column. 

        Args:
            folders (list): A list of folder names.

        Returns:
            dict: A dictionary where the keys are the folder names and the values are 
            pandas DataFrames containing the sorted data from the 'id.csv' files in each folder.
        """
        data = {}
        for folder in folders:
            id_file = os.path.join(self.data_path, folder, "id.csv")
            df = pd.read_csv(id_file, header=None)
            df = df.sort_values(by=1)  # type: ignore       # unfortunately the csv file is not sorted. We need to sort it here.
            # add self.data_path to the file name - the first column
            df[0] = df[0].apply(lambda x: os.path.join(self.data_path, x))
            data[folder] = df
        return data
    

    def train_test_val_data(self, test = [], val = [], ignore = []):
        if not self.train_list or not self.val_list or not self.test_list:
            self.train_test_val_dir(test, val, ignore)
        self.train_data = self.get_folder_data(self.train_list)
        self.val_data   = self.get_folder_data(self.val_list)
        self.test_data  = self.get_folder_data(self.test_list)
    
    
    def get_folder_number(self, folder):
        number = folder.split('_')[0]
        # check if the number is a digit
        if not number.isdigit():
            raise ValueError(f"Folder name '{folder}' is not in the correct format.")
        return int(number)
   
    
    def get_raw_image(self, folder, idx, list_type):
        '''
        Method to get the raw image from the folder, index, and list type.
        Raises errors if the list_type, folder, or idx are not valid.
        '''
        # Check if list_type is valid
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError(f"Invalid list_type '{list_type}'. Expected one of: {list(list_dict.keys())}")

        data = list_dict[list_type]

        # Check if folder is valid
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in {list_type} data.")

        # Check if idx is valid
        if not (0 <= idx < len(data[folder])):
            raise IndexError(f"Index {idx} is out of bounds for folder '{folder}' with size {len(data[folder])}.")

        filename = data[folder].iloc[idx, 0]
        id = data[folder].iloc[idx, 1]
        I = cv.imread(filename, cv.IMREAD_GRAYSCALE)

        # Check if the image was successfully loaded
        if I is None:
            raise FileNotFoundError(f"Image '{filename}' could not be loaded.")

        return I, id
       
            
    def get_raw_image_old(self, folder, idx, list_type):
        '''
        method to get the raw image from the folder, index and list type
        '''
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        data     = list_dict[list_type]
        filename = data[folder].iloc[idx, 0]
        id       = data[folder].iloc[idx, 1]
        I        = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        return I, id
    
    
    def get_random_image(self, list_type):
        """
        Get a random image from the specified list type.

        Args:
            list_type (str): The type of list to get the image from. Can be 'train', 'test', or 'val'.

        Returns:
            str: The path to the random image.
        """
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }

        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")

        data    = list_dict[list_type]
        rfolder = torch.randint(len(data.keys()), (1,)).item()
        # rfolder = random.choice(list(data.keys()))
        
        
        I, id, idx   = self.get_random_image_from_folder(rfolder, list_type)
        
        return I, id, rfolder, idx


    def get_random_image_from_folder(self, folder, list_type):
        """
        Get a random image from the specified list type.

        Args:
            list_type (str): The type of list to get the image from. Can be 'train', 'test', or 'val'.

        Returns:
            str: The path to the random image.
        """
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")

        data = list_dict[list_type]
        
        # Check if folder is not in data and raise an error if true
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in data.")
        
        idx = torch.randint(len(data[folder].index) - 1, (1,)).item()
        # idx = random.randint(0, len(data[folder].index) - 2)
        
        I, id = self.get_random_image_from_folder_idx(folder, idx, list_type)
        
        return I, id, idx
      

    def get_random_image_from_folder_idx(self, folder, idx, list_type):
        """
        Get a random image from the specified list type.

        Args:
            list_type (str): The type of list to get the image from. Can be 'train', 'test', or 'val'.

        Returns:
            str: The path to the random image.
        """
        I1, id1 = self.get_raw_image(folder, idx, list_type)
        I2, id2 = self.get_raw_image(folder, idx+1, list_type)
        
        # interpolate between I1 and I2. Also, we need to interpolate the id
        # get a random number between 0 and 1
        # alpha = random.uniform(0, 1)
        alpha = torch.rand(1).item()
        I     = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
        id    = alpha*id1 + (1-alpha)*id2
        
        # print id, id1, id2
        # print(f'Interpolated id: {id}, id1: {id1}, id2: {id2}')
        
        return I, id
    
    

    
# # used by tf pipeline
# class TFDataGenerator(SData):
#     def __init__(self, path, test = [], val = [], ignore = [], 
#                  size=(512, 512), padding=44, npoints=100, inward=40, outward=-24, length = None):
#         '''
#         This class is a subclass of SData. It is used to generate data for the model. 
#         It is an iterator that yields images and their corresponding ids.
#         ''' 
#         super().__init__(path, test=test, val=val, ignore=ignore)
        
#         self.size        = size
#         self.padding     = padding
#         self.npoints     = npoints
#         self.inward      = inward
#         self.outward     = outward
#         self.length      = length
        

#     def __call__(self, list_type):
#         list_dict = {   'train': self.train_data,
#                         'test': self.test_data,
#                         'val': self.val_data}
#         data = list_dict[list_type]
#         yes_shuffle = False
#         if list_type is 'train':
#             yes_shuffle = True
#         if yes_shuffle:               # Shuffle the folders only if list_type is 'train'
#             shuffled_data = list(data.items())
#             random.shuffle(shuffled_data)
#         else:
#             shuffled_data = data.items()

#         yes_shuffle = True
#         print(f"yes_shuffle: {yes_shuffle}")
#         print(f"list_type: {list_type}")

#         for folder, df in shuffled_data:
#             # Generate a list of indices based on the DataFrame's length
#             # And adjusted to avoid running over the last element
#             indices = list(range(len(df.index) - 1)) 
#             if yes_shuffle: 
#                 random.shuffle(indices) 
            
#             # Iterate over the DataFrame using the shuffled indices
#             for idx in indices:
#                 I, id = self.get_random_image_from_folder_idx(folder, idx, list_type)
#                 image = TFImage(CVImage(I, 
#                                         id, 
#                                         self.size, 
#                                         self.padding, 
#                                         False, 
#                                         self.npoints, 
#                                         self.inward, 
#                                         self.outward,
#                                         self.length).image, id)
#                 if list_type in ['train', 'val']:
#                     image.augment(seed=(random.randint(0, 1000), random.randint(0, 1000)))
#                 image.normalize()
#                 yield image.I, image.id
            
#     def __iter__(self, list_type):
#         return self(list_type)
    
#     def __next__(self, list_type):
#         return next(self(list_type))
        



# used by torch pipeline
class TorchDataset(SData, Dataset):
    def __init__(self, path, test=[], val=[], ignore=[], size=(512, 512), padding=44, npoints=100, inward=40, outward=-24, trunc_width=None, type='train'):
        super().__init__(path, test=test, val=val, ignore=ignore)
        self.size      = size
        self.padding   = padding
        self.npoints   = npoints
        self.inward    = inward
        self.outward   = outward
        self.trunc_width = trunc_width
        self.list_type = None
        self.data      = None
        self.indices   = None
        self.type      = type
        
        self.set_list_type(type)

    def set_list_type(self, list_type):
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")
        
        self.list_type = list_type
        self.data      = list_dict[list_type]
        self.indices   = [(folder, idx) for folder, df in self.data.items() for idx in range(len(df.index) - 1)]
        if list_type == 'train':
            random.shuffle(self.indices)
            # self.indices = [self.indices[i] for i in torch.randperm(len(self.indices))]

    def __len__(self):
        if self.indices is None:
            raise ValueError("List type not set. Call set_list_type() before using the dataset.")
        return len(self.indices)

    def __getitem__(self, index):
        if self.indices is None:
            raise ValueError("List type not set. Call set_list_type() before using the dataset.")
        
        folder, idx = self.indices[index]
        I, id = self.get_random_image_from_folder_idx(folder, idx, self.list_type)
        
        # Create CVImage instance with all parameters
        cv_image = CVImage(
            I=I,
            id=id,
            size=self.size,
            padding=self.padding,
            plot_images=False,
            npoints=self.npoints,
            inward=self.inward,
            outward=self.outward,
            trunc_width=self.trunc_width
        )
        
        # Create TorchImage from the processed image
        image = TorchImage(np.array(cv_image.image, dtype=np.float32), id)
        
        if self.list_type in ['train', 'val']:
            # image.I = image.augment(seed=(random.randint(0, 1000), random.randint(0, 1000)))
            # image.I = image.augment(seed=(torch.randint(0, 1000, (1,)).item(), torch.randint(0, 1000, (1,)).item()))
            pass
        
        image.I = image.normalize()
        return image.I, image.id
        
        

if __name__ == "__main__":
    path = "/Volumes/X2/Projects/staging/Data/data"
    f = SData(path)


    # create train, test, and val
    test = [6,7]
    val  = [21, 34]
    f.train_test_val_dir(test, val)
    # print test_list
    # print(f.test_list)
    # print(f.val_list)
    # print(f.train_list)
    
    
    # get folder data
    f.train_test_val_data()
    
    f.get_random_image('train')
    
    
    