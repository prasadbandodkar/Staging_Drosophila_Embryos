#
# This is a class to handle the files on the disc. The idea is to load a file from the id.csv file, and 
# use it to obtain synthetic images that are interpolated from two adjacent images. This class will also
# handle regression ids
#

import os, sys
import random

import pandas as pd
import cv2 as cv

from tfimage import TFImage
from cvimage import CVImage

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
        rfolder = random.choice(list(data.keys()))
        idx     = random.randint(0, len(data[rfolder].index) - 2)
        
        # print(len(data[rfolder].index))
        # print(f'Random folder: {rfolder}, Random idx: {idx}')
        
        I1, id1 = self.get_raw_image(rfolder, idx, list_type)
        I2, id2 = self.get_raw_image(rfolder, idx+1, list_type)
        
        # interpolate between I1 and I2. Also, we need to interpolate the id
        # get a random number between 0 and 1
        alpha = random.uniform(0, 1)
        I     = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
        id    = alpha*id1 + (1-alpha)*id2
        
        # print id, id1, id2
        # print(f'Interpolated id: {id}, id1: {id1}, id2: {id2}')
        
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
        
        idx = random.randint(0, len(data[folder].index) - 2)
        
        # print(len(data[rfolder].index))
        # print(f'Random folder: {rfolder}, Random idx: {idx}')
        
        I1, id1 = self.get_raw_image(folder, idx, list_type)
        I2, id2 = self.get_raw_image(folder, idx+1, list_type)
        
        # interpolate between I1 and I2. Also, we need to interpolate the id
        # get a random number between 0 and 1
        alpha = random.uniform(0, 1)
        I     = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
        id    = alpha*id1 + (1-alpha)*id2
        
        # print id, id1, id2
        # print(f'Interpolated id: {id}, id1: {id1}, id2: {id2}')
        
        return I, id
      

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
        alpha = random.uniform(0, 1)
        I     = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
        id    = alpha*id1 + (1-alpha)*id2
        
        # print id, id1, id2
        # print(f'Interpolated id: {id}, id1: {id1}, id2: {id2}')
        
        return I, id
    
    

    
# used by tf pipeline
class DataGenerator(SData):
    def __init__(self, path, test = [], val = [], ignore = [], 
                 size=(512, 512), padding=44, npoints=100, inward=40, outward=-24, length = None):
        '''
        This class is a subclass of SData. It is used to generate data for the model. 
        It is an iterator that yields images and their corresponding ids.
        ''' 
        super().__init__(path, test=test, val=val, ignore=ignore)
        
        self.size        = size
        self.padding     = padding
        self.npoints     = npoints
        self.inward      = inward
        self.outward     = outward
        self.length      = length
        

    def __call__(self, list_type):
        list_dict = {   'train': self.train_data,
                        'test': self.test_data,
                        'val': self.val_data}
        data = list_dict[list_type]
        yes_shuffle = False
        if list_type is 'train':
            yes_shuffle = True
        if yes_shuffle:               # Shuffle the folders only if list_type is 'train'
            shuffled_data = list(data.items())
            random.shuffle(shuffled_data)
        else:
            shuffled_data = data.items()

        yes_shuffle = True
        print(f"yes_shuffle: {yes_shuffle}")
        print(f"list_type: {list_type}")

        for folder, df in shuffled_data:
            # Generate a list of indices based on the DataFrame's length
            # And adjusted to avoid running over the last element
            indices = list(range(len(df.index) - 1)) 
            if yes_shuffle: 
                random.shuffle(indices) 
            
            # Iterate over the DataFrame using the shuffled indices
            for idx in indices:
                I, id = self.get_random_image_from_folder_idx(folder, idx, list_type)
                image = TFImage(CVImage(I, 
                                        id, 
                                        self.size, 
                                        self.padding, 
                                        False, 
                                        self.npoints, 
                                        self.inward, 
                                        self.outward,
                                        self.length).image, id)
                if list_type in ['train', 'val']:
                    image.augment(seed=(random.randint(0, 1000), random.randint(0, 1000)))
                image.normalize()
                yield image.I, image.id
            
    def __iter__(self, list_type):
        return self(list_type)
    
    def __next__(self, list_type):
        return next(self(list_type))
        



        

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
    
    
    