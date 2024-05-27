#
# This is a class to handle the files on the disc. The idea is to load a file from the id.csv file, and 
# use it to obtain synthetic images that are interpolated from two adjacent images. This class will also
# handle regression ids
#

import os, sys
import random

import pandas as pd
import cv2 as cv
import tensorflow as tf

class Data:
    def __init__(self, path):
        self.data_path  = path
        
        self.train_list = []
        self.val_list   = []
        self.test_list  = [] 
        
        self.train_data = {}
        self.val_data   = {}
        self.test_data  = {}   


    def train_test_val_dir(self, test, val):
        '''
        test and val are lists of numerical digits that correspond to the testing and validation files
        '''
        folders = os.listdir(self.data_path)
        for folder in folders:
            if not os.path.isdir(os.path.join(self.data_path, folder)):
                continue  # Skip files
            number_before_underscore = folder.split('_')[0]
            if not number_before_underscore.isdigit():
                continue  # Skip if the character before the underscore is not a number
            number_before_underscore = int(number_before_underscore)
            if number_before_underscore in test:
                self.test_list.append(folder)
            elif number_before_underscore in val:
                self.val_list.append(folder)
            else:
                self.train_list.append(folder)
    
    
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
    

    def train_test_val_data(self):
        self.train_data = self.get_folder_data(self.train_list)
        self.val_data   = self.get_folder_data(self.val_list)
        self.test_data  = self.get_folder_data(self.test_list)

            
    def get_cvimage(self, folder, idx, list_type):
        # first get the image idx from the folder reading the csv file
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        
        data = list_dict[list_type]
        
        filename1 = data[folder].iloc[idx, 0] 
        id1 = data[folder].iloc[idx, 1]
        I1 = cv.imread(filename1, cv.IMREAD_GRAYSCALE)
        
        filename2 = data[folder].iloc[idx+1, 0]
        id2 = data[folder].iloc[idx+1, 1]
        I2 = cv.imread(filename2, cv.IMREAD_GRAYSCALE)
        
        # interpolate between I and I2. Also, we need to interpolate the id
        # get a random number between 0 and 1
        alpha = random.uniform(0, 1)
        I = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
        id = alpha*id1 + (1-alpha)*id2
        
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

        data = list_dict[list_type]
        rfolder = random.choice(list(data.keys()))
        idx = random.randint(0, len(data[rfolder].index) - 1)
        
        # print(len(data[rfolder].index))
        # print(f'Random folder: {rfolder}, Random idx: {idx}')
        
        I, id = self.get_cvimage(rfolder, idx, list_type)
        
        return I, id
        
    
    def make_tf_data(self, list_type):
        
        
        

if __name__ == "__main__":
    path = "/Volumes/X2/Projects/staging/Data/data"
    f = Data(path)


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
    
    
    