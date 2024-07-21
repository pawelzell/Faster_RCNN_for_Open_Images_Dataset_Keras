# -*- coding: utf-8 -*-

import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from shutil import copyfile
import shutil
import sys
import time
import math
import tensorflow as tf


def process_labels(labels, image_size=1024., position_type='float'):
    labels.loc[labels['Abnormality'] == 'NORM', 'X'] = 511
    labels.loc[labels['Abnormality'] == 'NORM', 'Y'] = 511
    labels.loc[labels['Abnormality'] == 'NORM', 'Radius'] = 511
    labels.loc[labels['Abnormality'] == 'NORM', 'Severity'] = 'bg'
    
    labels['XMin'] = (labels['X'] - labels['Radius']) / image_size
    labels['XMax'] = (labels['X'] + labels['Radius']) / image_size
    
    labels['YMin'] = (image_size - labels['Y'] - labels['Radius']) / image_size
    labels['YMax'] = (image_size - labels['Y'] + labels['Radius']) / image_size
    labels['subset'] = 'train'
    return labels


def annotate_train_and_test(labels, test_size=0.2, validation_size=0.2, random_size=42):
    # Get unique values from the 'MIAS name' column
    unique_mias_names = labels['MIAS name'].unique()

    # Split the unique values into 80% train and 20% test
    train_names, test_names = train_test_split(unique_mias_names, test_size=test_size, random_state=random_size)
    labels.loc[labels['MIAS name'].isin(test_names), 'subset'] = 'test'
    labels.loc[labels['MIAS name'].isin(train_names),'subset'] = 'train'
    
    if validation_size is not None:
       validation_size = validation_size / (1.0 - test_size)
       unique_mias_names = labels.loc[labels['subset'] == 'train']['MIAS name'].unique()
       _, validation_names = train_test_split(unique_mias_names, test_size=validation_size, random_state=random_size)
       labels.loc[labels['MIAS name'].isin(validation_names),'subset'] = 'validation'
    
    return labels    


def get_src_img_path(img_id, image_dir='mias', img_ext='.pgm'):
    return os.path.join(image_dir, img_id + img_ext)


# TODO move img_ext to constants
def copy_imgs_to_subset_dirs(labels, dry_run=False, train_path='train', validation_path='validation', test_path='test', img_ext='.pgm'):
    def copy_files(row):
        img_id = row['MIAS name']
        subset = row['subset']
        src_img_path = get_src_img_path(img_id)

        if subset == 'train':
            dest_file = os.path.join(train_path, img_id + img_ext)
        elif subset == 'validation':
            dest_file = os.path.join(validation_path, img_id + img_ext)
        elif subset == 'test':
            dest_file = os.path.join(test_path, img_id + img_ext)
        else:
            raise Exception(f"Unrecognized subset {subset}")

        # Copy the file
        if not os.path.exists(src_img_path):
            print(f"Skip {img_id} for {subset} subset - src file not found")
            return
        if os.path.exists(dest_file):
            print(f"Skip {img_id} for {subset} subset - dst file exists")
            return
        
        if dry_run:
            print(f"Dry run skip copping {src_img_path} {dest_file} {img_id} - {subset}")
        else:
            shutil.copy(src_img_path, dest_file)
            print(f"Copping {src_img_path} {dest_file} {img_id} - {subset}")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Apply the function to each row in the DataFrame
    labels.apply(copy_files, axis=1)


def write_annotation_txt(directory, annotation_filename, labels, width=1024, heigth=1024, skip_bg=False):
    with open(os.path.join(directory, annotation_filename), "w+") as f:
        for _, row in labels[labels['Severity'].notnull()].iterrows():
            if skip_bg and row['Severity'] == 'bg':
                print(f"skip example {labels['MIAS name']}")
                continue
            if math.isnan(row['XMin']):
                continue
            x1 = int(row['XMin'] * width)
            x2 = int(row['XMax'] * width)
            y1 = int(row['YMin'] * heigth)
            y2 = int(row['YMax'] * heigth)
            f.write(row['MIAS name'] + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + row['Severity'] + '\n')
    print(f"annotation file {annotation_filename} written")

