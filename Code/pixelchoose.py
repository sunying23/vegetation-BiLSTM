# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:08:32 2019

@author: LDZ
"""

from shutil import copyfile
from sys import exit
import random
import os

source = "/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Data/2009/csv_pixel_2009"
target = "/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Data/2009/train_500"
test = "/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Data/2009/test_20"

filenames = os.listdir(source)
length = len(filenames)
index = [i for i in range(length)]
random.shuffle(index)

for i in range(500):
    fnum = index[i]
    in_fname = source + '/' + filenames[fnum]
    out_fname = target + '/' + filenames[fnum]
    try:
       copyfile(in_fname, out_fname)
    except IOError as e:
       print("Unable to copy file. %s" % e)
       exit(1)
    except:
       print("Unexpected error:", sys.exc_info())
       exit(1)


for i in range(20):
    fnum = index[i+500]
    in_fname = source + '/' + filenames[fnum]
    out_fname = test + '/' + filenames[fnum]
    try:
       copyfile(in_fname, out_fname)
    except IOError as e:
       print("Unable to copy file. %s" % e)
       exit(1)
    except:
       print("Unexpected error:", sys.exc_info())
       exit(1)

print("\nFile copy done!\n")
