import os
from datetime import datetime as dt
import cv2

def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_var(timestamp = dt.now().strftime("%Y-%m-%d-%H:%M:%S"))
def getTimestamp():
    return getTimestamp.timestamp

# If there is not input name, create the directory name with timestamp
def createNewDir(root_path, name=None):

    if name == None:
        print("[utils.py, createNewDir()] DirName is not defined in the arguments, define as timestamp")
        newpath = os.path.join(root_path, getTimestamp())
    else:
        newpath = os.path.join(root_path, name)

    """Create parent path if it doesn't exist"""
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    return newpath

def createTrainValidationDirpath(root_dir, createDir = False):
    
    if createDir == True:
        train_dir = createNewDir(root_dir, "train")
        val_dir = createNewDir(root_dir, "val")
    
    else:
        train_dir = os.path.join(root_dir, "train")
        val_dir = os.path.join(root_dir, "val") 

    return train_dir, val_dir

def writeHDR(arr, outfilename, imgshape):

    ext_name = outfilename.split(".")[1]
    
    if ext_name == "hdr":
        cv2.imwrite(outfilename, arr.copy())
