from ctypes import *
import cv2
import numpy as np
import numpy.ctypeslib as npct
import time

class Detector():
    def __init__(self,model_path,dll_path):
        self.yolov8 = CDLL(dll_path,winmode=0)
        self.yolov8.Detect.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte),npct.ndpointer(dtype = np.float32, ndim = 2, shape = (100, 6), flags="C_CONTIGUOUS")]
        self.yolov8.Init.restype = c_void_p
        self.yolov8.Init.argtypes = [c_void_p]
        self.c_point = self.yolov8.Init(model_path)

    def predict(self,img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((100,6),dtype=np.float32)
        self.yolov8.Detect(self.c_point,c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),res_arr)

        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array

    def visualize(self,bbox_array):

        arr2 = []
        for temp in bbox_array:

            arr = [int(temp[0]), int(temp[1]), int(temp[0] + temp[2]), int(temp[1] + temp[3]), int(temp[4]), temp[5]]
            arr2.append(arr)


        return arr2