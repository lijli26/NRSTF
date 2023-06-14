import numpy as np
from osgeo import gdal
import os 

def resize(img):
        (n,h,w) = img.shape
        ex = img[0].reshape(h,w,1)
        for i in range(1,n):
            
            ex2 = img[i].reshape(h,w,1)
            ex = np.concatenate((ex,ex2),axis=2)
        return ex
    
def imread(path,startX=0,startY=0,X=0,Y=0 ):
    
    ds = gdal.Open(path)
    im_width = ds.RasterYSize
    im_height = ds.RasterXSize

    if X!=0:
        im_height = X
    if Y!=0:
        im_width = Y

    img = np.array(ds.ReadAsArray(startX,startY,im_height,im_width))
    img = img.astype(np.float)
    return img

def imsave(img,path,Dtype):
    if len(img.shape) == 3:
        (n,h,w) = img.shape
    else:
        (h,w) = img.shape
        n = 1
    driver = gdal.GetDriverByName("GTiff")
    '''
    if 'uint8' in img.dtype.name:
        datatype = gdal.GDT_Byte    
            
    elif 'uint16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    '''
    if Dtype=='uint8':
        datatype = gdal.GDT_Byte
    elif Dtype=='uint16':
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    dataset = driver.Create(path, w,h,n, datatype)
    if len(img.shape) == 3:
        for t in range(n):
            dataset.GetRasterBand(t+1).WriteArray(img[t])
    else:
        dataset.GetRasterBand(1).WriteArray(img)
    
    del dataset

def read_multi_images(path):
    
    lista=os.listdir(path)
    M=[]
    for k in range(len(lista)):
        strpth=path+'/'+lista[k]
        im_data = imread(strpth)
        #im_data = im_data.astype(np.int16)
        M.append(im_data)
    if len(M[0].shape)==3:
        (n,h,w) = M[0].shape
        
        tensor = resize(M[0]).reshape(1,h,w,n)
        for t in range(1,len(M)):
            M[t] = resize(M[t]).reshape(1,h,w,n)
            tensor = np.concatenate((tensor,M[t]),axis=0)
        return tensor
        
    elif len(M[0].shape)==2:
        (h,w) = M[0].shape
        tensor = M[0].reshape(1,h,w,1)
        for t in range(1,len(M)):
            M[t] = resize(M[t]).reshape(1,h,w,1)
            tensor = np.concatenate((tensor,M[t]),axis=0)
        return tensor
    else :
        print('此函数只处理二维或三维图像')
        return

    
