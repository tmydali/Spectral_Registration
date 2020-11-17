import numpy as np
from math import ceil
from PIL import Image

def Split(img, size=160, plotEach=False):
    num = (np.int(img.shape[0]/size), np.int(img.shape[1]/size))
    imageSet = []
    for i in range(num[0]):
        for j in range(num[1]):
            imageSet.append(img[i*size:(i+1)*size, j*size:(j+1)*size])
            
    if plotEach:
        fig , ax = plt.subplots(num[0], num[1], sharex=True, sharey=True)

        index = 0
        for i in range(num[0]):
            for j in range(num[1]):
                plt.subplot(ax[i,j])
                plt.imshow(imageSet[index])
                index = index + 1

    return imageSet

def Merge(img_ls, func=lambda img: img):
    ### Merge image tiles from img_ls. Each tile would processed by function func.
    ### If func not specified, it would merely return original data.
    ### img_ls is a numpy array with [batch, width, height, channel]
    size = (0, 0)
    try:
        size = img_ls[0, :, :, 0].shape
    except:
        print('Empty image set')
        return None
    
    m_size = (1280, 960)
    size = img_ls[0, :, :, 0].shape
    colMax = ceil(m_size[0]/size[0])
    rowMax = ceil(m_size[1]/size[1])
    merged = Image.new('L', m_size)
    p = func(img_ls)
    
    index = 0
    for i in range(rowMax):
        for j in range(colMax):
            tile = p[index, :, :, 0]
            tile = Image.fromarray(np.uint8(tile*255))
            merged.paste(tile, (size[0]*j, size[1]*i))
            index += 1
        
    return merged

def Smooth(img, size):
    ### Use Gaussian kernel to smooth joint boundaries
    num = (np.int(img.shape[0]/size), np.int(img.shape[1]/size))
    