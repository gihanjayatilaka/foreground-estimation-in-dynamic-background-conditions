import cv2 as cv
import numpy as np

img = cv.imread("img1.tiff",cv.IMREAD_GRAYSCALE)

im=img.flatten() #Flat image

imgCol=np.ndarray(shape=(img.shape[0],img.shape[1],3),dtype=np.uint8) #To put the clusters

clusters=2
pixels=im.shape[0]

condProb_cluster_pixel=np.ndarray(shape=(clusters,pixels),dtype=np.float32)
condProb_pixel_cluster=np.ndarray(shape=(pixels,clusters),dtype=np.float32)
mean_cluster=np.ndarray(shape=(clusters),dtype=np.float32)
var_cluster=np.ndarray(shape=(clusters),dtype=np.float32)
prob_cluster=np.ndarray(shape=(clusters),dtype=np.float32)

#initializing with same probability for any pixel to be in any cluster
for c in range(clusters):
    for p in range(pixels):
        condProb_cluster_pixel[c][p]=1.0/clusters


#cluster means
for c in range(clusters):
    D=0
    N=0
    for p in range(pixels):
        N+=condProb_cluster_pixel[c][p]*float(im[p])
        D+=condProb_cluster_pixel[c][p]
    mean_cluster[c]=N/D


#cluster variances
for c in range(clusters):
    D=0
    N=0
    for p in range(pixels):
        N+=condProb_cluster_pixel[c][p]*(float(im[p])-mean_cluster[c])
        D+=condProb_cluster_pixel[c][p]
    var_cluster[c]=N/D


#cluster probabilitis
for c in range(clusters):
    D=0
    N=0
    for p in range(pixels):
        N+=condProb_cluster_pixel[c][p]
        D+=1
    prob_cluster[c]=N/D


for c in range(clusters):
    for p in range(pixels):
        condProb_pixel_cluster= (1/np.sqrt(2*np.pi*var_cluster[c])) * np.exp(-1* np.power(im[p]-mean_cluster[c],2) / (2*var_cluster[c]))




for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        imgCol[r][c][2]=img[r][c];
cv.imshow("Coloured Image" ,imgCol)
cv.waitKey(30000) & 0xff
