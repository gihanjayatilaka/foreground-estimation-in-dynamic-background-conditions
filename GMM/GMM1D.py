import cv2 as cv
import numpy as np
import sys





fileName=sys.argv[1]
clusters=int(sys.argv[2])
ITERATIONS=int(sys.argv[3])

img = cv.imread(fileName,cv.IMREAD_GRAYSCALE)

im=img.flatten() #Flat image
imgClusters=np.ndarray(shape=(clusters,img.shape[0],img.shape[1]),dtype=np.uint8) #To put the clusters

#im=np.array([1,2, 5 ,8 ,6 ,4 ,8 ,3 ,2])

pixels=im.shape[0]

condProb_cluster_pixel=np.ndarray(shape=(clusters,pixels),dtype=np.float32)
condProb_cluster_pixel.fill(0)
condProb_pixel_cluster=np.ndarray(shape=(pixels,clusters),dtype=np.float32)
mean_cluster=np.ndarray(shape=(clusters),dtype=np.float32)
var_cluster=np.ndarray(shape=(clusters),dtype=np.float32)
prob_cluster=np.ndarray(shape=(clusters),dtype=np.float32)

#initializing with same probability for any pixel to be in any cluster
'''
for c in range(clusters):
    for p in range(pixels):
        condProb_cluster_pixel[c][p]=0.5
'''


for p in range(pixels):
    condProb_cluster_pixel[np.random.randint(0,clusters)][p]=1

#print("start ",np.transpose(condProb_cluster_pixel).tolist())


for i in range(ITERATIONS): #ITERATIONS

    #cluster means
    for c in range(clusters):
        D=0
        N=0
        for p in range(pixels):
            N+=condProb_cluster_pixel[c][p]*float(im[p])
            D+=condProb_cluster_pixel[c][p]
 #       print("mean= "+str(N)+"/"+str(D))
        mean_cluster[c]=N/D


    #cluster variances
    for c in range(clusters):
        D=0
        N=0
        for p in range(pixels):
            N+=condProb_cluster_pixel[c][p]*np.power((1.0*float(im[p]))-mean_cluster[c],2)
            D+=condProb_cluster_pixel[c][p]
 #           print("ND",N,D)
#        print("var"+str(N)+"/"+str(D))
        var_cluster[c]=N/D


    #cluster probabilitis
    for c in range(clusters):
        D=0
        N=0
        for p in range(pixels):
            N+=condProb_cluster_pixel[c][p]
            D+=1
        prob_cluster[c]=N/D

    # prob(pixel/cluster)
    for c in range(clusters):
        for p in range(pixels):
            condProb_pixel_cluster[p][c]= (1/np.sqrt(2*np.pi*var_cluster[c])) * np.exp(-1* np.power(im[p]-mean_cluster[c],2) / (2*var_cluster[c]))

    # prob(cluster/pixel)
    for c in range(clusters):
        for p in range(pixels):
            N=condProb_pixel_cluster[p][c]*prob_cluster[c]
            D=0
            for cc in range(clusters):
                D+=condProb_pixel_cluster[p][cc]*prob_cluster[c]
            condProb_cluster_pixel[c][p]=N/D
    # output
    print("Iter "+str(i)+" over")
    print("Mean cluster",mean_cluster)
    print("Var cluster",var_cluster)

    print("Probability cluster",prob_cluster)


#    print("Cluster 1")
#    print(condProb_cluster_pixel[0][:])
#    print("Cluster 2")
#    print(condProb_cluster_pixel[1][:])

cv.imshow("Original image",img)

for c in range(clusters):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            imgClusters[c][y][x]=int(condProb_cluster_pixel[c][y*img.shape[0]+x]*255)
    cv.imshow("Cluster "+str(c),imgClusters[c])
'''
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(clusters):
            imgCol[y][x][c]=int(condProb_cluster_pixel[c][y*img.shape[0]+x]*255)
'''


cv.waitKey(0)
