import cv2 as cv
import numpy as np
import matplotlib.mlab as mlab
import pyqtgraph as pg
import sys

fileName='img4.tiff'#sys.argv[1]
clusters=4#int(sys.argv[2])
ITERATIONS=2000#int(sys.argv[3])

img = cv.imread(fileName,cv.IMREAD_GRAYSCALE)

#initializing with same probability for any pixel to be in any cluster
'''
for c in range(clusters):
    for p in range(pixels):
        condProb_cluster_pixel[c][p]=0.5
'''
app = pg.QtGui.QApplication([])
win = pg.GraphicsWindow()
pw = pg.GraphicsWindow()
cluster_pixel = pw.addPlot()
pixel_cluster = pw.addPlot()
p1 = win.addPlot()


def gmm(data, clusters, ITERATIONS):
    # init
    data.astype('float')
    length = data.size
    var_thresh = 1
    condProb_cluster_pixel = np.zeros(shape=(clusters, length), dtype=np.float32)
    condProb_pixel_cluster = np.ndarray(shape=(length, clusters), dtype=np.float32)

    mean_cluster = np.ndarray(shape=(clusters,), dtype=np.float32)
    var_cluster = np.ones(shape=(clusters,), dtype=np.float32)
    prob_cluster = np.ndarray(shape=(clusters,), dtype=np.float32)

    cluster_pixel_curves = []
    pixel_cluster_curves = []
    for c in range(clusters):
        for x in range(length):
            condProb_cluster_pixel[np.random.randint(0,clusters)][x]=1
        cluster_pixel_curves.append(cluster_pixel.plot(condProb_cluster_pixel[c], pen=(c,clusters)))
        pixel_cluster_curves.append(pixel_cluster.plot(condProb_pixel_cluster[:][c], pen=(c,clusters)))

    # plotter
    gaussiancurves = []
    p1.plot(np.arange(length), data/np.max(data), brush=(255, 0, 0, 80))
    for c in range(clusters):
        normpdf = mlab.normpdf(np.arange(length), mean_cluster[c], var_cluster[c])
        curve = p1.plot(np.arange(length),normpdf/np.max(normpdf),  pen=(c,clusters))
        gaussiancurves.append(curve)


    for i in range(ITERATIONS): #ITERATIONS

        #cluster means
        for c in range(clusters):
            if var_cluster[c] < var_thresh:
                continue
            D,N=0,0
            for x in range(length):
                N+=condProb_cluster_pixel[c][x]*float(x)*data[x]
                D+=condProb_cluster_pixel[c][x]*data[x]
            mean_cluster[c]=N/D


        #cluster variances
        for c in range(clusters):
            if var_cluster[c] < var_thresh:
                continue
            D,N=0,0
            for x in range(length):
                N+=condProb_cluster_pixel[c][x]*np.power((1.0*x)-mean_cluster[c],2)*data[x]
                D+=condProb_cluster_pixel[c][x]*data[x]
            var_cluster[c]=N/D
            if var_cluster[c]<var_thresh:
                var_cluster[c] = var_thresh
        #cluster probabilitis
        for c in range(clusters):
            if var_cluster[c] < var_thresh:
                continue
            D,N=0,0
            for x in range(length):
                N+=condProb_cluster_pixel[c][x]*data[x]
                D+=1*data[x]
            prob_cluster[c]=N/D

        # prob(pixel | cluster)
        for c in range(clusters):
            if var_cluster[c] < var_thresh:
                continue
            for x in range(length):
                condProb_pixel_cluster[x][c]= (1/np.sqrt(2*np.pi*var_cluster[c])) * np.exp(-1* np.power(x-mean_cluster[c],2) / (2*var_cluster[c]))
                
        # prob(cluster | pixel)
        for c in range(clusters):
            if var_cluster[c] < var_thresh:
                continue
            for x in range(length):
                N=condProb_pixel_cluster[x][c]*prob_cluster[c]*(data[x]+1)
                D=0
                for cc in range(clusters):
                    D+=condProb_pixel_cluster[x][cc]*prob_cluster[c]*(data[x]+1)

                condProb_cluster_pixel[c][x]=N/D
                if np.isnan(condProb_cluster_pixel[c][x]):
                    raise ArithmeticError
                    #condProb_cluster_pixel[c][x]=0.0
        # output
        print("Iter "+str(i)+" over")
        print("Mean cluster",mean_cluster)
        print("Var cluster",var_cluster)
        print("Probability cluster",prob_cluster)

        for c in range(clusters):
            normpdf = (1/np.sqrt(2*np.pi*var_cluster[c]))* np.exp(-1* np.power(np.arange(256)-mean_cluster[c],2) / (2*var_cluster[c]))
            gaussiancurves[c].setData(normpdf/np.max(normpdf)*prob_cluster[c])

            cluster_pixel_curves[c].setData(condProb_cluster_pixel[c])
            pixel_cluster_curves[c].setData(condProb_pixel_cluster[:,c])

            pg.QtGui.QApplication.processEvents()
        i=0
        while (i<1e6):
            i+=1
        """
        for c in range(clusters):
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    imgClusters[c][y][x] = int(condProb_cluster_pixel[c][img[y][x]] * 255)
            cv.imshow("Cluster " + str(c), imgClusters[c])
            #cv.imwrite("./results/"+fileName.strip().split(".")[0] + "- iter "+str(i)+" Cluster " + str(c) + ".tiff", imgClusters[c])
        """


    return condProb_cluster_pixel

    #    print("Cluster 1")
    #    print(condProb_cluster_pixel[0][:])
    #    print("Cluster 2")
    #    print(condProb_cluster_pixel[1][:])


for i in range(100):
    img = cv.imread(fileName,cv.IMREAD_GRAYSCALE)
    im=img.flatten()

y,x = np.histogram(im, 256)

condProb_cluster_pixel=gmm(y,clusters,ITERATIONS)

cv.imshow("Original image",img)

imgClusters = np.ndarray(shape=(clusters, img.shape[0], img.shape[1]), dtype=np.uint8)  # To put the clusters
for c in range(clusters):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            imgClusters[c][y][x]=int(condProb_cluster_pixel[c][img[y][x]] * 255)
    cv.imshow("Cluster "+str(c),imgClusters[c])
    #cv.imwrite("./results/"+fileName.strip().split(".")[0]+" Cluster "+str(c)+".tiff",imgClusters[c])

cv.waitKey(0)

#refresh pyqtgraphs
def myKeyPressEvent(e):
    if e.key() == pg.QtCore.Qt.Key_Enter or e.key() == pg.QtCore.Qt.Key_Return:
        global selectionFinished
        selectionFinished = True

# Monkey patch
selectionFinished = False
win.keyPressEvent = myKeyPressEvent

while not selectionFinished:
    pg.QtGui.QApplication.processEvents()