import numpy as np
import cv2
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.mlab as mlab
import threading, multiprocessing


class myThread (threading.Thread):
    def __init__(self, threadID, data, clusters, iterations):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.data = data
        self.clusters = clusters
        self.iter = iterations

    def run(self):
        print("Starting thread",self.threadID)
        # Get lock to synchronize threads
        threadLock.acquire()
        for i in range(len(self.data)):
            mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster = gmm(self.data[i], self.clusters, self.iter)
            large_var_clusters = []
            for c in range(len(mean_cluster)):
                if var_cluster[c] > 1000:
                    large_var_clusters.append(c)

            for i in range(N):
                val = history[i, y, x, 0]
                for c in large_var_clusters:
                    if condProb_cluster_pixel[c][val] > 0.95:
                        fg[i, y, x, :] = 1.0
                        print(i,y,x)
                        break
        # Free lock to release next thread
        print("Finished thread", self.threadID)
        threadLock.release()


def gmm(data, clusters, ITERATIONS):
    # init
    data.astype('float')
    data += 1
    
    length = data.size
    condProb_cluster_pixel = np.zeros(shape=(clusters, length), dtype=np.float32)
    condProb_pixel_cluster = np.ndarray(shape=(length, clusters), dtype=np.float32)

    mean_cluster = np.ndarray(shape=(clusters,), dtype=np.float32)
    var_cluster = np.ones(shape=(clusters,), dtype=np.float32)
    prob_cluster = np.ndarray(shape=(clusters,), dtype=np.float32)

    for x in range(length):
        for c in range(clusters):
            condProb_cluster_pixel[np.random.randint(0, clusters)][x] = 1

    for i in range(ITERATIONS):  # ITERATIONS

        # cluster means
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            D, N = 0, 0
            for x in range(length):
                N += condProb_cluster_pixel[c][x] * float(x) * data[x]
                D += condProb_cluster_pixel[c][x] * data[x]
            mean_cluster[c] = N / D

        # cluster variances
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            D, N = 0, 0
            for x in range(length):
                N += condProb_cluster_pixel[c][x] * np.power((1.0 * x) - mean_cluster[c], 2) * data[x]
                D += condProb_cluster_pixel[c][x] * data[x]
            var_cluster[c] = N / D
            if var_cluster[c] < 1e-4:
                var_cluster[c] = 1e-4

        # cluster probabilitis
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            D, N = 0, 0
            for x in range(length):
                N += condProb_cluster_pixel[c][x] * data[x]
                D += 1 * data[x]
            prob_cluster[c] = N / D

        # prob(pixel | cluster)
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            for x in range(length):
                condProb_pixel_cluster[x][c] = (1 / np.sqrt(2 * np.pi * var_cluster[c])) * np.exp(
                    -1 * np.power(x - mean_cluster[c], 2) / (2 * var_cluster[c]))

        # prob(cluster | pixel)
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            for x in range(length):
                N = condProb_pixel_cluster[x][c] * prob_cluster[c] * (data[x] + 1)
                D = 0
                for cc in range(clusters):
                    D += condProb_pixel_cluster[x][cc] * prob_cluster[c] * (data[x] + 1)

                condProb_cluster_pixel[c][x] = N / D
                if np.isnan(condProb_cluster_pixel[c][x]):
                    raise ArithmeticError
                    # condProb_cluster_pixel[c][x]=0.0
    return mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster

def gmm_helper(history, row, clusters, iter):
        print("Starting row",row)

        N, _height, _width = history.shape[0],history.shape[1],history.shape[2]


        data = np.zeros((_width,256))
        for i in range(N):
            for x in range(_width):
                data[x][history[i,row,x,0]] += 1
        for column in range(len(data)):
            if (column%20)==0:
                print("$$",row,column)
            mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster = gmm(data[column], clusters, iter)
            large_var_clusters = []
            for c in range(len(mean_cluster)):
                if var_cluster[c] > 1000:
                    large_var_clusters.append(c)

            for i in range(N):
                val = history[i, row, x, 0]
                for c in large_var_clusters:
                    if condProb_cluster_pixel[c][val] > 0.95:
                        fg[i, row, x, :] = 1.0
                        print(i,row,x)
                        break
        print("Finished row", row)
if __name__ == '__main__':


    frame_num = 1
    file_name = './dataset/dynamicBackground/fall/input/in{0:0>6}.jpg'.format(frame_num)
    print(file_name)
    frame = cv2.imread(file_name)


    clusters = 5

    _height, _width = frame.shape[0], frame.shape[1]
    image_name = '{} {}X{}'.format(file_name, _height, _width)

    N = 25
    history = np.zeros((N, _height, _width, 3), dtype=int)
    history[0, :, :, :] = frame

    pw = pg.GraphicsWindow()
    pl = pw.addPlot()
    pl.setYRange(0, 100, padding=0)
    pl2 = pw.addPlot()
    pl2.setYRange(0, 1, padding=0)

    c1,c2,c3 = pl2.plot(pen=(1,2)),pl2.plot(pen=(1,2)),pl2.plot(pen=(1,2))
    g1,g2,g3 = pl2.plot(pen=(0,2)),pl2.plot(pen=(0,2)),pl2.plot(pen=(0,2))

    while (1):
        print(frame_num)
        

        frame = cv2.imread('./dataset/dynamicBackground/fall/input/in{0:0>6}.jpg'.format(frame_num))
        try:
            history[frame_num % N, :, :, :] = frame
        except:
            break
        fg = np.zeros((N,_height,_width,3), dtype=float)
        if frame_num%N==0:
            nThreads = 4
            proc = []
            for y in range(360,370):
                
                proc.append(multiprocessing.Process(target=gmm_helper, args=(history, y, 3, 20)))
                
                if len(proc) == 4:
                    for p in proc:
                        p.start()

                    for t in proc:
                        t.join()
                    proc = []
        frame_num += 1
        cv2.imshow(image_name,  history[(frame_num-1) % N, :, :, :]/255.0)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    _ = input()
    cv2.destroyAllWindows()




'''
while (1):
    print(frame_num)
    

    frame = cv2.imread('./dataset/dynamicBackground/fall/input/in{0:0>6}.jpg'.format(frame_num))
    try:
        history[frame_num % N, :, :, :] = frame
    except:
        break
    fg = np.zeros((N,_height,_width,3), dtype=float)
    if frame_num%N == 0:
        for x in range(_width):
            print("checking",x)
            for y in range(350,360):
                data = np.zeros(256)
                for i in range(N):
                    data[history[i,y,x,1]] += 1

                mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster = gmm(data, 3, 10)
                
                #pl2.plot(history[:,y,x,0], clear=True)
                pl.plot(data, clear=True)

                c1.setData(condProb_cluster_pixel[0]/np.max(condProb_cluster_pixel))
                c2.setData(condProb_cluster_pixel[1]/np.max(condProb_cluster_pixel))
                c3.setData(condProb_cluster_pixel[2]/np.max(condProb_cluster_pixel))

                g1.setData(condProb_pixel_cluster[:,0]/np.max(condProb_pixel_cluster))
                g2.setData(condProb_pixel_cluster[:,1]/np.max(condProb_pixel_cluster))
                g3.setData(condProb_pixel_cluster[:,2]/np.max(condProb_pixel_cluster))

                pg.QtGui.QApplication.processEvents()

                large_var_clusters = []
                for c in range(len(mean_cluster)):
                    if var_cluster[c] > 1000:
                        large_var_clusters.append(c)

                for i in range(N):
                    val = history[i, y, x, 0]
                    for c in large_var_clusters:
                        if condProb_cluster_pixel[c][val] > 0.95:
                            fg[i, y, x, :] = 1.0
                            print(i,y,x)
                            break
        for i in range(N):
            cv2.imshow("fg", fg[i,:,:,:])

    frame_num += 1
    cv2.imshow(image_name,  history[(frame_num-1) % N, :, :, :]/255.0)
    if cv2.waitKey(20) & 0xFF == 27:
        break

_ = input()
cv2.destroyAllWindows()
'''