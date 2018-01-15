import numpy as np
import cv2
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.mlab as mlab
from sklearn import decomposition

frame_num = 1
file_name = './dataset/dynamicBackground/boats/input/in{0:0>6}.jpg'.format(frame_num)
print(file_name)
frame = cv2.imread(file_name)


clusters = 5

width, height = frame.shape[0], frame.shape[1]
image_name = '{} {}X{}'.format(file_name, width, height)

N = 700
history = np.zeros((N, width, height, 3), dtype=int)
history[0, :, :, :] = frame

pixel_update_queue = []


def gmm(data, clusters, ITERATIONS, gaussiancurves):
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
            D = np.sum(condProb_cluster_pixel[c] * data)
            N = np.sum(condProb_cluster_pixel[c] * data * np.arange(length))
            mean_cluster[c] = N / D

        # cluster variances
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            D = np.sum(condProb_cluster_pixel[c] * data)
            N = np.sum(condProb_cluster_pixel[c] * data * np.power(np.arange(length) - mean_cluster[c], 2))
            var_cluster[c] = N / D
            if var_cluster[c] < 1e-4:
                var_cluster[c] = 1e-4

        # cluster probabilitis
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            D = len(condProb_cluster_pixel[c])
            N = np.sum(condProb_cluster_pixel[c])
            prob_cluster[c] = N / D

        # prob(pixel | cluster)
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            condProb_pixel_cluster[:, c] = (1 / np.sqrt(2 * np.pi * var_cluster[c])) * np.exp(
                -1 * np.power(np.arange(length) - mean_cluster[c], 2) / (2 * var_cluster[c]))

        # prob(cluster | pixel)
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            _N = condProb_pixel_cluster[:, c] * data
            _D = np.sum(condProb_pixel_cluster * np.repeat(data, clusters).reshape((-1, clusters)), 1)
            condProb_cluster_pixel[c] = (_N / _D)

    for c in range(clusters):
        #normpdf = mlab.normpdf(np.arange(length), mean_cluster[c], var_cluster[c])
        gaussiancurves[c].setData(condProb_pixel_cluster[:,c])
        pg.QtGui.QApplication.processEvents()

class Visualizer3D(object):
    def __init__(self, x, y):
        self.traces = dict()
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 100
        self.w.setWindowTitle('RGB space for {}X{}'.format(x, y))
        self.w.setGeometry(100, 100, 500, 500)
        self.w.show()

        # set parameters for updating
        self.current = 0
        self.x = x
        self.y = y
        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(0, 10, 10)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(10, 0, 10)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(10, 10, 0)
        self.w.addItem(gz)

        self.curve1 = gl.GLLinePlotItem()
        self.curve2 = gl.GLLinePlotItem()
        self.curve3 = gl.GLLinePlotItem()
        self.w.addItem(self.curve1)
        self.w.addItem(self.curve2)
        self.w.addItem(self.curve3)
        self.X = np.ndarray((N,3),'float')
        self.pca = decomposition.PCA(n_components=3)

        for i in range(N):
            r = history[(i + frame_num + 1) % N, y, x, 0]
            g = history[(i + frame_num + 1) % N, y, x, 1]
            b = history[(i + frame_num + 1) % N, y, x, 2]
            pts = np.array([r, g, b])/14
            self.X[i,:] = pts
            self.traces[i] = gl.GLScatterPlotItem(pos=pts, color=pg.glColor(100,100,100,100))
            self.w.addItem(self.traces[i])

    def start(self):
        if not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, points, color):
        self.traces[name].setData(pos=points, color=color)
        self.X[name,:] = points
    def update(self, point, isfg):
        if isfg:
            clr = [255,0,0]
        else:
            clr = [255,255,255]
        self.set_plotdata(
            name=self.current, points=point/14,
            color=pg.glColor(clr[0],clr[1],clr[2],100)
        )


        self.pca.fit(self.X)
        V = self.pca.components_
        x_pca_axis, y_pca_axis, z_pca_axis = 30 * V.T
        X_tr = self.pca.transform(self.X)
        mean = np.average(self.X,axis=0)
        self.curve1.setData(pos=np.array([mean,[x_pca_axis[0],y_pca_axis[0],z_pca_axis[0]]]))
        self.curve2.setData(pos=np.array([mean,[x_pca_axis[1],y_pca_axis[1],z_pca_axis[1]]]))
        self.curve3.setData(pos=np.array([mean,[x_pca_axis[2],y_pca_axis[2],z_pca_axis[2]]]))

        self.current = (self.current + 1) % N


class Visualizer2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.iter = 20
        global clusters
        self.clusters = clusters

        self.r, self.g, self.b = np.arange(N), np.arange(N), np.arange(N)
        self.win = pg.GraphicsWindow(title="Pixel {} {}".format(x, y))

        self.time = self.win.addPlot()
        self.freq = self.win.addPlot()
        self.gaus = self.win.addPlot()

        self.time.setYRange(0, 256, padding=0)
        self.rcurve = self.time.plot(self.r, pen=(0, 3))
        self.gcurve = self.time.plot(self.g, pen=(1, 3))
        self.bcurve = self.time.plot(self.b, pen=(2, 3))

        self.avg = np.array([10, 10, 10])

        self.ravgcurve = self.time.plot(np.zeros(N) + self.avg[0], pen=(0, 3))
        self.gavgcurve = self.time.plot(np.zeros(N) + self.avg[1], pen=(1, 3))
        self.bavgcurve = self.time.plot(np.zeros(N) + self.avg[2], pen=(2, 3))

        self.rgaussiancurves,self.ggaussiancurves,self.bgaussiancurves = [],[],[]
        for c in range(self.clusters):
            self.rgaussiancurves.append(self.gaus.plot([], pen=(0, 3)))
            self.ggaussiancurves.append(self.gaus.plot([], pen=(1, 3)))
            self.bgaussiancurves.append(self.gaus.plot([], pen=(2, 3)))

        self.rfreq = np.zeros((256,))
        self.bfreq = np.zeros((256,))
        self.gfreq = np.zeros((256,))
        self.rfreq[0] = N
        self.bfreq[0] = N
        self.gfreq[0] = N

        self.rfreqcurve = self.freq.plot(self.rfreq, pen=(0, 3))
        self.gfreqcurve = self.freq.plot(self.gfreq, pen=(1, 3))        
        self.bfreqcurve = self.freq.plot(self.bfreq, pen=(2, 3))
        
        self.update()

    def update(self, point=None):
        self.rfreq = np.zeros((256,))
        self.bfreq = np.zeros((256,))
        self.gfreq = np.zeros((256,))

        for i in range(N):
            self.r[i] = history[(i + frame_num +1) % N, self.y, self.x, 0]
            self.g[i] = history[(i + frame_num +1) % N, self.y, self.x, 1]
            self.b[i] = history[(i + frame_num +1) % N, self.y, self.x, 2]
            self.rfreq[self.r[i]] += 1
            self.gfreq[self.g[i]] += 1
            self.bfreq[self.b[i]] += 1
        
        self.avg = np.average([self.r, self.g, self.b], axis=1)

        self.ravgcurve.setData(np.zeros(N) + self.avg[0])
        self.gavgcurve.setData(np.zeros(N) + self.avg[1])
        self.bavgcurve.setData(np.zeros(N) + self.avg[2])

        
        self.rfreqcurve.setData(self.rfreq)        
        self.gfreqcurve.setData(self.gfreq)
        self.bfreqcurve.setData(self.bfreq)

        if (frame_num%N==0):
            gmm(self.rfreq, self.clusters, self.iter, self.rgaussiancurves)
            gmm(self.gfreq, self.clusters, self.iter, self.ggaussiancurves)
            gmm(self.bfreq, self.clusters, self.iter, self.bgaussiancurves)

        self.rcurve.setData(self.r, pen=(0, 3))
        self.gcurve.setData(self.g, pen=(1, 3))
        self.bcurve.setData(self.b, pen=(2, 3))


# mouse callback function
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Selected pixel', x, y)
        #v2 = Visualizer2D(x, y)
        #pixel_update_queue.append(v2)
        v = Visualizer3D(x, y)
        v.start()
        pixel_update_queue.append(v)
    if event == cv2.EVENT_MOUSEMOVE:
        print(x,y)

cv2.namedWindow(image_name)
cv2.setMouseCallback(image_name, mouse)

class histogram(object):
    def __init__(self):
        self.win = pg.GraphicsWindow()

        self.plwin = self.win.addPlot()
        self.plwin.setYRange(0, 1000, padding=0)

        self.curver = self.plwin.plot([], [], fillLevel=0, brush=(255, 0, 0, 80))
        self.curveg = self.plwin.plot([], [], fillLevel=0, brush=(0, 255, 0, 80))
        self.curveb = self.plwin.plot([], [], fillLevel=0, brush=(0, 0, 255, 80))

    def update(self,frame):
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        ry, rx = np.histogram(r, bins=np.arange(255))
        gy, gx = np.histogram(g, bins=np.arange(255))
        by, bx = np.histogram(b, bins=np.arange(255))

        self.curver.setData(ry, fillLevel=0, brush=(255, 0, 0, 80))
        self.curveg.setData(gy, fillLevel=0, brush=(0, 255, 0, 80))
        self.curveb.setData(by, fillLevel=0, brush=(0, 0, 255, 80))

class simplefg(object):
    def __init__(self):
        self.avg = np.ndarray((width, height, 3))

    def update(self, frame):
        self.avg = np.average(history, axis=0)
        mask = (np.abs(self.avg - frame) > 50).astype('float')
        cv2.imshow("fg", mask)

#fg = simplefg()
#hist = histogram()

while (1):

    frame = cv2.imread('./dataset/dynamicBackground/boats/input/in{0:0>6}.jpg'.format(frame_num))
    gr_truth = cv2.imread('./dataset/dynamicBackground/boats/groundtruth/gt{0:0>6}.png'.format(frame_num))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    try:
        history[frame_num % N, :, :, :] = frame
    except:
        break

    #fg.update(frame)
    #hist.update(frame)
    for v in pixel_update_queue:
        r = history[(frame_num) % N, v.y, v.x, 0]
        g = history[(frame_num) % N, v.y, v.x, 1]
        b = history[(frame_num) % N, v.y, v.x, 2]
        if (gr_truth[v.y,v.x,0] >100):
            v.update(np.array([r, g, b]), True)
        else:
            v.update(np.array([r, g, b]), False)
    pg.QtGui.QApplication.processEvents()
    frame_num += 1
    cv2.imshow(image_name, frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
_ = input()
cv2.destroyAllWindows()
