import numpy as np
import cv2
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

file_name = 'test3.avi'
cap = cv2.VideoCapture(file_name)
_, frame = cap.read()

width, height = frame.shape[0], frame.shape[1]
image_name = '{} {}X{}'.format(file_name,width,height)

N = 100
history = np.ndarray((N,width,height,3))
history[0,:,:,:] = frame

pixel_update_queue = []

class Visualizer(object):
    def __init__(self,x,y):
        self.traces = dict()
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('RGB space for {}X{}'.format(x,y))
        self.w.setGeometry(0, 110, 500, 500)
        self.w.show()

        #set parameters for updating
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


        for i in range(N):
            pts = np.array([np.random.rand()*10,np.random.rand()*10,np.random.rand()*10])
            self.traces[i] = gl.GLScatterPlotItem(pos=pts, color=pg.glColor((i, N * 1.3)))
            self.w.addItem(self.traces[i])

    def start(self):
        if not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, points, color):
        self.traces[name].setData(pos=points, color=color)

    def update(self, point):
        self.set_plotdata(
            name=self.current, points=point/10,
            color=pg.glColor(point[0],point[1],point[2],150)
        )
        self.current = (self.current+1)%N

# mouse callback function
def mouse(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Selected pixel',x,y)
        plot_pixel_history(x,y)
        plot_pixel_history_in_rgb_space(x,y)

def plot_pixel_history(x,y):
    r,g,b = np.arange(N),np.arange(N),np.arange(N)
    for i in range(N):
        r[i] = history[(i+frame_num+1)%N,y,x,0]
        g[i] = history[(i+frame_num+1)%N,y,x,1]
        b[i] = history[(i+frame_num+1)%N,y,x,2]
    plotWidget = pg.plot(title="Pixel {} {}".format(x,y))
    plotWidget.setYRange(0, 256, padding=0)
    plotWidget.plot(r, pen=(0,3))
    plotWidget.plot(g, pen=(1,3))
    plotWidget.plot(b, pen=(2,3))

def plot_pixel_history_in_rgb_space(x,y):
    v = Visualizer(x,y)
    v.start()
    for i in range(N):
        r = history[(i + frame_num + 1) % N, y, x, 0]
        g = history[(i + frame_num + 1) % N, y, x, 1]
        b = history[(i + frame_num + 1) % N, y, x, 2]
        v.update(np.array([r,g,b]))
    pixel_update_queue.append(v)

cv2.namedWindow(image_name)
cv2.setMouseCallback(image_name,mouse)


app = QtGui.QApplication([])
win = pg.GraphicsWindow()

p1 = win.addPlot()
p2 = win.addPlot()
p3 = win.addPlot()
p1.setYRange(0, 1000, padding=0)
p2.setYRange(0, 1000, padding=0)
p3.setYRange(0, 1000, padding=0)

frame_num = 0
while(1):
    print(frame_num)
    frame_num+=1

    _, frame = cap.read()
    history[frame_num%N, :,:,:] = frame

    r,g,b = frame[:,:,0],frame[:,:,1],frame[:,:,2]
    ry, rx = np.histogram(r, bins=np.arange(255))
    gy, gx = np.histogram(g, bins=np.arange(255))
    by, bx = np.histogram(b, bins=np.arange(255))

    curver = p1.plot(rx, ry,  clear=True, stepMode=True, fillLevel=0, brush=(255, 0, 0, 80))
    curveg = p2.plot(gx, gy,  clear=True, stepMode=True, fillLevel=1, brush=(0,255, 0, 80))
    curveb = p3.plot(bx, by,  clear=True, stepMode=True, fillLevel=2, brush=(0, 0, 255, 80))

    for v in pixel_update_queue:
        r = history[(frame_num + 1) % N, v.y, v.x, 0]
        g = history[(frame_num + 1) % N, v.y, v.x, 1]
        b = history[(frame_num + 1) % N, v.y, v.x, 2]
        v.update(np.array([r, g, b]))
    pg.QtGui.QApplication.processEvents()

    cv2.imshow(image_name,frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()