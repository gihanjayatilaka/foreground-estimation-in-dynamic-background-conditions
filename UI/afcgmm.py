import numpy as np
import cv2,time
import pyqtgraph as pg
import pyqtgraph.opengl as gl
#import line_profiler
#import scipy.stats as stats
import scipy.linalg as la
import sys
#import matplotlib.pyplot as plt
from sklearn import decomposition

def update_cyl(X, lost_X, cyl, beta):
    D1 = 5
    D2 = 5

    # v1,v2,v3, R, L,c1,c2,c3, p
    #  0, 1, 2, 3, 4, 5, 6, 7, 8
    height, width, clusters, _ = cyl.shape
    p = cyl[:,:,:,8]
    R_2 = cyl[:,:,:,3]
    L_2 = cyl[:,:,:,4]
    v1 = cyl[:,:,:,0]
    v2 = cyl[:,:,:,1]
    v3 = cyl[:,:,:,2]
    c1 = cyl[:,:,:,5]
    c2 = cyl[:,:,:,6]
    c3 = cyl[:,:,:,7]

    s_ = np.rec.fromarrays([p*np.sqrt(L_2)/np.sqrt(R_2), p, L_2, R_2,v1,v2,v3,c1,c2,c3])
    s_.sort()
    p = s_.f1
    L_2 = s_.f2
    R_2 = s_.f3
    v1 = s_.f4
    v2 = s_.f5
    v3 = s_.f6
    c1 = s_.f7
    c2 = s_.f8
    c3 = s_.f9

    cs = np.array([c1,c2,c3])
    cs = cs.transpose([1,2,3,0])
    vs = np.array([v1,v2,v3])
    vs = vs.transpose([1,2,3,0])

    fg = np.zeros((height, width), float)

    for clu in range(clusters-1,-1,-1):
    
        centroids = cs[:,:,clu,:]
        
        d = np.abs(X - centroids)
        v = vs[:,:,clu,:]

        L_current_pixels = np.sum(d* (v / la.norm(v ,axis=2).reshape(height,width,1)),axis=2)
        L_2_current_pixels = L_current_pixels * L_current_pixels
        R_2_current_pixels = la.norm(d, axis=2)**2 - L_2_current_pixels

        cond = (L_2_current_pixels < D1 * L_2[:, :, clu]) * (R_2_current_pixels < D2 * R_2[:, :, clu])

        if clu == 0:
            cond = L_2_current_pixels == L_2_current_pixels    
        
        rho = beta / (1 + p[:,:,clu])
        fg += cond#*la.norm(d ,axis=2)*rho

        L_new_2 = L_2[:, :, clu] + cond * rho * (L_2_current_pixels - L_2[:, :, clu])
        R_new_2 = R_2[:, :, clu] + cond * rho * (R_2_current_pixels - R_2[:, :, clu])


        centroids_new = centroids + cond.reshape(height,width,1) * rho.reshape(height,width,1) * (X - centroids)

        v_new = v+(X-lost_X)*0.01*cond.reshape(height,width,1)
        v_new /= la.norm(v_new,axis=2).reshape(height,width,1)

        p_new = (1-rho)*p[:,:,clu] + rho*cond

        cyl[:, :, clu, :3] = v_new
        cyl[:, :, clu, 3] = R_new_2
        cyl[:, :, clu, 4] = L_new_2
        cyl[:, :, clu, 5:8] = centroids_new
        cyl[:, :, clu, 8] = p_new
    
    return cyl,1-(fg.astype('float')/fg.max())

def plot_helper(curves, data):
    for i in range(len(curves)):
        curves[i].setData(data[i])
    pg.QtGui.QApplication.processEvents()

class Window3d(object):

    def __init__(self,N,clusters):
        self.clusters = clusters

        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 100
        self.w.setWindowTitle('RGB space')
        self.w.setGeometry(100, 100, 500, 500)
        self.w.show()
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
        self.pca = decomposition.PCA(n_components=3)

        self.centers = [None for _ in range(clusters)]
        for i in range(clusters):
            pts = np.array([0, 0, 0])/14
            self.centers[i] = gl.GLScatterPlotItem(pos=pts, color=pg.glColor(0,0,255,255))
            self.w.addItem(self.centers[i])

        self.cyl=[]
        for i in range(clusters):
            CYL = gl.MeshData.cylinder(rows=10, cols=20, radius=[1., 1.0], length=5.)
            self.cyl.append(gl.GLMeshItem(meshdata=CYL, smooth=True, drawEdges=True, edgeColor=(1, 0, 0, 0.1), shader='balloon'))
            self.cyl[-1].setGLOptions('additive')
            self.w.addItem(self.cyl[-1])

        self.traces = dict()
        for i in range(N):
            pts = np.array([0, 0, 0])/14
            self.traces[i] = gl.GLScatterPlotItem(pos=pts, color=pg.glColor(100,100,100,100))
            self.w.addItem(self.traces[i])

    def set_plotdata(self, name, points, color):
        self.traces[name].setData(pos=points, color=color)

    def update(self,data):
        clr=[255,255,255]
        for i in range(len(data)):
            self.set_plotdata(
                name=i, points=data[i] / 14,
                color=pg.glColor(clr[0], clr[1], clr[2], 100)
            )

        self.pca.fit(data)
        V = self.pca.components_
        x_pca_axis, y_pca_axis, z_pca_axis = 30 * V.T
        #X_tr = self.pca.transform(data)
        mean = np.average(data/14, axis=0)
        self.curve1.setData(pos=np.array([mean, [x_pca_axis[0], y_pca_axis[0], z_pca_axis[0]]]))
        self.curve2.setData(pos=np.array([mean, [x_pca_axis[1], y_pca_axis[1], z_pca_axis[1]]]))
        self.curve3.setData(pos=np.array([mean, [x_pca_axis[2], y_pca_axis[2], z_pca_axis[2]]]))

    def update_cylinders(self, cyl_info):
        # v1,v2,v3, R, L,c1,c2,c3, p
        #  0, 1, 2, 3, 4, 5, 6, 7, 8
        for i in range(self.clusters):
            vec = cyl_info[i,:3]
            vec = vec/(np.sum(np.power(vec,2))**0.5)
            r = (cyl_info[i,3]**0.5)/14
            l = (cyl_info[i,4]**0.5)/14

            self.cyl[i].resetTransform()
            CYL = gl.MeshData.cylinder(rows=10, cols=20, radius=[r, r], length=2*l)
            self.cyl[i].setMeshData(meshdata=CYL)
            self.cyl[i].setColor(pg.glColor(255,255,255, cyl_info[i,8]*100))

            a = np.rad2deg(np.arccos(vec[2]/(np.sum(vec**2)**0.5)))
            self.cyl[i].rotate(a, -vec[0], vec[1], 0)
            c = [cyl_info[i,5]/14 - vec[0]*l,cyl_info[i,6]/14 - vec[1]*l,cyl_info[i,7]/14 - vec[2]*l]
            self.cyl[i].translate(c[0],c[1],c[2])

            for i in range(self.clusters):
                self.centers[i].setData(pos=cyl_info[i,5:8]/14)
        pg.QtGui.QApplication.processEvents()

def mouse(event, x, y, flags, param):
    global y_tmp,x_tmp,fg_mask_freq
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Selected pixel', x, y)
        y_tmp[0]=y
        x_tmp[0]=x

if __name__ == '__main__':
    file_name = sys.argv[1]
    clusters = int(sys.argv[2])
    verbose = eval(sys.argv[3])
    playin = eval(sys.argv[4])
    playout = eval(sys.argv[5])
    out_name = sys.argv[6]
    print(file_name, clusters, verbose, playin, playout, out_name)

    frame_num = 1000
    N = 100

    if file_name == 'None':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_name)
    _, frame = cap.read()
    
    y_tmp, x_tmp = [0], [0]
    if playout:
        cv2.namedWindow("fg")
        cv2.setMouseCallback("fg", mouse)

    
    _height, _width = frame.shape[0], frame.shape[1]
    image_name = '{} {}X{}'.format(file_name, _height, _width)

    # initializing matrices
    history = np.zeros((N, _height, _width, 3), dtype='int')
    history[0, :, :, :] = frame

    cylinder = np.random.rand(_height,_width,clusters,9) # v1,v2,v3,R,L,c1,c2,c3,p
    freq = np.zeros((_height,_width,3, 256))
    # v1,v2,v3, R, L,c1,c2,c3, p
    #  0, 1, 2, 3, 4, 5, 6, 7, 8
    cylinder[:,:,:,3:5] = np.random.random()*25
    cylinder[:,:,:,:3] = 1

    for c in range(clusters):
        cylinder[:,:,c,5:8] = (c*255/(clusters - 1))
    
    # initializing plotting windows and curves
    if verbose:
        pw = pg.GraphicsWindow()
        w3d = Window3d(N,clusters)

    # initializing video writers
    if out_name != "None":
        writer = cv2.VideoWriter(out_name, -1, 24, (_width, _height), False)

    print("Initializing first {} frames".format(N))
    for _ in range(N):
        _, frame = cap.read()
        try:
            history[frame_num % N, :, :, :] = frame
            freq[np.arange(_height).repeat(_width*3).reshape(_height,_width,3),
                 np.arange(_width).repeat(_height *3).reshape(_width,_height,3).transpose([1,0,2]),
                 np.arange(3).repeat(_width * _height).reshape(3,_height, _width).transpose([1,2,0]),
                 frame]+=1
        except:
            break
        frame_num+=1
    print("Done!")

    while (frame_num<2000):
        
        _, frame = cap.read()
    
        lost_frame = None
        try:
            lost_frame = history[frame_num % N, :, :, :]
            freq[np.arange(_height).repeat(_width * 3).reshape(_height, _width, 3),
                 np.arange(_width).repeat(_height * 3).reshape(_width, _height, 3).transpose([1, 0, 2]),
                 np.arange(3).repeat(_width * _height).reshape(3, _height, _width).transpose([1, 2, 0]),
                 lost_frame] -= 1
            history[frame_num % N, :, :, :] = frame
            freq[np.arange(_height).repeat(_width * 3).reshape(_height, _width, 3),
                 np.arange(_width).repeat(_height * 3).reshape(_width, _height, 3).transpose([1, 0, 2]),
                 np.arange(3).repeat(_width * _height).reshape(3, _height, _width).transpose([1, 2, 0]),
                 frame] += 1

        except:
            break

        cylinder,fg = update_cyl(frame, lost_frame, cylinder, 0.01)
        if playout:
            cv2.imshow("fg", fg)
        if playin:
            cv2.imshow("ori", frame)
        #plotting and other stuff
        x = x_tmp[-1]
        y = y_tmp[-1]

        if verbose:
            w3d.update(history[:, y, x, :])
            w3d.update_cylinders(cylinder[y, x, :, :])
        if out_name != "None":
            writer.write((255 * fg).astype('uint8'))

    
        frame_num += 1

        if cv2.waitKey(20) & 0xFF == 27:
            break
    if out_name != "None":
        writer.release()
    cv2.destroyAllWindows()
