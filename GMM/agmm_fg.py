import numpy as np
import cv2,time
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import line_profiler
import scipy.stats as stats
import signal
import sys

from sklearn import decomposition


def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

def gmm(freq, clusters, ITERATIONS):
    # init

    f = np.add(freq, 1)

    height,width,length = f.shape
    condProb_cluster_pixel = np.random.randint(2, size=height*width*clusters * length).reshape(height,width,clusters, length).astype(np.float32)
    condProb_pixel_cluster = np.ndarray(shape=(height,width,length, clusters), dtype=np.float32)

    mean_cluster = np.ndarray(shape=(height,width,clusters,), dtype=np.float32)
    var_cluster = np.ones(shape=(height,width,clusters,), dtype=np.float32)
    prob_cluster = np.ndarray(shape=(height,width,clusters,), dtype=np.float32)

    linspace = np.arange(length).repeat(height*width).reshape(length,width, height).transpose()

    for i in range(ITERATIONS):  # ITERATIONS
        print(i)
        # cluster means
        for c in range(clusters):

            D = np.sum(condProb_cluster_pixel[:,:,c,:] * f, axis=2)
            N = np.sum(condProb_cluster_pixel[:,:,c,:] * f * linspace, axis=2)
            mean_cluster[:,:,c] = N / D

        # cluster variances
        for c in range(clusters):

            D = np.sum(condProb_cluster_pixel[:,:,c,:] * f, axis=2) + 1
            diff =  linspace- mean_cluster[:,:,c].repeat(256).reshape(height,width,-1)
            N = np.sum(condProb_cluster_pixel[:,:,c,:] * f * np.power(diff , 2), axis=2)
            var_cluster[:,:,c] = N / D

        # cluster probabilities
        for c in range(clusters):
            D = length
            N = np.sum(condProb_cluster_pixel[:,:,c,:], axis=2)
            prob_cluster[:, :, c] = N / D

        # prob(pixel | cluster)
        for c in range(clusters):
            condProb_pixel_cluster[:,:,:, c] = stats.norm(mean_cluster[:, :, c].reshape(height,width,1), np.sqrt(var_cluster[:, :, c].reshape(height,width,1))).pdf(np.arange(length))
            #for x in range(width):
            #    for y in range(height):
            #        condProb_pixel_cluster[y,x,:,c] =(1 / np.sqrt(2 * np.pi * var_cluster[y,x,c])) * np.exp(-1 * np.power(np.arange(length) - mean_cluster[y,x,c], 2) / (2 * var_cluster[y,x,c]))

        # prob(cluster | pixel)
        D = np.sum(condProb_pixel_cluster * f.repeat(clusters).reshape((height, width, length, clusters)), 3)
        for c in range(clusters):
            N = condProb_pixel_cluster[:,:,:, c] * f
            condProb_cluster_pixel[:,:,c] = (N / D)

    return mean_cluster, var_cluster, prob_cluster

def update_gaussians(X, mean, var, prob, alpha, clusters):

    _height, _width = X.shape

    x_new_3d = np.repeat(X, clusters).reshape(_height,_width,clusters)
    #cond_within_std_dev = np.where(x_new_3d - mean < 2.5*np.sqrt(var),0,1)

    min_dist = np.abs(x_new_3d - mean)
    cond_min_dist = (min_dist == np.repeat(min_dist.min(axis=2),clusters).reshape(_height,_width,clusters))

    # updating gaussians

    # updating probability of clusters
    prob = prob + alpha*(cond_min_dist - prob)*0.1

    rho = 1e-2 + alpha*stats.norm(mean,np.sqrt(var)).pdf(x_new_3d)

    # updating mean
    mean = mean + rho * (x_new_3d - mean) * cond_min_dist
    #mean = mean + rho * (x_new_3d - mean) * (1 - cond_min_dist)*1

    # updating variance
    var = var + rho * ((x_new_3d - mean) ** 2 - var) * cond_min_dist
    #var = var + rho * ((x_new_3d - mean) ** 2 - var) * (1-cond_min_dist) * 0.1

    return mean,var,prob

def find_fg(frame, mean, var, prob, clusters, var_offset=10, amp_gain=3, power_gain=40,mode="all"):
    # finding the gaussian with min var ---and max probability---
    _height, _width = frame.shape

    if mode=="all":
        tmp = np.sum(stats.norm(mean,np.sqrt(var)+var_offset).pdf(frame.repeat(clusters).reshape(_height,_width,clusters))*prob/np.sqrt(var),axis=2)

        fg_mask_frame = (1 - tmp*amp_gain)**power_gain

        return fg_mask_frame
    elif mode=="best":
        w = prob / var
        cond_best_gaus = (w == np.repeat(w.max(axis=2), clusters).reshape(_height, _width, clusters))

        t = np.sqrt(var * cond_best_gaus).max(axis=2) * 2.5
        mean_given = (mean * cond_best_gaus).max(axis=2)

        fg_mask_frame = (np.abs(mean_given - frame) > t).astype('float')
        return fg_mask_frame
    elif mode=="best continuous":
        w = prob / var
        cond_best_gaus = (w == np.repeat(w.max(axis=2), clusters).reshape(_height, _width, clusters))

        v = np.sqrt(var * cond_best_gaus).max(axis=2)
        m = (mean * cond_best_gaus).max(axis=2)
        p = (prob * cond_best_gaus).max(axis=2)
        tmp = stats.norm(m, np.sqrt(v) + var_offset).pdf(frame)*p
        return (1 - tmp*amp_gain)**power_gain

def plot_helper(curves, data):
    for i in range(len(curves)):
        curves[i].setData(data[i])
    pg.QtGui.QApplication.processEvents()

def getCurves(window, clusters, clusters_layer2):
    pl = window.addPlot()
    pl.setYRange(0, 1, padding=0)

    curves = {}

    curves['fg_mask_variation'] = pl.plot(fillLevel=0, pen=None, brush=(255, 255, 255, 10))
    curves['hist'] = pl.plot(fillLevel=0, pen=None, brush=(255, 0, 255, 40))
    curves['min_var_tot'] = pl.plot(brush=(255, 0, 255, 100))

    fg_g_curves = []
    for i in range(clusters_layer2):
        fg_g_curves.append(pl.plot(brush=(255, 255, 255, 100)))
    curves['fg_g_curves'] = fg_g_curves

    g_curves = []
    for i in range(clusters):
        g_curves.append(pl.plot(pen=(i, clusters)))
    g_curves.append(pl.plot(pen=(0, clusters)))
    curves['g_curves'] = g_curves

    pg.QtGui.QApplication.processEvents()

    return curves

class Window3d(object):

    def __init__(self,N):
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
        X_tr = self.pca.transform(data)
        mean = np.average(data, axis=0)
        self.curve1.setData(pos=np.array([mean, [x_pca_axis[0], y_pca_axis[0], z_pca_axis[0]]]))
        self.curve2.setData(pos=np.array([mean, [x_pca_axis[1], y_pca_axis[1], z_pca_axis[1]]]))
        self.curve3.setData(pos=np.array([mean, [x_pca_axis[2], y_pca_axis[2], z_pca_axis[2]]]))



def mouse(event, x, y, flags, param):
    global y_tmp,x_tmp,fg_mask_freq
    if event == cv2.EVENT_MOUSEMOVE:
        print('Selected pixel', x, y)
        y_tmp[0]=y
        x_tmp[0]=x

if __name__ == '__main__':
    frame_num = 1
    clusters = 5
    iterations = 10
    N = 100
    amp_gain = 20
    power_gain = 5
    var_offset = 10

    clusters_layer2=5

    y_tmp, x_tmp = [0], [0]

    signal.signal(signal.SIGINT, signal_handler)

    cv2.namedWindow("fg")
    cv2.setMouseCallback("fg", mouse)

    file_name = './dataset/dynamicbackground/boats/input/in{0:0>6}.jpg'
    frame = cv2.imread(file_name.format(frame_num))
    _height, _width = frame.shape[0], frame.shape[1]
    image_name = '{} {}X{}'.format(file_name, _height, _width)

    # initializing matrices
    history = np.zeros((N, _height, _width, 3), dtype='int')
    history[0, :, :, :] = frame
    fg_mask_freq = np.zeros((_height, _width, 256),np.int16)

    mean_dict, var_dict, prob_dict = np.zeros((3, _height, _width, clusters)), np.zeros(
        (3, _height, _width, clusters)), np.zeros((3, _height, _width, clusters))
    fg_mean, fg_var, fg_prob = np.zeros((_height, _width, clusters_layer2)), np.zeros(
        (_height, _width, clusters_layer2)), np.zeros((_height, _width, clusters_layer2))
    for x in range(_width):
        for y in range(_height):
            for c in range(clusters_layer2):
                fg_mean[y,x,c], fg_var[y,x,c], fg_prob[y,x,0] = 255 / 2 + np.random.randint(-100,100), 10, 1 / clusters_layer2

            for c in range(clusters):
                mean_dict[:,y,x,c] = (c*255/(clusters - 1))
                var_dict[:,y,x,c] = ((255/(clusters - 1)/2)**1)
                prob_dict[:,y,x,c] = (1/clusters)

    # initializing plotting windows and curves
    pw = pg.GraphicsWindow()
    curves0 = getCurves(pw, clusters, clusters_layer2)
    curves1 = getCurves(pw, clusters, clusters_layer2)
    curves2 = getCurves(pw, clusters, clusters_layer2)
    w3d = Window3d(N)

    # initializing video writers
    writer = cv2.VideoWriter('agmmfg.avi', -1, 24, (_width, _height), False)
    writer2 = cv2.VideoWriter('agmm2fg.avi', -1, 24, (_width, _height), False)


    while (frame_num<1500):

        frame = cv2.imread(file_name.format(frame_num))
        #_,frame = cap.read()
        try:
            history[frame_num % N, :, :, :] = frame
        except:
            break
        # frame = np.average(history[frame_num%N, :, :, :],2)

        # update gaussians
        mean_dict[0], var_dict[0], prob_dict[0] = update_gaussians(frame[:, :, 0], mean_dict[0], var_dict[0],
                                                                   prob_dict[0], 0.1, clusters)
        mean_dict[1], var_dict[1], prob_dict[1] = update_gaussians(frame[:, :, 1], mean_dict[1], var_dict[1],
                                                                   prob_dict[1], 0.1, clusters)
        mean_dict[2], var_dict[2], prob_dict[2] = update_gaussians(frame[:, :, 2], mean_dict[2], var_dict[2],
                                                                   prob_dict[2], 0.1, clusters)

        fg = find_fg(frame[:, :, 0], mean_dict[0], var_dict[0], prob_dict[0],
                     clusters, var_offset, amp_gain, power_gain, mode="all")
        cv2.imshow("fg", fg)
        '''fg = find_fg(frame[:, :, 1], mean_dict[1], var_dict[1], prob_dict[1],
                     clusters, var_offset, amp_gain, power_gain, mode="all")
        cv2.imshow("fg GREEN", fg)
        fg = find_fg(frame[:, :, 2], mean_dict[2], var_dict[2], prob_dict[2],
                     clusters, var_offset, amp_gain, power_gain, mode="all")
        cv2.imshow("fg BLUE", fg)'''

        #fg_mean, fg_var, fg_prob = update_gaussians((255*fg).astype('int8'), fg_mean, fg_var, fg_prob, 0.0001, clusters_layer2)

        if (frame_num%500==0):
            fg_mean, fg_var, fg_prob= gmm(fg_mask_freq,clusters_layer2,10)

        # for a given pixel fg varies betwen 0 and 1.
        # now we need to find how this varies. if it vavries all the time. it may be a dynamic background
        fg2 = find_fg(fg * 255, fg_mean, fg_var, fg_prob, clusters_layer2,var_offset=20, amp_gain=10, power_gain=80,mode="all")




        # updating the fg frequencies (0 to 1 transformed to 0 to 256)
        i = np.arange(_height).repeat(_width).reshape(_height,_width).flatten()
        j = np.arange(_width).repeat(_height).reshape(_width,_height).transpose().flatten()
        k = np.round(fg.flatten()*255).astype('int8')
        fg_mask_freq[[i,j,k]]+=1

        #plotting and other stuff
        x = x_tmp[-1]
        y = y_tmp[-1]
        data = np.zeros((3,256))
        for i in range(N):
            data[0,history[i, y, x, 0]] += 1
            data[1, history[i, y, x, 1]] += 1
            data[2, history[i, y, x, 2]] += 1

        w3d.update(history[:,y,x,:])
        plot_helper([curves0['hist'],curves1['hist'],curves2['hist']], [data[0] / np.max(data[0]),data[1] / np.max(data[1]),data[2] / np.max(data[2])])

        tot = np.zeros(256)
        for c in range(clusters):
            tmp = prob_dict[0,y,x,c]/np.sqrt(var_dict[0,y,x,c])*stats.norm(mean_dict[0,y,x,c],np.sqrt(var_dict[0,y,x,c]+var_offset)).pdf(np.arange(256))
            tot += tmp
            plot_helper([curves0['g_curves'][c]],[tmp*amp_gain])
        plot_helper([curves0['g_curves'][-1]], [(1-amp_gain*tot)**power_gain])
        tot = np.zeros(256)
        for c in range(clusters):
            tmp = prob_dict[1,y,x,c]/np.sqrt(var_dict[1,y,x,c])*stats.norm(mean_dict[1,y,x,c],np.sqrt(var_dict[1,y,x,c]+var_offset)).pdf(np.arange(256))
            tot += tmp
            plot_helper([curves1['g_curves'][c]],[tmp*amp_gain])
        plot_helper([curves1['g_curves'][-1]], [(1-amp_gain*tot)**power_gain])
        tot = np.zeros(256)
        for c in range(clusters):
            tmp = prob_dict[2,y,x,c]/np.sqrt(var_dict[2,y,x,c])*stats.norm(mean_dict[2,y,x,c],np.sqrt(var_dict[2,y,x,c]+var_offset)).pdf(np.arange(256))
            tot += tmp
            plot_helper([curves2['g_curves'][c]],[tmp*amp_gain])
        plot_helper([curves2['g_curves'][-1]], [(1-amp_gain*tot)**power_gain])

        for c in range(clusters_layer2):
            plot_helper([curves0['fg_g_curves'][c]], [500*fg_prob[y,x,c]/np.sqrt(fg_var[y,x,c])*stats.norm(fg_mean[y,x,c],np.sqrt(fg_var[y,x,c])).pdf(np.arange(256))])
        plot_helper([curves0['fg_mask_variation']],[fg_mask_freq[y,x]/fg_mask_freq[y,x].max()])

        writer.write((255 * fg).astype('uint8'))
        writer2.write((255 * fg2).astype('uint8'))


        cv2.imshow("fg2", fg2)
        print(frame_num)
        frame_num += 1

        if cv2.waitKey(20) & 0xFF == 27:
            break

    writer.release()
    writer2.release()
    cv2.destroyAllWindows()
