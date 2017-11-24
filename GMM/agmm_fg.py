import numpy as np
import cv2,time
import pyqtgraph as pg
import line_profiler
import scipy.stats as stats
import signal
import sys

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


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

    return mean_cluster, var_cluster, prob_cluster


def update_gaussians_and_find_fg_color_mask_for_pixel(x_new, mean, var, prob, alpha, var_thresh, prob_thresh):
    """
    param:
    x_new - new value for the given pixel
    mean - array of mean values for the given pixel
    var - array of variance values for the given pixel
    prob - probability of the gaussian array for the given pixel

    alpha - learning rate
    var_thresh - variance threshold value
    prob_thresh - minimum probability threshold value
    """
    
    # finding the best gaussian match for x_new
    # x_new in 2.5 std deviation
    close_gaussians = []
    closest_gaussian, min_std_dev = 0,1e10
    for c in range(len(var)):
        if abs(x_new - mean[c]) < 2.5*np.sqrt(var[c]):
            close_gaussians.append(c)
            if abs(x_new - mean[c]) < min_std_dev:
                min_std_dev = abs(x_new - mean[c])
                closest_gaussian = c
    if (var[closest_gaussian]>var_thresh):
        print(var[closest_gaussian], closest_gaussian, var)
    # updating gaussians

    # updating probability of clusters
    for c in range(len(mean)):
        if c == closest_gaussian:
            prob[c] = prob[c] + alpha * (1 - prob[c])
        else:
            prob[c] = prob[c] + alpha * (0 - prob[c])

    rho = 1e-2 + alpha* (1 / np.sqrt(2 * np.pi * var[closest_gaussian])) * np.exp(-1 * np.power(x_new - mean[closest_gaussian], 2) / (2 * var[closest_gaussian]))
    
    #print(x_new, "gaussian", "mean", mean[closest_gaussian], "var", var[closest_gaussian],"rho",rho)

    # updating mean
    mean[closest_gaussian] = mean[closest_gaussian] + rho * (x_new - mean[closest_gaussian]) * 2

    # updating variance
    var[closest_gaussian] = var[closest_gaussian] + rho * ((x_new - mean[closest_gaussian])**2 - var[closest_gaussian])
    if var[closest_gaussian]<1:
        var[closest_gaussian] = 1
    #print("new gaussian", "mean", mean[closest_gaussian], "var", var[closest_gaussian])


    # finding the gaussian with min var ---and max probability---    
    mask = np.ones(256)
    tmp = 0
    idx = -1
    for i in range(len(prob)):
        q = prob[i]/var[i]
        if (q>tmp):
            idx = i
            tmp = q

    t = np.sqrt(var[idx])*3
    for i in range(256):
        if i > mean[idx] - t and i < mean[idx] + t:
            mask[i] = 0

    return mask, []#(1 / np.sqrt(2 * np.pi * var[idx])) * np.exp(-1 * np.power(np.arange(256) - mean[idx], 2) / (2 * var[idx]))

@profile
def update_gaussians(X, mean, var, prob, alpha, clusters):
    """
    param:
    x_new - 2d new values | shape - (height, width)
    mean - 3d array of mean values
    var - 3d array of variance values
    prob - probability of the gaussian array

    alpha - learning rate
    var_thresh - variance threshold value
    prob_thresh - minimum probability threshold value
    """
    
    # finding the best gaussian match for x_new
    # x_new in 2.5 std deviation

    _height, _width = X.shape

    x_new_3d = np.repeat(X, clusters).reshape(_height,_width,clusters)
    #cond_within_std_dev = np.where(x_new_3d - mean < 2.5*np.sqrt(var),0,1)

    min_dist = np.abs(x_new_3d - mean)
    cond_min_dist = (min_dist == np.repeat(min_dist.min(axis=2),clusters).reshape(_height,_width,clusters))

    # updating gaussians

    # updating probability of clusters
    prob = prob + alpha*(cond_min_dist - prob)

    rho = 1e-2 + alpha*stats.norm(mean,var).pdf(x_new_3d)

    # updating mean
    mean = mean + rho * (x_new_3d - mean) * cond_min_dist

    # updating variance
    var = var + rho * ((x_new_3d - mean)**2 - var)*cond_min_dist

    return mean,var,prob
@profile
def find_fg(frame, mean, var, prob, clusters):
    # finding the gaussian with min var ---and max probability---
    _height, _width = frame.shape
    w = prob/var
    cond_best_gaus = (w == np.repeat(w.max(axis=2),clusters).reshape(_height,_width,clusters))

    t = np.sqrt(var*cond_best_gaus).max(axis=2)*2.5
    mean_given = (mean*cond_best_gaus).max(axis=2)

    fg_mask_frame = (np.abs(mean_given-frame)>t).astype('float')
    
    return fg_mask_frame

def plot_helper(curves, data):
    for i in range(len(curves)):
        curves[i].setData(data[i])
    pg.QtGui.QApplication.processEvents()

def load_log( clusters, iter, start, end, file, height, width):
    log = open("log_agmm.txt",'r')

    load = False
    mean,var,prob = {},{},{}
    for x in range(width):
        for y in range(height):
            mean["{},{}".format(x,y)] = None
            var["{},{}".format(x,y)] = None
            prob["{},{}".format(x,y)] = None

    for line in log:
        v = line.strip().split()
        if v[0] == "NE":
            if v[1:] == list(map(str,[clusters, iter, start, end, file])):
                load = True
            else:
                load = False

        if load:
            if v[0] == "MEAN":
                mean[v[1]] = list(map(float,v[2].split(',')))
            elif v[0] == "VAR":
                var[v[1]] = list(map(float, v[2].split(',')))
            elif v[0] == "PROB":
                prob[v[1]] = list(map(float, v[2].split(',')))
    return mean, var, prob

def display_results(start, end, _input, _output):
    cap = cv2.VideoCapture(_output)
    for i in range(start, end):
        i_frame = cv2.imread(_input.format(i))
        _, o_frame = cap.read()
        cv2.imshow("input", i_frame)
        cv2.imshow("output", o_frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame_num = 1
    clusters = 3
    iterations = 10
    N = 100
    y_tmp, x_tmp = 40, 200

    signal.signal(signal.SIGINT, signal_handler)

    file_name = './dataset/dynamicBackground/boats/input/in{0:0>6}.jpg'
    frame = cv2.imread(file_name.format(frame_num))
    _height, _width = frame.shape[0], frame.shape[1]
    image_name = '{} {}X{}'.format(file_name, _height, _width)

    history = np.zeros((N, _height, _width, 3), dtype='int')
    history[0, :, :, :] = frame

    pixel_history = []


    pw = pg.GraphicsWindow()
    pl = pw.addPlot()
    pl.setYRange(0, 1, padding=0)

    color_mask,hist,min_var_tot = pl.plot(fillLevel=0, brush=(255, 255, 255, 40)),pl.plot(pen=(1,2)),pl.plot( brush=(255, 0, 255, 100))

    g_curves = []
    for i in range(clusters):
        g_curves.append(pl.plot(fillLevel=i,pen=(i,clusters)))
    pg.QtGui.QApplication.processEvents()

    #writer = cv2.VideoWriter('agmmfg.avi', -1, N, (_width, _height), False)

    mean_dict,var_dict, prob_dict = np.zeros((_height,_width,clusters)),np.zeros((_height,_width,clusters)),np.zeros((_height,_width,clusters))
    for x in range(_width):
        for y in range(_height):
            for c in range(clusters):
                mean_dict[y,x,c] = (c*255/(clusters - 1))
                var_dict[y,x,c] = ((255/(clusters - 1)/2)**1)
                prob_dict[y,x,c] = (1/clusters)

    while (1):

        frame = cv2.imread(file_name.format(frame_num))
        #_,frame = cap.read()
        try:
            history[frame_num % N, :, :, :] = frame
            pixel_history.append(frame[y_tmp,x_tmp,1])
        except:
            break

        # update gaussians
        mean_dict, var_dict, prob_dict=update_gaussians(history[frame_num%N, :, :, 1],mean_dict,var_dict,prob_dict, 0.1,clusters)
        if (frame_num>N):
            fg=find_fg(history[frame_num%N, :, :, 1],mean_dict,var_dict,prob_dict,clusters)
            x = x_tmp
            y = y_tmp
            data = np.zeros(256)
            for i in range(N):
                data[history[i,y,x,1]] += 1
            plot_helper([hist], [data / np.max(data)])

            for c in range(clusters):
                plot_helper([g_curves[c]],[prob_dict[y,x,c]*3*(1 / np.sqrt(2 * np.pi * var_dict[y,x,c])) * np.exp(-1 * np.power(np.arange(256) -
                                                                                                             mean_dict[y,x,c], 2) / (2 * var_dict[y,x,c]))])


            #writer.write((255*fg).astype('uint8'))
            cv2.imshow("fg", fg)
        print(frame_num)
        frame_num += 1

        if cv2.waitKey(20) & 0xFF == 27:
            break

    #writer.release()
    _ = input()
    cv2.destroyAllWindows()
