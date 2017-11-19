import numpy as np
import cv2
import pyqtgraph as pg


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
            D = np.sum(data)
            N = np.sum(condProb_cluster_pixel[c] * data)
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
            for x in range(length):
                N = condProb_pixel_cluster[x][c] * prob_cluster[c] * data[x]
                D = np.sum(condProb_pixel_cluster[x] * prob_cluster[c] * data[x])

                condProb_cluster_pixel[c][x] = N / D
                if np.isnan(condProb_cluster_pixel[c][x]):
                    raise ArithmeticError

    return mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster

def find_fg(mean, var, var_thresh, prob_thresh):
    small_var_cluster = []

    condProb_pixel_cluster = np.zeros((256, len(mean)))

    for c in range(len(mean)):
        if var[c] < var_thresh:
            small_var_cluster.append(c)
            condProb_pixel_cluster[:,c] = (1 / np.sqrt(2 * np.pi * var[c])) * np.exp(-1 * np.power(np.arange(256) - mean[c], 2) / (2 * var[c]))

    mask = np.ones(256)
    tot = np.zeros(256)

    for val in range(256):
        tmp = 0
        for c in small_var_cluster:
            tmp += condProb_pixel_cluster[val,c]
        tot[val] = tmp

    tot = tot/np.max(tot)
    for i in range(256):
        if tot[i] > prob_thresh:
            mask[i] = 0

    return mask, tot

def build_fg_mask(history, fg_color_mask):
    height, width = len(history[0]), len(history[0][0])
    fg_mask = np.zeros((len(history), height, width, 1))

    print("Finding fg matrix")
    for i in range(len(history)):
        for x in range(width):
            for y in range(height):
                fg_mask[i, y, x, 0] = fg_color_mask[y,x,history[i,y,x,1]]
    print("Completed")
    return fg_mask

def plot_helper(curves, data):
    for i in range(len(curves)):
        curves[i].setData(data[i])
    pg.QtGui.QApplication.processEvents()

def load_log( clusters, iter, start, end, height, width):
    log = open("log.txt",'r')

    load = False
    mean,var = {},{}
    for x in range(width):
        for y in range(height):
            mean["{},{}".format(x,y)] = None

    for line in log:
        v = line.strip().split()
        if v[0] == "NE":
            if v[1:] == list(map(str,[clusters, iter, start, end])):
                load = True
                print("log match found")
            else:
                load = False

        if load:
            if v[0] == "MEAN":
                mean[v[1]] = (list(map(float,v[2].split(','))))
            elif v[0] == "VAR":
                var[v[1]] = (list(map(float, v[2].split(','))))
    return mean, var

if __name__ == '__main__':


    frame_num = 1
    file_name = './dataset/dynamicBackground/boats/input/in{0:0>6}.jpg'.format(frame_num)
    print(file_name)
    frame = cv2.imread(file_name)


    clusters = 5
    iterations = 10

    _height, _width = frame.shape[0], frame.shape[1]
    image_name = '{} {}X{}'.format(file_name, _height, _width)

    N = 700
    history = np.zeros((N, _height, _width, 3), dtype=int)
    history[0, :, :, :] = frame
    '''
    pw = pg.GraphicsWindow()
    pl = pw.addPlot()
    pl.setYRange(0, 1, padding=0)

    color_mask,hist,min_var_tot = pl.plot(fillLevel=0, brush=(255, 255, 255, 40)),pl.plot(pen=(1,2)),pl.plot( brush=(255, 0, 255, 100))

    g_curves = []
    for i in range(clusters):
        g_curves.append(pl.plot(fillLevel=i,pen=(i,clusters)))
    pg.QtGui.QApplication.processEvents()
    '''
    start = 0
    end = 60



    while (1):
        print(frame_num)
        

        frame = cv2.imread('./dataset/dynamicBackground/boats/input/in{0:0>6}.jpg'.format(frame_num))
        try:
            history[frame_num % N, :, :, :] = frame
        except:
            break

        if frame_num%N == 0:

            mean_dict, var_dict = load_log(clusters, iterations, frame_num - N, frame_num, _height,_width)
            log = open('log.txt', 'a')
            log.write("NE " + str(clusters) + " " + str(iterations) + " " + str(frame_num - N)+" "+ str(frame_num)+'\n')

            # estimating gaussians
            for y in range(start,end):
                print("checking",y)
                for x in range(_width):
                    if (mean_dict["{},{}".format(x,y)]!=None):
                        print("skipping", "{},{}".format(x,y))
                        continue

                    if (x%20)==0:
                        print("$$",y,x)
                    data = np.zeros(256)
                    for i in range(N):
                        data[history[i,y,x,1]] += 1

                    mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster = gmm(data, clusters, iterations)
                    log.write("MEAN "+"{},{} ".format(x,y)+",".join(map(str,mean_cluster))+'\n')
                    log.write("VAR " + "{},{} ".format(x, y) + ",".join(map(str, var_cluster))+'\n')

            # finding fg color masks
            fg_color_mask = np.zeros((_height,_width,256))
            for y in range(start, end):
                for x in range(_width):
                    index = "{},{}".format(x,y)
                    if mean_dict[index]==None:
                        continue
                    fg_color_mask[y,x,:],tot = find_fg(mean_dict[index], var_dict[index],750,0.1)

                    data = np.zeros(256)
                    for i in range(N):
                        data[history[i, y, x, 1]] += 1
                    #plot_helper([hist, min_var_tot, color_mask],[data/np.max(data), tot, fg_color_mask[y,x,:]])
            # finding fg
            fg = build_fg_mask(history, fg_color_mask)

            writer = cv2.VideoWriter('test1.avi', -1, N, (_width, _height), False)
            for i in range(N):
                writer.write((255*fg[i]).astype('uint8'))
            writer.release()
        frame_num += 1
        #cv2.imshow(image_name,  history[(frame_num-1) % N, :, :, :]/255.0)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    _ = input()
    cv2.destroyAllWindows()
