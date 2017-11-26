import numpy as np
import cv2,sys,time
#import pyqtgraph as pg
#import line_profiler
import scipy.stats as stats

def gmm(data, clusters, ITERATIONS):
    # init
    data.astype('float')
    data += 1
    
    length = data.size
    condProb_cluster_pixel = np.random.randint(2,size=clusters*length).reshape(clusters, length).astype(np.float32)
    condProb_pixel_cluster = np.ndarray(shape=(length, clusters), dtype=np.float32)

    mean_cluster = np.ndarray(shape=(clusters,), dtype=np.float32)
    var_cluster = np.ones(shape=(clusters,), dtype=np.float32)
    prob_cluster = np.ndarray(shape=(clusters,), dtype=np.float32)

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
            condProb_pixel_cluster[:, c] = (1 / np.sqrt(2 * np.pi * var_cluster[c])) * np.exp(-1 * np.power(np.arange(length) - mean_cluster[c], 2) / (2 * var_cluster[c]))

        # prob(cluster | pixel)
        for c in range(clusters):
            if var_cluster[c] < 1e-4:
                continue
            N = condProb_pixel_cluster[:, c] * data
            D = np.sum(condProb_pixel_cluster * np.repeat(data, clusters).reshape((-1, clusters)), 1)
            condProb_cluster_pixel[c] = (N / D)

    return mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster


def find_fg(mean, var, var_thresh, prob_thresh):
    small_var_cluster = []

    condProb_pixel_cluster = np.zeros((256, len(mean)))
    

    for c in range(len(mean)):
        if var[c] < var_thresh:
            small_var_cluster.append(c)
            condProb_pixel_cluster[:,c] = (1 / np.sqrt(2 * np.pi * var[c])) * np.exp(-1 * np.power(np.arange(256) - mean[c], 2) / (2 * var[c]))

    # finding the gaussian with min var ---and max probability---
    '''
    mask = np.ones(256)
    m = min(var)
    for i in range(len(var)):
        if var[i] == m:
            idx = i
    t = np.sqrt(var[idx])*3
    for i in range(256):
        if i > mean[idx] - t and i < mean[idx] + t:
            mask[i] = 0

    return mask, condProb_pixel_cluster[:,idx]
    '''
    mask = np.ones(256)

    tot = np.sum(condProb_pixel_cluster[:,small_var_cluster], axis=1)

    max_tot = np.max(tot)
    if max_tot == 0:
        max_tot = 1
    tot = tot/max_tot

    for i in range(256):
        if tot[i] > prob_thresh:
            mask[i] = (1-tot[i])**40

    return mask, tot


def build_fg_mask(history, fg_color_mask):
    height, width = len(history[0]), len(history[0][0])
    fg_mask = np.zeros((len(history), height, width, 1))

    print("Finding fg matrix")
    tot_iter = width*height
    counter = 0
    comp = 0
    start_t = time.time()

    for x in range(width):
        for y in range(height):
            counter += 1
            if (counter % int(tot_iter/100) == 0):
                sys.stdout.write('\r')
                comp += 1
                t_per_iter = (time.time()-start_t)/int(tot_iter/100)
                sys.stdout.write("[%-20s] %d%% ETA=%.2f" % ('='*int(comp/5), comp , (tot_iter - counter)*t_per_iter))
                sys.stdout.flush()
                start_t = time.time()

            fg_mask[:, y, x, 0] = fg_color_mask[y,x,history[:,y,x,1]]
    print("")
    return fg_mask

'''def plot_helper(curves, data):
    for i in range(len(curves)):
        curves[i].setData(data[i])
    pg.QtGui.QApplication.processEvents()'''

def load_log( clusters, iter, start, end, file, height, width):
    log = open("log.txt",'r')

    load = False
    mean,var = {},{}
    for x in range(width):
        for y in range(height):
            mean["{},{}".format(x,y)] = None
            var["{},{}".format(x,y)] = None

    for line in log:
        v = line.strip().split()
        if v[0] == "NE":
            if v[1:] == list(map(str,[clusters, iter, start, end, file])):
                load = True
            else:
                load = False

        if load:
            if v[0] == "MEAN":
                mean[v[1]] = (list(map(float,v[2].split(','))))
            elif v[0] == "VAR":
                var[v[1]] = (list(map(float, v[2].split(','))))
    return mean, var

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

    cap = cv2.VideoCapture("output_video.avi")
    frame_num = 4901
    clusters = 5
    iterations = 10
    N = 700

    file_name = './dataset/dynamicBackground/boats/input/in{0:0>6}.jpg'

    frame = cv2.imread(file_name.format(frame_num))
    #_,frame = cap.read()


    _height, _width = frame.shape[0], frame.shape[1]
    image_name = '{} {}X{}'.format(file_name, _height, _width)
    print(_height, _width)

    history = np.zeros((N, _height, _width, 3), dtype=int)
    history[0, :, :, :] = frame
    
    '''pw = pg.GraphicsWindow()
    pl = pw.addPlot()
    pl.setYRange(0, 1, padding=0)

    color_mask,hist,min_var_tot = pl.plot(fillLevel=0, brush=(255, 255, 255, 40)),pl.plot(pen=(1,2)),pl.plot( brush=(255, 0, 255, 100))

    g_curves = []
    for i in range(clusters):
        g_curves.append(pl.plot(fillLevel=i,pen=(i,clusters)))
    pg.QtGui.QApplication.processEvents()'''
    
    start = 0
    end = _height



    while (1):
        sys.stdout.write('\r')
        sys.stdout.write("%d" %frame_num)
        

        frame = cv2.imread(file_name.format(frame_num))
        #_,frame = cap.read()
        try:
            history[frame_num % N, :, :, :] = frame
        except:
            break

        if frame_num%N == 0:

            mean_dict, var_dict = load_log(clusters, iterations, frame_num - N, frame_num, file_name, _height,_width)
            log = open('log.txt', 'a')
            log.write("NE " + str(clusters) + " " + str(iterations) + " " + str(frame_num - N)+" "+ str(frame_num)+" " + file_name+'\n')

            # estimating gaussians
            print("\nestimating gaussians")
            tot_iter = (end-start)*_width
            counter,start_t = 0,time.time()
            for y in range(start,end):
                for x in range(_width):
                    counter+=1
                    if (counter % int(tot_iter/100) == 0):
                        sys.stdout.write('\r')
                        comp = (y*_width + x + 1)/tot_iter
                        t_per_iter = (time.time()-start_t)/int(tot_iter/100)
                        sys.stdout.write("[%-20s] %d%% ETA=%.2f" % ('='*int(comp*20), 100*comp , (tot_iter - counter)*t_per_iter))
                        sys.stdout.flush()
                        start_t = time.time()
                    
                    index = "{},{}".format(x,y)
                    if (mean_dict[index]!=None and var_dict[index]!=None):
                        continue

                    data = np.zeros(256)
                    for i in range(N):
                        data[history[i,y,x,1]] += 1

                    mean_cluster, var_cluster, condProb_cluster_pixel, condProb_pixel_cluster = gmm(data, clusters, iterations)
                    
                    mean_dict[index] = mean_cluster
                    var_dict[index] = var_cluster

                    log.write("MEAN "+index+" "+",".join(map(str,mean_cluster))+'\n')
                    log.write("VAR " + index +" "+ ",".join(map(str, var_cluster))+'\n')
            print('\n')

            # finding fg color masks
            print("finding color masks")
            counter,start_t = 0,time.time()
            fg_color_mask = np.zeros((_height,_width,256))
            for y in range(start, end):
                for x in range(_width):
                    counter += 1
                    if (counter % int(tot_iter/100) == 0):
                        sys.stdout.write('\r')
                        comp = (y*_width + x + 1)/tot_iter
                        t_per_iter = (time.time()-start_t)/int(tot_iter/100)
                        sys.stdout.write("[%-20s] %d%% ETA=%.2f" % ('='*int(comp*20), 100*comp , (tot_iter - counter)*t_per_iter))
                        sys.stdout.flush()
                        start_t = time.time()
                        
                    index = "{},{}".format(x,y)
                    if mean_dict[index] is None:
                        continue
                    fg_color_mask[y,x,:],tot = find_fg(mean_dict[index], var_dict[index],500,0.02)

                    '''
                    if index == "240,200":
                        data = np.zeros(256)
                        for i in range(N):
                            data[history[i, y, x, 1]] += 1
                        plot_helper([hist, min_var_tot, color_mask],[data/np.max(data), tot, fg_color_mask[y,x,:]])'''
            print('\n')

            # finding fg
            fg = build_fg_mask(history, fg_color_mask)

            writer = cv2.VideoWriter('fg{}.avi'.format(frame_num-N), -1, 24, (_width, _height), False)
            for i in range(N):
                writer.write((255*fg[i]).astype('uint8'))
            writer.release()

            #display results
            display_results(frame_num-N, frame_num, file_name,'fg{}.avi'.format(frame_num-N) )

        frame_num += 1
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
