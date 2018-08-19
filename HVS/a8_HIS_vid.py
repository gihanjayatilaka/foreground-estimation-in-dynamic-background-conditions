from collections import defaultdict
# from queue import PriorityQueue
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
from scipy.stats import normaltest
from scipy import stats
def save_vid(output, name):
    n_frame, rows, cols = output.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(name, fourcc, 20.0, (cols, rows), False)

    for i in range(n_frame):
        img = output[i, :, :]
        out_vid.write(img)
        cv2.imshow('boat', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

    out_vid.release()


class Video:
    def __init__(self, m, n, nf):
        self.rows = m
        self.cols = n
        self.n_frame = nf
        self.V = m * n * nf
        self.n_par = 0
        self.label = [0 for i in range(self.V)]
        self.adj = None
        self.segment = None

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def dfs(self):
        self.segment = defaultdict(list)
        self.adj = defaultdict(set)
        for i in range(self.n_frame):
            for j in range(self.rows):
                for k in range(self.cols):
                    a = self.label[i][j][k]
                    self.segment[a].append((i, j, k))       #add to segment dict

                    try:
                        b = self.label[i + 1][j][k]
                        if a != b:
                            self.adj[a].add(b)
                    except:
                        pass
                    try:
                        c = self.label[i][j + 1][k]
                        if a != c:
                            self.adj[a].add(c)
                    except:
                        pass
                    try:
                        d = self.label[i][j][k + 1]
                        if a != d:
                            self.adj[a].add(d)
                    except:
                        pass

    def mak_label(self, parent):
        uniq = defaultdict()  # dic to map unique parents
        u = 0

        for i in range(self.V):
            par = self.find(parent, i)
            if par in uniq:
                self.label[i] = uniq[par]
            else:
                self.label[i] = u
                uniq[par] = u
                u += 1

        self.n_par = u

        self.label = np.reshape(np.asarray(self.label, dtype=int), (self.n_frame, self.rows, self.cols))

        print('Constructing adjacency list and segmenting...')
        self.dfs()  # construct adjacency list

        return self.label

    def mak_label_util(self, parent, lab):

        self.label = np.reshape(np.asarray(self.label, dtype=int), (self.n_frame, self.rows, self.cols))

        uniq = defaultdict()  # dic to map unique parents
        u = 0

        for i in range(self.n_frame):
            for j in range(self.rows):
                for k in range(self.cols):
                    a = lab[i][j][k]
                    par = self.find(parent, a)

                    if par in uniq:
                        self.label[i][j][k] = uniq[par]
                    else:
                        self.label[i][j][k] = u
                        uniq[par] = u
                        u += 1

        self.n_par = u

        print('Constructing adjacency list and segmenting...')
        self.dfs()  # construct adjacency list

        return self.label

    def remove_pix(self, pix):
        print('Removing smaller segments...')
        parent = [i for i in range(self.n_par)]
        #for i in self.segment:
        #    if len(self.adj[i]) == 1 and len(video.segment[i]) <= pix:
        #        parent[i] = next(iter(self.adj[i]))
        for i in self.segment:
            if len(video.segment[i]) <= pix:
                par = self.find(parent, next(iter(self.adj[i])))
                for e in self.adj[i]:
                    if len(video.segment[par]) < len(video.segment[e]):
                        par = self.find(parent, e)
                parent[i] = par

        return self.mak_label_util(parent, self.label)

    def mak_vid(self, name, dist):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(name, fourcc, 20.0, (self.cols, self.rows), 1)

        #try:
            #dist = np.random.rand(self.n_par, 3)  # give random RGB values to each parent
            #output = np.reshape(np.asarray([dist[i] for i in self.label]), (self.n_frame, self.rows, self.cols, 3)) * 255
            #output = output.astype(np.uint8)

        l = len(dist)
        output = np.reshape(np.asarray([dist[i%l] for i in self.label]), (self.n_frame, self.rows, self.cols, 3))



        for i in range(n_frame):
            img = output[i, :, :, :]
            out_vid.write(img)
            cv2.imshow('boat', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.destroyAllWindows()

        out_vid.release()

class Graph:
    def __init__(self, V, k):
        self.V = V
        self.graph = []
        self.k = k

    def addEdge(self, u, v, w):
        self.graph.append((u, v, w))

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, x, y, parent, rank, weight, number, max_w, num):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
            par = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
            par = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
            par = xroot

        max_w[par] = weight
        num[par] = number

    def mst(self):

        i = 0  # An index variable, used for sorted edges
        e = 0  # An index variable, used for result[]
        parent = [i for i in range(self.V)]
        rank = [0 for i in range(self.V)]
        max_w = [0 for i in range(self.V)]
        num = [1 for i in range(self.V)]

        self.graph = sorted(self.graph, key=lambda item: item[2])
        l = len(self.graph)

        for i in range(l):
            # print(i)
            u, v, w = self.graph[i]
            x = self.find(parent, u)
            y = self.find(parent, v)

            # No cycle
            if x != y:
                mw = max(max_w[x], max_w[y])
                count = num[x] + num[y]
                cost = mw + self.k / count
                if cost > w:
                    e = e + 1
                    self.union(x, y, parent, rank, mw, count, max_w, num)

        return parent

'''
st_frame = 1970
n_frame = 20
k0 = 15
k1 = 100
k_fact = 1.2
n_bin = 100
pix = 4
pix_fact = 1.2
n_reps = 15
out
'''
out_name = 'b'
st_frame = 1970
n_frame = 10
k0 = 80
k1 = 100
k_fact = 1.2
n_bin = 100
pix = 4
pix_fact = 1.2
n_reps = 5

a = np.linspace(0, 255, 10).astype(int)
b = np.hstack([a, a, a])
perm = (list(set(permutations(b, 3))))
shuffle(perm)
colour = np.array(perm, dtype = np.uint8)



name = 'boats\input\in' + str(1).zfill(6) + '.jpg'
img = cv2.imread(name, 1)

m, n, o = img.shape
in_vid = np.ndarray(shape=(n_frame, m, n, o), dtype=np.uint8)

for i in range(n_frame):
    name = 'boats\input\in' + str(st_frame + i).zfill(6) + '.jpg'
    img = cv2.imread(name, 1)
    in_vid[i, :, :, :] = img
    cv2.imshow('boat', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()

vid = in_vid.astype(float) / 255

# Graph
print('Constructing graph...')
video = Video(m, n, n_frame)

g = Graph(video.V, k0)

w1 = np.sum(np.square(vid[:, :, 0:n-1, :] - vid[:, :, 1:n, :]), axis=3)
w2 = np.sum(np.square(vid[:, 0:m-1, :, :] - vid[:, 1:m,:, :]), axis=3)
w3 = np.sum(np.square(vid[0:n_frame-1, :, :, :] - vid[1:n_frame, :, :, :]), axis=3)
#print(n_frame, m, n)
#print(w1.shape)
for k in range(n_frame):
    for i in range(m):
        for j in range(n):
            u = k * m * n + i * n + j
            try:
                g.addEdge(u, u + 1, w1[k, i, j])
            except:
                pass
            try:
                g.addEdge(u, u + n, w2[k, i, j])
            except:
                pass
            try:
                g.addEdge(u, u + n * m, w3[k, i, j])
            except:
                pass
            #w = sum((vid[k][i][j][:] - vid[k][i][j + 1][:]) ** 2)
            #w = sum((vid[k][i][j][:] - vid[k][i + 1][j][:]) ** 2)
            #w = sum((vid[k][i][j][:] - vid[k + 1][i][j][:]) ** 2)

print('Number of nodes = ', g.V)
print('Number of edges = ', len(g.graph))

weight = [w for (u, v, w) in g.graph]
n_bin = 100
plt.hist(weight, bins = n_bin, range = (min(weight), max(weight)/400))
plt.show()


# MST
print('MST...')
parent = g.mst()
video.mak_label(parent)
lab = video.remove_pix(pix)
print(video.n_par)

video.mak_vid('output.avi', colour)

print('Creating bins...')
bins = np.linspace(0, 1, n_bin+1)


for x in range(n_reps):
    pix *= pix_fact
    k1 *= k_fact

    print('\nrepetition ' + str(x+1))

    print('Constructing histogram...')
    hist = np.ndarray(shape=(video.n_par, n_bin, 3), dtype = np.float64)

    for par in range(video.n_par):
        for col in range(3):
            data = [vid[i][j][k][col] for (i, j, k) in video.segment[par]]
            hist[par, :, col] = np.histogram(data, bins, density=True)[0]

    g = Graph(video.n_par, k1)
    eps = np.amin(hist)

    print('Constructing graph...')
#    try:
#        h1 = (np.reshape(hist, (video.n_par, 1, n_bin, 3)))
#        h2 = (np.reshape(hist, (1, video.n_par, n_bin, 3)))
#        print(h1.shape)
#        print(h2.shape)
#        w = np.sum(np.sum(np.square(h1 - h2), axis=3), axis=2)
#        print(w.shape)
#        for i in video.adj:
#            for j in video.adj[i]:
#                g.addEdge(i, j, w[i][j])
#    except:
    for i in video.adj:
        for j in video.adj[i]:
            chi = sum(sum(np.square((hist[i, :, :] - hist[j, :, :]))))#/(hist[i, :] + hist[j, :] + eps)))
            g.addEdge(i, j, chi)

    video = Video(m, n, n_frame)

    print('MST...')
    parent = g.mst()

    print('Making label...')
    lab = video.mak_label_util(parent, lab)
    lab = video.remove_pix(pix)
    print(video.n_par)
    video.mak_vid('output' + str(x).zfill(3) + '.avi', colour)


#data = [len(video.segment[i]) for i in video.segment]
#n_bin = int(max(data)/1000)
#print(n_bin)
#bins = np.linspace(min(data), max(data), int(max(data)/1000))
#hist = np.histogram(data, bins)
#print(hist)
#plt.hist(data, bins = n_bin)
#plt.show()
#for i in video.segment:
#    print(i, len(video.segment[i]), len(video.adj[i]))
segs = list(range(video.n_par))
segs = sorted(segs, key = lambda x:len(video.segment[x]), reverse = True)
#for i in segs:
    #print(i, len(video.segment[i]))

fig, ax = plt.subplots(10, 1)

#print(data.shape)
#print(data)
for g in range(10):
    f = segs[g]
    data = np.array([vid[i, j, k, :] for(i, j, k) in video.segment[f]])

    mask = np.zeros((n_frame, m, n))

    for i in range(n_frame):
        for j in range(m):
            for k in range(n):
                if video.label[i][j][k] == f:
                    mask[i, j, k] = 250

    mask = mask.astype(np.uint8)

    save_vid(mask, 'mask' + str(g) + '.avi')

    n_bin = 100
    ax[g].hist(data[:, 0], bins = n_bin, range = (0, 1))
    ax[g].hist(data[:, 1], bins = n_bin, range = (0, 1))
    ax[g].hist(data[:, 2], bins = n_bin, range = (0, 1))

    print(g, stats.skewtest(data[:, 0]), stats.skewtest(data[:, 1]), stats.skewtest(data[:, 2]))

plt.show()

