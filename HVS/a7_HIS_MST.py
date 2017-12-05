from collections import defaultdict
from queue import PriorityQueue
import cv2
import numpy as np

class Graph:

    def __init__(self,m, n, k):
        self.rows = m
        self.cols = n
        self.V= m*n
        self.graph = []
        self.parent = [i for i in range(self.V)]
        self.rank = [0 for i in range(self.V)]
        self.max_w = [0 for i in range(self.V)]
        self.k = k
        self.num = [1 for i in range(self.V)]

    def addEdge(self,u,v,w):
        self.graph.append((u, v, w))

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, x, y, weight, num):
        xroot = self.find(self.parent, x)
        yroot = self.find(self.parent, y)

        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
            par = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
            par = xroot
        else :
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
            par = xroot

        self.max_w[par] = weight
        self.num[par] = num

    def MST(self):

        i = 0 # An index variable, used for sorted edges
        e = 0 # An index variable, used for result[]

        self.graph =  sorted(self.graph,key=lambda item: item[2])
        #print(self.graph)
        l = len(self.graph)
        print(len(self.graph))

        while i < l:
            #print(i)
            u,v,w =  self.graph[i]
            i = i + 1
            x = self.find(self.parent, u)
            y = self.find(self.parent,v)

            # No cycle
            if x != y:
                mw = max(self.max_w[x], self.max_w[y])
                num = self.num[x] + self.num[y]
                cost = mw + self.k/num
                if cost > w:
                    e = e + 1
                    self.union(x, y, mw, num)

        out = np.random.rand(self.V, 3)

        for i in range(self.V):
            out[i, :] = out[self.find(self.parent, i), :]

        return np.reshape(out, (m, n, 3))


st_frame = 980
n_frame = 10

name = 'boats\input\in' + str(st_frame).zfill(7)+ '.jpg'
img = cv2.imread(name,1)

for i in range(n_frame):
    name = 'boats\input\in' + str(st_frame+i).zfill(7)+ '.jpg'
    img = cv2.imread(name,1)
    cv2.imshow('boat',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#cv2.imshow('1',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

'''
img = cv2.imread(name,0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
im = img.astype(float)/255

m, n, o =  (im.shape)
print(im.dtype)

# Driver code
g = Graph(m, n, 10)

for i in range(m-1):
    for j in range(n-1):
        u = i*n+j
        w = sum((im[i][j][:] - im[i][j+1][:])**2)
        g.addEdge(u, u+1, w)
        w = sum((im[i][j][:] - im[i+1][j][:])**2)
        g.addEdge(u, u+n, w)

out = g.MST()

#for i in g.parent:
#    if i != g.parent[i]:
#        print (i)

out = out*255
output = out.astype(np.uint8)
cv2.imshow('2',output)
cv2.waitKey(0)
#cv2.destroyAllWindows()

