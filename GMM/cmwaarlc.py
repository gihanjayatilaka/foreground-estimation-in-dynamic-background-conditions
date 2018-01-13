import numpy as np
import numpy.linalg as la

class cmwaarlc:






    def __init__(self,CLUSTERS,COLORS):
        self.CLUSTERS=CLUSTERS
        self.COLORS=COLORS
        self.dataPoints=0

        #Cylindrical model with adaptive AXES,RADII,LENGTHS and CENTROIDS
        self.axis=np.zeros((CLUSTERS,COLORS),np.float32)
        self.radius=np.zeros((CLUSTERS),np.float32)
        self.halfLength=np.zeros((CLUSTERS),np.float32)
        self.centroid=np.zeros((CLUSTERS,COLORS),np.float32)

    #to be used in a(dvanced)cmwaarlc
    #    self.clusterWeight=np.zeros((CLUSTERS),np.float32)


    def adapt(self,dataPoint):
        #The first few data points are used to initialize the cylinders
        if self.dataPoints ==0:
            self.tempPreviousDataPoint=dataPoint
            self.dataPoints+=1
        elif self.dataPoints <= self.CLUSTERS:
            self.axis[self.dataPoints-1]=(dataPoint-self.tempPreviousDataPoint)/la.norm(dataPoint-self.tempPreviousDataPoint)
            self.radius[self.dataPoints-1]=1
            self.halfLength[self.dataPoints - 1] = 1
            self.centroid[self.dataPoints-1]=(dataPoint+self.tempPreviousDataPoint)/2
            self.tempPreviousDataPoint=dataPoint
            self.dataPoints+=1



        else:
            print("Adapting.....")
            matchingCluster=-1
            rlDistanceMin=10

            for cl in range(self.CLUSTERS):
                rDistanceFromCentroid=np.linalg.norm(np.dot(dataPoint-self.centroid[cl],self.axis[cl])/self.radius[cl])
                lDistanceFromCentroid=np.abs(np.cross(dataPoint-self.centroid[cl],self.axis[cl])/self.halfLength[cl])
                rlDistanceFromCentroid=np.sqrt(np.sum(np.square(np.array([rDistanceFromCentroid,lDistanceFromCentroid]))))

                if (rDistanceFromCentroid<2.0) & (lDistanceFromCentroid<2.0):
                    if rlDistanceFromCentroid<rlDistanceMin:
                        rlDistanceMin=rlDistanceFromCentroid
                        matchingCluster=cl

            if matchingCluster >= 0:
                rDistanceFromCentroid=np.abs(np.dot(dataPoint-self.centroid[cl],self.axis[cl]))
                lDistanceFromCentroid=np.abs(np.cross(dataPoint-self.centroid[cl],self.axis[cl]))

                #<<<<<<<<<<<<<LEARNING STEP>>>>>>>>>>>>>
                self.centroid[matchingCluster]=self.centroid[matchingCluster]*0.9 + dataPoint*0.1
                self.length=self.halfLength[matchingCluster]*0.9 + lDistanceFromCentroid*0.1
                self.radius=self.radius[matchingCluster]*0.9 + rDistanceFromCentroid*0.1

                if np.dot(self.axis[matchingCluster],dataPoint-self.centroid[matchingCluster]) < -1:
                    self.axis[matchingCluster]*=-1

                self.axis[matchingCluster]=self.axis[matchingCluster]/np.sum(np.square(self.axis[matchingCluster]))
                temp=(dataPoint-self.centroid[matchingCluster])/np.sum(np.square((dataPoint-self.centroid[matchingCluster])))

                self.axis[matchingCluster]=self.axis[matchingCluster]*0.9 + temp*0.1
                #<<<<<<<<< LEARNING STEP OVER>>>>>>>>>>>>>>

        print("Data point count",self.dataPoints)
        print("Centroids",self.centroid)
        print("Axes",self.axis)
        print("R",self.radius)
        print("L",self.halfLength)




if __name__ == '__main__':
    a=cmwaarlc(3,2)
    for x in [0,10,20,30,40]:
        a.adapt(np.array([x,x]))
