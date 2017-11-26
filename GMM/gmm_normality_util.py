import numpy as np


def normality(means,vars,pr_cluster_data,data):
    #means[CLUSTERS]
    #vars[CLUSTERS]
    #pr_cluster_data[CLUSTERS][256]
    #data[256]

    CLUSTERS=len(means)
    residue=np.zeros((CLUSTERS))

    for cl in range(CLUSTERS):

        normalFit=dnpmf(means[cl],vars[cl])

        thisCluster=np.multiply(data,pr_cluster_data[cl][:]) #element by element multiplication
        np.divide(thisCluster,np.sum(thisCluster)) #normalize

        residue[cl]=squareDifference(normalFit,thisCluster)



    print(means,vars,residue) #print all these 3 for every cluster







def squareDifference(ar1,ar2):
    return np.sum(np.power(np.subtract(ar1,ar2),2)) #individual element to power 2 and then sum,


def gaussian(x, mean, var):
    return np.exp(-np.power(x - mean, 2.) / (2 * var))

def dnpmf(mean,var): #discrete normal probability mass function
    pmf=np.zeros((256))
    for x in range(256):
        pmf[x]=gaussian(x,mean,var)
    return pmf/np.sum(pmf)




