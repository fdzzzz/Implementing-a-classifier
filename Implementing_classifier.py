import numpy as np
import matplotlib.pyplot as plt



def Euclidean_distance(point1, point2): 
    Edistance = np.sqrt(np.sum((point1 - point2) ** 2))
    return Edistance


def get_Euclidean_distance(point, dataSet): 
    distanceSet = []
    for sample in dataSet:
        if Euclidean_distance(point, sample):
            distanceSet.append(Euclidean_distance(point, sample))
    return distanceSet

class KNN():
    k = 0 #initialize the k value 

    def train(self, position, value): 
        self.position = position
        self.value = value
        Kmax = len(position)
        best_k = 0 
        best_accurrcy = 0  
        for k in range(1, Kmax):
            labelSet = self.predict(position, k)
            count = np.sum(np.logical_xor(labelSet, value) == 1)
            precision = 1 - count / value.shape[0]
            if precision > best_accurrcy:  
                best_accurrcy = precision
                best_k = k
        return best_k, best_accurrcy #find the most optimoal k with its accurracy

    def predict(self, predictSet, k):
        labelSet = [] 
        for point in predictSet:
            distanceSet = get_Euclidean_distance(point, self.position)
            sorted_index = sorted(range(len(distanceSet)), key=lambda k: distanceSet[k], reverse=False)
            count1 = list(self.value[sorted_index[:k]]).count(1)  
            count0 = list(self.value[sorted_index[:k]]).count(0)
            if count0 < count1:
                labelSet.append(1)
            else:
                labelSet.append(0)
        return labelSet

if __name__ == "__main__":
    #create the dataset 
    mean1 = [0, 0]
    mean2 = [4, 4]
    cov = [[1, 0], [1, 1]]
    points0 = np.random.multivariate_normal(mean1, cov, 20)
    label0 = np.zeros(20)
    data0 = np.c_[points0, label0]
    points1 = np.random.multivariate_normal(mean2, cov, 20)
    label1 = np.ones(20)
    data1 = np.c_[points1, label1]
    data = np.vstack((data0, data1))
    

    #get positions and values
    position = data[:, 0:2]
    value = data[:, 2]
    
    #draw the graph
    data1_position = data1[:, 0:2]
    data0_position = data0[:, 0:2]
    plt.xlabel('x-value')
    plt.ylabel('y-value')
    plt.scatter(data0_position[:,0], data0_position[:,1],color="green",marker='+')
    plt.scatter(data1_position[:,0], data1_position[:,1],color="blue",marker='.')
    plt.show()



