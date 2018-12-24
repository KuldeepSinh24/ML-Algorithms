import numpy as np 

class KMeans:
    def __init__(self,k=3,max_iterations=1000,tolerance=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def train(self,inputs):
        self.centroids = {}

        for i in range(self.k):
            self.centroids = inputs[i]

        for i in range(self.max_iterations):
            self.classes ={}
            for i in range(self.k):
                self.classes[i] = []
            for features in inputs:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
                
            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)
            
            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

			#break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
            if isOptimal:
                break

    def test(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

