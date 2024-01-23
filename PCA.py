
import numpy as np

class PCA:
    def __init__(self, n_component):
        self.n_component = n_component
        self.component = None
        self.mean = None


    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        
        x = x - self.mean

        cov = x.T.dot(x)    
        eign_values, eign_vectors = np.linalg.eig(cov)

        indxs = np.argsort(eign_values)[:: -1] # To reversing.
        eign_values = eign_values[indxs]       
        eign_vectors = eign_vectors[indxs]     

        self.component = eign_vectors[: , : self.n_component]

    def transform(self, x):
        x = x - self.mean
        return x.dot(self.component) 
    
    def fit_transform(self, x):
        self.fit(x)
        x_transformed = self.transform(x)

        return x_transformed



