import numpy as np

class KalmanFilter:
    def __init__(self, x0, P, F, Q, H, R):
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.x = x0
        self.P = P
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.P.T) + self.Q
    
    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        Sinv = np.linalg.inv(S)
        self.K = np.dot(np.dot(self.P, self.H.T), Sinv)

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, y)

        I = np.eye(self.n)
        self.P = np.dot((I - np.dot(self.K, self.H)), self.P)
        
        return self.x
    
    def updateTransitionFucntion(self, F):
        self.F = F

    def normalizeX(self, x):
        self.x = x


