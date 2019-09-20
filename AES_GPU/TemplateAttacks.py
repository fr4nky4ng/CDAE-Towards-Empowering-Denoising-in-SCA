import numpy as np
from TimeBar import *
from utils import snr,corr,cal_HMweight


class TemplateAttacks:
    # TA module，including profiling phase and calculating single trace probability, not including POI selection and normalization.
    """
    Parameters:
    matrix_type: 'covariance','pooled','reduced'
    class_num: the number of classes
    """
    # Example:
    # TA_clf = TemplateAttacks(matrix_type='pooled',n_classes=256)
    # TA_clf.fit(X_profiling,Y_profiling)
    # prob,y_pred = TA_clf.predict(X_attack)


    def __init__(self, matrix_type, n_classes):
        self.matrix_type = matrix_type
        self.n_classes = n_classes
        self.X_m = []
        self.X_c = []
        self.X_c_inv = []
        self.X_c_det = []

    def fit(self, X, y):
        # X: array-like traces to train, shape (n_samples, n_features)
        # y: lables , shape (n_samples,)
        
        print("Profiling start")
        
        if self.matrix_type == 'covariance':
            for i in range(self.n_classes):
                idx = np.where(y == i)[0]
                self.X_m.append(np.mean(X[idx,:],axis=0).reshape(-1))
                self.X_c.append(np.cov(X[idx,:].T))
                self.X_c_det.append(np.linalg.det(self.X_c[-1]))
                self.X_c_inv.append(np.linalg.inv(self.X_c[-1]))
        elif self.matrix_type == 'pooled':
            for i in range(self.n_classes):
                idx = np.where(y == i)[0]
                self.X_m.append(np.mean(X[idx,:],axis=0).reshape(-1))
                self.X_c.append(np.cov(X[idx,:].T))
            self.X_c = [np.mean(self.X_c, axis=0)]*self.n_classes
            self.X_c_det = [np.linalg.det(self.X_c[0])]*self.n_classes
            self.X_c_inv = [np.linalg.inv(self.X_c[0])]*self.n_classes
        elif self.matrix_type == 'reduced':
            for i in range(self.n_classes):
                idx = np.where(y == i)[0]
                self.X_m.append(np.mean(X[idx,:],axis=0).reshape(-1))
                self.X_c.append(np.eye(X[idx,:].shape[0]))
        else:
            print('Error: unknown matrix type')
        
        return self

    def predict(self, X):
        # X: array-like traces to test, shape (n_samples, n_features)
        # y: lables , shape (n_samples,)
        prob = np.zeros((X.shape[0], self.n_classes))
        
        print("Prob estimating start")
        timebar = TimeBar()
        
        if self.matrix_type == 'covariance':
            for i in range(X.shape[0]):
                print(timebar(i, X.shape[0]-1), end='')
                for j in range(self.n_classes):
                    prob[i][j] = -1 / 2 * (
                        np.log(self.X_c_det[j]) + (X[i] - self.X_m[j]).dot(
                            self.X_c_inv[j]).dot(X[i].T - self.X_m[j].T))
        elif self.matrix_type == 'pooled':
            for i in range(X.shape[0]):
                print(timebar(i, X.shape[0]-1), end='')
                for j in range(self.n_classes):
                    prob[i][j] = -1 / 2 * ((X[i] - self.X_m[j]).dot(
                        self.X_c_inv[j]).dot(X[i].T - self.X_m[j].T))
        elif self.matrix_type == 'reduced':
            for i in range(X.shape[0]):
                print(timebar(i, X.shape[0]-1), end='')
                for j in range(self.n_classes):
                    prob[i][j] = -1 / 2 * np.dot(self.X_m[j], self.X_m[j].T) + np.dot(
                        self.X_m[j], X[i].T)
        else:
            print('error: unknown matrix type')
        
        y_pred = [np.argmax(prob[i,:]) for i in range(X.shape[0])]
        
        return prob,y_pred