import numpy as np
import pandas as pd
import matplotlib
from sklearn.preprocessing import normalize
from helper_functions import get_feature_data, get_label_data, calculate_accuracy
from sklearn.metrics import accuracy_score


def sigma(K, Y, b):
    KY = K*Y
    shape = KY.shape
    inner = KY + b*np.ones(shape)
    return np.tanh(inner)

def sigma_prime(K, Y, b):
    """
    Inverse of tanh(x) is 1 - tanh(x)^2
    source: http://ronny.rest/blog/post_2017_08_16_tanh/
    """
    a = sigma(K, Y, b)
    shape = a.shape
    return np.ones(shape) - np.multiply(a, a)

def J_K(K, Y, b, v):
    shape = K.shape
    mat_v = np.matrix(v.reshape(shape))
    sig_prime = sigma_prime(K,Y, b)
    result = np.multiply(sig_prime, mat_v * Y)
    return np.matrix(result.reshape((shape[0]*Y.shape[1], 1)))

def J_K_T(K, Y, b, u):
    shape = (K.shape[0], Y.shape[1])
    mat_u = np.matrix(u.reshape(shape))
    sig_prime = sigma_prime(K, Y, b)
    result = np.multiply(sig_prime, mat_u) * Y.T
    return np.matrix(result.reshape(K.shape[0]*K.shape[1], 1))


def J_b(K, Y, b, v):
    return sigma_prime(K, Y, b) * v

def J_b_T(K, Y, b, u):
    z = sigma_prime(K, Y, b)
    if u.shape == z.shape:
        nm = z.shape[0]*z.shape[1]
        return sigma_prime(K, Y, b).T.reshape((1, nm)) * u.reshape((nm, 1))[0,0]
    return sigma_prime(K, Y, b).T * u


def J_Y(K, Y, b, v):
    nf = Y.shape[0]
    n = Y.shape[1]
    shape = Y.shape

    sig_prime = sigma_prime(K, Y, b)
    t2 = K * np.matrix(v.reshape(shape))
    return np.multiply(sig_prime, t2).reshape(K.shape[0]*n, 1)

def J_Y_T(K, Y, b, u):
    m = K.shape[0]
    nf = K.shape[1]
    n = Y.shape[1]
    sig_prime = sigma_prime(K, Y, b)
    result = K.T * np.multiply(sig_prime, u.reshape((m,n)))
    return result.reshape((nf*n, 1))


def verify_gradient(K, Y, b):
    c1_results = []
    c2_results = []
    c3_results = []
    c4_results = []
    c5_results = []
    c6_results = []
    h_vector = []
    for i in range(-20, 1, 1):
        v_k = np.matrix(np.random.random(K.shape))
        v_b = np.matrix(np.random.random((Y.shape[1], 1)))
        v_y = np.matrix(np.random.random(Y.shape))
        h = 10**i
        c1 = (sigma(K+ h*v_k, Y, b) - sigma(K,Y,b)).trace()[0,0]
        c2 = (sigma(K+ h*v_k, Y, b) - sigma(K,Y,b) - h * J_K(K, Y, b, v_k).reshape((K.shape[0],Y.shape[1]))).trace()[0,0]

        c3 = np.linalg.norm((sigma(K, Y, b+h) - sigma(K, Y, b)) * v_b)
        c4 = np.linalg.norm((sigma(K, Y, b+h) - sigma(K, Y, b)) * v_b - h * J_b(K, Y, b, v_b))

        c5 = (sigma(K, Y + h*v_y, b) - sigma(K, Y, b)).trace()[0,0]
        c6 = (sigma(K, Y + h*v_y, b) - sigma(K, Y, b) - h * J_Y(K, Y, b, v_y).reshape((K.shape[0],Y.shape[1]))).trace()[0,0]

        c1_results.append(np.abs(c1))
        c2_results.append(np.abs(c2))
        c3_results.append(np.abs(c3))
        c4_results.append(np.abs(c4))
        c5_results.append(np.abs(c5))
        c6_results.append(np.abs(c6))

        h_vector.append(h)
    return h_vector, c1_results, c2_results, c3_results, c4_results, c5_results, c6_results


def entropy(c, Y, K, b, w): 
    """
    w: nc x m matrix
    y: nf x n matrix
    c: nc x n matrix
    K: m x nf
    b: scalar
    """

    # S is nc x n
    S = w * sigma(K, Y, b)
    nc = c.shape[0]
    n = c.shape[1]

    # Row vector of ones
    encT = np.matrix(np.ones((1,nc)))
    # np.multiply is the Hadamard product
    hadamard = np.multiply(c,S)
    t1 = (-1 / n) * (encT * hadamard * np.matrix(np.ones((n,1))))[0,0]

    t2 = (1 / n) * (encT * c * np.log(encT * np.exp(S)).T)[0,0]


    return t1 + t2

def entropy_gradient(c, Y, K, b, w):
    """
    w: nc x m matrix
    y: nf x n matrix
    c: nc x n matrix
    K: m x nf
    b: scalar
    """
    z = sigma(K, Y, b)
    S = w * z
    nc = c.shape[0]
    n = c.shape[1]
    enc = np.matrix(np.ones((nc,1)))
    encT = enc.T

    S_gradient = (-1/n)*(c - np.multiply(np.exp(S),enc*np.divide(1, encT*np.exp(S))))

    print(S_gradient.shape)
    z_gradient = w.T * S_gradient
    print(z_gradient.shape)
    K_gradient = J_K_T(K, Y, b, z_gradient).reshape(K.shape)
    b_gradient = J_b_T(K, Y, b, z_gradient)
    w_gradient = S_gradient * z.T
    return {"z": z_gradient, "K": K_gradient, "b": b_gradient, "w": w_gradient}


class Deep_NN():
    def __init__(self, c, Y, Ks, bs, layers=1, gamma = 0.5, w_init = None):
        self.c = c
        self.Y = Y
        assert len(Ks) == len(bs)
        self.Ks = Ks
        self.bs = bs
        self.layers = layers
        self.gamma = gamma
        if w_init is not None:
            self.w = w_init
        else:
            shape = (c.shape[0], Ks[-1].shape[0])
            self.w = np.random.random(shape)

    def forward_propagate(self):
        y_i = self.Y
        for K_i, b_i in zip(self.Ks[:-1], self.bs[:-1]):
            # This won't run until we know what p_i is
            y_i_1 = y_i + np.multiply(self.gamma, sigma(K_i, y_i, b_i))
            y_i = y_i_1

        return entropy(self.c, y_i, self.Ks[-1], self.bs[-1], self.w)

    def classify(self, Ytest):
        y_i = Ytest
        for K_i, b_i in zip(self.Ks[:-1], self.bs[:-1]):
            # This won't run until we know what p_i is
            y_i_1 = y_i + np.multiply(self.gamma, sigma(K_i, y_i, b_i))
            y_i = y_i_1

        final_probabilities = self.w * sigma(self.Ks[-1], y_i, self.bs[-1])
        labels = pd.DataFrame(final_probabilities).idxmax(axis=1)
        return labels

    def score(self, Ytest, cTest):
        labels = self.classify(Ytest)
        true_labels = pd.DataFrame(cTest).idxmax(axis=1)
        accuracy = accuracy_score(true_labels.values, labels.values)
        return accuracy








