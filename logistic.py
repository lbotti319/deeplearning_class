import numpy as np
import matplotlib
from sklearn.preprocessing import normalize

# Use if running from command line
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helper_functions import get_feature_data, get_label_data, calculate_accuracy

def softmax(c, y, w):
    """
    w: nc x nf matrix
    y: nf x n matrix
    c: nc x n matrix
    """

    # S is nc x n
    S = w * y
    nc = c.shape[0]
    n = c.shape[1]

    # Row vector of ones
    encT = np.matrix(np.ones((1,nc)))
    # np.multiply is the Hadamard product
    hadamard = np.multiply(c,S)
    t1 = (-1 / n) * (encT * hadamard * np.matrix(np.ones((n,1))))[0,0]

    t2 = (1 / n) * (encT * c * np.log(encT * np.exp(S)).T)[0,0]


    return t1 + t2

def softmax_gradient(c, y, w):
    """
    w: nc x nf matrix
    y: nf x n matrix
    c: nc x n matrix
    """
    S = w * y

    nc = c.shape[0]
    n = c.shape[1]
    enc = np.matrix(np.ones((nc,1)))
    encT = enc.T

    s1 = encT * np.exp(S)
    s2 = enc * np.divide(1, s1)
    # Assume multiplication before addition (order of operations)
    t2 = np.multiply(np.exp(S), s2)
    m1 = (1 / n) * ((-1 * c) + t2)
    m2 = y.T
    return m1 * m2

def verify_gradient(c, y, w):
    """
    Verfies the gradient calculation for a given c and y with inital w
    """
    c1_results = []
    c2_results = []
    c3_results = []
    h_vector = []
    for i in range(-20, 1, 1):
        D = np.matrix(np.random.random(w.shape))
        h = 10**i
        c1 = softmax(c, y, w + (h * D)) - softmax(c,y,w)
        c2 = softmax(c, y, w + (h * D)) - softmax(c,y,w) - h * (D.T * softmax_gradient(c, y, w)).trace()[0,0]
        c3 = softmax(c, y, w + (h * D)) - softmax(c,y,w) - h * (D.T * softmax_gradient(c, y, w)).trace()[0,0] - ((h**2)/2) * (D.T * hessian_sub(y,w,D)).trace()[0,0]

        c1_results.append(np.abs(c1))
        c2_results.append(np.abs(c2))
        c3_results.append(np.abs(c3))
        h_vector.append(h)
    return h_vector, c1_results, c2_results, c3_results


def steepest_descent(c,y,w,params = {}):
    alpha_0 = params.get('alpha', 1)
    maxIter = params.get("maxIter", 10)
    gamma = params.get("gamma", 1.2)

    alpha = alpha_0
    for _ in range(maxIter):
        D = -1 * softmax_gradient(c,y,w)
        line_search = False
        while softmax(c,y,w) <= softmax(c,y, (w + (alpha * D))):
            # Line search
            alpha = alpha/2
            line_search = True

        # Set new w
        w = w + (alpha * D)
        
        if line_search:
            alpha = alpha_0
        else:
            alpha_0 = alpha * gamma
            alpha = alpha_0

    return w


def newton_cgls(c, y, w, **kwargs):
    alpha_0 = kwargs.get('alpha', 1)
    maxIter = kwargs.get("maxIter", 10)
    gamma = kwargs.get("gamma", 1.2)
    margin = kwargs.get('stop_condition', 12)
    cgls_stop = kwargs.get('cgls_stop', 0.0005)
    scores = []
    old_score = float('inf')
    current_score = softmax(c,y,w)

    alpha = alpha_0
    iteration = 0
    while (old_score - current_score) / current_score > margin:
        hessian_handle = lambda x: hessian_sub(y, w, x)
        D = cgls_wiki(
            hessian_handle,
            -1*softmax_gradient(c, y, w),
            k=maxIter,
            x_0=softmax_gradient(c, y, w),
            tolerance=cgls_stop
        )
        line_search = False
        old_softmax = softmax(c,y,w)

        new_softmax = softmax(c,y, (w + (alpha * D)))

        while softmax(c,y,w) <= softmax(c,y, (w + (alpha * D))):
            # Line search
            alpha = alpha/2
            line_search = True
        w = w + (alpha * D)
        if line_search:
            alpha = alpha_0
        else:
            alpha_0 = alpha * gamma
            alpha = alpha_0

        old_score = current_score
        current_score = softmax(c,y,w)
        scores.append(current_score)
        iteration += 1
        # print("iteration", iteration)
        # print("score difference", old_score - current_score)
        # print("old_score", old_score)
        # print("current_score", current_score)
        # print("relative error", (old_score - current_score) / current_score)
    return w, scores


def cgls(hessian_handle, c, k=10, tolerance=0.005, x_0=None):
    """
    CGLS from notes, fails to converge
    """
    if x_0 is not None:
        x = x_0
    else:
        x = np.zeros((c.shape))
    d_matrix = hessian_handle(c)
    r = c
    normr2 = np.linalg.norm(d_matrix, 'fro')
    
    for i in range(k):
        Ad = hessian_handle(d_matrix)
        alpha = normr2 / np.linalg.norm(Ad, 'fro')
        # print("alpha", alpha)
        x = x + (alpha*d_matrix)
        r = r - (alpha * Ad)
        d_matrix_new = hessian_handle(r)
        new_norm = np.linalg.norm(d_matrix_new, 'fro')
        # print("norms", normr2, new_norm)
        if new_norm <= tolerance:
            break
        beta = new_norm / normr2
        normr2 = new_norm
        d_matrix = r + (beta * d_matrix)

    return x



def cgls_wiki_mat(hessian_handle, c, k=10, tolerance=0.005, x_0=None):
    """
    CGLS algorithm from wikipedia using vector norms, gives NaNs
    """
    if x_0 is not None:
        x = x_0
    else:
        x = np.zeros((c.shape))

    r = c - hessian_handle(x)
    p = r
    
    for i in range(k):
        Ap = hessian_handle(p)
        normr2 = np.array((r.T * r).diagonal())[0]
        alpha = normr2 / np.array((p.T*Ap).diagonal()[0])
        # print("alpha", alpha)
        x = x + (alpha*np.array(p))
        r = r - (alpha * np.array(Ap))

        new_norm = np.array((r.T * r).diagonal()[0])
        # print("norms", normr2, new_norm)
        if i%20==0:
            print("cgls iteration", i)
            print("new_norm", new_norm.sum())
        if new_norm.sum() <= tolerance:
            break
        beta = new_norm / normr2
        normr2 = new_norm
        p = r + (beta * np.array(p))

    return x

def cgls_wiki(hessian_handle, c, k=10, tolerance=0.005, x_0=None):
    """
    CGLS algorithm from wikipedia using matrix norms
    Might not work for matrices
    """
    if x_0 is not None:
        x = x_0
    else:
        x = np.zeros((c.shape))

    r = c - hessian_handle(x)
    p = r
    
    for i in range(k):
        Ap = hessian_handle(p)
        normr2 = np.linalg.norm(r)**2
        alpha = normr2 / ((p.T*Ap).trace()[0,0])
        # print("alpha", alpha)
        x = x + (alpha*p)
        r = r - (alpha * Ap)
        d_matrix_new = hessian_handle(r)
        new_norm = np.linalg.norm(r, 'fro')**2
        # print("norms", normr2, new_norm)
        # if i%50==0:
        #     print("cgls iteration", i)
        #     print("new_norm", new_norm)
        # if (normr2 - new_norm)/normr2 <= tolerance:
        #     break
        beta = new_norm / normr2
        normr2 = new_norm
        p = r + (beta * p)

    return x

def hessian_sub(y , w, V):
    S = w * y
    nc = S.shape[0]
    n = S.shape[1]
    enc = np.matrix(np.ones((nc,1)))
    encT = enc.T

    s1 = np.divide(np.exp(S), enc * encT * np.exp(S))
    s2 = np.multiply(s1,V*y)
    H1V = s2 * y.T

    # Because encT * exp(S) is not a square matrix
    # I'm going to assume that we should be squaring each element
    s1 = -1 * np.divide(np.exp(S), enc * np.square(encT * np.exp(S)))
    s2 = enc * encT * np.multiply(np.exp(S), V * y)
    H2V = np.multiply(s1, s2) * y.T
    return (1/n)*(H1V + H2V)


def get_weight_matrix(train_features, train_labels):
    A = train_features
    c = train_labels.T

    square = A.T * A

    # If the matrix is non invertible take the psuedo inverse
    try:
        A_sharp = square.I
    except np.linalg.linalg.LinAlgError:
        A_sharp = np.linalg.pinv(square)

    weights =  A_sharp* A.T * c
    return weights


def plot_derivative_checks():
    c = get_label_data("t10k-labels-idx1-ubyte")
    y = get_feature_data("t10k-images-idx3-ubyte")

    # Normalize to prevent overflow
    y = normalize(y, axis=1)
    y = np.matrix(y)
    w = get_weight_matrix(y,c)

    # The initial_guess could be the minimum given by the normal equations or random
    initial_guess = w.T
    y = y.T

    h, c1, c2, c3 = verify_gradient(c,y,initial_guess)

    O1 = [h_i for h_i in h]
    O2 = [h_i**2 for h_i in h]
    O3 = [h_i**3 for h_i in h]
    plt.loglog(h, c1, label="column 1")
    plt.plot(h,O1)
    plt.savefig("hw2/derivative_verification_O1.png")
    plt.clf()
    plt.loglog(h, c2, label="column 2")
    plt.plot(h, O2)
    plt.savefig("hw2/derivative_verification_O2.png")
    plt.clf()
    plt.loglog(h, c3, label="hessian check")
    plt.plot(h, O3)
    plt.savefig("hw2/derivative_verification_O3.png")

if __name__ == '__main__':
    # plot_derivative_checks()

    c = get_label_data("t10k-labels-idx1-ubyte")
    y = get_feature_data("t10k-images-idx3-ubyte")
    # cTrain = get_label_data("train-labels-idx1-ubyte")[:,:10000]
    # yTrain = get_feature_data("train-images-idx3-ubyte")[:10000, :]
    cTrain = get_label_data("train-labels-idx1-ubyte")
    yTrain = get_feature_data("train-images-idx3-ubyte")
    y = np.matrix(normalize(y, axis=1))
    yTrain = np.matrix(normalize(yTrain, axis=1))


    # print(yTrain.shape)
    # print(cTrain.shape)
    w_init = get_weight_matrix(yTrain, cTrain)

    score1 = calculate_accuracy(y, w_init, c)
    print("initial accuracy", score1)


    # Testing CGLS
    # hessian_handle = lambda x: hessian_sub(yTrain.T, w_init.T, x)
    # D = cgls_wiki(
    #     hessian_handle,
    #     -1 * softmax_gradient(cTrain, yTrain.T, w_init.T),
    #     k=200,
    #     x_0=w_init.T
    # )

    # diff_matrix = -1 * softmax_gradient(cTrain, yTrain.T, w_init.T) - hessian_sub(yTrain.T,w_init.T, D)
    # print("norm of diff_matrix", np.linalg.norm(diff_matrix))

    w_trained, scores = newton_cgls(
        cTrain,
        yTrain.T,
        w_init.T,
        stop_condition=0.0005,
        maxIter=200,
        cgls_stop=0.0005
    )

    score = calculate_accuracy(y, w_trained.T, c)
    print("test score:", score)








