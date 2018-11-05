"""
Implementation of Cox process with GP intensitie. Following Adams et all:
'Tractable Nonparametric Bayesian Inference in Poisson Process with Gaussian Process Intensities'

Xinglong Li
Oct, 2018
"""

import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm



# lam_star: upper bound of intensity of Cox process
# tau: range of random field S

# using squared exponential kernel
def Ker_sqexp(x_0, x_1, sig_f, sig_l):
    """ 
    Squared exponential kernel.
    Computes the covariance matrix from points in x_0 and x_1
    x_0: n*d array
    x_1: m*d array
    return : n*m covariance array
    """
    sqdist = np.sum(x_0**2, 1).reshape(-1,1) + np.sum(x_1**2, 1) - 2 * np.dot(x_0, x_1.T)
    return sig_f**2 * np.exp(-0.5 * sqdist / sig_l**2) 



def Tran_sig(x):
    """
    sigmoid transformation function
    scaled by upper bound lam_star
    """
    return lam_star / (1 + np.exp(-x))



def Thinning(lam_star, tau, d, Kernel, Transform):
    """
    sample from Cox process on finite random filed limited by tau, using thinning algorithm
    intensity is given by Gaussian process with kernel = Kernel, transformed by Transform
    """
    # first generate number of points, which follows Poisson distribution
    N = np.random.poisson(lam_star * tau)
    
    # given number of points, sample random postions uniformally in finite field 
    points = tau *  np.array([np.random.rand(d) for i in range(N)])
    
    # given positions of random points, sample from GP at these points
    # add noise to the diagnal to keep the kernel positive definite
    K = Kernel(points, points, sig_f, sig_l) + 0.01*np.eye(N)
    
    Y = np.random.multivariate_normal(np.zeros(N), K)
    
    # transform samples from GP to intensity of Poisson process
    Lam = Transform(Y)
    
    # accept random points according to their intensities
    U = lam_star * np.random.rand(N)
    S = points[U < Lam]
    
    return S


    
########### sample from the full likelihood of augmented Cox process ##########

def Samp(S_K, T, lam_star, tau, Kernel, sig_f, sig_l):
    """
    Genrate samples from complete likelihood of augmentated Cox process from Equation(4) in
    'Tractable Nonparametric Bayesian Inference in Poisson Processes 
     with Gaussian Process Intensities'
     
     T (int): sample size
     S_K (n*d array): postions of observed points
    """
    d = S_K.shape[1]
    # array of number of thinned points in each sample from a Poisson process
    M = []
    # array of matrix, each matrix is a M*d matrix of M positions of thinned points
    S_M = []
    # array of vector, each vector's length is K+M, saving the random sample from GP 
    G = []
    
    # firstly generate the initial sample 
    
    # number of thinned points
    m_t = np.random.poisson(lam_star * tau)
    # positions of thinned points
    sm_t = tau * np.array([np.random.rand(d) for i in range(m_t)])
    # augmented positions (thinned and observed)
    s = np.concatenate((S_K, sm_t))
    # sample from GP at augmented positions
    kernel_0 = Kernel(s, s, sig_f, sig_l) + 0.01 * np.eye(len(s)) # avoid singularity
    g_t = np.random.multivariate_normal(np.zeros(len(s)), kernel_0)
    # compute the inverse of covariance for later use
    SigInv = np.linalg.solve(kernel_0, np.eye(len(s)))
    
    # initalize the sample sequence
    M.append(m_t)
    S_M.append(sm_t)
    G.append(g_t)
    
    # iterate over M, S_M, and G
    
    for t in np.arange(T)+1:
        
        # new state of M, other variables would also possibly change accordingly
        # for each transition of S_M and G, make 10 transition of M
        for i in range(10):
            m_t, sm_t, g_t, SigInv = Samp_M(m_t, sm_t, g_t, SigInv, S_K, lam_star, tau, Kernel, sig_f, sig_l)
        
        # new state of positions of thinneds points S_M , SigInv would also change accordingly
        sm_t, SigInv = Samp_S(sm_t, g_t, SigInv, S_K, Kernel, sig_f, sig_l)
        
        # generate GP likelihood of all points S_M & S_K
        g_t = Samp_G(g_t, len(S_K), SigInv)
        
        M.append(m_t)
        S_M.append(sm_t)
        G.append(g_t)
    
    M = np.array(M)
    S_M = np.array(S_M)
    G = np.array(G)
    
    return {'M':M, 'S_M':S_M, 'G':G}



def Samp_M(m_0, sm_0, g_0, SigInv, S_K, lam_star, tau, Kernel, sig_f, sig_l):
    """
    generate new state of number of thinned points M,
    if M change, sm_0, g_0, SigInv would also change accordingly
    """
    d = S_K.shape[1]
    s = np.concatenate((S_K, sm_0))
    b = np.random.rand()
    if b > 0.5:
        # generate a new thinned position uniformally in random field
        s_new = tau * np.array([np.random.rand(d)])
        # sample from GP at s_new given g_0
        cov = Kernel(s_new, s, sig_f, sig_l)
        mu_new = np.dot(np.dot(cov, SigInv), g_0)
        sd_new = np.sqrt(Kernel(s_new, s_new, sig_f, sig_l) - np.dot(np.dot(cov, SigInv),cov.T))
        g_new = np.random.normal(mu_new, sd_new)
        # accept prob of M -> M+1
        accept_inc = tau * lam_star / ((m_0 + 1) * (1 + np.exp(g_new)))
        u = np.random.rand()
        if u < accept_inc:
            m_t = m_0 + 1
            sm_t = np.concatenate((sm_0, s_new))
            g_t = np.concatenate((g_0, g_new[0]))
            s_t = np.concatenate((S_K, sm_t)).reshape(-1,d)
            ker_new = Kernel(s_t, s_t, sig_f, sig_l) + 0.01 * np.eye(len(s_t))
            SigInv = np.linalg.solve(ker_new, np.eye(len(s_t)))
        else:
            m_t = m_0
            sm_t = sm_0
            g_t = g_0
    else:
        # random select a point to delete
        m = np.random.choice(range(m_0))
        g_del = g_0[len(S_K) + m]
        # accept prob of M -> M-1
        accept_del = (m_0 * (1+np.exp(g_del))) / (tau * lam_star)
        u = np.random.rand()
        if u < accept_del:
            m_t = m_0 - 1
            sm_t = np.delete(sm_0, m, 0)
            g_t = np.delete(g_0, len(S_K)+m, 0)
            s_t = np.concatenate((S_K, sm_t)).reshape(-1,d)
            ker_new = Kernel(s_t, s_t, sig_f, sig_l) + 0.01 * np.eye(len(s_t)) # avoid singularity
            SigInv = np.linalg.solve(ker_new, np.eye(len(s_t)))
        else:
            m_t = m_0
            sm_t = sm_0
            g_t = g_0
        
    return m_t, sm_t, g_t, SigInv


    
def Samp_S(sm_0, g_0, SigInv, S_K, Kernel, sig_f, sig_l):
    """
    new position samples of thinned points
    after positions change, the SigInv changes accordingly
    """
    d = S_K.shape[1]
    s = np.concatenate((S_K, sm_0)).reshape(-1, d)
    for m in range(len(sm_0)):
        if d == 1:
            a = (0 - sm_0[m]) / (sig_f * 0.5)
            b = (tau - sm_0[m]) / (sig_f * 0.5)
            s_new = truncnorm.rvs(a, b, sm_0[m], sig_f)
        elif d == 2:
            x_low = np.max([sm_0[m,0]-sig_f, 0])
            x_up = np.min([sm_0[m,0]+sig_f, tau])
            y_low = np.max([sm_0[m,1]-sig_f, 0])
            y_up = np.min([sm_0[m,1]+sig_f, tau])
            new_x = np.random.uniform(x_low, x_up)
            new_y = np.random.uniform(y_low, y_up)
            s_new = np.array([new_x, new_y])
        
        s_new = s_new.reshape(-1,d)
        # sample from GP at s_new given g_0
        cov = Kernel(s_new, s, sig_f, sig_l)
        mu_new = np.dot(np.dot(cov, SigInv), g_0)
        sd_new = np.sqrt(Kernel(s_new, s_new, sig_f, sig_l) - np.dot(np.dot(cov, SigInv),cov.T))
        g_new = np.random.normal(mu_new, sd_new)
        p_accept = (1 + np.exp(g_0[len(S_K)+m])) / (1 + np.exp(g_new))
        u = np.random.rand()
        if u < p_accept:
            sm_0[m] = s_new
    
    # change SigInv accordingly    
    s_t = np.concatenate((S_K, sm_0)).reshape(-1,d)
    ker_new = Kernel(s_t, s_t, sig_f, sig_l) + + 0.01 * np.eye(len(s_t)) # avoid singularity
    SigInv = np.linalg.solve(ker_new, np.eye(len(s_t)))
        
    return sm_0, SigInv



def Samp_G(g_0, K, SigInv):
    L = 100
    epsilon = 0.1
    return HMC(U, grad_U, epsilon, L, g_0, K, SigInv)
    


def U(g, K, SigInv):
    # the potential energy in HMC, which is negative log-likelihood(up to a constant)
    term_1 = np.matmul(np.matmul(g.transpose(), SigInv), g) / 2
    term_2 = np.sum(np.log(1 + np.exp(-g[:K])))
    term_3 = np.sum(np.log(1 + np.exp(g[K:])))
    
    return term_1 + term_2 + term_3


    
def grad_U(g, K, SigInv):
    # the gradient of poetential function at g
    term_1 = np.matmul(SigInv, g)
    grad_K = -1 / (1 + np.exp(g[:K])) 
    grad_M = 1 / (1 + np.exp(-g[K:]))
    term_2 = np.concatenate((grad_K, grad_M))
    
    return term_1 + term_2


 
def HMC(U, grad_U, epsilon, L, current_q, K, SigInv):
    q = current_q
    p = np.random.randn(len(q))
    current_p = p
    
    # make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q, K, SigInv) / 2
    
    # alternate full steps for position and momentum
    for i in range(L):
        # make a full step for the position
        q = q + epsilon * p 
        # make a full step for the momentum, except at the end of trajectory
        if i < L-1:
            p = p - epsilon * grad_U(q, K, SigInv)
    
    # make a half step for the momentum at the end
    p = p - epsilon * grad_U(q, K, SigInv) / 2
    
    # negate momentum at the end of trajectory to make the proposal symmetric
    p = -p
    
    # evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q, K, SigInv)
    current_K = np.sum(current_p**2) / 2
    proposed_U = U(q, K, SigInv)
    proposed_K = np.sum(p**2) / 2
    
    # accept or reject the state at end of trajectory, returning either 
    # the position at the end of trajectory or the initial position
    u = np.random.rand()
    accept_p = np.exp(current_U - proposed_U + current_K - proposed_K)
    if u < accept_p:
        return q
    else:
        return current_q
    


############################## Simulations  part ##############################



# get the estimated intensity function given in the form of vector 
        
def Get_intensity(S_K, samples, Transform, tau, burn_in= 1000, norm=False, draw=True):
    """
    after sampling, get the estimated intensity function by averaging over 
    posterior samples.
    the function is given in the form of a of vector averaged values
    """
    d = S_K.shape[1]
    T = len(samples['M'])
    K = len(S_K)
    
    # firstly average over all the GP values at observed values 
    sum_g_K = np.zeros(K)
    for i in range(burn_in, T):
        sum_g_K += samples['G'][i][:K]
        
    avg_g_K = sum_g_K / (T - burn_in)
    
    # combine all sample positions and corresponding GP values
    all_points = S_K
    all_g = avg_g_K
    for i in range(burn_in, T):
        all_points = np.concatenate((all_points, samples['S_M'][i]))
        all_g = np.concatenate((all_g, samples['G'][i][K:]))  
    
    # transform the GP values     
    all_g = Transform(all_g)
    
    if d == 1:
        # position where function being evaluated, with each interval length 0.5
        x = np.linspace(0,tau,100)
        inter = tau / 100
        # vector for function values at x_value
        y = np.zeros(100)  
        for i in range(100):
            idx = ((all_points > i*inter) & (all_points < i*inter+inter))
            idx = idx[:,0]
            values = all_g[idx]
            y[i] = np.mean(values)
            
    elif d == 2:
        x = None # in 2d case, we no longer need x to plot function
        y = np.zeros((50,50))
        inter = tau / 50
        all_x = all_points[:, 0]
        all_y = all_points[:, 1]
        for i in range(50):
            for j in range(50):
                idx = ((all_x > i*inter) & (all_x < i*inter+inter))
                idy = ((all_y > j*inter) & (all_y < j*inter+inter ))
                values = all_g[(idx & idy)] + 0
                y[i,j] = np.mean(values)
            
    if norm == True:
        # then normalize the obtained intensity
        mu = np.sum(y) * inter**2
        y = y / mu
        
    if draw == True:
        if d == 1:
            plt.plot(x, y, linewidth = 3, label="Sigmoid")
        elif d == 2:
            plt.imshow(y, cmap='jet', extent=[0,tau,tau,0])
    
    return x, y



# hyperparameters selection by 5 fold CV, selecting the best on a 3-d grid

def Select_hyp(lam_range, sig_f_range, sig_l_range, tau, S_K, Kernel, Transform):
    """
    3 hyperparameters to select, 
    lam_range, sig_f_range, sig_l_range are grids of values to select from
    use 5 fold cross validation
    """
    d = S_K.shape[1]
    if d==1:
        inter = tau / 100
    elif d==2:
        inter = tau / 50
    minimum = 10000
    best_lam = best_sig_f = best_sig_l = 0
    K = len(S_K)
    num_test = int(K // 5)
    np.random.shuffle(S_K)
    for lam_star in lam_range:
        for sig_f in sig_f_range:
            for sig_l in sig_l_range:
                print(sig_l)
                neg_log_lik = 0
                for t in range(1):
                    test_idx = range(t*num_test, (t+1)*num_test)
                    test = S_K[test_idx]
                    train = np.delete(S_K, test_idx, 0)
                    samples = Samp(train, 1500, lam_star, tau, Ker_sqexp, sig_f, sig_l)
                    x, y = Get_intensity(train, samples, Transform, tau, burn_in= 500, norm=True, draw=False)
                    if d == 1:
                        idxs = (test // inter).astype(int)
                        neg_log_lik += - np.sum(np.log(y[idxs]))
                    elif d == 2:
                        idx = (test[:, 0] // inter).astype(int)
                        idy = (test[:, 1] // inter).astype(int)
                        neg_log_lik += - np.sum(np.log(y[idx, idy]))
                        
                if neg_log_lik < minimum:
                    minimum = neg_log_lik
                    best_lam = lam_star
                    best_sig_f = sig_f
                    best_sig_l = sig_l
    
    return best_lam, best_sig_f, best_sig_l
    


################################# simulation 1 ################################
    

        
def Intensity_1(x):
    """
    intensity function defined on the interval [0, 50]
    with 53 events
    """
    return 2 * np.exp(-x / 15) + np.exp(-((x-25)/10)**2)

x = np.linspace(0,50,5000)
y = Intensity_1(x)

tau = 50
lam_star = 2

# generate the observed positions from true Poisson process
"""
N = np.random.poisson(tau*2)
points = tau * np.random.rand(N)
p_accept = Intensity_1(points)
u = 2 * np.random.rand(N)
S_K = points[u < p_accept]
"""
d = 1
S_K = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/int1.txt',delimiter=',')
S_K = S_K.reshape(-1,d)

plt.plot(x, y, linewidth=3, label="True Intensity")
height = np.ones(len(S_K)) * 2.5
plt.plot(S_K, height, '|', markersize = 30, markeredgewidth=1.5, markeredgecolor='black')

#         Samp(S_K,  T,   lam_star, tau, Kernel,    sig_f, sig_l)
samples = Samp(S_K, 6000, lam_star, tau, Ker_sqexp, sig_f=2.5, sig_l=9)
x_sig, y_sig = Get_intensity(S_K, samples, Tran_sig, tau, burn_in= 500, norm=False, draw=True)

lam_range = np.array([2])
#sig_f_range = np.array([1, 2, 3])
#sig_l_range = np.array([6, 9, 12])
sig_f_range = np.array([1.5, 2, 2.5])
sig_l_range = np.array([8, 9, 10])

lam0, sig_f0, sig_l0 = Select_hyp(lam_range, sig_f_range, sig_l_range, tau, S_K, Ker_sqexp, Tran_sig)

# selection result for sigmoid transform
# best_sig_f = 2.5
# best_sig_l = 9



############################### simulation 2 ##################################
    
def Intensity_2(x):
    """
    intensity function denfined on the interval [0, 5]
    with 29 events
    """
    return 5 * np.sin(x**2) + 6

x = np.linspace(0,5,500)
y = Intensity_2(x)

tau = 5
lam_star = 12
sig_f = 2
sig_l = 0.5

# generate the observed positions from true Poisson process
"""
N = np.random.poisson(tau*12)
points = tau * np.random.rand(N)
p_accept = Intensity_2(points)
u = 12 * np.random.rand(N)
S_K = points[u < p_accept]
"""
d=1
S_K = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/int2.txt',delimiter=',')
S_K = S_K.reshape(-1,d)

plt.plot(x,y, linewidth=3)
height = np.ones(len(S_K)) * 14
plt.plot(S_K, height, '|', markersize = 30, markeredgewidth=1.5, markeredgecolor='black')

lam_range = np.array([12])
sig_f_range = np.array([2, 2.5, 3])
sig_l_range = np.array([0.3, 0.5, 0.8])
sig_f_range = np.array([2.5])
sig_l_range = np.array([0.6, 0.8, 1.0])

lam0, sig_f0, sig_l0 = Select_hyp(lam_range, sig_f_range, sig_l_range, tau, S_K, Ker_sqexp, Tran_sig)
# selection result for sigmoid transform
# best_sig_f = 2.5
# best_sig_l = 0.6

samples = Samp(S_K, 6000, lam_star, tau, Ker_sqexp, 2.5, 0.6)
x_sig, y_sig = Get_intensity(S_K, samples, Tran_sig, tau, burn_in= 500, norm=False, draw=True)


################################ simulation 3 #################################


    
def Intensity_3(x):
    """
    intensity function defined on the interval [0,100]
    wiht 235 events
    """
    y = np.zeros(len(x))
    y += ((x>=0 ) & (x<25)) * (2 + 0.04 * x)
    y += ((x>=25) & (x<50)) * (5 - 0.08 * x)
    y += ((x>=50) & (x<75)) * (-2 + 0.06 * x)
    y += ((x>=75) & (x<=100))* (1 + 0.02 *x)
    
    return y

x = np.linspace(0,100,1000)
y = Intensity_3(x)


tau = 100
lam_star = 3.5

# generate the observed positions from true Poisson process
"""
N = np.random.poisson(tau*3)
points = tau * np.random.rand(N)
p_accept = Intensity_3(points)
u = 3 * np.random.rand(N)
S_K = points[u < p_accept]
"""
d=1
S_K = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/int3.txt',delimiter=',')
S_K = S_K.reshape(-1,d)

plt.plot(x,y, linewidth=3)
height = np.ones(len(S_K)) * 4
plt.plot(S_K, height, '|', markersize = 30, markeredgewidth=1.5, markeredgecolor='black')

lam_range = np.array([3.5])
sig_f_range = np.array([3, 5, 8])
sig_l_range = np.array([15, 20, 25])
sig_f_range = np.array([2,3])
sig_l_range = np.array([25, 28])
sig_f_range = np.array([3])
sig_l_range = np.array([28, 31])
lam0, sig_f0, sig_l0 = Select_hyp(lam_range, sig_f_range, sig_l_range, tau, S_K, Ker_sqexp, Tran_sig)

# selection result for sigmoid transform
# best_sig_f = 3
# best_sig_l = 28

#         Samp(S_K,  T,   lam_star, tau, Kernel,    sig_f, sig_l)
samples = Samp(S_K, 6000, lam_star, tau, Ker_sqexp, 3, 28)
x_sig, y_sig = Get_intensity(S_K, samples, Tran_sig, tau, burn_in= 500, norm=False, draw=True)



############################# Coal Mining #####################################

coal_mining = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/coal_mining.txt',delimiter=',')
coal_mining = coal_mining - 1850
d=1
coal_mining = coal_mining.reshape(-1,d)
height = np.ones(len(coal_mining)) * 3
plt.plot(coal_mining, height, '|', markersize = 30, markeredgewidth=1.5, markeredgecolor='black')

test_idx = np.arange(0, 191, 5)
test = coal_mining[test_idx]
train = np.delete(coal_mining, test_idx, 0)

tau = 120
lam_star = 2

lam_range = np.array([2])
sig_f_range = np.array([2, 3, 5])
sig_l_range = np.array([23, 28, 33])
sig_f_range = np.array([5, 8])
sig_l_range = np.array([20, 23])
sig_f_range = np.array([5, 7])
sig_l_range = np.array([18, 20])
sig_f_range = np.array([5])
sig_l_range = np.array([16, 18])
sig_l_range = np.array([10, 13, 16])
sig_l_range = np.array([12,13,14])

lam0, sig_f0, sig_l0 = Select_hyp(lam_range, sig_f_range, sig_l_range, tau, train, Ker_sqexp, Tran_sig)
# # selection result for sigmoid transform
# best_sig_f = 5
# best_sig_l = 13

samples = Samp(train, 4000, lam_star, tau, Ker_sqexp, 5, 13)
x_sig, y_sig = Get_intensity(train, samples, Tran_sig, tau, burn_in= 500, norm=False, draw=True)



############################### 2d simulation 1 ###############################
"""
x = np.linspace(0,5,50)
y = np.linspace(0,5,50)
xx, yy = np.meshgrid(x,y)

X = xx.reshape(1,-1)
Y = yy.reshape(1, -1)

points = np.array(list(zip(X[0],Y[0])))
K = Ker_sqexp(points, points,1,1) + np.eye(2500)*0.01

Y = np.random.multivariate_normal(mean=np.zeros(2500), cov=K)
Y = Y**2
Y = Y.reshape(50,50)
plt.imshow(Y, cmap='jet')
"""

tau = 5
lam_star = 8

"""
N = np.random.poisson(lam_star * tau**2)
Sx = tau * np.random.rand(N)
Sy = tau * np.random.rand(N)
S = np.array(list(zip(Sx,Sy)))

idx = (Sx // 0.1).astype(int)
idy = (Sy // 0.1).astype(int)
values = intensity_2d[idx, idy]
u = lam_star * np.random.rand(N)
points = S[u < values]
"""

intensity_2d = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/int2d.txt')
sk2d = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/sk2d.txt')
# x axis corresponding to col and y axis corresponding to reversed row
plt.scatter(sk2d[:,1], sk2d[:,0],marker='o', color='black',s=75)
plt.imshow(intensity_2d, cmap='jet',extent=[0,5,5,0])

lam_range = np.array([8])
sig_f_range = np.array([0.5, 0.7, 0.9])
sig_l_range = np.array([0.7, 1, 1.3])
sig_f_range = np.array([0.8, 0.9, 1.1])
sig_l_range = np.array([0.3, 0.5, 0.7])
sig_f_range = np.array([0.9])
sig_l_range = np.array([0.2, 0.3, 0.4])

lam0, sig_f0, sig_l0 = Select_hyp(lam_range, sig_f_range, sig_l_range, tau, sk2d, Ker_sqexp, Tran_sig)
# # selection result for sigmoid transform
# best_sig_f = 0.9
# best_sig_l = 0.4

sig_f = 0.9
sig_l = 0.4
samples = Samp(sk2d, 6000, lam_star, tau, Ker_sqexp, 0.9, 0.4)
x_sig, y_sig = Get_intensity(sk2d, samples, Tran_sig, tau, burn_in= 500, norm=False, draw=True)



################################ red woods data ###############################
redwood = np.loadtxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/redwood.txt',delimiter=',')
plt.scatter(redwood[:,0], redwood[:,1],marker='o', color='black',s=75)

test_idx = np.arange(0, 195, 5)
test = redwood[test_idx]
train = np.delete(redwood, test_idx, 0)

tau = 1
lam_star = 500

lam_range = np.array([500])
sig_f_range = np.array([0.1, 0.2, 0.3])
sig_l_range = np.array([0.1, 0.2, 0.3])
sig_f_range = np.array([0.05, 0.1, 0.15])
sig_l_range = np.array([0.3, 0.4])
sig_f_range = np.array([0.03, 0.04, 0.05])
sig_l_range = np.array([0.4, 0.5])

lam0, sig_f0, sig_l0 = Select_hyp(lam_range, sig_f_range, sig_l_range, tau, train, Ker_sqexp, Tran_sig)
# # selection result for sigmoid transform
# best_sig_f = 0.04
# best_sig_l = 0.4

samples = Samp(train, 6000, lam_star, tau, Ker_sqexp, 0.04, 0.4)
x_sig, y_sig = Get_intensity(train, samples, Tran_sig, tau, burn_in= 500, norm=False, draw=True)


#np.savetxt('C:/Users/lxlandsj/Desktop/Stat547P/Class Project/int1.txt', S_K, delimiter=',')
