import numpy as np
import random
import copy
import scipy
from random import sample
import matplotlib.pyplot as plt
from SUGAR.synthetic import *
import tensorflow as tf
import SUGAR.utils_tools as utils_tools
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from sklearn import neural_network
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict

# =================================================================================================================
# This contains the helper functions for p value calculation in simulations
# =================================================================================================================


def index_to_graph(num,d):
    """
    This function converts the number index to the edge index.
    
    Input
    ----------
    num: the number index
    d: the dimension of the DAG graph
    
    Output
    ----------
    i,j: the edge index
    """
    i = int(num/d)
    j = num%d
    return i,j


def root_relationship(b_):
    """
    This function calculates the ancestors for each node.
    
    Input
    ----------
    b_: the graph matrix
    
    Output
    ----------
    root: list of ancestors for each node
    """
    
    root=[]
    d=int(b_.shape[0])
    for m in range(d):
        # iterate over all the nodes
        if(np.sum(b_[m,:]!=0)==0):
            # if there is no parent for the current node, the ancestor is empty
            root.append([])
        else:
            basis=[]
            extra=[]
            new=[]
            for k in range(d):
                if b_[m,k]!=0:
                    new.append(k)
            basis=copy.deepcopy(new)
            stop=1
            while stop==1:
                # iteratively search the ancestors for each parent of the current node
                new=[]
                for k in range(len(basis)):
                    root_m=basis[k]
                    extra.append(root_m)
                    if (root_m < m):
                        for u in range(len(root[root_m])):
                            extra.append(root[root_m][u])
                    else:
                        for u in range(d):
                            if b_[root_m,u]!=0:
                                new.append(u)
                basis=copy.deepcopy(new)
                if basis==[]:
                    stop=0
            extra=sorted(set(extra))
            root.append(extra)
    return root

def root_relationship_all(b_all):
    """
    This function calculates the ancestors for the graph matrix of each sample split
    
    Input
    ----------
    b_all: the graph matrix of each sample split
    
    Output
    ----------
    root_all: list of ancestors for each node of each sample split
    """
    root_all=[]
    for l in range(2):
        root_all.append(root_relationship(b_all[l]))
    return(root_all)    

def cal_ancestor(root,j,k):
    """
    This function calculates the ancestors (without k) for each node j given the root object
    
    Input
    ----------
    root: the ancestors for each node
    j: the current node
    k: the node to exclude
    
    Output
    ----------
    act_j: the ancestors (without k) for each node j 
    """
    act_j=copy.deepcopy(root[j])
    try:
        act_j.remove(k)
    except ValueError:
        pass
    return(act_j)

def generate_f(x,choose_seed):
    """
    This function specifies the data generating function with equal probability to be sine or cosine.
    
    Input
    ----------
    x: the input value of the function
    choose_seed: the random seed for using sine or cosine function
    
    Output
    ----------
    value: the transformed value of the function
    """
    if choose_seed[0]<1:
        value=np.sin(x)
    else:
        value=np.cos(x)
    return value

def generate_iteraction(d,W,N,T,c,delta,y_ls):
    """
    This function generates the data and graph matrix with interactions

    Input
    ----------
    d: dimension of the graph 
    W: original graph 
    N: number of subjects 
    T: time length of each series
    c and delta: paramethers to control edge strength 
    y_ls: original AR process before transformation
    
    Output
    ----------
    xs: generated series
    W_new: graph matrix
    """
    # we fix a seed for the generating process so that the interaction setting and graph matrix will be the same across the repetitions. The only randomness will come from the original AR process (y_ls).
    random.seed(8)
    np.random.seed(8)
    W_new=np.zeros([d,d])
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    xs = np.zeros((size, d))
    for i in range(d):
        y=y_ls[i]
        parents= (W[i,:]!=0)
        # total number of parents for node i
        n_parents=int(np.sum(W[i,:]!=0))
        # find the positions of the parents
        parents_position=[]
        for m in range(d):
            if W[i,m]!=0:
                parents_position.append(m)
        if n_parents==0:
            xs[:,i]=y
        else:
            xs[:,i]=y
            xs_parents=xs[:,parents]
            # set the sparsity of parent edges 
            n_U=int(n_parents*(n_parents+1)/2+n_parents)
            # the strength is generated by uniform distribution
            U=np.random.uniform(0.5,1.5,n_U)*delta
            select=np.random.uniform(-1,1,n_U)
            U[select<0]=-U[select<0]
            n_select=int(c*n_U/n_parents)
            if n_select < n_U:
                select_sample=sample(range(n_U),int(n_U-n_select))
                U[select_sample]=0
            count=0
            for k in range(n_parents):
                for m in range(int(n_parents-k)):
                    # generate interaction terms by using sine or cosine function with equal probability
                    choose=int(m+k)
                    choose_seed=sample(range(2),1)
                    part1=generate_f(xs_parents[:,k],choose_seed)
                    choose_seed=sample(range(2),1)
                    part2=generate_f(xs_parents[:,choose],choose_seed)
                    xs[:,i]= xs[:,i]+U[count]*part1*part2
                    # update the structure of the graph because of the added interaction
                    if(U[count]!=0):
                        W_new[i,parents_position[k]]=1
                        W_new[i,parents_position[choose]]=1
                    count=count+1
            for k in range(n_parents):
                    # generate linear terms by using sine or cosine function with equal probability
                    choose_seed=sample(range(2),1)
                    part=generate_f(xs_parents[:,k],choose_seed)
                    xs[:,i]= xs[:,i]+U[count]*part
                    if(U[count]!=0):
                        W_new[i,parents_position[k]]=1
                    count=count+1  
    return xs, W_new  


def cal_true_graph(args,W0):
    """
    This function is used to return the true structure of DAG graph

    Input
    ----------
    args: the arguments for the hyperparameters
    W0: the structure of the initialzied graph
    
    Output
    ----------
    W: the true structure of DAG graph
    """
    d = args.d
    N = args.N
    T = args.T
    c = args.c
    delta = args.delta
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500
    
    # generate the AR process of the noise to construct the structure of the DAG graph with interactions
    y_ls=[]
    for i in range(d):
        for n in range(N):
            if n==0:
                y=arma_generate_sample(arparams, maparam, T + begin)
                y=y[begin:(T+begin)]
            else:
                q=arma_generate_sample(arparams, maparam, T + begin)
                put=copy.deepcopy(q[begin:(T+begin)])
                y=np.concatenate((y, put), axis=0)
        y_ls.append(y)        
    xs, W =generate_iteraction(d,W0,N,T,c,delta,y_ls)

    return W

def split_data(xs,l):
    """
    This function splits the whole sample into two splits

    Input
    ----------
    xs: the time series data
    l: the index for the splitted piece
    
    Output
    ----------
    train: the training data
    test: the testing data
    each_size: the size for each split
    """
    # split the subjects into two groups
    size=xs.shape[0]
    each_size=int(size/2)
    xs1=copy.deepcopy(xs[0:each_size,:])
    xs2=copy.deepcopy(xs[each_size:size,:])    

    # set one group as training (estimate learners) and another as testing (evaluate learners)
    if l==0:
        train=copy.deepcopy(xs1)
        test=copy.deepcopy(xs2)
    else:
        train=copy.deepcopy(xs2)
        test=copy.deepcopy(xs1)    
        
    return train,test,each_size 

def cal_diff2(GAN_result,xk_train,B,each_size,M):
    """
    This function calculates the second component of the test statistics in SUGAR

    Input
    ----------
    GAN_result: the oject contains the result from the fitted GAN model
    xk_train: the training data for node k
    B: the total number of transformation functions
    each_size: the size of each sample split
    M: the number of pseudo samples
    
    Output
    ----------
    diff2: the calculated values for the second component of the test statistics under different choices of pseudo samples
    u_ls: the selected pseudo samples
    """
    # generate the pseudo samples u,v
    u=np.random.normal(0,1,B)
    u_ls=[]
    
    # construct diff2
    part1_cos=[]
    part1_sin=[]
    part2_cos=[]
    part2_sin=[]
    extract=[]
    extract=np.zeros([each_size,M])
    for m in range(each_size):
        # extract the generated samples by the fitted GAN model
        q=GAN_result[0][0][m].numpy()
        q=q.reshape(q.shape[0],)
        extract[m,:]=q
        
    for b1 in range(B):
        # for each choice of the pseudo samples, we calculate it corresponded transformation functions for the test statistics
        part1_first=u[b1]*xk_train
        part1_cos.append(np.cos(part1_first))
        part1_sin.append(np.sin(part1_first))
        record=u[b1]*extract
        # cosine function part
        q1=np.sum(np.cos(record),axis=1)/M
        q1=q1.reshape(q1.shape[0],1)
        part2_cos.append(q1)
        # sine function part
        q2=np.sum(np.sin(record),axis=1)/M
        q2=q2.reshape(q2.shape[0],1)
        part2_sin.append(q2)
        
    diff2=[]
    for b1 in range(B):
        # we take the difference for the transformation funtions between the traning samples and the generated samples
        # cosine function part
        part1=part1_cos[b1]
        part2=part2_cos[b1]
        diff2.append(part1-part2)
        u_ls.append(u[b1])
        # sine function part
        part1=part1_sin[b1]
        part2=part2_sin[b1]
        diff2.append(part1-part2) 
        u_ls.append(u[b1])
    
    return diff2, u_ls

def cal_diff2_mean(xk_train,B,each_size,M):
    """
    This function calculates the second component of the test statistics based on the averaging

    Input
    ----------
    xk_train: the training data for node k
    B: the total number of transformation functions
    each_size: the size of each sample split
    M: the number of pseudo samples
    
    Output
    ----------
    diff2: the calculated values for the second component of the test statistics under different choices of pseudo samples
    u_ls: the selected pseudo samples
    """
    # generate the pseudo samples u,v
    u=np.random.normal(0,1,B)
    u_ls=[]
    
    # construct diff2
    part1_cos=[]
    part1_sin=[]
    part2_cos=[]
    part2_sin=[]
    for b1 in range(B):
        part1_first=u[b1]*xk_train
        part1_cos.append(np.cos(part1_first))
        part1_sin.append(np.sin(part1_first))
        part2_cos.append(np.cos(part1_first))
        part2_sin.append(np.sin(part1_first))
        
    diff2=[]
    for b1 in range(B):
            # we use the averaging in this case to calculate the test statistics component
            part1=part1_cos[b1]
            part2=np.mean(part2_cos[b1])
            diff2.append(part1-part2)
            u_ls.append(u[b1])
            part1=part1_sin[b1]
            part2=np.mean(part2_sin[b1])
            diff2.append(part1-part2) 
            u_ls.append(u[b1])
    
    return diff2, u_ls

def cal_position(diff1,diff2,B,N,T,each_size,K):
    """
    This function returns the position of the selected pseudo sample that maximize the test statsitics. The reason for this function is because of the sample-splitting technique used in the simulation so we need to specify the location of the selected pseudo sample.

    Input
    ----------
    diff1: the calculated first component of the test statistics
    diff2: the calculated second component of the test statisticss
    each_size: the size of each sample split
    K: the number of observations in the batched standard error estimators
    
    Output
    ----------
    b_position: the position of the selected pseudo sample that maximize the test statsitics
    """
    # calculate I_bt
    I_bt=[]
    for b in range(int(2*B)):
        I_bt.append(diff1*(diff2[b]))

    # calculate scaled and averaged version of I_bt
    I_b=[]
    for b in range(int(2*B)):
        I_b.append(2*np.sum(I_bt[b])/(N*T))
        
    # extract the average from I_bt for standard deviation calculation
    divide = int(each_size/ K)
    I_bt0 = []
    for m in range(2 * B):
        I_bt0.append((I_bt[m] - I_b[m]) / np.sqrt(K))
    
    # calculates each part for constructing batched standard error estimators
    store = np.zeros([divide, 2 * B])
    store = np.asmatrix(store)
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        for mm in range(0, 2 * B):
            store[m, mm] = np.sum(I_bt0[mm][choose])
    # calculate the standard deviation and select b        
    sigma=store.T*store
    sigma = sigma / divide
    diag = [sigma[qq,qq] for qq in range(len(sigma))]
    diag=np.sqrt(diag)
    b_value=np.abs(I_b)/diag
    b_position=np.argmax(b_value)
    
    return b_position


def cal_pvalue_SG(GAN_result,u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K):
    """
    This is the helper function to calculate the p value of hypothesis testing for SUGAR method.

    Input
    ----------
    GAN_result: the oject contains the result from the fitted GAN model
    u_ls: the list of the generated pseudo samples
    xp2: the prediction of the surpervised machine learning model
    xj_test: the test data for node j
    xk_test: the test data for node k
    b_position: the position of the selected pseudo sample
    each_size: the size of each sample split
    M: the number of pseudo samples
    N: the total number of subjects
    T: the length of the time sequence
    K: the number of observations in the batched standard error estimators
    
    Output
    ----------
    pvalue_SG: the calculated p value for the testing based on the SUGAR method.
    """
    
    # extract the selected pseudo sample
    uu=u_ls[b_position]
    
    # calculate the first component of the test statistics
    diff1=xj_test-xp2
    diff1=diff1.reshape(diff1.shape[0],1)
    
    # extract the generated samples from GAN
    extract=[]
    for m in range(each_size):
        q=GAN_result[1][0][m].numpy()
        extract.append(q)
        
    # calculate the second component of the test statistics based on sine and cosine transformation functions
    part1_first=uu*xk_test
    part1_cos=np.cos(part1_first)
    part1_sin=np.sin(part1_first)
    record=[]
    for m in range(each_size):
        record.append(np.cos(uu*(extract[m])))
    part2_cos=np.sum(record,axis=1)/M
    record=[]
    for m in range(each_size):
        record.append(np.sin(uu*(extract[m])))
    part2_sin=np.sum(record,axis=1)/M
    if (b_position%2==0):
        part1=part1_cos
        part2=part2_cos
        diff2=part1-part2
    else:
        part1=part1_sin
        part2=part2_sin
        diff2=part1-part2
        
    # calculate the test statistics corresponded to each sample
    I_bt=diff1*(diff2)
    I_b=2*np.sum(I_bt)/(N*T)
    divide = int(each_size/ K)
    # calculate the standardized version of the test statistics
    I_bt0=(I_bt - I_b) / np.sqrt(K)
    # calculate the batched standard deviation
    res=0
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        ww=np.sum(I_bt0[choose])
        res = res + ww*ww
    statistics=np.sqrt(each_size)*np.abs(I_b)/np.sqrt(res/divide)
    # calculate the pvalue
    pvalue_SG = 2*scipy.stats.norm(0, 1).cdf(-statistics)
    
    return pvalue_SG

def cal_pvalue_SG_mean(u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K):
    """
    This is the helper function to calculate the p value of hypothesis testing based on the simple averaging.

    Input
    ----------
    u_ls: the list of the generated pseudo samples
    xp2: the prediction of the surpervised machine learning model
    xj_test: the test data for node j
    xk_test: the test data for node k
    b_position: the position of the selected pseudo sample
    each_size: the size of each sample split
    M: the number of pseudo samples
    N: the total number of subjects
    T: the length of the time sequence
    K: the number of observations in the batched standard error estimators
    
    Output
    ----------
    pvalue_SG: the calculated p value for the testing based on the simple averaging.
    """
    
    # extract the selected pseudo sample
    uu=u_ls[b_position]
    
    # calculate the first component of the test statistics
    diff1=xj_test-xp2
    diff1=diff1.reshape(diff1.shape[0],1)
    
    # calculate the second component of the test statistics based on sine and cosine transformation functions
    part1_first=uu*xk_test
    part1_cos=np.cos(part1_first)
    part1_sin=np.sin(part1_first)
    part2_cos=np.cos(part1_first)
    part2_sin=np.sin(part1_first)
    if (b_position%2==0):
        part1=part1_cos
        part2=np.mean(part2_cos)
        diff2=part1-part2
    else:
        part1=part1_sin
        part2=np.mean(part2_sin)
        diff2=part1-part2
        
    # calculate the test statistics corresponded to each sample
    I_bt=diff1*(diff2)
    I_b=2*np.sum(I_bt)/(N*T)
    divide = int(each_size/ K)
    
    # calculate the standardized version of the test statistics
    I_bt0=(I_bt - I_b) / np.sqrt(K)
    
    # calculate the batched standard deviation
    res=0
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        ww=np.sum(I_bt0[choose])
        res = res + ww*ww
    statistics=np.sqrt(each_size)*np.abs(I_b)/np.sqrt(res/divide)
    
    # calculate the pvalue
    pvalue_SG = 2*scipy.stats.norm(0, 1).cdf(-statistics)

    return pvalue_SG

def cal_pvalue_DRT(diff1,diff2,N,T,each_size,K):
    """
    This is the helper function to calculate the p value of hypothesis testing based on the DRT.

    Input
    ----------
    diff1: the calculated first component of the test statistics
    diff2: the calculated second component of the test statistics
    N: the total number of subjects
    T: the length of the time sequence
    K: the number of observations in the batched standard error estimators
    
    Output
    ----------
    pvalue_DRT: the calculated p value for the testing based on DRT.
    """
    # calculate the test statistics corresponded to each sample
    I_bt=diff1*(diff2)
    I_b=2*np.sum(I_bt)/(N*T)
    divide = int(each_size/ K)
    
    # calculate the standardized version of the test statistics
    I_bt0=(I_bt - I_b) / np.sqrt(K)
    
    # calculate the batched standard deviation
    res=0
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        ww=np.sum(I_bt0[choose])
        res = res + ww*ww
    statistics=np.sqrt(each_size)*np.abs(I_b)/np.sqrt(res/divide)
    
    # calculate the pvalue
    pvalue_DRT = 2*scipy.stats.norm(0, 1).cdf(-statistics)
    return pvalue_DRT
    
def cal_infer_SG_DRT(j,k,root_all,K,b_all,xs,d,M,B,N,T,n_iter,h_size,v_dims,h_dims):
    """
    This is the utils function to calculate the p value of hypothesis testing for both SUGAR and DRT method at the same time to save the computational cost.

    Input
    ----------
    j, k: the node where its edge will be tested
    root_all: the list of ancestors for current node j excluding node k
    b_all: the estimated DAG graph for each splitted sample
    xs: time series data
    d: the dimension of the DAG graph
    M: the number of pseudo samples
    B: the number of transformation functions
    N: the total number of subjects
    T: the length of the time sequence
    n_iter: the number of iterations to train the neural network of GAN
    h_size: the number of hidden units for the neural network in surpervised ML part
    v_dims: the dimension of the input noise for the generative model of GAN
    h_dims: the number of hidden units for the neural network of GAN
    
    Output
    ----------
    result_ls: [pvalue of SG, pvalue of DRT, [j,k]]
    """

    # set random seed
    np.random.seed(8)
    random.seed(8)
    tf.random.set_seed(8)
    
    pvalue_SG = []
    pvalue_DRT = []
    for l in range(2):
        # split the training and testing set
        train, test, each_size = split_data(xs,l)
        
        # calculate the ancestors
        act_j=cal_ancestor(root_all[l],j,k)
        
        # extract data for training and testing
        z_train=train[:,act_j]
        z_test=test[:,act_j]
        xj_train=train[:,j]
        xj_test=test[:,j]
        xk_train=train[:,[k]]
        xk_test=test[:,[k]]
        dim_z=z_train.shape[1]

        if (k in root_all[l][j]) == False:
            '''
            k doesn't belong to ancestors of j so the p-value is set as 1 according the algorithm in the paper
            '''
            
            '''
            SUGAR Test 
            '''
            pvalue_SG.append(1)
            
            '''
            Double Rregression Test 
            '''
            pvalue_DRT.append(1)
            
        elif (cal_ancestor(root_all[l],j,k)==[]) and (root_all[l][j]!=[]):
            '''
            when k is the only ancestor of j 
            '''  
            xp1=np.mean(xj_train)
            xp2=np.mean(xj_test)
            
            '''
            SUGAR Test 
            '''
            # calculate the first part of the test statistics
            diff1=xj_train-xp1
            diff1=diff1.reshape(diff1.shape[0],1)
            
            # calculate the second part of the test statistics based on the simple average
            diff2, u_ls = cal_diff2_mean(xk_train,B,each_size,M)

            # get the position
            b_position = cal_position(diff1,diff2,B,N,T,each_size,K)
            
            # calculate pvalue for Sugar
            pvalue_SG.append(cal_pvalue_SG_mean(u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K))

            '''
            Double Rregression Test 
            '''
            # update the first part of the test statistics for another split
            diff1=xj_test-xp2
            diff1=diff1.reshape(diff1.shape[0],1)
            
            # update the second part of the test statistics for another split based on the simple average
            xk_train=xk_train.reshape(xk_train.shape[0],)
            xk_test=xk_test.reshape(xk_test.shape[0],)
            xp2=np.mean(xk_test)
            diff2=xk_test-xp2
            diff2=diff2.reshape(diff2.shape[0],1)
            
            pvalue_DRT.append(cal_pvalue_DRT(diff1,diff2,N,T,each_size,K))   
        else:
            '''
            for the other more general cases
            '''  
            # supervised learning algorithm
            nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(h_size,),activation='relu', solver='adam', max_iter=2000)
            regressormodel = nn_unit.fit(z_train, xj_train)
            
            # get the prediction for both traning and testing data
            xp1 = nn_unit.predict(z_train)
            xp2 = nn_unit.predict(z_test)       

            '''
            SUGAR Test 
            '''
            
            # CGAN
            GAN_result=utils_tools.gcit_tools(x_train=xk_train,z_train=z_train,x_test=xk_test,z_test=z_test,v_dims=v_dims,h_dims=h_dims,M = M, batch_size=64, n_iter=n_iter, standardise =False,normalize=True)        

            # calculate the first part of the test statistics
            diff1=xj_train-xp1
            diff1=diff1.reshape(diff1.shape[0],1)

            # calculate the second part of the test statistics
            diff2, u_ls = cal_diff2(GAN_result,xk_train,B,each_size,M)

            # get the position
            b_position = cal_position(diff1,diff2,B,N,T,each_size,K)

            # calculate pvalue for Sugar
            pvalue_SG.append(cal_pvalue_SG(GAN_result,u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K))

            '''
            Double Rregression Test 
            '''
            # update the first part of the test statistics for another split
            diff1=xj_test-xp2
            diff1=diff1.reshape(diff1.shape[0],1)

            # supervised learning algorithm
            nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(h_size,),activation='relu', solver='adam', max_iter=2000)
            xk_train=xk_train.reshape(xk_train.shape[0],)
            xk_test=xk_test.reshape(xk_test.shape[0],)
            regressormodel = nn_unit.fit(z_train, xk_train)
            xp2 = nn_unit.predict(z_test)

            # update the second part of the test statistics for another split
            diff2=xk_test-xp2
            diff2=diff2.reshape(diff2.shape[0],1)
            
            # calculate the p value for DRT
            pvalue_DRT.append(cal_pvalue_DRT(diff1,diff2,N,T,each_size,K))
        
    # save 4 digits of the p value and record the corresponded edge
    result_ls=[]
    result_ls.append(round(2*np.minimum(pvalue_SG[0],pvalue_SG[1]),4))
    result_ls.append(round(2*np.minimum(pvalue_DRT[0],pvalue_DRT[1]),4))
    result_ls.append([j,k])
    return(result_ls)           
        