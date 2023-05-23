from SUGAR.infer_utils import *
from SUGAR.nonlinear_learning import *


# =================================================================================================================
# This contains the major functions to implement DAG structure learning and p value calculation for simulations
# =================================================================================================================

def cal_pvalue(seed,j,k,W0,b_ls,args):     
    """
    This function calculates the p value for hypothesis testing for simulations.
    
    Input
    ----------
    seed: the random seed
    j: the first index of the edge for testing
    k: the second index of the edge for testing
    W0: the initialized weights of the DAG networks
    b_ls: the estimated graph of the DAG
    args: arguments for the hyperparameters
    
    Output
    ----------
    result: contains the weight value of the tested edge and its corresponded p value
    """
    
    # set the random seed and specify the hyperprameters
    np.random.seed(100+seed)
    K = args.K
    d = args.d
    N = args.N
    T = args.T
    c = args.c
    M = args.M
    B = args.B
    delta = args.delta
    h_size = args.h_size
    n_iter = args.n_iter
    v_dims = args.v_dims
    h_dims = args.h_dims
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500
    
    # consider different model types
    if args.graph_type == "nonlinear":
        # if it is nonlinear, generate signals with nonlinear interactions
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
        # add interactions to the model structure
        xs, W =generate_iteraction(d,W0,N,T,c,delta,y_ls)
    if args.graph_type == "linear":
        # if it is linear, generate signals with linear model
        W = W0
        xs = np.zeros((size, d))
        for i in range(d):
            for n in range(N):
                if n==0:
                    y=arma_generate_sample(arparams, maparam, T + begin)
                    y=y[begin:(T+begin)]
                else:
                    q=arma_generate_sample(arparams, maparam, T + begin)
                    put=copy.deepcopy(q[begin:(T+begin)])
                    y=np.concatenate((y, put), axis=0)
            xs[:, i] = y + xs.dot(W[i, :])
    b_all=b_ls[seed]
    # calculate the ancestors of each edge for hypothesis testing
    root_all=root_relationship_all(b_all)
    # implement the testing to calculate the pvalue value for the selected edge
    result=cal_infer_SG_DRT(j=j,k=k,root_all=root_all,K = K,b_all=b_all,xs=xs,d=d,M=M,B=B,N=N,T=T,n_iter=n_iter,h_size=h_size,v_dims=v_dims,h_dims=h_dims)
    result.append(W[j,k])
#     print("null value: ",W[j,k])
#     print("p value: ",result)
    return result


def cal_graph(random0,W0,graph_type,args):
    """
    This function implements the DAG structure learning under cross-fitting for simulations.
    
    Input
    ----------
    random0: the random seed
    W0: the initialized weights of the DAG network
    graph_type: the type of the graph model
    args: arguments for the hyperparameters
    
    Output
    ----------
    b_all: the estimated DAG structure for each fold of cross-fitted sample.
    """
    
    print("Iter: "+str(random0))
    
    # specify the random seed and hyperparameters
    np.random.seed(100+random0)
    N=args.N
    T=args.T
    c=args.c
    delta=args.delta
    n_ratio=0
    n_generate=2
    max_iter = args.max_iter
    d = args.d
    
    if  d==50 and graph_type == "nonlinear":
        lambda0 = 0.025
        w_threshold = 0.15
        h_units = 3
    elif  d==100 and graph_type == "nonlinear":
        lambda0 = 0.025
        w_threshold = 0.15
        h_units = 3     
    elif  d==150 and graph_type == "nonlinear":
        lambda0 = 0.025
        w_threshold = 0.15
        h_units = 3 
    elif  d==50 and graph_type == "linear":
        lambda0 = 0.025
        w_threshold = 0.25
        h_units = 10 
    
    # generate the data for different model types 
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500
    
    if graph_type == "nonlinear":
        # if it is nonlinear, generate signals with nonlinear interactions
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
        value=generate_iteraction(d,W0,N,T,c,delta,y_ls)
        xs=value[0]
        W=value[1]
    elif graph_type == "linear":
        # if it is linear, generate signals with linear model
        W = W0
        xs = np.zeros((size, d))
        for i in range(d):
            for n in range(N):
                if n==0:
                    y=arma_generate_sample(arparams, maparam, T + begin)
                    y=y[begin:(T+begin)]
                else:
                    q=arma_generate_sample(arparams, maparam, T + begin)
                    put=copy.deepcopy(q[begin:(T+begin)])
                    y=np.concatenate((y, put), axis=0)
            xs[:, i] = y + xs.dot(W[i, :])   
    
    # implement sample splitting
    size=xs.shape[0]
    each_size=int(size/2)
    xs1=copy.deepcopy(xs[0:each_size,:])
    xs2=copy.deepcopy(xs[each_size:size,:])
    
    torch.manual_seed(0)
    
    # implement the structure learning based on the neural network for each splitted fold
    model=NotearsMLP(dims=[d, h_units, 1], bias=True)
    W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est1)==False:
        # if a wrong penalization is used and yields out the error, we would adaptively update it.
        revise=0
        while revise==0:
            W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est1)==True:
                revise=1
            revise_times=revise_times+1

    W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est2)==False:
        # if a wrong penalization is used and yields out the error, we would adaptively update it.
        revise=0
        while revise==0:
            W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est2)==True:
                revise=1
            revise_times=revise_times+1

    b_all=[copy.deepcopy(W_est1.T),copy.deepcopy(W_est2.T)]
    return(b_all)

def struct_learn(W0,graph_type,args):
    """
    This function implements the DAG structure learning for 200 repetitions.
    
    Input
    ----------
    W0: the initialized weights of the DAG network
    graph_type: the type of the graph model
    args: arguments for the hyperparameters
    
    Output
    ----------
    b_ls: the estimated DAG structure for 200 repetitions.
    """
    b_ls=[cal_graph(random0,W0,graph_type,args) for random0 in range(200)]
    return b_ls