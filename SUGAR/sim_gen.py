from SUGAR.infer_utils import *
from SUGAR.nonlinear_learning import *

# =================================================================================================================
# This contains the utils functions to generate data for simulations and is sourced by main_lrt.R file for implementing
# the LRT.
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

def sim_gen(seed,d,prob,N,T,delta):   
    """
    This generates the synthetic data for simulations.

    Input
    ----------
    seed: random seed 
    d: the dimension of the DAG network
    prob: density of the DAG network
    N: the number of the subjects of the data
    T: the length of the time sequence
    delta: paramether to control edge strength
    
    Output
    ----------
    xs: generated series
    W: graph matrix
    """
    
    # initialize the weight matrix
    np.random.seed(8)
    W0 = generate_W(d=d, prob=prob)
    
    np.random.seed(100+seed)
    K = 20
    c = 2
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500
    
    # generate the AR noise 
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
        
    # generate time series and calculate its corresponed graph structure
    xs, W =generate_iteraction(d,W0,N,T,c,delta,y_ls)
    
    return xs, W
    
def get_edges(d,W):
    """
    randomly select 100 edges for testing the null or alterative hypothesis seperately

    Input
    ----------
    d: the dimension of the DAG network
    W: graph of the DAG
    
    Output
    ----------
    edge_null_pair: selected edges for null hypothesis
    edge_alter_pair: selected edges for alternative hypothesis
    """
    np.random.seed(8)
    arr_index = np.arange(d**2).reshape([d,d])
    edge_null = arr_index[W==0]
    edge_alter = arr_index[W!=0]
    edge_null_select = np.random.choice(edge_null,100,replace=False)
    edge_alter_select = np.random.choice(edge_alter,100,replace=False)
    edge_null_pair = [index_to_graph(item,d) for item in edge_null_select]
    edge_alter_pair = [index_to_graph(item,d) for item in edge_alter_select]
    
    return edge_null_pair,edge_alter_pair