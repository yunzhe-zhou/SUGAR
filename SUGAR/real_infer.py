from SUGAR.infer_utils import *
from SUGAR.nonlinear_learning import *
import networkx as nx
from tqdm import tqdm

# =================================================================================================================
# This contains the major functions to implement DAG structure learning and p value calculation for the brain connectivity data
# =================================================================================================================

def calculate_interaction_graph_real(xs,max_iter,d,lambda0,w_threshold):
    """
    This function implements the DAG structure learning under cross-fitting for the brain connectivity data
    
    Input
    ----------
    xs: time series data
    max_iter: maximum number of iterations
    d: the dimension of the graph
    lambda0: parameter for regulization
    w_threshold: the cutoff for the graph weights
    
    Output
    ----------
    b_all: the estimated DAG structure for each fold of cross-fitted sample.
    """
    
    # organize the real data into multiple subjects 
    judge1=0
    judge2=0
    N=int(xs.shape[0]/316)
    T=316
    size=xs.shape[0]
    each_size=int(size/2)
    # sample splitting
    xs1=copy.deepcopy(xs[0:each_size,:])
    xs2=copy.deepcopy(xs[each_size:size,:])
    # use Notears MLP for the DAG structual learning
    model=NotearsMLP(dims=[d, 3, 1], bias=True)
    W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est1)==False:
        # if a wrong penalization is used and yields out the error, we would adaptively update it.
        judge1=1
        revise=0
        while revise==0:
            print("revise: ",revise_times)
            W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est1)==True:
                revise=1
            revise_times=revise_times+1
    W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est2)==False:
        # if a wrong penalization is used and yields out the error, we would adaptively update it.
        judge2=1
        revise=0
        while revise==0:
            print("revise: ",revise_times)
            W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est2)==True:
                revise=1
            revise_times=revise_times+1
    b_all=[copy.deepcopy(W_est1.T),copy.deepcopy(W_est2.T),judge1,judge2]
    return(b_all)

def cal_pvalue_real(j,k,K,b_ls,xs):   
    """
    This function calculates the p value for hypothesis testing for the brain connectivity data
    
    Input
    ----------
    j: the first index of the edge for testing
    k: the second index of the edge for testing
    K: the number of observations in the batched standard error estimators
    b_ls: the estimated graph of the DAG
    xs: time series data
    
    Output
    ----------
    result: contains the weight value of the tested edge and its corresponded p value
    """
    
    T=316
    d = 127
    N=int(xs.shape[0]/316)
    b_all=b_ls
    ##### calculate the ancestors for the first sample split
    G=nx.DiGraph()
    G.add_nodes_from(range(d))
    for m in range(d):
        for n in range(d):
            if b_all[0][m,n]!=0:
                G.add_edge(n,m)
    root1=[]
    for m in range(d):
        root1.append(list(nx.ancestors(G, m)))
    ##### calculate the ancestors for the second sample split
    G=nx.DiGraph()
    G.add_nodes_from(range(d))
    for m in range(d):
        for n in range(d):
            if b_all[1][m,n]!=0:
                G.add_edge(n,m)
    root2=[]
    for m in range(d):
        root2.append(list(nx.ancestors(G, m)))
    ###### the ancestors for all the split folds
    root_all=[root1,root2]
   
    ##### calculate the p value for both SUGAR and DRT
    result = cal_infer_SG_DRT(j=j,k=k,root_all=root_all,K = K,b_all=b_all,xs=xs,d=d,M=100,B=1000,N=N,T=T,n_iter=300,h_size=100,v_dims=5,h_dims=2000)
    return result


def run_real(data_type):
    """
    This function calculates the p value for brain connectivity data under low or high performance groups.
    
    Input
    ----------
    data_type: the performance groups for the data (low or high)
    
    Output
    ----------
    There is no output but the function saves the results into NumPy file.
    """
    # load the data
    if data_type == "low":
        HCP_test=np.load("data/HCP_low.npy",allow_pickle=True)
    elif data_type == "high":
        HCP_test=np.load("data/HCP_high.npy",allow_pickle=True)
    xs=HCP_test
    # standardize the data
    xs=(xs-np.mean(xs))/np.std(xs)
    d=xs.shape[1]
    
    # structual learning for the DAG graph
    max_iter=20
    lambda0=0.025
    w_threshold=0.3
    
    # we directly load the estimated DAG graph here to save computational cost but you can uncomment it when you want to 
    # use structural learning. It might take a little bit time and require GPU resources since it is a quite large network
    
    # b_ls = calculate_interaction_graph_real(xs,max_iter,d,lambda0,w_threshold) 
    b_ls = np.load("data/HCP_"+data_type+"_graph.npy",allow_pickle=True)
    
    # calculate the p value for each edge
    K = 20
    result_all = []
    for j in tqdm(range(d)):
        result_iter=[]
        for k in range(d):           
            result=cal_pvalue_real(j,k,K,b_ls,xs)
            result_iter.append(result)
        result_all.append(result_iter)
        # save the results
        if data_type == "low":
            np.save("data/HCP_low_pvalue.npy",result_all)
        elif data_type == "high":
            np.save("data/HCP_high_pvalue.npy",result_all)

def print_real(data_type):
    """
    This function is used to print out the number of identified significant within-module and between-module connections
    
    Input
    ----------
    data_type: the performance score (low or high)
    
    Output
    ----------
    pring out the number of identified significant within-module and between-module connections
    """
    
    # load the results
    if data_type == "low":
        result_all = np.load("data/HCP_low_pvalue.npy",allow_pickle=True)
    elif data_type == "high":
        result_all = np.load("data/HCP_high_pvalue.npy",allow_pickle=True)
    
    # append the calculated pvalues into a list
    pvalue_ls=[]
    for m in range(len(result_all)):
        for k in range(len(result_all[m])):
            pvalue_ls.append(result_all[m][k][0])
    
    # determine the order of the pvalues for using Benjamini–Hochberg (BH) procedure
    order=np.argsort(pvalue_ls)

    pvalue_order=[]
    for m in range(len(pvalue_ls)):
        pvalue_order.append(pvalue_ls[order[m]])

    M=len(pvalue_order)
    
    # we consider the FDR in the general case with Euler–Mascheroni constant
    R=0
    for m in range(M):
        R=R+1/(1+m)
    
    # specify the q value for BH
    q=0.05
    for m in range(M):
        k=M-m-1
        if pvalue_order[k]<= k*q/(M*R):
            M0=k
            break
            
    # record the number of identified significant within-node and between-node connections
    W=np.zeros([127,127])
    count=0
    for m in range(len(result_all)):
        for k in range(len(result_all[m])):
            if pvalue_ls[count]<=pvalue_order[M0]:
                W[m,k]=1
            count=count+1
    
    d=W.shape[0]
    G=nx.DiGraph()
    G.add_nodes_from(range(d))
    for m in range(d):
        for n in range(d):
            if W[m,n]!=0:
                G.add_edge(n,m)
                
    # record the number of identified significant within-module and between-module connections based on the module name info
    module_name=np.load("data/module_name.npy",allow_pickle=True)
    color_map = []
    node1=[]
    node2=[]
    node3=[]
    node4=[]
    for node in G:
        if module_name[node]=="Auditory":
            color_map.append('blue') 
            node1.append(node)
        if module_name[node]=="Default mode":
            color_map.append('red')    
            node2.append(node)
        if module_name[node]=="Visual":
            color_map.append('green')   
            node3.append(node)
        if module_name[node]=="Fronto-parietal Task Control":
            color_map.append('yellow')  
            node4.append(node)
    relationship=np.zeros([4,4])
    
    def return_position(m):
        if m in node1:
            value=0
        elif m in node2:
            value=1
        elif m in node3:
            value=2
        elif m in node4:
            value=3
        return int(value)
    for i in range(127):
        for j in range(127):
            if W[i,j]!=0:
                relationship[return_position(i),return_position(j)]=relationship[return_position(i),return_position(j)]+1
    
    # convert the matrix into a symmetric type
    show=relationship+relationship.T
    for m in range(4):
        show[m,m]=show[m,m]/2
    print(show)
