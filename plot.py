import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Main function to generate plots for the simulation resultss
# =============================================================================


def generate_plot_null(ax1,d,N,T,delta):
    """
    This generates the plot for the simulation results under the null.
    
    Input
    ----------
    d: the dimension of the DAG network
    N: the number of the subjects of the data
    T: the length of the time sequence
    delta: paramether to control edge strength
    
    Output
    ----------
    There is no output but corresponded plots are generated.
    """
    
    pvalue_sugar_ls_null = []
    pvalue_drt_ls_null = []     
    path = "data/nonlinear_"+str(d)+"_null_N_"+str(N)+"_T_"+str(T)+"_delta_"+str(delta)+".npy"
    pvalue_all1=np.load(path,allow_pickle=True)
    for i in range(len(pvalue_all1)):
        pvalue_sugar_null = np.mean(np.array([item[0] for item in pvalue_all1[i]])<0.05)
        pvalue_drt_null = np.mean(np.array([item[1] for item in pvalue_all1[i]])<0.05)
        pvalue_sugar_ls_null.append(pvalue_sugar_null)
        pvalue_drt_ls_null.append(pvalue_drt_null)
    pvalue_sugar_ls_null = np.array(pvalue_sugar_ls_null)
    pvalue_drt_ls_null = np.array(pvalue_drt_ls_null)
   

    path = "data/LRT_nonlinear_"+str(d)+"_null_N_"+str(N)+"_T_"+str(T)+"_delta_"+str(delta)+".npy"
    pvalue_all2=np.load(path,allow_pickle=True)
    pvalue_lrt_null = np.mean(pvalue_all2<0.05,1)

    bp1 = ax1.boxplot(pvalue_sugar_ls_null,positions=[1],widths=0.6)
    bp2 = ax1.boxplot(pvalue_drt_ls_null,positions=[2],widths=0.6)
    bp3 = ax1.boxplot(pvalue_lrt_null,positions=[3],widths=0.6)

    ax1.axhline(y=0.05, color='red', linestyle='--')
    ax1.set_xticks(np.arange(1, 4, 1));
    ax1.set_xticklabels(["SUGAR","DRT","LRT"],fontsize= 12)

    ax1.set_title("d = "+str(d)+", N= "+str(N)+", T = "+str(T) + r", $\delta$ = " + str(delta), fontsize=12) 
    
def generate_plot_alter(ax1,d,N,T,delta):
    """
    This generates the plot for the simulation results under the alternative.
    
    Input
    ----------
    d: the dimension of the DAG network
    N: the number of the subjects of the data
    T: the length of the time sequence
    delta: paramether to control edge strength
    
    Output
    ----------
    There is no output but corresponded plots are generated.
    """
    
    pvalue_sugar_ls_alter = []
    pvalue_drt_ls_alter = []     
    path = "data/nonlinear_"+str(d)+"_alter_N_"+str(N)+"_T_"+str(T)+"_delta_"+str(delta)+".npy"
    pvalue_all1=np.load(path,allow_pickle=True)
    for i in range(len(pvalue_all1)):
        pvalue_sugar_alter = np.mean(np.array([item[0] for item in pvalue_all1[i]])<0.05)
        pvalue_drt_alter = np.mean(np.array([item[1] for item in pvalue_all1[i]])<0.05)
        pvalue_sugar_ls_alter.append(pvalue_sugar_alter)
        pvalue_drt_ls_alter.append(pvalue_drt_alter)
    pvalue_sugar_ls_alter = np.array(pvalue_sugar_ls_alter)
    pvalue_drt_ls_alter = np.array(pvalue_drt_ls_alter)
    
    bp1 = ax1.boxplot(pvalue_sugar_ls_alter,positions=[1],widths=0.6)
    bp2 = ax1.boxplot(pvalue_drt_ls_alter,positions=[2],widths=0.6)
    
    ax1.axhline(y=0.05, color='red', linestyle='--')
    ax1.set_xticks(np.arange(1, 3, 1));
    ax1.set_xticklabels(["SUGAR","DRT"],fontsize= 12)
    
    ax1.set_title("d = "+str(d)+", N= "+str(N)+", T = "+str(T) + r", $\delta$ = " + str(delta), fontsize=12) 

def generate_plot_diff(ax1,d,N,T,delta):
    """
    This generates the power difference plot for the simulation results under the alternative.
    
    Input
    ----------
    d: the dimension of the DAG network
    N: the number of the subjects of the data
    T: the length of the time sequence
    delta: paramether to control edge strength
    
    Output
    ----------
    There is no output but corresponded plots are generated.
    """
    
    pvalue_sugar_ls_alter = []
    pvalue_drt_ls_alter = []     
    path = "data/nonlinear_"+str(d)+"_alter_N_"+str(N)+"_T_"+str(T)+"_delta_"+str(delta)+".npy"
    pvalue_all1=np.load(path,allow_pickle=True)
    for i in range(len(pvalue_all1)):
        pvalue_sugar_alter = np.mean(np.array([item[0] for item in pvalue_all1[i]])<0.05)
        pvalue_drt_alter = np.mean(np.array([item[1] for item in pvalue_all1[i]])<0.05)
        pvalue_sugar_ls_alter.append(pvalue_sugar_alter)
        pvalue_drt_ls_alter.append(pvalue_drt_alter)
    pvalue_sugar_ls_alter = np.array(pvalue_sugar_ls_alter)
    pvalue_drt_ls_alter = np.array(pvalue_drt_ls_alter)
    
    bp1 = ax1.boxplot(pvalue_sugar_ls_alter - pvalue_drt_ls_alter,widths=0.6)
    
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xticks(np.arange(1, 2, 1));
    ax1.set_xticklabels(["Difference between SUGAR and DRT"],fontsize= 12)
    
    ax1.set_title("d = "+str(d)+", N= "+str(N)+", T = "+str(T) + r", $\delta$ = " + str(delta), fontsize=12) 
    
# different scenarios for null hypothesis testing
f, ax = plt.subplots(4, 3, figsize=(18,20))
generate_plot_null(ax[0,0],50,10,100,1)
generate_plot_null(ax[0,1],50,20,100,1)
generate_plot_null(ax[0,2],50,40,100,1)
generate_plot_null(ax[1,0],50,20,50,1)
generate_plot_null(ax[1,1],50,20,100,1)
generate_plot_null(ax[1,2],50,20,200,1)
generate_plot_null(ax[2,0],50,20,100,0.5)
generate_plot_null(ax[2,1],50,20,100,1)
generate_plot_null(ax[2,2],50,20,100,2)
generate_plot_null(ax[3,0],50,20,100,1)
generate_plot_null(ax[3,1],100,20,100,1)
generate_plot_null(ax[3,2],150,20,100,1)
f.savefig('sim_null.png')

# different scenarios for alternative hypothesis testing
f, ax = plt.subplots(4, 3, figsize=(18,20))
generate_plot_alter(ax[0,0],50,10,100,1)
generate_plot_alter(ax[0,1],50,20,100,1)
generate_plot_alter(ax[0,2],50,40,100,1)
generate_plot_alter(ax[1,0],50,20,50,1)
generate_plot_alter(ax[1,1],50,20,100,1)
generate_plot_alter(ax[1,2],50,20,200,1)
generate_plot_alter(ax[2,0],50,20,100,0.5)
generate_plot_alter(ax[2,1],50,20,100,1)
generate_plot_alter(ax[2,2],50,20,100,2)
generate_plot_alter(ax[3,0],50,20,100,1)
generate_plot_alter(ax[3,1],100,20,100,1)
generate_plot_alter(ax[3,2],150,20,100,1)
f.savefig('sim_alter.png')

# different scenarios for alternative hypothesis testing and plot the difference of power
f, ax = plt.subplots(4, 3, figsize=(18,20))
generate_plot_diff(ax[0,0],50,10,100,1)
generate_plot_diff(ax[0,1],50,20,100,1)
generate_plot_diff(ax[0,2],50,40,100,1)
generate_plot_diff(ax[1,0],50,20,50,1)
generate_plot_diff(ax[1,1],50,20,100,1)
generate_plot_diff(ax[1,2],50,20,200,1)
generate_plot_diff(ax[2,0],50,20,100,0.5)
generate_plot_diff(ax[2,1],50,20,100,1)
generate_plot_diff(ax[2,2],50,20,100,2)
generate_plot_diff(ax[3,0],50,20,100,1)
generate_plot_diff(ax[3,1],100,20,100,1)
generate_plot_diff(ax[3,2],150,20,100,1)
f.savefig('sim_diff.png')