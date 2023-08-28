import uproot as uproot
import numpy as np
import awkward as ak
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.special import exp10
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

latex_legends = [r'True $\nu_{e}~CC0\pi$', r"True entering $\gamma$", r'True NC $\gamma$', r'True NC $\pi^0$',  r'True NC $~$', r'True $\nu_{\mu, ws}$', r'True $\nu_{\mu}$', r'True $\nu_{e}~CC~other$', r'True $\nu_{e, ws}$', "other"]

def read_folder(fpath = "/home/amisery/Analysis/RecoAnalysis/outputs/data/data_all_precut_likelihood/", flim = 90):
    df_list = []
    for i in range(1,flim):
        if i not in [18,19,20]:
            file = fpath + "chunk_%d.root"%(i)
            root_file = uproot.open(file)
            dict_vals = {}
            for key in root_file["data"].keys():
                dict_vals[key] = np.array(root_file["data"][key])[0]
            df_chunk = pd.DataFrame(dict_vals)
            df_list.append(df_chunk)
            
    df = pd.concat(df_list)
    return df

def plot_sig_bg_2d_comp(df, key1, key2, xlim =(0,2500), ylim = (1e-12,1), logbin_y = False, binnum_x = 100, binnum_y = 200, xname = "mom", yname = "pmu", plot_line = "cutname", show = False):
    sig_cut = (df["sig"] == 1.0)
    bg_cut = (df["sig"] == 0.0)
        
    df_sig = df.loc[sig_cut]
    df_bg = df.loc[bg_cut]
    # Define the number of bins for x and y axes
    if logbin_y:
        bins_y = [10**(x) for x in np.linspace(-12,0,50)]
        yscale = "log"
    else :
        bins_y = binnum_y
        yscale = "linear"
    bins_x = binnum_x
    fig, axes = plt.subplots(1,2, figsize = (20,8))
    #fig.tight_layout(pad = 3)
    plt.subplot(121,)
    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_sig[key1], df_sig[key2], range = [list(xlim), list(ylim)], bins=[bins_x, bins_y])
    plt.yscale(yscale)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(r" $\nu_e$ $CC0\pi$ events (signal)")
    # Plot the histogram
    plt.gca().set_aspect("auto")
    plt.pcolor(x_edges, y_edges, hist.T, norm =mpl.colors.LogNorm())
    plt.colorbar()
    
    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_bg[key1], df_bg[key2], range = [list(xlim), list(ylim)],  bins=[bins_x, bins_y])
    
    plt.subplot(122,)
    plt.yscale(yscale)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title("Background")
    plt.gca().set_aspect("auto")
    # Plot the histogram
    plt.pcolor(x_edges, y_edges, hist.T, norm =mpl.colors.LogNorm())
    plt.colorbar()
    plt.subplots_adjust(wspace = 0.25)
#     fig.colorbar(im)
    if show :
        plt.show()
    return fig, axes
    

def plot_class_frac(df):
    label_list = [0,2,3,4,6,7,8]
    names = [latex_legends[i] for i in label_list]
    counts = [len(df.loc[df["reac"]==i])/len(df) for i in label_list]
    plt.xticks(rotation=30, ha='right')
    plt.title("Class fraction before basic cuts")
    plt.ylabel("Class fraction [a.u.]")
    plt.bar(names, counts)
    plt.show()
    plt.clf()

def plot_2d_tp_tn_fp_fn(df, key1, key2, xlim =(0,2500), ylim = (1e-12,1), logbin_y = False, vmax = 20, binnum_x = 100, binnum_y = 50, xname = "mom", yname = "pmu"):
    
    tp_cut = (df["sig_pred"] == 1.0) & (df["sig"] == 1.0)
    fp_cut = (df["sig"] == 0.0) & (df["sig_pred"] == 1.0)

    tn_cut = (df["sig"] == 0.0) & (df["sig_pred"] == 0.0)
    fn_cut = (df["sig"] == 1.0) & (df["sig_pred"] == 0.0)
        
    df_tp = df.loc[tp_cut]
    df_fp = df.loc[fp_cut]
    df_tn = df.loc[tn_cut]
    df_fn = df.loc[fn_cut]
    # Define the number of bins for x and y axes
    if logbin_y:
        bins_y = [10**(x) for x in np.linspace(-12,0,50)]
        yscale = "log"
    else :
        bins_y = binnum_y
        yscale = "linear"
    bins_x = binnum_x
    fig = plt.figure(figsize = (12,12))
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    #fig.tight_layout(pad = 3)
#     plt.subplot(121,)
    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_tp[key1], df_tp[key2],  bins=[bins_x, bins_y])
    ax1.pcolor(x_edges, y_edges, hist.T, vmin = 0, vmax = vmax, label = "True positives")
    ax1.set_yscale(yscale)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)
    ax1.legend(handletextpad=-0.3, handlelength=0)
#     pl1.title(r"TP events")
    # Plot the histogram
#     plt.gca().set_aspect("auto"
    
    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_fp[key1], df_fp[key2],  bins=[bins_x, bins_y])
    
#     plt.subplot(122,)
#     plt.gca().set_aspect("auto")
    # Plot the histogram
    ax2.pcolor(x_edges, y_edges, hist.T, vmin = 0, vmax = vmax, label = "False Positives")
    ax2.set_yscale(yscale)
    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.legend(handletextpad=-0.3, handlelength=0)
#     ax2.set_xlabel(xname)
#     ax2.set_ylabel(yname)
#     plt.title("FP events")
    
    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_tn[key1], df_tn[key2],  bins=[bins_x, bins_y])
    
#     plt.subplot(221,)
#     plt.gca().set_aspect("auto")
    # Plot the histogram
    ax3.pcolor(x_edges, y_edges, hist.T, vmin = 0, vmax = vmax, label = "True negatives")
    ax3.set_yscale(yscale)
    ax3.set_ylim(ylim)
    ax3.set_xlim(xlim)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname)
    ax3.legend(handletextpad=-0.3, handlelength=0)
#     plt.title("TN events")
    
    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_fn[key1], df_fn[key2],  bins=[bins_x, bins_y])
    
#     plt.subplot(222,)
#     plt.gca().set_aspect("auto")
    # Plot the histogram
    im = ax4.pcolor(x_edges, y_edges, hist.T, vmin = 0, vmax = vmax, label = "False Negatives")
    ax4.set_yscale(yscale)
    ax4.set_ylim(ylim)
    ax4.set_xlim(xlim)
    ax4.set_xlabel(xname)
    ax4.legend(handletextpad=-0.3, handlelength=0)
#     ax4.set_ylabel(yname)
#     title("FN events")
    
#     plt.subplots_adjust(wspace = 0.25)
    fig.colorbar(im, ax = [ax2,ax4], shrink = 0.7)
    plt.show()

def return_cuts(df):
    #Return precomputed ML visual selection cuts for the analysis
    pmu_cut = ((df["pmu"]<0.0004) & (df["mom"]>300)) | ((df["pmu"]< 10**((df["mom"]-100)/200*(6+np.log10(0.0004)) - 6)) & (df["mom"] <= 300))
    pi0_cut = (((df["ppi0"]<0.3) & (df["mom"]<=1000)) | (df["mom"]>1000)) & (df["pi0mass"]<100)
    pe_cut = ((df["pe"]>0.35) & (df["mom"]<250)) | (df["mom"]>=250)

    #Precomputed pi0 fitqun cut
    pi0_fitqun_cut = (df["pi0fitqun"] == 1)

    return  pmu_cut, pi0_cut, pe_cut, pi0_fitqun_cut

def print_sel_comp(df, key1, key2):
    mom_val_1 = []
    mom_val_2 = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    dict_frac = {}
    dict_vals = {}
    for i in range(10):
        reac_sel_1 = df[(df["reac"] == i) & (df[key1]==1)]
        reac_sel_2 = df[(df['reac']==i) & (df[key2] == 1)]
        mom_val_1.append(reac_sel_1["mom"])
        mom_val_2.append(reac_sel_2["mom"])
    fig, axs = plt.subplots(2,2, figsize = (16,8))
    
    for i in [0,3,4]:
        axs[0,0].hist(mom_val_1[i], range = (0,2500), label = "%s, %s"%(key1, latex_legends[i]), histtype = "step", linewidth = 1.5, color = colors[i], linestyle = "-")
        axs[0,0].hist(mom_val_2[i], range = (0,2500), label = "%s, %s"%(key2, latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = ":", color = colors[i])
        dict_frac[latex_legends[i]] = [len(mom_val_1[i])/len(mom_val_2[i])]
    axs[0,0].legend()
    axs[0,0].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
    axs[0,0].set_ylabel("Event count")
    
    for i in [2,6,7,8]:
        axs[0,1].hist(mom_val_1[i], range = (0,2500), label = "%s, %s"%(key1, latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = "-", color = colors[i])
        axs[0,1].hist(mom_val_2[i], range = (0,2500), label = "%s, %s"%(key2, latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = ":", color = colors[i])
        dict_frac[latex_legends[i]] = [len(mom_val_1[i])/len(mom_val_2[i])]
    axs[0,1].legend()
    axs[0,1].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
    axs[0,1].set_ylabel("Event count")
    
    for i in [0,3,4]:
        histo1, x_edges = np.histogram(mom_val_1[i], range = (0,2500), bins=5)
        histo2, x_edges = np.histogram(mom_val_2[i], range = (0,2500), bins=5)
        vals_x = np.array([])
        vals_y = np.array([])
        for j in range(5):
            vals_x = np.concatenate((vals_x, np.linspace(x_edges[j],x_edges[j+1],10)))
            vals_y = np.concatenate((vals_y,np.array([histo1[j]/histo2[j] for i in range(10)])))
        axs[1,0].plot(vals_x, vals_y, label = "%s, %s"%(key1, latex_legends[i]), linewidth = 1.5, linestyle = ":", color = colors[i])
    
        dict_frac[latex_legends[i]] = [len(mom_val_1[i])/len(mom_val_2[i])]
        dict_vals[latex_legends[i]] = [len(mom_val_1[i]),len(mom_val_2[i])]
    axs[1,0].legend()
    axs[1,0].hlines(1, 0, 2500, colors='black', linewidth=3)
    axs[1,0].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
    axs[1,0].set_ylabel("%s/%s event ratio"%(key1, key2))
    axs[1,0].set_ylim(0,2)
    
    for i in [2,6,7,8]:
        histo1, x_edges = np.histogram(mom_val_1[i], range = (0,2500), bins = 5)
        histo2, x_edges = np.histogram(mom_val_2[i], range = (0,2500), bins = 5)
        print(histo1/histo2)
        vals_x = np.array([])
        vals_y = np.array([])
        for j in range(5):
            vals_x = np.concatenate((vals_x, np.linspace(x_edges[j],x_edges[j+1],10)))
            vals_y = np.concatenate((vals_y,np.array([histo1[j]/histo2[j] for i in range(10)])))
        axs[1,1].plot(vals_x, vals_y, label = "%s, %s"%(key1, latex_legends[i]), linewidth = 1.5, linestyle = ":", color = colors[i])
        dict_frac[latex_legends[i]] = [len(mom_val_1[i])/len(mom_val_2[i])]
        dict_vals[latex_legends[i]] = [len(mom_val_1[i]),len(mom_val_2[i])]
    axs[1,1].legend()
    axs[1,1].hlines(1, 0, 2500, colors='black', linewidth=3)
    axs[1,1].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
    axs[1,1].set_ylabel("%s/%s event ratio"%(key1, key2))
    axs[1,1].set_ylim(0,2)
    plt.show()
    plt.clf()
    plt.show()
    plt.clf()
    return dict_frac, dict_vals