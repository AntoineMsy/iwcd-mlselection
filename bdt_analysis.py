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
from train_test_tree import CutEngine
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from utils import *

class BDTAnalysis():
    def __init__(self, df, train_col):
        self.df = df
        self.train_col = train_col
        self.cut_engine = CutEngine(self.df, self.train_col)

        self.latex_legends = [r'True $\nu_{e}~CC0\pi$', r"True entering $\gamma$", r'True NC $\gamma$', r'True NC $\pi^0$',  r'True NC $~$', r'True $\nu_{\mu, ws}$', r'True $\nu_{\mu}$', r'True $\nu_{e}~CC~other$', r'True $\nu_{e, ws}$', "other"]

    def create_engine_and_train(self, weight = 0.4):
        self.cut_engine = CutEngine(self.df, train_col = self.train_col, lr = 0.1)
        self.cut_engine.train()
        self.cut_engine.test(weight_rec = weight)
        self.df["sig_gbdt"] = self.cut_engine.get_pred_labels()

    def error_on_split(self):
        # Runs a series of training and splitting to estimate the variance in the output
        # outputs precision, recall, feature importance and class fraction selection
        dict_collection = []
        prec_agg, rec_agg = [], []
        df = self.df
        weight_span = np.linspace(0.1, 1, 10)
        for i in range(10):
            self.create_engine_and_train()
            # cut_engine = CutEngine(df, train_col = self.train_col, lr = 0.1)
            # cut_engine.train()
        #     cut_engine.make_calibration_curve()
            prec_l, rec_l = [], []
            dict_pred = {}
            thresh_chosen = []
            for weight_rec in weight_span:
                test_f1, prec, rec = self.cut_engine.test(weight_rec = weight_rec)
                prec_l.append(prec)
                rec_l.append(rec)
                thresh_chosen.append(self.cut_engine.best_thresh)

                df["sig_gbdt"] = self.cut_engine.get_pred_labels()
                df_cut = df.iloc[self.cut_engine.X_v.index]
    #             basic_scoring(df_cut,"sig_gbdt")
    #             basic_scoring(df_cut,"sig_fitqun")
    #             basic_scoring(df_cut,"sig_mixed")

    #             print_sel_comp(df_cut,"sig_gbdt", "sig_mixed")
                dict_frac = self.get_event_frac(df_cut,"sig_gbdt", "sig_fitqun")
                for key in list(dict_frac.keys()):
                    if key not in list(dict_pred.keys()):
                        dict_pred[key] = []
                    else : 
                        dict_pred[key] += dict_frac[key]
                dict_collection.append(dict_pred)
   
            prec_agg.append(prec_l)
            rec_agg.append(rec_l)

        prec_agg = np.array(prec_agg)
        rec_agg = np.array(rec_agg)
        print(prec_agg)
        prec_mean, rec_mean = np.mean(prec_agg, axis = 0), np.mean(rec_agg, axis = 0)
        prec_err, rec_err = np.std(prec_agg, axis = 0), np.std(rec_agg, axis =0)
        plt.errorbar(weight_span, prec_mean, prec_err, label = "Precision")
        plt.errorbar(weight_span, rec_mean, rec_err, label = "Recall")
        plt.plot(weight_span, [0.567 for i in range(10)], label = "FitQun precision", color = "blue", linestyle = ":")
        plt.plot(weight_span, [0.695 for i in range(10)], label = "FitQun recall", color = "orange", linestyle = ":")
        plt.xlabel("Recall weight in F1 score")
        plt.legend()
        plt.show()
        plt.clf()
        return self.cut_engine, df_cut, dict_collection, thresh_chosen, prec_l, rec_l

    def error_on_train(self):
        #Keeps same splitting but runs a batch of training to estimate the variance in model output
        # outputs precision, recall, feature importance and class fraction selection   
        return True

    def get_event_frac(self, df, key1, key2, return_full_count = False):
        #Plots event selection as a function of momentum with a given weight_rec
        mom_val_1 = []
        mom_val_2 = []
        dict_frac = {}
        dict_vals = {}
        for i in range(10):
            reac_sel_1 = df[(df["reac"] == i) & (df[key1]==1)]
            reac_sel_2 = df[(df['reac']==i) & (df[key2] == 1)]
            mom_val_1.append(reac_sel_1["mom"])
            mom_val_2.append(reac_sel_2["mom"])
        for i in [0,3,4]:
            dict_frac[self.latex_legends[i]] = [len(mom_val_1[i])/len(mom_val_2[i])]
            dict_vals[self.latex_legends[i]] = [len(mom_val_1[i]),len(mom_val_2[i])]
        for i in [2,6,7,8]:
            dict_frac[self.latex_legends[i]] = [len(mom_val_1[i])/len(mom_val_2[i])]
            dict_vals[self.latex_legends[i]] = [len(mom_val_1[i]),len(mom_val_2[i])]
        if return_full_count :
            return dict_frac, dict_vals
        else :
            return dict_frac
    
    def train_at_fixed_weight(self, weight = 0.4):
        for i in range(1):
            self.create_engine_and_train(weight = weight)
            test_f1, prec, rec = self.cut_engine.test(weight_rec = weight)
            self.print_metrics()
        return precision_score(self.df["sig"], self.df["sig_gbdt"]), recall_score(self.df["sig"], self.df["sig_gbdt"])
    
    def print_metrics(self, df = None):
        if df == None :
            df = self.df
        dict_frac, dict_count = self.get_event_frac(df, "sig_gbdt", "sig_fitqun", return_full_count = True)
        df_cut = df.iloc[self.cut_engine.X_v.index]
        print("prec ML on test : " + str(precision_score(df_cut["sig"], df_cut["sig_gbdt"])), "recall ML on test = " + str(recall_score(df_cut["sig"], df_cut["sig_gbdt"])) )
        print("prec FitQun on test : " + str(precision_score(df_cut["sig"], df_cut["sig_fitqun"])), "recall FitQun on test : " + str(recall_score(df_cut["sig"], df_cut["sig_fitqun"])))
        print_sel_comp(self.df, "sig_gbdt", "sig_fitqun")
        if "sig_mixed" in list(self.df.keys()):
            print("prec mixed on test : " + str(precision_score(df_cut["sig"], df_cut["sig_mixed"])), "recall mixed on test : " + str(recall_score(df_cut["sig"], df_cut["sig_mixed"])))
            print_sel_comp(self.df, "sig_gbdt", "sig_mixed")
        if "sig_ml" in list(self.df.keys()):
            print("prec ML only on test : " + str(precision_score(df_cut["sig"], df_cut["sig_ml"])), "recall ML only on test : " + str(recall_score(df_cut["sig"], df_cut["sig_ml"])))
            print_sel_comp(self.df, "sig_gbdt", "sig_ml")
        print(dict_frac, dict_count)
        

    def plot_selection(self, df, key1, key2):
        #Plots event selection as a function of momentum with a given weight_rec
        mom_val_1 = []
        mom_val_2 = []
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        dict_frac = {}
        for i in range(10):
            reac_sel_1 = df[(df["reac"] == i) & (df[key1]==1)]
            reac_sel_2 = df[(df['reac']==i) & (df[key2] == 1)]
            mom_val_1.append(reac_sel_1["mom"])
            mom_val_2.append(reac_sel_2["mom"])
        fig, axs = plt.subplots(2,2, figsize = (16,8))
        
        for i in [0,3,4]:
            axs[0].hist(mom_val_1[i], range = (0,2500), label = "%s, %s"%(key1, self.latex_legends[i]), histtype = "step", linewidth = 1.5, color = colors[i], linestyle = "-")
            axs[0].hist(mom_val_2[i], range = (0,2500), label = "%s, %s"%(key2, self.latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = ":", color = colors[i])
    
        axs[0].legend()
        axs[0].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
        axs[0].set_ylabel("Event count")
        
        for i in [2,6,7,8]:
            axs[1].hist(mom_val_1[i], range = (0,2500), label = "%s, %s"%(key1, self.latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = "-", color = colors[i])
            axs[1].hist(mom_val_2[i], range = (0,2500), label = "%s, %s"%(key2, self.latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = ":", color = colors[i])
            
        axs[1].legend()
        axs[1].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
        axs[1].set_ylabel("Event count")
    
        for i in [0,3,4]:
            axs[2].hist(mom_val_1[i]/mom_val_2[i], range = (0,2), label = "%s, %s"%(key1, self.latex_legends[i]), histtype = "step", linewidth = 1.5, color = colors[i], linestyle = "-")
          
        axs[2].legend()
        axs[2].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
        axs[2].set_ylabel("Event count")
            
        for i in [2,6,7,8]:
            axs[3].hist(mom_val_1[i]/mom_val2[i], range = (0,2), label = "%s, %s"%(key1, self.latex_legends[i]), histtype = "step", linewidth = 1.5, linestyle = "-", color = colors[i])
       
        axs[3].legend()
        axs[3].set_xlabel(r'$Reconstructed~{e}~Momentum~[MeV/c]$')
        axs[3].set_ylabel("Event count")
        plt.show()
        plt.clf()
        plt.show()
        plt.clf()
        return True
    
    def plot_2d_tp_tn_fp_fn(self,df, sig_key, key1, key2, xlim =(0,2500), ylim = (1e-12,1), logbin_y = False, vmax = 20, binnum_x = 100, binnum_y = 50, xname = "mom", yname = "pmu"):
         
        tp_cut = (df[sig_key] == 1.0) & (df["sig"] == 1.0)
        fp_cut = (df["sig"] == 0.0) & (df[sig_key] == 1.0)

        tn_cut = (df["sig"] == 0.0) & (df[sig_key] == 0.0)
        fn_cut = (df["sig"] == 1.0) & (df[sig_key] == 0.0)
            
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
    