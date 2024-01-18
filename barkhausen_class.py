# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:31:16 2023

@author: gh457
"""
import powerlaw
import numpy as np
import pickle
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin_fit(x,a,b):
    return a*x+b

def load_bark(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj

def plot_datasets(objects,nrofbins=50):
    
        
    params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
    plt.rcParams.update(params)
    plt.figure(figsize=(10,6))
    plt.grid(which='major',axis='both')
    plt.xscale('log')
    plt.yscale('log')
    for obj in objects:
        
        plt.scatter(obj.bins,obj.hist/max(obj.hist))
        
    plt.show()

class Barkhausen_data:
    def __init__(self,eventsize,eventenergy,threshold,findmaxavg,folder):
        self.sizes=eventsize
        self.energies=eventenergy
        self.threshold=threshold
        self.findmax=findmaxavg
        self.folder=folder
        
        self.xminsize=0
        self.xmaxsize=1
        
        self.xminenergy=0
        self.xmaxenergy=1
        
        self.MLxminsize=0
        self.MLxminenergy=0
        
        self.bins=[]
        self.hist=[]
        
        
        lastslash=0
        slashindex=0
        for char in self.folder:
            if char == '\\':
                lastslash=slashindex+1
            slashindex=slashindex+1
        name=self.folder[lastslash:]
        self.foldername=name
        
        self.data=self.sizes
    
    def set_style(self):  #set the plotting fontsizes etc
        plt.style.use('seaborn-v0_8-whitegrid')

        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            'text.latex.preamble': r'\usepackage{upgreek}',
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 15,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,

            'legend.frameon' : True
            #"axes.grid" : True,
            #"grid.color": "grey"
        }
        plt.rcParams.update(tex_fonts)
        return
        
    def switchdata(self): #switch between biggest event amplitude and integrated event size
        if self.data==self.sizes:
            self.data=self.energies
        elif self.data==self.energies:
            self.data=self.sizes
        
    def MLexponent(self): #determine the power law exponent using the Maximum Likelihood method
        fit=powerlaw.Fit(self.data)
        self.MLxmin=fit.xmin
        return fit.alpha , fit.sigma
    
    def Linexponent(self,xmin,xmax): #determine the power law exponent usng linear fit by setting the boundaries in which the power law is good
        if self.data==self.sizes:
            self.xminsize=xmin
            self.xmaxsize=xmax
        elif self.data==self.energies:
            self.xminenergy=xmin
            self.xmaxenergy=xmax
        
        edges, hist = powerlaw.pdf(self.data,number_of_bins=50)
        
        bin_centers = (edges[1:]+edges[:-1])/2.0
        
        hist=hist[xmin<bin_centers]
        bin_centers=bin_centers[xmin<bin_centers]

        hist=hist[bin_centers<xmax]
        bin_centers=bin_centers[bin_centers<xmax]

        bin_centers=bin_centers[hist!=0]
        hist=hist[hist!=0]

        popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))
        
        exponent=popt[0]
        error=np.sqrt(pcov[0][0])
        fitdata=popt
        return np.array([error,exponent,fitdata],dtype=object)
    
    def bins_hist(self,binnr=50):
        edges, hist = powerlaw.pdf(self.data,number_of_bins=binnr)
        
        bin_centers = (edges[1:]+edges[:-1])/2.0
                
        self.hist=hist
        self.bins=bin_centers
    
    def plot_data(self,nrofbins=50,MLFit=True,MLmin=False,MLxmin=1e-2,LinFit=False,xmin=1e-7,xmax=5e1):
        
        self.set_style()
        
        # plt.hist(self.data,np.logspace(np.log10(min(self.data)),np.log10(max(self.data)),nrofbins),density=True)
        
        if MLFit==True:
            if MLmin==True:    
                fit=powerlaw.Fit(self.data,xmin=MLxmin)
            elif MLmin==False:
                fit=powerlaw.Fit(self.data)
            fit.power_law.plot_pdf(label=r'$\alpha_{ML}=$'+str(round(fit.alpha,2))+r'$\pm$'+str(round(fit.sigma,2)))
            
        if LinFit==True:
            values=self.Linexponent(xmin,xmax)[2]
            manualfit=np.exp(lin_fit(np.log(self.bins),*values))
            plt.plot(self.bins[self.bins>=xmin],manualfit[self.bins>=xmin])
        try:
            plt.scatter(self.bins,self.hist)
            plt.plot(self.bins[self.bins>=xmin*0.5],manualfit[self.bins>=xmin*0.5],label=r'$\alpha_{Lin}=$'+str(round(np.abs(self.Linexponent(xmin, xmax)[1],2)))+r'$\pm$'+str(round(self.Linexponent(xmin, xmax)[0],2)))
            # fit.power_law.plot_pdf()
            
            plt.xscale('log')
            plt.yscale('log')
            
            plt.xlabel('biggest amplitude in an event')
            plt.ylabel('probability P')
            plt.legend()
            plt.show()
        except: 
            plt.scatter(self.bins,self.hist)
            exp=round(self.Linexponent(xmin,xmax)[1],2)
            err=round(self.Linexponent(xmin,xmax)[0],2)
            plt.plot(self.bins[self.bins>=xmin*0.5],manualfit[self.bins>=xmin*0.5],label=r'$\alpha_{Lin}=$'+str(np.abs(exp))+r'$\pm$'+str(err))
         
            plt.xscale('log')
            plt.yscale('log')
            
            plt.xlabel('biggest amplitude in an event')
            plt.ylabel('probability density')
            plt.legend()
            plt.show()
            return
        if LinFit==False and MLFit==True:
            plt.xscale('log')
            plt.yscale('log')
            
            plt.xlabel('biggest amplitude in an event')
            plt.ylabel('probability density')
            
            plt.legend()
            plt.show()
        else:
            return
        
    def save(self,name='',overwrite=False):
        if overwrite == True:
            if name == '':    
                with open(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc'+'\\'+self.foldername+'.pkl', 'wb') as output: 
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            if name != '':
                with open(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc'+'\\'+self.foldername+' '+name+'.pkl', 'wb') as output: 
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            return
        if name == '':    
            if os.path.exists(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc'+'\\'+self.foldername+'.pkl')==False:
                with open(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc'+'\\'+self.foldername+'.pkl', 'wb') as output: 
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            else:
                print('Data has previously been analyzed')
        if name != '':
            if os.path.exists(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc'+'\\'+self.foldername+' '+name+'.pkl')==False:   
                with open(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc'+'\\'+self.foldername+' '+name+'.pkl', 'wb') as output: 
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            else:
                print('Data has previously been analyzed')
            return