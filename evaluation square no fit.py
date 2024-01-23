# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:34:56 2023

@author: gh457
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import PPoly
from scipy.interpolate import pchip_interpolate as Pchip
from scipy.special import erf
from scipy.signal import find_peaks
import os
import powerlaw
from barkhausen_class import Barkhausen_data, load_bark, plot_datasets
import sys
from mpl_axes_aligner import align

plt.style.use('seaborn-v0_8-whitegrid')


params = {'axes.labelsize': 18,'axes.titlesize':26,  'legend.fontsize': 16, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(params)
#%%functions

def gauss(x, sigma, mu):
    return np.exp(-((((x)-mu)**2/np.sqrt(2*np.pi)*sigma)))

def skewed_normal(x, mu, alpha, d, c, b):
    return 1/2*c*(1+erf(alpha*((x-mu)/b)/np.sqrt(2)))*1/np.sqrt(2*np.pi)*np.exp(-((x-mu)/b)**2/2)+d

def Lorentz(x, mu, gamma, a, b, c):
    return a/np.pi*(1/2*gamma/(((x-mu)/c)**2+(1/2*gamma)**2))+b

def lin_fit(x, a, b):
    return a*x + b

def analyze(data, xmin, xmax, binnumber, datatype='eventsize'):
    params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
    plt.rcParams.update(params)
    
    plt.figure(figsize=(10,6))
    edges, hist = powerlaw.pdf(data,number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    
    fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    hist=hist[xmin<bin_centers]
    bin_centers=bin_centers[xmin<bin_centers]

    hist=hist[bin_centers<xmax]
    bin_centers=bin_centers[bin_centers<xmax]

    bin_centers=bin_centers[hist!=0]
    hist=hist[hist!=0]

    popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))

    fitlin=np.exp(lin_fit(np.log(bin_centers),*popt))
    plt.plot(bin_centers,fitlin,label=r'$\alpha_{Lin}$'+r'={} $\pm${}'.format(np.abs(round(float(popt[0]),2)),round(float(np.sqrt(pcov[0][0])),2)))
    plt.xscale('log')
    plt.yscale('log')
    if datatype=='eventsize':
        plt.xlabel(r'Slew-Rate $S=\left[\frac{A^2}{s^2}\right]$',fontsize=18)
    if datatype=='eventenergy':
        plt.xlabel('integrated eventsize')
    if datatype=='intereventtime':
        plt.xlabel('interevent time [s]')
    plt.ylabel(r'probability P($S=S_i$)',fontsize=18)
    plt.legend(fontsize=18)
    # plt.grid(True)
    plt.show()
    startvalue=xmin

    return fit, popt, pcov, bin_centers, hist

def analyze_setxmin(data, xmin, xmax, binnumber, datatype='eventsize'):
    
    plt.rc('font', size=16)
    plt.figure(figsize=(10, 6))
    plt.grid(which='major', axis='both')
    
    edges, hist = powerlaw.pdf(data, number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+"={}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    hist=hist[xmin<bin_centers]
    bin_centers=bin_centers[xmin<bin_centers]

    hist=hist[bin_centers<xmax]
    bin_centers=bin_centers[bin_centers<xmax]

    bin_centers=bin_centers[hist!=0]
    hist=hist[hist!=0]

    popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))

    fitlin=np.exp(lin_fit(np.log(bin_centers),*popt))
    plt.plot(bin_centers,fitlin,label=r'$\alpha_{Lin}$'+r'={} $\pm${}'.format(np.abs(round(float(popt[0]),2)),round(float(np.sqrt(pcov[0][0])),2)))
    plt.xscale('log')
    plt.yscale('log')
    if datatype=='eventsize':
        plt.xlabel('biggest amplitude in an event')
    if datatype=='eventenergy':
        plt.xlabel('integrated eventsize')
    if datatype=='intereventtime':
        plt.xlabel('interevent time [s]')
    plt.ylabel(r'probability P($S=S_i$)',fontsize=18)
    plt.xlabel(r'Slew-Rate $S=\left[\frac{A^2}{s^2}\right]$',fontsize=18)
    plt.legend(fontsize=18)
    plt.legend()
    plt.show()
    return fit,popt,pcov,bin_centers,hist

def compare(datalist,labellist=['labels']):
    
    plt.figure(figsize=(10,6))
    params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
    plt.rcParams.update(params)
    
    names=[r'$S_{min} = 3\cdot 10^{-3} \frac{A^2}{s^2}$',r'$S_{min} = 3\cdot 10^{-2} \frac{A^2}{s^2}$',r'$S_{min} = 3\cdot 10^{-1} \frac{A^2}{s^2}$']

    
    for index in range(0,len(datalist)):
        
        edges, hist = powerlaw.pdf(np.abs(datalist[index]),number_of_bins=50)
    
        bin_centers = (edges[1:]+edges[:-1])/2.0
    
        # fit=powerlaw.Fit(data)
        # fit.power_law.plot_pdf(label="{}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
        plt.xscale('log')
        plt.yscale('log')
        try:
            plt.scatter(bin_centers,hist,label=labellist[index])
        except:
            plt.scatter(bin_centers,hist)
    plt.xlabel(r'Slew rate S $\left[\frac{A^2}{s^2}\right]$')
    plt.ylabel(r'P($S=S_i$)')
    # plt.xlim(3e-2,3e3)
    # plt.ylim((1e-7,1e2))
    # plt.xlabel(r'interevent time $\delta t$')
    # plt.ylabel(r'P($\delta t = \delta t_i $)')
    plt.legend()
    # plt.grid(True)
    plt.show()
    return
#%% load data create lists for the resulting barkhausen pulses

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\29_09_23 samples julius 1\action at the start'


sizelist=[] 
maxlist=[]
waitlist=[]    
durationlist=[]

varlist=[[],[]]

lastidx=0
#%% settings for the threshold, input resistance and what is supposed to be plotted
upperlimit=3e-3
k=0
var=0
resistance=50
plot=True
plotoriginal=True
ploteventdists=False

split=False
#%% Auswertung
for file in os.listdir(folder):
    filename,ext=os.path.splitext(file)  
    if ext=='' or ext=='pkl':
        print('image or classdata')
        continue
    
    
    try:    
        colnamesmfli = ['Time [s]', 'Vout [V]']
        data = pd.read_csv(folder+'\\'+file, comment='%', sep='; ', names=colnamesmfli).values
            
        try:
            t=data[0:,0]
            vout=data[0:,1]
            vin=np.ones(len(t))
            I=vout/resistance
            I=I-I[0]
        except TypeError:
            t=data[0:,0]
            vout=data[0:,1]
            for i in range(0,len(vout)):
                vout[i]=vout[i][0:-1]
                vout[i]=float(vout[i])
            vin=np.ones(len(t))
            I=vout/resistance
            I=I-I[0]
        dt=t[1]-t[0]    
        lastidx=0
    except (UnicodeDecodeError):
        print('Unicodeerror')
        continue
    
        
    try:    
        colnamesmfli = ['Time [s]', 'Vout [V]', 'Current [A]']
        data = pd.read_csv(folder+'\\'+file, comment='%', sep=';', names=colnamesmfli).values
            
        t=data[0:,0]
        vout=data[0:,1]
        vin=data[0:,2]
        vin=np.ones(len(t))
        I=vout/resistance
        dt=t[1]-t[0]

    
    except (UnicodeDecodeError):
        continue
#%% slice up the different parts of the measurement
    # if k!=1:
    #     k=k+1
    #     continue
    # else:
    k=k+1
#%%
    if plotoriginal==True:    
        plt.plot(t,I*1000,label=k-1)
        # plt.plot(tcut,Icut*1000)
        # plt.plot(t,fit(tcut)*1000,label='derivative from fit')
        plt.xlabel('time [s]')
        plt.ylabel('current [mA]')
        plt.legend()
        plt.show()
        #%%
    #this is to choose the right width over which is analyzed, typically just where the switching peak is located and dependent on samplingrate
    #in case of fitting this also set the degree of the polynomial
    while True:
        if 1/dt < 4000:        
            length=lengthright=lengthleft=50  #3.66kHz
            break
        elif 1/dt <8000:
            length=lengthright=lengthleft=4000#7.32kHz
            break
        elif 1/dt < 15000 :
            length=lengthright=lengthleft=7000   #14.64kHz
            break
        elif 1/dt < 30000 :
            length=lengthright=1500  #29.3kHz
            lengthleft=1500
            break
        elif 1/dt < 60000 :
            length=lengthright=6000  #29.3kHz
            lengthleft=6000
            break
        elif 1/dt < 120000 :
            length=lengthright=1500  #29.3kHz
            lengthleft=1500
            break
        elif 1/dt < 240000 :
            length=lengthright=3000  #29.3kHz
            lengthleft=3000
            break
        else:
            lengthleft=1400
            addition=0
            lengthright=1600#938kHz
            break
        
    peakidx=find_peaks(np.abs(I),height=max(np.abs(I)/5),distance=(lengthright+lengthleft))[0]
    if plotoriginal==True:
        plt.plot(t,np.abs(I))
        plt.plot([min(t),max(t)],np.ones(2)*[max(np.abs(I)/5)])
        plt.scatter(t[peakidx],np.abs(I[peakidx]),color='r')
        plt.show()
    
    var=0   #variable to select certain maybe interesting cases put in a legend to find said cases and make settings below to fit     
    for  idxext in peakidx:  
        var = var+1
        # if var!=0:
        #     var=var+1
        #     continue
        # else:
        #     var=var+1 
            
#%% make the polynomial fit
        shift=0
            
        idxext=idxext+shift
        if idxext<lengthleft:
            tcut,Icut=t[0:idxext+lengthright],I[0:idxext+lengthright]
        elif len(t)-idxext<lengthright:
            continue
        
        
        else:
            tcut,Icut=t[idxext-lengthleft:idxext+lengthright],I[idxext-lengthleft:idxext+lengthright]
            # tcut = list(tcut)
            # Icut = list(Icut)
        
        check=0
        while check==0:

#%% slice up the different parts of the measurement
        
            if plotoriginal==True:    
                # plt.plot(t,I*1000)
                plt.plot(tcut,Icut*1000,label=[var-1,k-1])
                # plt.plot(t,fit(tcut)*1000,label='derivative from fit')
                plt.xlabel('time [s]')
                plt.ylabel('current [mA]')
                plt.legend()
                plt.show()
            
            if split==True and var==1:
                Icutsave=Icut
                tcutsave=tcut
                tcut,Icut=tcutsave[0:2*lengthleft+addition],Icutsave[0:2*lengthleft+addition]
            if split==True and var==2:
                tcut,Icut=tcutsave[2*lengthleft+addition:],Icutsave[2*lengthleft+addition:]
                check=1
            
            if split==False:
                check=2
                            
                    
                
    # %% analysis: calculate derivative for each point and with polynomial fit calculates fit derivative 
    #              to subtract as welll as a baseline from minima to prevent negative values for the squared slew rate difference 
            
            dI=np.array([])
            dt=t[1]-t[0]
            dFit=np.array([])       
            for i in range(0,len(Icut[0:])-1):
                dI=np.append(dI,(Icut[i+1]-Icut[i]))
                # dFit=np.append(dFit,(fit(tcut)[i+1]-fit(tcut)[i]))
                # dFit=np.append(dFit,(skewed_normal(tcut[i+1],*popt)-skewed_normal(tcut[i],*popt)))
                
            Jerkspectrum=(dI/dt)
            Jerkspectrum= np.append(Jerkspectrum,0) 
            # Jerksandbase=np.abs(Jerkspectrum)
            Jerksandbase=Jerkspectrum**2
            
            # Fitspectrum=(dFit/dt)
            # Fitspectrum=np.append(Fitspectrum,0)
            # Fitbase=Fitspectrum**2
            # Fitbase=np.abs(Fitspectrum)
            
            jerksminusfit=Jerksandbase#-Fitbase
            
            minlist=[]
            timelist=[]
            
            #remove baseline made from local minima can be set to either only take negative minima for baseline or everything 
            for l in range(0,len(jerksminusfit)):
                 if l==0:
                     if jerksminusfit[l]<jerksminusfit[l+1]:# and jerksminusfit[l]<0:
                         minlist.append(jerksminusfit[l])
                         timelist.append(tcut[l])
             
                 if l==len(jerksminusfit)-1:
                     if jerksminusfit[l]<jerksminusfit[l-1]:# and jerksminusfit[l]<0:
                         minlist.append(jerksminusfit[l])
                         timelist.append(tcut[l])
             
                 elif l!=0 and l!=len(jerksminusfit)-1:
                     if jerksminusfit[l]<jerksminusfit[l-1] and jerksminusfit[l]<jerksminusfit[l+1]:# and jerksminusfit[l]<0:
                         minlist.append(jerksminusfit[l])
                         timelist.append(tcut[l])
    
            # try:
            baseline=Pchip(timelist, minlist, tcut)
            # except:
            #     baseline=np.zeros(len(tcut))
            jerks=jerksminusfit-baseline
            
    
    #%%     filter out the data that is below a certain limit that was initially set
            filtered=[]
            # upperlimit=1
            for j in range(0,len(jerks)):
                if jerks[j]<upperlimit:
                # if Jerks[j]<2e-14:
                    filtered.append(0)
                else:
                    filtered.append(jerks[j])
    # %%     
            # plot the data to see slew rates over time compared to the current response
            if plot==True:
                fig, ax1 = plt.subplots()
                ax1.plot(tcut,Icut,label=var-1)
                ax1.set_ylabel('current [A]',color='blue')
                ax1.set_xlabel('time[s]')
                ax2=ax1.twinx()
                ax2.plot(tcut,filtered,c='r')
                ax2.set_ylabel(r'slew rate $[\frac{A^2}{s^2}]$',c='r')
                ax2.grid(False)
                plt.legend()
                plt.show()
                #%% show where in the measurement is being analyzed
                plt.plot(t,I,c='r',label='')
                plt.plot(tcut,Icut,c='g',label='')
                plt.xlabel('time [t]')
                plt.ylabel('current [A]')
                plt.legend()
                plt.show()
    # %%    identify the jerks where they breach the threshold and calculate the corresponding values that are to be analyzed
            #the indizes within the cutlist are saved
            indexlist=[]
            wait = 0
            duration = 0
            for j in range(0,len(filtered)):
                if filtered[j]==0:
                    wait=wait+1
                    if len(indexlist)!=0:
                        indexlist.append(j)
                        #integrate the slew rate over the jerk and put it into list
                        energy=scipy.integrate.cumulative_trapezoid(np.take(filtered,indexlist),np.take(t,indexlist))
                        sizelist.append(energy[len(energy)-1])
                        
                        #take the amplitude as maximum and put it into the list
                        maxlist.append(max(np.take(filtered,indexlist)))
                        
                        #in the case that different parts of the peak are to be analyzed different lists for parts are also filled
                        if split==True and var==1:
                            varlist[0].append(max(np.take(filtered,indexlist)))
                        
                        if split==True and var==2:
                            varlist[1].append(max(np.take(filtered,indexlist)))
                        
                        #calculates duration of the jerks
                        durationlist.append((duration+2)*dt)
                        
                        #calculates interevent time
                        waitlist.append(wait*dt)
                        
                        #reset for next jerk
                        indexlist=[]
                        wait=0
                        duration=0
                    continue
                else:
                    if j!=0:
                        indexlist.append(j-1)
                        duration=duration+1
                    indexlist.append(j)
#%% plot the different data for jerks, often the best as scatter with logarithmic y axis for better visibility
if ploteventdists==True:
    plt.scatter(np.linspace(0,len(sizelist),len(sizelist)),sizelist)
    plt.ylabel('integrated Eventsize')
    plt.yscale('log')
    plt.show()
    plt.scatter(np.linspace(0,len(maxlist),len(maxlist)),maxlist)
    plt.ylabel('biggest Amplitude in an Event')
    plt.yscale('log')
    plt.show()
    plt.scatter(np.linspace(0,len(waitlist),len(waitlist)),waitlist)
    plt.ylabel('Interevent time')
    plt.show()
# %% 
#%% make class object
avg=7e-5 #not used anymore but here for functionality of the class saving mechanism
barkdatameas=Barkhausen_data(maxlist, sizelist, upperlimit, avg, folder)
#%%
barkdatameas.bins_hist()
# barkdatameas.Linexponent(1e-2,1e2)
#%%
barkdatameas.plot_data(MLFit=True,MLmin=True,MLxmin=1e-12,LinFit=True,xmin=1e-2,xmax=1e0)
# barkdatameas.Linexponent(1e-2,1e1)
# barkdatameas.plot_data(MLFit=True,LinFit=True,xmin=1e-2,xmax=1e1)
# barkdatameas.plot_data(LinFit=True,xmin=1e-10,xmax=1e-6)
# %% used to load data saved from the class object and compare different datasets
# barkdata1=load_bark(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Thesis data\barkhausen part\30V big deviation\backup\30V big deviation.pkl')
# barkdata2=load_bark(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc\30V big deviation.pkl')
# barkdata3=load_bark(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc\30V big deviation shift=500.pkl')
# barkdata4=load_bark(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc\30V shift=600.pkl')
# barkdata5=load_bark(r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\folder to save data for exponents settings etc\30V shift=2000.pkl')
# plot_datasets([barkdatameas,barkdata1,barkdata2,barkdata3,barkdata4])

#%% save the data
# if shift==0:
#     barkdatameas.save(overwrite=False)
# if shift!=0:
#     barkdatameas.save(name='shift='+str(shift))

# %% analysis without using the class
analyze(barkdatameas.sizes,xmin=1e-2,xmax=1e1,binnumber=50)
#%% Maximum likelihood method exponents for the different minimum settings maximum boundary can be set as xmax but takes very long
fit=powerlaw.Fit(maxlist)
plt.plot(fit.xmins,fit.alphas,marker='.',linewidth=0)
plt.xscale('log')
plt.show()
# %% produce the Maximum Likelihood result for different minima settings to see if a good plateau will be there
#simpler variant that is possible  after a fit without fixed minimum and fit.alphas with all points from the list
explist=[]
sigmalist=[]
minlist=np.logspace(np.log10(min(maxlist)),np.log10(max(maxlist)/6),100)
for xmin in minlist:
    fit=powerlaw.Fit(maxlist,xmin=xmin)
    explist.append(fit.alpha)
    sigmalist.append(fit.sigma)
fit=powerlaw.Fit(sizelist)
plt.errorbar(minlist,explist,sigmalist,elinewidth=0.3,barsabove=True,color='b')
plt.hlines(fit.alpha,min(maxlist),max(maxlist)/6,color='red',label='exponent')
plt.hlines(fit.alpha+fit.sigma,min(maxlist),max(maxlist)/6,color='green',label='standard deviation')
plt.hlines(fit.alpha-fit.sigma,min(maxlist),max(maxlist)/6,color='green')
plt.plot(minlist,explist)
plt.xscale('log')
plt.xlabel(r'point at which the powre law behaviour starts $x_{min}$')
plt.ylabel(r'Power law exponent estimate $\hat{\alpha}$')
plt.legend()
plt.show()
