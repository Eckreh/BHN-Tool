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
from mpl_axes_aligner import align

import os
import powerlaw

#plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')

params = {'axes.labelsize': 18,'axes.titlesize':16,  'legend.fontsize': 16, 'legend.frameon': True, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
plt.rcParams.update(params)

# plt.rcParams.update(plt.rcParamsDefault)
#%%functions

def gauss(x,sigma,mu):
    return np.exp(-((((x)-mu)**2/np.sqrt(2*np.pi)*sigma)))

def skewed_normal(x,mu,alpha,d,c,b):
    return 1/2*c*(1+erf(alpha*((x-mu)/b)/np.sqrt(2)))*1/np.sqrt(2*np.pi)*np.exp(-((x-mu)/b)**2/2)+d

def Lorentz(x,mu,gamma,a,b,c):
    return a/np.pi*(1/2*gamma/(((x-mu)/c)**2+(1/2*gamma)**2))+b

def lin_fit(x,a,b):
    return a*x+b

def analyze(data,xmin,xmax,datatype='eventsize',binnumber=50):
    
    params = {'axes.labelsize': 18,'axes.titlesize':16,  'legend.fontsize': 16, 'legend.frameon': True, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
    plt.rcParams.update(params)
    
    edges, hist = powerlaw.pdf(data,number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    fit.power_law.plot_pdf(label=r'$\alpha_{ML}$'+r'{}$\pm {}$'.format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    hist=hist[fit.xmin<bin_centers]
    bin_centers=bin_centers[fit.xmin<bin_centers]

    hist=hist[bin_centers<xmax]
    bin_centers=bin_centers[bin_centers<xmax]

    bin_centers=bin_centers[hist!=0]
    hist=hist[hist!=0]

    popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))

    fitlin=np.exp(lin_fit(np.log(bin_centers),*popt))
    plt.plot(bin_centers,fitlin,label=r'$\alpha_{Lin}$'+r'={} $\pm${}'.format(round(float(popt[0]),2),round(float(np.sqrt(pcov[0][0])),2)))
    plt.xscale('log')
    plt.yscale('log')
    if datatype=='eventsize':
        plt.xlabel('biggest amplitude in an event',fontsize=15)
    if datatype=='eventenergy':
        plt.xlabel('integrated eventsize')
    if datatype=='intereventtime':
        plt.xlabel('interevent time [s]')
    plt.ylabel(r'probability density $P(S=S_i)$',fontsize=15)
    plt.legend()
    plt.show()
    
    # fitstart=np.logspace(np.log10(min(data)),np.log10(max(data)),20)
    # alphalist=[]
    # for k in fitstart:
    #     fit=powerlaw.Fit(data,xmin=k)
    #     alphalist.append(fit.alpha)
    # plt.plot(fitstart,alphalist,label='a')
    # plt.xscale('log')
    # plt.legend()
    # plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    return fit,popt,pcov,bin_centers,hist

def compare(datalist):

    plt.figure(figsize=(10,6))
    
    for index in range(0,len(datalist)):
        
        edges, hist = powerlaw.pdf(np.abs(datalist[index]),number_of_bins=50)
    
        bin_centers = (edges[1:]+edges[:-1])/2.0
    
        # fit=powerlaw.Fit(data)
        # fit.power_law.plot_pdf(label="{}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
        
        plt.scatter(bin_centers,hist)
    
    plt.xlabel('eventsize S [A^2/s^2]')
    plt.ylabel('P(S_i=S)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return



#%% set folder to load data

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Daten\evaltest'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\12.05.2023\test 14.6'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\05_05_2023\Measurements for Bark analysis with DWM after some settling time\7.32kHz'
folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\05_05_2023\Measurements for Bark analysis with DWM\29.3kHz'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\05_05_2023\tests with new samples\7.32 medium big'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\05_05_2023\tests with new samples\7.32 medium big'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\14_4 or group meeting (both)\12_04_23 tests for baseline subtraction\better results\50 Ohm'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\12.05.2023\test 14.6'

#dont make stack use this for all the new ones
folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\archive\measurement data shown in thesis\the ramped measurements\measurements that did show events\29.3kHz'


sizelist=[]
maxlist=[]
waitlist=[]    
durationlist=[]
#%%

k=0
resistance=50

for file in os.listdir(folder):
    
    name,ext=os.path.splitext(file)  
    
    if ext=='':
        continue
    
    try:    
        colnamesmfli = ['Time [s]', 'Vout [V]', 'Current [A]']
        data = pd.read_csv(folder+'\\'+file, comment='%', sep=';', names=colnamesmfli).values
            
        t=data[0:,0]
        vout=data[0:,1]
        vin=data[0:,2]
        vin=np.ones(len(t))
        I=vout
        dt=t[1]-t[0]
    except (UnicodeDecodeError):
        print('error loading the data')
        continue
    # if k!=8:
    #     k=k+1
    #     continue
    # else:
    k=k+1
#%% extract local extrema
    # l=int(len(t)/9)
    l=int(len(t)/2)
    ll=0
    while ll < len(vout)-l:
        Itemp=I[ll:ll+l]
        ttemp=t[ll:ll+l]
        if np.abs(max(Itemp)-np.mean(Itemp)) > np.abs(min(Itemp)-np.mean(Itemp)):
            if np.abs(max(Itemp)-np.mean(Itemp))<7e-6:
                ll=ll+l
                continue
            idx=np.where(Itemp==max(Itemp))
            idxloc=idx[0][0]
        elif np.abs(max(Itemp)-np.mean(Itemp)) < np.abs(min(Itemp)-np.mean(Itemp)):
            if np.abs(min(Itemp)-np.mean(Itemp))<7e-6:
                ll=ll+l
                continue
            idx=np.where(Itemp==min(Itemp))
            idxloc=idx[0][0]
        idxext=ll+idxloc     

# %% make the polynomial fit and choose a cut that mainly takes the switching peak for better fit frequency correlates to width
        while True:
            if 1/dt < 4000:        
                length=50  #3.66kHz
                shift=0
                degval=60
                break
            elif 1/dt <8000:
                length=100#7.32kHz
                shift=0
                degval=80
                upperlimit=1.5e-9
                break
            elif 1/dt < 15000 :
                length=150   #14.64kHz
                shift=0
                degval=100
                break
            elif 1/dt < 30000 :
                length=500   #29.
                shift=0
                degval=50
                upperlimit=1e-7
                break
            
            elif 1/dt < 1000000 :
                length=1500   #29.3kHz
                degval=8
                upperlimit=1e-9
                break
        if idxext<length:
            tcut,Icut=t[0:idxext+length+shift],I[0:idxext+length+shift]
        else:
            tcut,Icut=t[idxext-length+shift:idxext+length+shift],I[idxext-length+shift:idxext+length+shift]
        
        
        fit=np.polyfit(tcut,Icut,deg=degval)#::-1],deg=degval)
        fit=np.poly1d(fit)
#%% plot hte current response with the fit        
        fig, ax1 = plt.subplots(figsize=(8,6))
        
        ax1.plot(tcut,Icut*1000,label='current response')
        ax1.plot(tcut,fit(tcut)*1000,label='fit')#k-1)
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('current [mA]')
        ax1.legend()
        plt.show()
        
        ll=l+ll   #go to the next switching peak
# %%    calculate the derivatives and subtract the fit
        dI=np.array([])
        dt=t[1]-t[0]
        dFit=np.array([])
                 
        for i in range(0,len(Icut[0:])-1):
            dI=np.append(dI,(Icut[i+1]-Icut[i]))
            dFit=np.append(dFit,(fit(tcut)[i+1]-fit(tcut)[i]))
            # dFit=np.append(dFit,(skewed_normal(tcut[i+1],*popt)-skewed_normal(tcut[i],*popt)))
            
        Jerkspectrum=(dI/dt)
        Jerkspectrum= np.append(Jerkspectrum,0) 
        Jerksandbase=Jerkspectrum
        
        Fitspectrum=(dFit/dt)
        Fitspectrum=np.append(Fitspectrum,0)
        Fitbase=Fitspectrum
        
        jerksminusfit=Jerksandbase-Fitbase
        jerksminusfit=jerksminusfit**2
        
        minlist=[]
        timelist=[]
        
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
                 if jerksminusfit[l]<jerksminusfit[l-1] and jerksminusfit[l]<jerksminusfit[l+1]: # and jerksminusfit[l]<0:
                     minlist.append(jerksminusfit[l])
                     timelist.append(tcut[l])
#%% plot the derivative of both fit and measurement data
        baseline=Pchip(timelist, minlist, tcut)
        jerks=jerksminusfit-baseline
        
        plt.xlabel('time [ms]')        
        plt.ylabel(r'slew rate $[\frac{A^2}{s^2}]$')
        plt.plot(tcut,Jerksandbase)
        plt.plot(tcut,Fitbase)
        plt.show()        
#%% remove data blelow the threshold
        filtered=[]
        for j in range(0,len(jerks)):
            if jerks[j]<upperlimit:
            # if Jerks[j]<2e-14:
                filtered.append(0)
            else:
                filtered.append(jerks[j])
        plt.plot(tcut,jerks,label='Jerks')
        # plt.plot(tcut,jerksminusfit)
        # plt.plot(tcut,Fitbase)
        plt.plot(tcut,np.ones(len(tcut))*upperlimit,label='threshold') 
        plt.xlabel('time[s]')
        plt.ylabel(r'Slew rate squared $\left[\frac{A^2}{s^2}\right]$')
        plt.legend()
        plt.ylim(0,0.00001)
        plt.show()
#%% plot current response and corresponding slew rate
        fig, ax1 = plt.subplots(figsize=(9,6))
        ax2=ax1.twinx()
        
        ax1.plot(tcut,Icut*1000,label='current response')
        # ax1.plot(tcut,fit(tcut)*1000)
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('current [mA]')
        
        ax2.plot(tcut,filtered,label='slew rate',color='r')
        # ax2.axhline(upperlimit,min(t),max(t),color='orange')
        ax2.set_xlabel('time [s]')
        ax2.set_ylabel(r'slew rate $[\frac{A^2}{s^2}]$',color='red')
        ax2.grid(visible=False)
        
        plot1,label1=ax1.get_legend_handles_labels()
        plot2,label2=ax2.get_legend_handles_labels()
        
        ax1.legend(plot1 + plot2,label1+label2,loc='upper right')
        
        plt.tight_layout()
        plt.show()
#%% 
        indexlist=[]
        wait = 0
        duration = 0
        for j in range(0,len(filtered)):
            if filtered[j]==0:
                wait=wait+1
                if len(indexlist)!=0:
                    indexlist.append(j)
                    energy=scipy.integrate.cumulative_trapezoid(np.take(filtered,indexlist),np.take(t,indexlist))
                    sizelist.append(energy[len(energy)-1])
                    maxlist.append(max(np.take(filtered,indexlist)))
                    durationlist.append((duration+2)*dt)
                    waitlist.append(wait*dt)
                    indexlist=[]
                    wait=0
                    duration=0
                continue
            else:
                if j!=0:
                    indexlist.append(j-1)
                    duration=duration+1
                indexlist.append(j)

#%% plot distributions of the different event properties
plt.plot(sizelist)
plt.xlabel('Number')
plt.ylabel('integrated Eventsize')
plt.show()
#%%
plt.plot(maxlist,marker='.',linewidth=0)
plt.xlabel('Number')
plt.ylabel('biggest Amplitude in an Event')
plt.yscale('log')
plt.show()
#%%
# plt.plot(durationlist)
# plt.show()
plt.plot(waitlist)
plt.ylabel('Interevent time')
plt.show()
#%%

#%%
# analyse(sizelist,1e-12,3e-11,datatype='eventenergy')

# analyse(maxlist,1e-8,3e-8,datatype='eventsize')

# analyse(waitlist,min(waitlist),max(waitlist),datatype='intereventtime')

analyze(sizelist,1e-11,5e-8,datatype='eventenergy',binnumber=20)
#%%
analyze(maxlist,3e-8,3e-6,datatype='eventsize',binnumber=50)
#%%
bins,edges,stuff=plt.hist(maxlist,bins=np.logspace(-7,-5.8,20),density=True)
bin_centers=(edges[1:]-edges[:-1])/2
plt.show()
plt.scatter(bin_centers,bins)
plt.xlabel('biggest amplitude in an event')
plt.ylabel('probability density P($S=S_i$)')
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
stuff=plt.hist(maxlist,bins=np.logspace(-7,-5.8,30),density=True)
plt.show()
edges=stuff[1]
hist=stuff[0]
bin_centers = (edges[1:]+edges[:-1])/2.0
plt.scatter(bin_centers,hist)
plt.xlabel('biggest amplitude in an event')
plt.ylabel('probability density P($S=S_i$)')
plt.xscale('log')
plt.yscale('log')
plt.show()

#%%
analyze(waitlist,5e-5,1e-3,datatype='intereventtime')

# analyze(durationlist,1e-4,2e-3,datatype='intereventtime')
# analyse(waitlist,min(waitlist),max(waitlist),datatype='intereventtime')
#%%
# compare([maxlist,savelst])

#%%
fit=powerlaw.Fit(maxlist)
print(fit.alpha,fit.sigma)
print(fit.xmin)
# %% produce some fancy shit for the ML results
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
# %%
explist=[]
sigmalist=[]
minlist=np.logspace(np.log10(min(maxlist)),np.log10(max(maxlist)/6),50)
for xmin in minlist:
    fit=powerlaw.Fit(maxlist,xmin=xmin)
    explist.append(fit.alpha)
    sigmalist.append((fit.sigma))
fit=powerlaw.Fit(maxlist)
plt.errorbar(minlist,explist,sigmalist,linewidth=0.2,elinewidth=0.4,barsabove=True)
plt.scatter(minlist,explist,color='blue',s=0.2)
# maxlist=np.log(maxlist)
plt.hlines(fit.alpha,min(maxlist),max(maxlist)/6,linestyle='-',color='red',label=r'$\alpha =$ {}'.format(round(fit.alpha,2)))
# plt.hlines(fit.alpha+fit.sigma,min(maxlist),max(maxlist)/6,color='green',label='standard deviation')
# plt.hlines(fit.alpha-fit.sigma,min(maxlist),max(maxlist)/6,color='green')
plt.xscale('log')
plt.xlabel('start of powerlaw behaviour')
plt.ylabel('estimated Exponent')
# plt.xlim(1e-8,1e-4)
# plt.ylim(1.4,2.2)
plt.legend()
plt.show()