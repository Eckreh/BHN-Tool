# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:23:03 2023

@author: gh457
"""

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
from barkhausen_class import Barkhausen_data , load_bark , plot_datasets
import sys

params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(params)
#%%functions

# gauss(x,sigma,mu):
#     return np.exp(-((((x)-mu)**2/np.sqrt(2*np.pi)*sigma)))

# def skewed_normal(x,mu,alpha,d,c,b):
#     return 1/2*c*(1+erf(alpha*((x-mu)/b)/np.sqrt(2)))*1/np.sqrt(2*np.pi)*np.exp(-((x-mu)/b)**2/2)+d

# def Lorentz(x,mu,gamma,a,b,c):
#     return a/np.pi*(1/2*gamma/(((x-mu)/c)**2+(1/2*gamma)**2))+b

def compare(datalist,labellist=['labels']):
    plt.style.use('seaborn-whitegrid')
    
    
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
    # plt.xlabel(r'interevent time $\delta t$')
    # plt.ylabel(r'P($\delta t = \delta t_i $)')
    plt.legend()
    plt.show()
    return

def lin_fit(x,a,b):
    return a*x+b

def analyze(data,xmin,xmax,binnumber,datatype='eventsize'):
    params = {'axes.labelsize': 22,'axes.titlesize':26,  'legend.fontsize': 20, 'legend.frameon': True, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
    plt.rcParams.update(params)
    
    plt.figure(figsize=(10,6))
    edges, hist = powerlaw.pdf(data,number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    
    fit.power_law.plot_pdf(label=r'$\epsilon_{ML}$'+"={}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    hist=hist[xmin<bin_centers]
    bin_centers=bin_centers[xmin<bin_centers]

    hist=hist[bin_centers<xmax]
    bin_centers=bin_centers[bin_centers<xmax]

    bin_centers=bin_centers[hist!=0]
    hist=hist[hist!=0]

    popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))

    fitlin=np.exp(lin_fit(np.log(bin_centers),*popt))
    plt.plot(bin_centers,fitlin,label=r'$\epsilon_{Lin}$'+r'={} $\pm${}'.format(np.abs(round(float(popt[0]),2)),round(float(np.sqrt(pcov[0][0])),2)))
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
    plt.show()
    startvalue=xmin

    return fit,popt,pcov,bin_centers,hist

def analyze_setxmin(data,xmin,xmax,binnumber,datatype='eventsize'):
    
    plt.rc('font',size=16)
    plt.figure(figsize=(10,6))
    plt.grid(which='major',axis='both')
    
    edges, hist = powerlaw.pdf(data,number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    fit.power_law.plot_pdf(label="{}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    hist=hist[xmin<bin_centers]
    bin_centers=bin_centers[xmin<bin_centers]

    hist=hist[bin_centers<xmax]
    bin_centers=bin_centers[bin_centers<xmax]

    bin_centers=bin_centers[hist!=0]
    hist=hist[hist!=0]

    popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))

    fitlin=np.exp(lin_fit(np.log(bin_centers),*popt))
    plt.plot(bin_centers,fitlin,label='fit $bx^{}$, a={} $\pm${}'.format('-a',round(float(popt[0]),2),round(float(np.sqrt(pcov[0][0])),2)))
    plt.xscale('log')
    plt.yscale('log')
    if datatype=='eventsize':
        plt.xlabel('biggest amplitude in an event')
    if datatype=='eventenergy':
        plt.xlabel('integrated eventsize')
    if datatype=='intereventtime':
        plt.xlabel('interevent time [s]')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    return fit,popt,pcov,bin_centers,hist

#%%

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\11_10_23 hopefully better systematic stuff\44 7.5Hz 29V 938kHz'


sizelist=[] 
maxlist=[]
waitlist=[]    
durationlist=[]
lastidx=0

varlist=[]
varindex=0
dumbindex=0

prepeaklist=[]

peaklist=[]

shortafterpeaklist=[]

afterpeaklist=[]

fullpeaklist=[]

allfiltered=np.array([])

#%% modify plot style
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

    'legend.frameon' : True,
    "axes.grid" : True,
    #"grid.color": "grey"
}
plt.rcParams.update(tex_fonts)

# %%
avg=3e-9
upperlimit=1e-3
k=0
var=0
resistance=50
plot=False
plotoriginal=False
ploteventdists=False
plot_split=True



#%%
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
        except TypeError:
            t=data[0:,0]
            vout=data[0:,1]
            for i in range(0,len(vout)):
                vout[i]=vout[i][0:-1]
                vout[i]=float(vout[i])
            vin=np.ones(len(t))
            I=vout/resistance
        dt=t[1]-t[0]    
        lastidx=0
    except (UnicodeDecodeError):
        print('Unicodeerror')
        continue
    
    # if k!=3:
    #     dumbindex=dumbindex+1
    #     k=k+1
    #     continue
    # else:
    k=k+1
    dumbindex=dumbindex+1 #comment if you want to look at single stuff
        
    while True:
        if 1/dt < 4000:        
            length=lengthright=lengthleft=50  #3.66kHz
            degval=200
            break
        elif 1/dt <8000:
            length=lengthright=lengthleft=200#7.32kHz
            degval=200
            break
        elif 1/dt < 15000 :
            length=lengthright=lengthleft=150   #14.64kHz
            degval=100
            break
        elif 1/dt < 30000 :
            length=lengthright=lengthleft=2000   #29.3kHz
            degval=100
            break
        else:
            lengthleft=500
            lengthright=1000#938kHz
            degval=10
            break

    
    #for 7.5
    position=-0.62/1000
    shift=1500
    
    #for 15
    # position=-0.3/1000
    # shift=750    


    for thresh in range(0,len(t)):
        if t[thresh] < position:
            continue
        else:
            shiftindex=thresh
            break

        #%%
    var=0
    if dumbindex==2:
        varindex=1
    
    while True:
                
        var=var+1
        
        if plot_split==True:
            if var==2:
                shift=1500
            if shiftindex+shift<len(t):
                plt.plot(t[shiftindex:shiftindex+shift],I[shiftindex:shiftindex+shift])
            else:
                newshift=len(t)-shiftindex
                plt.plot(t[shiftindex:shiftindex+newshift],I[shiftindex:shiftindex+newshift])

        
        tcut,Icut=t[shiftindex+(var-1)*shift:shiftindex+var*shift],I[shiftindex+(var-1)*shift:shiftindex+var*shift]
        
        if var==2:
            shift=1500
        
        
        if shiftindex+shift<len(t):
            tcut,Icut=t[shiftindex:shiftindex+shift],I[shiftindex:shiftindex+shift]
            shiftindex=shiftindex+shift
            if varindex==0:
                varlist.append([])
            
        elif shiftindex!=len(t):
            shift=len(t)-shiftindex
            tcut,Icut=t[shiftindex:shiftindex+shift],I[shiftindex:shiftindex+shift]
            shiftindex=shiftindex+shift
            if varindex==0:    
                varlist.append([])
        
        
        else:
            break
#%%
        
        if plotoriginal==True:    
            plt.plot(t,I*1000)
            plt.plot(tcut,Icut*1000,label=[var-1,k-1],c='r')
            # plt.plot(t,fit(tcut)*1000,label='derivative from fit')
            plt.xlabel('time [s]')
            plt.ylabel('current [mA]')
            plt.legend()
            plt.show()
        
# %%
        dI=np.array([])
        dFit=np.array([])       
        for i in range(0,len(Icut[0:])-1):
            dI=np.append(dI,(Icut[i+1]-Icut[i]))
            
        Jerkspectrum=(dI/dt)
        Jerkspectrum= np.append(Jerkspectrum,0) 
        Jerksandbase=Jerkspectrum**2
        
        Fitspectrum=(dFit/dt)
        Fitspectrum=np.append(Fitspectrum,0)
        Fitbase=Fitspectrum**2
        
        jerksminusfit=Jerksandbase-Fitbase
        
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
                 if jerksminusfit[l]<jerksminusfit[l-1] and jerksminusfit[l]<jerksminusfit[l+1]:# and jerksminusfit[l]<0:
                     minlist.append(jerksminusfit[l])
                     timelist.append(tcut[l])
    
        # baseline=Pchip(timelist, minlist, tcut)  
        jerks=jerksminusfit
    
# %%     
        # jerks = np.abs(jerksminusfit)
        if plot==True:
            fig, ax1 = plt.subplots()
            ax1.plot(tcut,Icut,label=[var-1,k-1])
            ax1.set_ylabel('current [A]')
            
            ax2=ax1.twinx()
            ax2.plot(tcut,jerks,color='r')
            plt.legend()
            plt.show()
        
        filtered=[]
            # upperlimit=1
        for j in range(0,len(jerks)):
            if jerks[j]<upperlimit:
            # if Jerks[j]<2e-14:
                filtered.append(0)
            else:
                filtered.append(jerks[j])
        
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
                    varlist[var-1].append(max(np.take(filtered,indexlist)))
                    if var==1 or var==2:
                        fullpeaklist.append(max(np.take(filtered,indexlist)))
                    
                    else:
                        afterpeaklist.append(max(np.take(filtered,indexlist)))
                    
                    if var==1:
                        prepeaklist.append(max(np.take(filtered,indexlist)))
                    if var==2:
                        peaklist.append(max(np.take(filtered,indexlist)))
                    
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
        allfiltered=np.append(allfiltered,filtered)
#%%
        if plot==True:
            
            fig, ax1 = plt.subplots()
            ax1.plot(tcut,Icut,label=[var-1,k-1])
            ax1.set_ylabel('current [A]')
            
            ax2=ax1.twinx()
            ax2.axhline(upperlimit,min(tcut),max(tcut),color='g')
            ax2.plot(tcut,filtered,color='r')
            ax1.legend()
            plt.show()
#%%
    if plot==True:
        fig, ax1 = plt.subplots()
        ax1.plot(t[thresh:thresh+8*shift],I[thresh:thresh+8*shift])
        ax2=ax1.twinx()
        ax2.plot(t[thresh:thresh+8*shift],allfiltered,color='r')
        ax1.set_xlabel('t in [s]')
        ax1.set_ylabel('I in [A]',color='b')
        ax2.set_ylabel(r'slew rate squared in $[\frac{A^2}{s^2}]$',color='r')
        plt.show()
    plt.xlabel('t[s]',fontsize=18)
    plt.ylabel('I [A]',fontsize=18,)
    plt.tight_layout()
    plt.show()
#%%
if ploteventdists==True:
    plt.plot(sizelist,marker='.',linewidth=0)
    plt.ylabel('integrated Eventsize')
    plt.yscale('log')
    plt.show()
#%%   
    plt.plot(maxlist,marker='.',linewidth=0)
    plt.xlabel('Events',fontsize=15)
    plt.ylabel('biggest Amplitude in an Event',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.yscale('log')
    plt.show()
    
    spaace=np.linspace(0,len(maxlist),len(maxlist))
    maxlist=np.array(maxlist)
    col=np.where(spaace<0,'k',np.where(maxlist>3e-2,'b','r'))
    newmaxlist=np.array(maxlist)
    newmaxlist=newmaxlist[3e-2<newmaxlist]
    plt.scatter(spaace,maxlist,marker='.',linewidth=0,c=col)
    # plt.plot(newmaxlist,marker='.',linewidth=0,color='red')
    plt.xlabel('Events',fontsize=15)
    plt.ylabel('biggest Amplitude in an Event',fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale('log')
    plt.show()
#%%   
    plt.plot(waitlist)
    plt.ylabel('Interevent time')
    plt.show()
#%% 
result=analyze(maxlist,xmin=1e-2,xmax=3e2,binnumber=50)
#%%
analyze(prepeaklist,xmin=3e-1,xmax=8e2,binnumber=50)
analyze(peaklist,xmin=5e-2,xmax=3e2,binnumber=50)
analyze(afterpeaklist,xmin=5e-2,xmax=8e2,binnumber=50)
#%%
lenlist=[]
meanlist=[]
for j in range(0,len(varlist)):
    print(len(varlist[j]))
    varlist[j].sort
    lenlist.append(len(varlist[j]))
    meanlist.append(np.mean([varlist[j]]))
    #analyze(varlist[j],xmin=5e-2,xmax=1e2,binnumber=50)
    
plt.scatter(np.linspace(1,len(lenlist),len(lenlist)),lenlist)
plt.xlabel('region number')
plt.ylabel('number of events')
plt.ylim((0,2500))
plt.grid(True)
plt.show()
#%%
plt.scatter(np.linspace(1,len(lenlist),len(lenlist)),meanlist)
plt.ylim((1e-1,1e1))
plt.yscale('log')
plt.xlabel('region number')
plt.ylabel('mean of Slew rate')
plt.show()

#%%
compare([varlist[0],varlist[3],varlist[6]],[1,4,7])
#%%
compare([prepeaklist,peaklist,afterpeaklist],['ramp','peak','after peak'])
# %%
ML=True
if ML == True:
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