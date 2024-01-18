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
from mpl_axes_aligner import align


params = {'axes.labelsize': 18,'axes.titlesize':16,  'legend.fontsize': 16, 'legend.frameon': True, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
plt.rcParams.update(params)
#%%functions



def gauss(x,sigma,mu):
    return np.exp(-((((x)-mu)**2/np.sqrt(2*np.pi)*sigma)))

def skewed_normal(x,mu,alpha,d,c,b):
    return 1/2*c*(1+erf(alpha*((x-mu)/b)/np.sqrt(2)))*1/np.sqrt(2*np.pi)*np.exp(-((x-mu)/b)**2/2)+d

def Lorentz(x,mu,gamma,a,b,c):
    return a/np.pi*(1/2*gamma/(((x-mu)/c)**2+(1/2*gamma)**2))+b

def lin_fit(x,a,b):
    return a*x+b

def analyze(data,xmin,xmax,binnumber,datatype='eventsize'):
    plt.style.use('seaborn-whitegrid')

    params = {'axes.labelsize': 18,'axes.titlesize':16,  'legend.fontsize': 16, 'legend.frameon': True, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
    plt.rcParams.update(params)
    
    plt.figure(figsize=(10,6))
    edges, hist = powerlaw.pdf(data,number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    fit.power_law.plot_pdf(label=r'$\tau_{ML}$'+"={}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    histsave=hist
    bin_centerssave=bin_centers
    
    print(bin_centers,hist)
     
    for i in range(0,len(xmin)):
        hist=histsave
        bin_centers=bin_centerssave
        hist=hist[xmin[i]<bin_centers]
        bin_centers=bin_centers[xmin[i]<bin_centers]
    
        hist=hist[bin_centers<xmax[i]]
        bin_centers=bin_centers[bin_centers<xmax[i]]
        
        bin_centers=bin_centers[hist!=0]
        hist=hist[hist!=0]
    
        
        popt,pcov=curve_fit(lin_fit,np.log(bin_centers),np.log(hist))
    
        fitlin=np.exp(lin_fit(np.log(bin_centers),*popt))
        plt.plot(bin_centers,fitlin,label=r'$\tau_{Lin}$'+r'={} $\pm${}'.format(np.abs(round(float(popt[0]),2)),round(float(np.sqrt(pcov[0][0])),2)))
    plt.xscale('log')
    plt.yscale('log')
    if datatype=='eventsize':
        plt.xlabel('Charge [C]')
        # plt.xlabel('event size in e')
    if datatype=='eventenergy':
        plt.xlabel('integrated eventsize')
    if datatype=='intereventtime':
        plt.xlabel('interevent time [s]')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
    startvalue=fit.xmin
    return fit,popt,pcov,bin_centerssave,histsave

def compare(datalist,binnrlist):
    if len(datalist)!=len(binnrlist):
           print('no')

    plt.figure(figsize=(10,6))
    
    for index in range(0,len(datalist)):
        
        edges, hist = powerlaw.pdf(np.abs(datalist[index]),number_of_bins=binnrlist[index])
    
        bin_centers = (edges[1:]+edges[:-1])/2.0
    
        # fit=powerlaw.Fit(data)
        # fit.power_law.plot_pdf(label="{}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
        
        plt.scatter(bin_centers,hist)
    plt.xlabel('Polarization C [e]')
    plt.ylabel('P(C_i=C)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return
    

def analyze_setxmin(data,xmin,xmax,binnumber,datatype='eventsize'):
    
    plt.rc('font',size=16)
    plt.figure(figsize=(10,6))
    plt.grid(which='major',axis='both')
    
    edges, hist = powerlaw.pdf(data,number_of_bins=binnumber)

    bin_centers = (edges[1:]+edges[:-1])/2.0

    fit=powerlaw.Fit(data)
    fit.power_law.plot_pdf(label="{}$\pm {}$".format(round(fit.alpha,2),round(fit.sigma,2)))
    
    plt.scatter(bin_centers,hist)
    
    for i in range(0,len(xmin)):
        hist=hist[xmin[i]<bin_centers]
        bin_centers=bin_centers[xmin[i]<bin_centers]
    
        hist=hist[bin_centers<xmax[i]]
        bin_centers=bin_centers[bin_centers<xmax[i]]
    
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
    return fit,popt,pcov,bin_centers,hist

#%%

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Daten\evaltest'


folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\05_05_2023\Measurements for Bark analysis with DWM after some settling time\7.32kHz'


# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\12.07.23\trashtest'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\05_05_2023\tests with new samples\7.32 medium big'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\14_4 or group meeting (both)\12_04_23 tests for baseline subtraction\better results\50 Ohm'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\12.05.2023\test 14.6'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\28.06 new samples\small x small'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\28.06 new samples\medium x big\29.3kHz'
folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\17_07_23\different old sample keysight small big\reduced voltage'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\31.07.23 more barkhausen analysis sample 2-7 tmbb\rise time for 30 V 32Hz 300 us'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\better data for presenting\17_07_23\all 938 kHz'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\17_07_23\newer sample shows different results keysight\recorded with 10mV res'


# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\31.07.23 more barkhausen analysis sample 2-7 tmbb\everything'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Thesis data\barkhausen part\30V big deviation'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\21_07_23 different step voltages AFG AMP\30V'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\09.08 more barktests with some older samples\truly big events only close to switching peak'

folder =r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\09.08 more barktests with some older samples\35V 16,025Hz 600us rise 20ms plateau'

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\27_07_23 systematic rise times\bark test 300 us'
# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\26.05.23\29.3 keysight 50ms ramps'

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\11_10_23 hopefully better systematic stuff\32 7.5Hz 22V 938kHz'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Thesis data\barkhausen part\30V big deviation'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\29_09_23 samples julius 1\noise from increase'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\10_10_23 very systematic stuff\1_240Hz_30V'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\11_10_23 hopefully better systematic stuff\23 1Hz 16V 14.6kHz actual DWM'

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\19.11.23 measurements with samples that arent crazy\7 sample 5tbbs 938kHz 25V+30Hz less repetitions'

#%%

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Daten\combine lots of data for some more or less satisfying result'

# folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\Daten\combine 1 high freq one low freq'

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\19.11.23 measurements with samples that arent crazy\14 DWM with filter 30V 1Hz 29.3kHz'

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\19.11.23 measurements with samples that arent crazy\7 sample 5tbbs 938kHz 25V+30Hz less repetitions'

folder=r'\\heivol-i.ad.uni-heidelberg.de\cam\Research\Groups\AG_Kemerink\Group Members\Seiler, Toni\barkhausen measurements\11_10_23 hopefully better systematic stuff\46 30Hz 29V 938kHz'

sizelist=[] 
maxlist=[]
waitlist=[]    
durationlist=[] 
Pollist=np.array([])
Chargelist=np.array([])

Chargelistslow=[]
Chargelistfast=[]
    
frequenzyindex=0

lastidx=0

print(frequenzyindex)
# %%
area=0.1e-3*0.5e-3
avg=7e-5
upperlimit=3e-2
k=0
var=0
resistance=50
plot=False
plotoriginal=False
ploteventdists=False

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
        I=vout
        dt=t[1]-t[0]

    
    except (UnicodeDecodeError):
        print('2')
        continue
    # if k!=10:
    #     k=k+1
    #     continue
    # else:
    k=k+1
    
    if plotoriginal==True:    
        plt.plot(t,I*1000,label=k-1)
        # plt.plot(tcut,Icut*1000)
        # plt.plot(t,fit(tcut)*1000,label='derivative from fit')
        plt.xlabel('time [s]')
        plt.ylabel('current [mA]')
        plt.legend()
        plt.show()
        
    var=0
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
            
            frequenzyindex=1
            
            length=lengthright=2000  #29.3kHz
            lengthleft=500
            degval=100
            
            break
        
        elif 1/dt < 60000 :
            
            frequenzyindex=1
            
            length=lengthright=2000  #29.3kHz
            lengthleft=500
            degval=100
            
            break

        else:
            
            frequenzyindex=2
            
            lengthleft=200
            lengthright=200#938kHz
            degval=10
            break
        
    peakidx=find_peaks(np.abs(I),height=max(np.abs(I)/20),distance=(lengthright+lengthleft))[0]
    plt.plot(t,np.abs(I))
    plt.plot([min(t),max(t)],np.ones(2)*[max(np.abs(I)/20)])
    plt.scatter(t[peakidx],np.abs(I[peakidx]),color='r')
    plt.show()
    print(frequenzyindex)
    for  idxext in peakidx:  
        # if idxext<4000:
        #     continue
        # if var!=0:
        #     var=var+1
        #     continue
        # else:
        var=var+1   
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
        while True:
            try:
                fited=np.polyfit(tcut,Icut,deg=degval)       #::-1],deg=degval
                break
            except np.linalg.LinAlgError :
                degval=degval-5
        
        fited=np.poly1d(fited)
        
        lastidx=idxext
#%%
        
        if plotoriginal==True:    
            # plt.plot(t,I*1000)
            plt.plot(tcut,Icut*1000,label=[var-1,k-1])
            # plt.plot(t,fit(tcut)*1000,label='derivative from fit')
            plt.xlabel('time [s]')
            plt.ylabel('current [mA]')
            plt.legend()
            plt.show()
            
        # if plot==True:    
        #     plt.plot(tcut,Icut*1000)
        #     plt.plot(tcut,fit(tcut)*1000,)
        #     plt.xlabel('time [s]')
        #     plt.ylabel('current [mA]')
        #     # plt.legend()
        #     plt.show()
        
# %%       
        
        dI=np.array([])
        dt=t[1]-t[0]
        dFit=np.array([])       
        for i in range(0,len(Icut[0:])-1):
            dI=np.append(dI,(Icut[i+1]-Icut[i]))
            dFit=np.append(dFit,(fited(tcut)[i+1]-fited(tcut)[i]))
            # dFit=np.append(dFit,(skewed_normal(tcut[i+1],*popt)-skewed_normal(tcut[i],*popt)))
            
        Jerkspectrum=(dI/dt)
        Jerkspectrum= np.append(Jerkspectrum,0) 
        # Jerksandbase=np.abs(Jerkspectrum)
        Jerksandbase=Jerkspectrum**2
        
        Fitspectrum=(dFit/dt)
        Fitspectrum=np.append(Fitspectrum,0)
        # Fitbase=np.abs(Fitspectrum)
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

        baseline=Pchip(timelist, minlist, tcut)  
        jerks=jerksminusfit-baseline
        
# %%     
        # jerks = np.abs(jerksminusfit)
        if plot==True:            
            fig, ax1 = plt.subplots()
            ax1.plot(tcut,np.abs(Icut),label=var-1)
            ax1.plot(tcut,np.abs(fited(tcut)))
            ax1.set_ylabel('current [A]')
            ax2=ax1.twinx()
            # plt.plot(tcut,jerks)
            # plt.plot(tcut,Jerksandbase,c='r',label='data')
            # plt.plot(tcut,Fitbase,c='g',label='fit')
            ax2.plot(tcut,jerks,c='r')
            ax2.set_xlabel('time [s]')
            ax2.set_ylabel(r'slew rate $[\frac{A^2}{s^2}]$',c='r')
            ax2.grid(False)
            align.yaxes(ax1,0,ax2,0,0.05)
            plt.legend()
            plt.show()
#%%
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


#%%        
        polarpeaks=find_peaks(jerks,height=max(jerks)/20,distance=20)[0]
        plt.plot(tcut,Icut)
        plt.scatter(tcut[polarpeaks],Icut[polarpeaks],color='red')
        plt.show()
        
        Icut=np.abs(Icut)
        
        polarpeaks=polarpeaks[polarpeaks>20]
        polarpeaks=polarpeaks[polarpeaks<len(tcut)-20]
#%%
        check1=0
        check2=0
        idxnr=0
        for polarnr in polarpeaks:
            
                    check1=0
                    check2=0        
                    eventlengthleft=20
                    eventlengthright=20
                    
                    while eventlengthleft!=0 and eventlengthright!=0:
                        # print(eventlengthleft,eventlengthright,check1,check2)
                        checkleft=0
                        checkright=0
                        Inew=Icut[polarnr-eventlengthleft:polarnr+eventlengthright]
                        for d in range(0,eventlengthleft):
                            if Inew[0]>Inew[d+1]:
                                checkleft=1
                        if checkleft==1:
                            eventlengthleft=eventlengthleft-1
                        else:
                            check1=1
                        for d in range(1,eventlengthright):
                            if Inew[-1]>Inew[-d]:
                                checkright=1
                        if checkright==1:
                            eventlengthright=eventlengthright-1
                        else:
                            check2=1
                        if check1==1 and check2==1:
                            tnew=tcut[polarnr-eventlengthleft:polarnr+eventlengthright]
                            # print([eventlengthleft,eventlengthright])
                            break
                    if eventlengthleft==0 or eventlengthright==0:
                        continue
                    # try:
                    #     Inew=Icut[polarnr-listformanualstuff[idxnr][0]:polarnr+listformanualstuff[idxnr][1]]
                    #     tnew=tcut[polarnr-listformanualstuff[idxnr][0]:polarnr+listformanualstuff[idxnr][1]]
                        
                    #     idxnr=idxnr+1
                        
                    #     if len(Inew)==0:
                    #         continue
                    # except:
                    #     continue
                    tfit,Ifit=tnew[0],Inew[0]
                    tfit,Ifit=np.append(tfit,tnew[-1]),np.append(Ifit,Inew[-1])
                    
                    try:
                        popt,pcov=curve_fit(lin_fit,tfit,Ifit)
                    except RuntimeError:
                        continue
                    Iintegrate=Inew-lin_fit(tnew,*popt)    
                    
                    Charge=scipy.integrate.cumulative_trapezoid(Iintegrate,tnew)[-1]
                    Pol=Charge/area
                    
                    if Charge>1e-10:
                        Chargelist=np.append(Chargelist,Charge)
                        Pollist=np.append(Pollist,Pol)
                    
                        plt.plot(tnew,lin_fit(tnew,*popt))
                        plt.plot(tnew,Inew)
                        plt.plot(tnew,Iintegrate)
                        
                    else:
                        continue
                    
                    if frequenzyindex==1:
                        Chargelistslow.append(Charge)
                    if frequenzyindex==2:
                        Chargelistfast.append(Charge)
                        
                    # print(max(tnew)-min(tnew),max(Iintegrate))
                # except:
                #     print('trash',len(tnew),len(Inew))
                #     continue
        plt.grid(visible=True)
        plt.xlabel('time [s]')
        plt.ylabel('Current [mA]')
        plt.show()

#%%
plt.scatter(np.linspace(0,len(Chargelist),len(Chargelist)),Chargelist)
plt.xlabel('#')
plt.ylabel(r'Charge $\left[C\right]$')
plt.yscale('log')
plt.show()
#%%
Chargelistslow=np.array(Chargelistslow)
Chargelistfast=np.array(Chargelistfast)

Chargelistbig=Chargelist/1.602e-19

# Chargebigfast=Chargelistfast/1.602e-19

#%%
Chargelistbig=Chargelistbig[Chargelistbig>2e3]
# Chargelistbig=Chargelistbig[Chargelistbig>3e8]

# Chargebigfast=Chargebigfast[Chargebigfast>3e3]


# Chargelistbig=np.append(Chargelistbig,Chargebigfast)
# Chargelistbig=np.append(Chargelistbig,datalst[0])

results=analyze(np.abs(Chargelistbig), [1e9], [1e11], 50)

# print(results)
#%%
# compare([Chargelistbig,chargelistDWM],[30,30])
#%%looking for different shit
plt.hist(Chargelistbig,bins=np.logspace(np.log10(min(Pollistbig)),np.log10(max(Pollistbig)),30),density=True)
plt.xscale('log')
plt.yscale('log')
plt.show()


#%% handpicking shit
# Chargelistfastarchive=Chargelistfast
# Chargelistslowarchive=Chargelistslow
# Simlistarchive=datalst[0]


        # polarpeaks=find_peaks(jerks,height=max(jerks)/50,distance=10)[0]
        # plt.plot(tcut,Icut)
        # plt.scatter(tcut[polarpeaks],Icut[polarpeaks])
        # plt.show()


        # Pollist=[]
        # # plt.plot(tcut,Icut)
        # listformanualstuff=[[1,10],[1,10],[1,10],[5,15],[5,10],[2,8],[13,5],[5,5],[5,20],[15,7],[2,30],[2,15],[3,10],[1,13],[14,12],[5,10]]
        
        # idxnr=0
        # for polarnr in polarpeaks:
        #     try:
        #         Inew=Icut[polarnr-listformanualstuff[idxnr][0]:polarnr+listformanualstuff[idxnr][1]]
        #         tnew=tcut[polarnr-listformanualstuff[idxnr][0]:polarnr+listformanualstuff[idxnr][1]]
                
        #         idxnr=idxnr+1
                
        #         if len(Inew)==0:
        #             continue
        #     except:
        #         continue
        #     try:
        #         tfit,Ifit=tnew[0],Inew[0]
        #         tfit,Ifit=np.append(tfit,tnew[-1]),np.append(Ifit,Inew[-1])
                
        #         popt,pcov=curve_fit(lin_fit,tfit,Ifit)
                    
        #         Iintegrate=Inew-lin_fit(tnew,*popt)    
                
        #         plt.plot(tnew,lin_fit(tnew,*popt))
        #         plt.plot(tnew,Inew)
        #         plt.plot(tnew,Iintegrate)
                
        #         Charge=np.sum(scipy.integrate.cumulative_trapezoid(tnew,Iintegrate))
        #         Pol=Charge/area
        #         Pollist=np.append(Pollist,Pol[len(Pol)-1])
        #     except:
        #         continue
        # plt.show()
        

#%% attempts at getting actually good baselines usin assymetric leas squares smoothing
#
#             ASLSbaseline=baseline_als(I,0.001,1.5,niter=10)
#             fig,ax1=plt.subplots()
#             ax1.plot(t,I-ASLSbaseline,c='g')
#             plt.show()
# #5%
#             fig,ax1=plt.subplots()
#             ax1.plot(t,ASLSbaseline)
#             ax2=ax1.twinx()
#             ax2.plot(t,I)
            # plt.show()