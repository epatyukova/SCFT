#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 00:31:26 2022

@author: patyukoe
"""


import numpy as np


###########################################################
# solution of diffusion equations, output: fq and bq      #
###########################################################

# diffusion operator
def cal_L(D):
    """
    cal_L calculates L when D is given

    Parameters
    ----------
    D : float
    
    Returns
    -------
    L : float 3D array
    
    """
    if(struct=='LAM'):
        L=D*6.0**(0.5),D*6.0**(0.5)/float(nm[0])*2.0,D*6.0**(0.5)/float(nm[0])*2.0
    elif(struct=='HEX'):
        L=D*6.0**(0.5),D*6**(0.5)*3.0**0.5,D*6.0**(0.5)/float(nm[0])*2.0
    else:
        L=D*6.0**(0.5),D*6.0**(0.5),D*6.0**(0.5)
    
    return L 

def dif_cal(dif,L):  
    """
    dif_cal calculates diffusion operator when L is given

    Parameters
    ----------
    L : 3d array
    dif: empty numpy array for ddiffusion operator
    
    Returns
    -------
    dif : array with diffusion operator values
    
    """
    for i in range(0,round(nm[0]/2)):
        for j in range(0,round(nm[1]/2)):
            for k in range(0,round(nm[2]/2)):
                dif[i,j,k]=exp(-4*pi**2*ds*i**2/L[0]**2)*exp(-4*pi**2*ds*j**2/L[1]**2)*exp(-4*pi**2*ds*k**2/L[2]**2)
            for k in range(round(nm[2]/2),nm[2]):
                dif[i,j,k]=exp(-4*pi**2*ds*i**2/L[0]**2)*exp(-4*pi**2*ds*j**2/L[1]**2)*exp(-4*pi**2*ds*(nm[2]-k)**2/L[2]**2)
        for j in range(round(nm[1]/2),nm[1]):
            for k in range(0,round(nm[2]/2)):
                dif[i,j,k]=exp(-4*pi**2*ds*i**2/L[0]**2)*exp(-4*pi**2*ds*(nm[1]-j)**2/L[1]**2)*exp(-4*pi**2*ds*k**2/L[2]**2)
            for k in range(round(nm[2]/2),nm[2]):
                dif[i,j,k]=exp(-4*pi**2*ds*i**2/L[0]**2)*exp(-4*pi**2*ds*(nm[1]-j)**2/L[1]**2)*exp(-4*pi**2*ds*(nm[2]-k)**2/L[2]**2)
    for i in range(round(nm[0]/2),nm[0]):
        for j in range(0,round(nm[1]/2)):
            for k in range(0,round(nm[2]/2)):
                dif[i,j,k]=exp(-4*pi**2*ds*j**2/L[1]**2)*exp(-4*pi**2*ds*(nm[0]-i)**2/L[0]**2)*exp(-4*pi**2*ds*k**2/L[2]**2)
            for k in range(round(nm[2]/2),nm[2]):
                dif[i,j,k]=exp(-4*pi**2*ds*j**2/L[1]**2)*exp(-4*pi**2*ds*(nm[0]-i)**2/L[0]**2)*exp(-4*pi**2*ds*(nm[2]-k)**2/L[2]**2)
        for j in range(round(nm[1]/2),nm[1]):
            for k in range(0,round(nm[2]/2)):
                dif[i,j,k]=exp(-4*pi**2*ds*(nm[0]-i)**2/L[0]**2)*exp(-4*pi**2*ds*(nm[1]-j)**2/L[1]**2)*exp(-4*pi**2*ds*k**2/L[2]**2)
            for k in range(round(nm[2]/2),nm[2]):
                dif[i,j,k]=exp(-4*pi**2*ds*(nm[0]-i)**2/L[0]**2)*exp(-4*pi**2*ds*(nm[1]-j)**2/L[1]**2)*exp(-4*pi**2*ds*(nm[2]-k)**2/L[2]**2)
    return dif



def fequation(fq,fields,dif):
    """
        fequaiton solves diffusion equation for forward propogator
    
        Parameters
        ----------
        fq : TYPE
            fq is a forward propogator (array with initial condition specified).
        wa : TYPE
            self-consistent field acting on A segments.
        wb : TYPE
            self-consistent field acting on B segments.
        dif : TYPE
            diffusion operator.
    
        Returns
        -------
        fq : TYPE
            forward propogator.
    """
    q0=np.zeros(nm)
    q13=np.zeros(nm)
    q23=np.zeros(nm)
    c=np.zeros(nm)
    
    if(nblocks==2):    
        for i in range(nchains):
            for j in range(0,int(ns*cf[i])):
                q0=fq[i,j,:,:,:]
                q13=q0*exp(-fields['wa']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wa']*ds/2) 
                fq[i,j+1,:,:,:]=q23.real
                
            for j in range(int(ns*cf[i]),ns):
                q0=fq[i,j,:,:,:]
                q13=q0*exp(-fields['wb']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wb']*ds/2)
                fq[i,j+1,:,:,:]=q23.real
    elif(nblocks==3):
        # first block calculation
        for j in range(0,nl):
            q0=fq[0,j,:,:,:]
            q13=q0*exp(-fields['wa']*ds/2)
            c=fftn(q13)
            c=c*dif
            q23=ifftn(c)
            q23=q23*exp(-fields['wa']*ds/2) 
            fq[0,j+1,:,:,:]=q23.real
        fq[1,:,:,:,:]=fq[0,:,:,:,:]
        fq[2,:,:,:,:]=fq[0,:,:,:,:]
        
        for j in range(0,nl):
            q0=fq[5,j,:,:,:]
            q13=q0*exp(-fields['wb']*ds/2)
            c=fftn(q13)
            c=c*dif
            q23=ifftn(c)
            q23=q23*exp(-fields['wb']*ds/2) 
            fq[5,j+1,:,:,:]=q23.real
        fq[3,:,:,:,:]=fq[5,:,:,:,:]
        fq[4,:,:,:,:]=fq[5,:,:,:,:]
        
        # second block calculation
        for i in [0,3,4]:
            for j in range(nl,2*nl):
                q0=fq[i,j,:,:,:]
                q13=q0*exp(-fields['wa']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wa']*ds/2) 
                fq[i,j+1,:,:,:]=q23.real
        
        for i in [1,2,5]:    
            for j in range(nl,2*nl):
                q0=fq[i,j,:,:,:]
                q13=q0*exp(-fields['wb']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wb']*ds/2) 
                fq[i,j+1,:,:,:]=q23.real
                    
        # third block calculation        
        for i in [0,2,4]:
            for j in range(2*nl,3*nl):
                q0=fq[i,j,:,:,:]
                q13=q0*exp(-fields['wa']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wa']*ds/2) 
                fq[i,j+1,:,:,:]=q23.real
        
        for i in [1,3,5]:
            for j in range(2*nl,3*nl):
                q0=fq[i,j,:,:,:]
                q13=q0*exp(-fields['wb']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wb']*ds/2) 
                fq[i,j+1,:,:,:]=q23.real
        
    
    return fq



def bequation(bq,fields,dif):
    """
        bequation solves diffusion equation for backward propogator 
        
        Parameters
        ----------
        bq : TYPE
            bq is a backward propogator (array with initial condition specified).
        wa : TYPE
            self-consistent field acting on A segments.
        wb : TYPE
            self-consistent field acting on B segments.
        dif : TYPE
            diffusion operator.
    
        Returns
        -------
        bq : TYPE
            backward propogator.
    
    """
    q0=np.zeros(nm)
    q13=np.zeros(nm)
    q23=np.zeros(nm)
    c=np.zeros(nm)
    
    if(nblocks==2):
        for i in range(nchains):    
            for j in range(0,ns-int(ns*cf[i])):
                q0=bq[i,ns-j,:,:,:]
                q13=q0*exp(-fields['wb']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wb']*ds/2)
                bq[i,ns-j-1,:,:,:]=q23.real
                
            for j in range(ns-int(ns*cf[i]),ns):
                q0=bq[i,ns-j,:,:,:]
                q13=q0*exp(-fields['wa']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wa']*ds/2)
                bq[i,ns-j-1,:,:,:]=q23.real
    elif(nblocks==3):
        # third block calculation
        for j in range(0,nl):
            q0=bq[0,ns-j,:,:,:]
            q13=q0*exp(-fields['wa']*ds/2)
            c=fftn(q13)
            c=c*dif
            q23=ifftn(c)
            q23=q23*exp(-fields['wa']*ds/2)
            bq[0,ns-j-1,:,:,:]=q23.real
        bq[4,:,:,:,:]=bq[0,:,:,:,:]
        bq[2,:,:,:,:]=bq[0,:,:,:,:]
            
        for j in range(0,nl):
            q0=bq[5,ns-j,:,:,:]
            q13=q0*exp(-fields['wb']*ds/2)
            c=fftn(q13)
            c=c*dif
            q23=ifftn(c)
            q23=q23*exp(-fields['wb']*ds/2)
            bq[5,ns-j-1,:,:,:]=q23.real
        bq[1,:,:,:,:]=bq[5,:,:,:,:]
        bq[3,:,:,:,:]=bq[5,:,:,:,:]
        
        # second block calculation
        for i in [0,3,4]:
            for j in range(nl,2*nl):
                q0=bq[i,ns-j,:,:,:]
                q13=q0*exp(-fields['wa']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wa']*ds/2)
                bq[i,ns-j-1,:,:,:]=q23.real
        for i in [1,2,5]:
            for j in range(nl,2*nl):
                q0=bq[i,ns-j,:,:,:]
                q13=q0*exp(-fields['wb']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wb']*ds/2)
                bq[i,ns-j-1,:,:,:]=q23.real
                
        # first block calculation
        for i in [0,1,2]:
            for j in range(2*nl,ns):
                q0=bq[i,ns-j,:,:,:]
                q13=q0*exp(-fields['wa']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wa']*ds/2)
                bq[i,ns-j-1,:,:,:]=q23.real
        for i in [3,4,5]:
            for j in range(2*nl,ns):
                q0=bq[i,ns-j,:,:,:]
                q13=q0*exp(-fields['wb']*ds/2)
                c=fftn(q13)
                c=c*dif
                q23=ifftn(c)
                q23=q23*exp(-fields['wb']*ds/2)
                bq[i,ns-j-1,:,:,:]=q23.real
               
    return bq

















