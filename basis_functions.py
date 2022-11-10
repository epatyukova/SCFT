#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:49:21 2022

@author: patyukoe
"""


import numpy as np


# reference for basis functions is Matsen, Bates, Macromolecules 1996, 29, 4, 1091–1098
# for other structures the best way is to use symmetry adapted basis functions 
# suggested by David Morse

def lamellar(nm_lam):
    wa=np.zeros(nm_lam)
    wb=np.zeros(nm_lam)
    
    for i in range(0,nm_lam[0]):
        wb[i,:,:] = (1+2.0**0.5*np.cos(2*np.pi*i/nm_lam[0])+2.0**0.5*np.cos(4*np.pi*i/nm_lam[0]))
        wa[i,:,:] = -wb[i,:,:] 
       
    return wa,wb
    

def hexagonal(nm_hex):
    nm=nm_hex
    wa=np.zeros(nm)
    wb=np.zeros(nm)
    
    for i in range(0,nm[0]):
        for j in range(0,nm[1]):
            for k in range(0,nm[2]):
            # inital field for HEX               
                wb[i,j,k]=(1+(2/3)**0.5*(np.cos(2*2*np.pi*j/nm[1])+2*np.cos(2*np.pi*i/nm[0])*np.cos(2*np.pi*j/nm[1]))+\
                           (2/3)**0.5*(np.cos(2*2*np.pi*i/nm[0])+2*np.cos(2*np.pi*i/nm[0])*np.cos(3*2*np.pi*j/nm[1]))+\
                           (2/3)**0.5*(np.cos(4*2*np.pi*j/nm_hex[1])+2*np.cos(2*2*np.pi*i/nm_hex[0])*np.cos(2*2*np.pi*j/nm_hex[1]))+\
                           (4/3)*0.5*(np.cos(3*2*np.pi*i/nm_hex[0])*np.cos(2*np.pi*j/nm_hex[1])+\
                           np.cos(2*2*np.pi*i/nm_hex[0])*np.cos(4*2*np.pi*j/nm_hex[1])+\
                           np.cos(2*np.pi*i/nm_hex[0])*np.cos(5*2*np.pi*j/nm_hex[1])))
                wa[i,j,k]=-wb[i,j,k]
    
    
    return wa,wb


def gyroid(nm_gyr):
 
    nm=nm_gyr
    wa=np.zeros(nm)
    wb=np.zeros(nm)
    
    for i in range(0,nm[0]):
        for j in range(0,nm[1]):
            for k in range(0,nm[2]):
                wb[i,j,k]=1+(8/3)**0.5*(np.cos(2*np.pi*i/nm[0])*np.sin(2*np.pi*j/nm[1])*np.sin(2*2*np.pi*k/nm[2])+\
                                        np.cos(2*np.pi*j/nm[1])*np.sin(2*np.pi*k/nm[2])*np.sin(2*2*np.pi*i/nm[0])+\
                                        np.cos(2*np.pi*k/nm[2])*np.sin(2*np.pi*i/nm[0])*np.sin(2*2*np.pi*j/nm[1]))+\
                           (4/3)**0.5*(np.cos(2*2*np.pi*i/nm[0])*np.cos(2*2*np.pi*j/nm[1])+np.cos(2*2*np.pi*k/nm[2])*np.cos(2*2*np.pi*j/nm[1])+\
                                       np.cos(2*2*np.pi*i/nm[0])*np.cos(2*2*np.pi*k/nm[2]))+\
                           (4/3)**0.5*(np.sin(2*2*np.pi*i/nm[0])*(np.cos(3*2*np.pi*j/nm[1])*np.sin(2*np.pi*k/nm[2])-np.sin(3*2*np.pi*j/nm[1])*np.cos(2*np.pi*k/nm[2]))+\
                                       np.sin(2*2*np.pi*j/nm[1])*(np.cos(3*2*np.pi*k/nm[2])*np.sin(2*np.pi*i/nm[0])-np.sin(3*2*np.pi*k/nm[2])*np.cos(2*np.pi*i/nm[0]))+\
                                       np.sin(2*2*np.pi*k/nm[2])*(np.cos(3*2*np.pi*i/nm[0])*np.sin(2*np.pi*j/nm[1])-np.sin(3*2*np.pi*i/nm[0])*np.cos(2*np.pi*j/nm[1])))+\
                           (2/3)**0.5*(np.cos(4*2*np.pi*i/nm[0])+np.cos(4*2*np.pi*j/nm[1])+np.cos(4*2*np.pi*k/nm[2]))    
                wa[i,j,k]=-wb[i,j,k]
    
    return wa,wb



def bcc(nm_bcc):
    nm=nm_bcc
    wa=np.zeros(nm)
    wb=np.zeros(nm)
    
    for i in range(0,nm[0]):
        for j in range(0,nm[1]):
            for k in range(0,nm[2]):
                wb[i,j,k]=1+(4/3)**0.5*(np.cos(2*np.pi*i/nm[0]+2*np.pi*j/nm[1])*np.cos(2*np.pi*0/nm[0]+2*np.pi*k/nm[2])+np.cos(np.pi*2*i/nm[0]+2*np.pi*j/nm[1])*np.cos(2*np.pi*j/nm[1]+2*np.pi*k/nm[2])+\
                          np.cos(2*np.pi*i/nm[0]+2*np.pi*k/nm[2])*np.cos(2*np.pi*j/nm[1]+2*np.pi*k/nm[2]))+\
                          (2/3)**0.5*(np.cos(4*np.pi*(i/nm[0]+j/nm[1]))+np.cos(4*np.pi*(i/nm[0]+k/nm[2]))+\
                          np.cos(4*np.pi*(j/nm[1]+k/nm[2])))
                wa[i,j,k]=-wb[i,j,k]
    
    return wa,wb
  
    

    
    
    
    