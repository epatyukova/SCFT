#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:34:25 2022

@author: patyukoe
"""

import numpy as np
from basis_functions import lamellar, hexagonal, gyroid, bcc
from diff_eq_solver import dif_cal, fequation


class Supervisor:
    def __init__(self,**kwargs):
        self._kwargs = kwargs
        self.ensemble = kwargs.get('ensemble')
        self.chi = int(kwargs.get('chi')) # chiN product
        self.struct =  kwargs.get('structure') # type of structure
        self.nl = int(kwargs.get('nl')) # length of one block
        self.nm = int(kwargs.get('nm')) # number of collocation points
        self.D_init = float(kwargs.get('D')) # initial guess for the period of microstructure 
        self.nblocks = int(kwargs.get('nblocks'))
        self.ns = float(self.nl * self.nblocks)
        self.ds = 1.0/self.ns
        self.nh = int(kwargs.get('nh'))
        
        self.f = float(kwargs.get('comp')) # average compostion, should be in the range (0,1)
        self.lamb = float(kwargs.get('lambda')) # parameter describing
        
        if(self.nblocks == 2):
            self.nchains = 3
        elif(self.nblocks == 3):
            self.nchains = 6 
            
        self.Q = np.zeros(self.nchains) # chain partition function  
        
        self.t = self.f*(1.0-self.f)*(1.0-self.lamb) # parameter theta
        self.vol0 = np.zeros(self.nchains) # volume fractions of components in the initial mixture
        
        self.fA = self.f
        self.fB = 1.0 - self.f
        
        if(self.nblocks == 2):
            self.vol0[0] = self.f - self.t
            self.vol0[1] = 2.0*self.t
            self.vol0[2] = 1.0 - self.f - self.t
        elif(self.nblocks == 3):
            self.vol0[0] = (self.f-self.t)**2/self.f
            self.vol0[1]=2.0*self.t*(1.0-self.f-self.t)/(1.0-self.f)
            self.vol0[2]=self.t**2/(1.0-self.f)
            self.vol0[3]=self.t**2/self.f
            self.vol0[4]=2*self.t*(self.f-self.t)/self.f
            self.vol0[5]=(1.0-self.f-self.t)**2/(1.0-self.f)
            
        self.cf=np.zeros(self.nchains)
        if(self.nblocks==2):
            self.cf[0]=1.0 #AA
            self.cf[1]=1.0/2.0 #AB
            self.cf[2]=0.0 #BB
        elif(self.nblocks==3):
            self.cf[0]=1.0 #AAA
            self.cf[1]=1.0/3.0 #ABB
            self.cf[2]=2.0/3.0 #ABA
            self.cf[3]=1.0/3.0 #BAB
            self.cf[4]=2.0/3.0 #BAA
            self.cf[5]=0.0 #BBB
        
        self.z = np.zeros(self.nchains)
        self.z_file = kwargs.get('z_file')
        

        self.wa_file = kwargs.get('A_field_file')
        self.wb_file = kwargs.get('B_field_file')
        
        self.wa_savefile = kwargs.get('A_save_field_file')
        self.wb_savefile = kwargs.get('B_save_field_file')

        if(self.ensemble == 'grand'):
            try:
                self.z = np.loadtxt(self.z_file, delimiter=' ')
            except:
                print("No z-file!")
        elif(self.ensemble == 'canonical'):
            self.vol=self.vol0
            
        if(self.struct=='LAM'):
            self.nm = self.nm,2,2 # numbers of collocation points
            self.L = self.D_init *6.0**(0.5),\
                     self.D_init*6.0**(0.5)/float(self.nm)*2.0,\
                     self.D_init*6.0**(0.5)/float(self.nm)*2.0
            if (self.wa_file ==('False' or 'false' or None)) or (self.wb_file ==('False' or 'false' or None)):       
                self.wa,self.wb=lamellar(self.nm)
            else:
                self.wa=np.load(self.wa_file)
                self.wb=np.load(self.wb_file)
                
        elif(self.struct=='HEX'):
            self.nm=self.nm,int(self.nm*3.0**0.5),2 # numbers of collocation points
            self.L=self.D_init*6.0**(0.5),\
                   self.D_init*6**(0.5)*3.0**0.5,\
                   self.D_init*6.0**(0.5)/float(self.nm)*2.0 # sizes of the cell for cylinder
            if (self.wa_file==('False' or 'false' or None)) or (self.wb_file== ('False' or 'false' or None)):       
                self.wa,self.wb=hexagonal(self.nm)
            else:
                self.wa=np.load(self.wa_file)
                self.wb=np.load(self.wb_file)
        
        elif(self.struct == 'BCC'):
            self.nm=self.nm,self.nm,self.nm 
            self.L=self.D_init*6.0**(0.5),self.D_init*6.0**(0.5),self.D_init*6.0**(0.5) 
            if (self.wa_file==('False' or 'false' or None)) or (self.wb_file== ('False' or 'false' or None)): 
                self.wa,self.wb=bcc(self.nm)
            else:
                self.wa=np.load(self.wa_file)
                self.wb=np.load(self.wb_file)
                
        elif(self.struct == 'GYR'):
            self.nm=self.nm,self.nm,self.nm 
            self.L=self.D_init*6.0**(0.5),self.D_init*6.0**(0.5),self.D_init*6.0**(0.5) 
            if (self.wa_file==('False' or 'false')) or (self.wb_file== ('False' or 'false')): 
                self.wa,self.wb=gyroid(self.nm)
            else:
                self.wa=np.load(self.wa_file)
                self.wb=np.load(self.wb_file)
        
                
        self.dwa=np.zeros(self.nm) # difference in field between successive iterations
        self.dwb=np.zeros(self.nm)
        
        self.fields=dict(wa=self.wa,wb=self.wb,dwa=self.dwa,dwb=self.dwb)

        self.rhoA=np.zeros(self.nm) # density of A-segments
        self.rhoB=np.zeros(self.nm) # density of B-segments
        self.rho=np.zeros(self.nm) # total density
        self.rhoAch=np.zeros((self.nchains,*self.nm)) # density of A-segments
        self.rhoBch=np.zeros((self.nchains,*self.nm)) # density of B-segments
        self.dens=dict(rhoA=self.rhoA,rhoB=self.rhoB,rho=self.rho,rhoAch=self.rhoAch,rhoBch=self.rhoBch)

        self.hiswa=np.zeros((self.nh+1,*self.nm))  # arrays to save histories
        self.hiswb=np.zeros((self.nh+1,*self.nm))
        self.hisdwa=np.zeros((self.nh+1,*self.nm))
        self.hisdwb=np.zeros((self.nh+1,*self.nm))
        self.his=dict(hiswa=self.hiswa,hiswb=self.hiswb,hisdwa=self.hisdwa,hisdwb=self.hisdwb)
        
        self.fq=np.zeros((self.nchains,self.ns+1,*self.nm)) # forward propogator
        self.bq=np.zeros((self.nchains,self.ns+1,*self.nm)) # backward propagator

        self.fq[:,0,:,:,:]=1.0  # initial condition for forward propogator
        self.bq[:,ns,:,:,:]=1.0 # initial condition for backward propogator
        
        self.dif=np.zeros(self.nm)
        
    def run(self):
        return
        
    
    
    
    
    
    
    
    
    
    
    