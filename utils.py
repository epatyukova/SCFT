#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:13:22 2022

@author: patyukoe
"""

import numpy as np



def simplemixing(fields,dens):
    """
    Makes one iteration of simple mixing method for solution of 
    self-consistent equations
    
    """
    lam=0.1 # parameter for iteration method
    
    wta=np.zeros(nm) # auxilary field arrays
    wtb=np.zeros(nm)
    
    if(ens=='grand'):
        wta=chi*(2*dens['rhoB']-1)+fields['wb']
        wtb=-chi*(1-2*dens['rhoA'])+fields['wa']
    elif(ens=='canonical'):
        # wta=chi*(dens['rhoB']-fB)+1/2*(fields['wa']+fields['wb'])
        # wtb=chi*(dens['rhoA']-fA)+1/2*(fields['wa']+fields['wb'])
        wta=chi*(2*dens['rhoB']-1)+fields['wb']
        wtb=-chi*(1-2*dens['rhoA'])+fields['wa']
                
    dwa=wta-fields['wa']
    dwb=wtb-fields['wb']
        
    fields['wa']=fields['wa']+lam*dwa
    fields['wb']=fields['wb']+lam*dwb 
    fields['dwa']=dwa
    fields['dwb']=dwb
    
    # if(ens=='canonical'):    
    #     fields['wa']=fields['wa']-np.sum(fields['wa'])*1/prod(nm)
    #     fields['wb']=fields['wb']-np.sum(fields['wb'])*1/prod(nm)
    
    return fields



def history(nn,fields,his):
    """
    the function updates histories

    """
    
    if(nn<nh+1):
        his['hisdwa'][nn,:,:,:]=fields['dwa']
        his['hisdwb'][nn,:,:,:]=fields['dwb']
        his['hiswa'][nn,:,:,:]=fields['wa']
        his['hiswb'][nn,:,:,:]=fields['wb']
    else:
        for i in range(nh):
            his['hiswa'][i,:,:,:]=his['hiswa'][i+1,:,:,:]
            his['hiswb'][i,:,:,:]=his['hiswb'][i+1,:,:,:]
            his['hisdwa'][i,:,:,:]=his['hisdwa'][i+1,:,:,:]
            his['hisdwb'][i,:,:,:]=his['hisdwb'][i+1,:,:,:]
        his['hisdwa'][nh,:,:,:]=fields['dwa']
        his['hisdwb'][nh,:,:,:]=fields['dwb']
        his['hiswa'][nh,:,:,:]=fields['wa']
        his['hiswb'][nh,:,:,:]=fields['wb'] 
        
    
    return his



def andersonmixing(fields,dens,his):
    """
    function does Anderson mixing iteration

    """
    da=np.zeros((nh+1,nh+1))        
    ua=np.zeros((nh,nh))
    va=np.zeros(nh)
    cma=np.zeros(nh) 
    wta=np.zeros(nm)
    wtb=np.zeros(nm)
        
    lam=1.0
    
    # if(ens=='grand'):        
    #     dwa=-fields['wa']+fields['wb']+chi*(2*dens['rhoB']-1)
    #     dwb=fields['wa']-fields['wb']-chi*(1-2*dens['rhoA'])
    # elif(ens=='canonical'):
    #     dwa=chi*(dens['rhoB']-fB)+1/2*(fields['wb']-fields['wa'])
    #     dwb=chi*(dens['rhoA']-fA)+1/2*(fields['wa']-fields['wb'])
    
            
    for i in range(nh+1):
        for j in range(nh+1):
            da[i,j]=1/prod(nm)*np.sum(his['hisdwa'][i,:,:,:]*his['hisdwa'][j,:,:,:])+\
                    +1/prod(nm)*np.sum(his['hisdwb'][i,:,:,:]*his['hisdwb'][j,:,:,:])
    
    for i in range(1,nh+1):
          va[i-1]=da[0,0]-da[0,i]
          for j in range(1,nh+1):
              ua[i-1,j-1]=da[0,0]+da[i,j]-da[0,i]-da[0,j]
    
   
    # ui=np.linalg.inv(ua)   
    # cma=np.matmul(ui,va)
    
    q,r=np.linalg.qr(ua)
    b=np.dot(q.T,va)
    for i in range(nh):
        s=0
        for j in range(i):
            s=s+cma[nh-1-j]*r[nh-1-i,nh-1-j]    
        cma[nh-1-i]=(b[nh-1-i]-s)/r[nh-1-i,nh-1-i]
   
    cmaa=np.zeros(nh+1)
    cmaa[0]=1-np.sum(cma)
    for i in range(1,nh+1):
        cmaa[i]=cma[i-1]
    
    wta=0.0
    wtb=0.0
    for i in range(0,nh+1):
        wta=wta+cmaa[i]*(his['hiswa'][i,:,:,:]+lam*his['hisdwa'][i,:,:,:])
        wtb=wtb+cmaa[i]*(his['hiswb'][i,:,:,:]+lam*his['hisdwb'][i,:,:,:])
        # print('wta',i,1/prod(nm)*np.sum(wta))
    # wta=wta+cmaa[nh]*(fields['wa']+lam*dwa)
    # wtb=wtb+cmaa[nh]*(fields['wb']+lam*dwb)
    # wta=wta+cmaa[nh]*(fields['wa']+lam*his['hisdwa'][nh,:,:,:])
    # wtb=wtb+cmaa[nh]*(fields['wb']+lam*his['hisdwb'][nh,:,:,:])
    # print('wta',i+1,1/prod(nm)*np.sum(wta))
    fields['wa']=wta
    fields['wb']=wtb
    
    # if(ens=='canonical'):    
    #     fields['wa']=fields['wa']-np.sum(fields['wa'])*1/prod(nm)
    #     fields['wb']=fields['wb']-np.sum(fields['wb'])*1/prod(nm)
        
    return fields

def error(fields):
    """
    Error function calculates error

    Parameters
    ----------
    dwa : TYPE
        DESCRIPTION.
    dwb : TYPE
        DESCRIPTION.
    er : TYPE
        DESCRIPTION.

    Returns
    -------
    er : TYPE
        DESCRIPTION.

    """
    b1=1/prod(nm)*np.sum(fields['dwa']**2)
    b2=1/prod(nm)*np.sum(fields['dwb']**2)
        
    er=1/chi*(b1+b2)**(1/2) 
    return er

def selfconsistent(dens,fields,fq,bq,his,dif,L,z):
    """
    function solves self-consistent equations

    """
    er=1
    nn=0

    while er>10**(-9) and nn<200:
        # print(nn)
        fq=fequation(fq,fields,dif)
        bq=bequation(bq,fields,dif)
        
        # partition function of a single chain        
        for j in range(nchains):
            Q[j]=1.0/prod(nm)*np.sum(fq[j,ns,:,:,:])
        

        rhoAch[:,:,:,:]=0.0
        rhoBch[:,:,:,:]=0.0
        
        if(ens=='grand'):
            if(nblocks==2):
                for j in range(0,nm[0]):
                    for k in range(0,nm[1]):
                        for l in range(0,nm[2]):
                            rhoAch[0,j,k,l]=z[0]*ds*simps(fq[0,0:ns+1,j,k,l]*bq[0,0:ns+1,j,k,l])
                            rhoAch[1,j,k,l]=z[1]*ds*simps(fq[1,0:nl+1,j,k,l]*bq[1,0:nl+1,j,k,l])
                        
                            rhoBch[1,j,k,l]=z[1]*ds*simps(fq[1,nl:ns+1,j,k,l]*bq[1,nl:ns+1,j,k,l])
                            rhoBch[2,j,k,l]=z[2]*ds*simps(fq[2,0:ns+1,j,k,l]*bq[2,0:ns+1,j,k,l])
                                                   
            
            elif(nblocks==3):
                for j in range(0,nm[0]):
                    for k in range(0,nm[1]):
                        for l in range(0,nm[2]):
                            rhoAch[0,j,k,l]=z[0]*ds*simps(fq[0,0:ns+1,j,k,l]*bq[0,0:ns+1,j,k,l])
                            rhoAch[1,j,k,l]=z[1]*ds*simps(fq[1,0:nl+1,j,k,l]*bq[1,0:nl+1,j,k,l])
                            rhoAch[2,j,k,l]=z[2]*ds*(simps(fq[2,2*nl:ns+1,j,k,l]*bq[2,2*nl:ns+1,j,k,l])+\
                                                                simps(fq[2,0:nl+1,j,k,l]*bq[2,0:nl+1,j,k,l]))
                            rhoAch[3,j,k,l]=z[3]*ds*simps(fq[3,nl:2*nl+1,j,k,l]*bq[3,nl:2*nl+1,j,k,l])    
                            rhoAch[4,j,k,l]=z[4]*ds*simps(fq[4,nl:ns+1,j,k,l]*bq[4,nl:ns+1,j,k,l])
                            
                                                        
                            rhoBch[1,j,k,l]=z[1]*ds*simps(fq[1,nl:ns+1,j,k,l]*bq[1,nl:ns+1,j,k,l])
                            rhoBch[2,j,k,l]=z[2]*ds*simps(fq[2,nl:2*nl+1,j,k,l]*bq[2,nl:2*nl+1,j,k,l])
                            rhoBch[3,j,k,l]=z[3]*ds*(simps(fq[3,2*nl:ns+1,j,k,l]*bq[3,2*nl:ns+1,j,k,l])+\
                                                                simps(fq[3,0:nl+1,j,k,l]*bq[3,0:nl+1,j,k,l]))
                            rhoBch[4,j,k,l]=z[4]*ds*simps(fq[4,0:nl+1,j,k,l]*bq[4,0:nl+1,j,k,l])
                            rhoBch[5,j,k,l]=z[5]*ds*simps(fq[5,0:ns+1,j,k,l]*bq[5,0:ns+1,j,k,l])
                            
                            
                            
        if(ens=='canonical'): 
            if(nblocks==2):
                for j in range(0,nm[0]):
                    for k in range(0,nm[1]):
                        for l in range(0,nm[2]):
                                rhoAch[0,j,k,l]=vol0[0]*1.0/Q[0]*ds*simps(fq[0,0:ns+1,j,k,l]*bq[0,0:ns+1,j,k,l])
                                rhoAch[1,j,k,l]=vol0[1]*1.0/Q[1]*ds*simps(fq[1,0:nl+1,j,k,l]*bq[1,0:nl+1,j,k,l])
                        
                                rhoBch[1,j,k,l]=vol0[1]*1.0/Q[1]*ds*simps(fq[1,nl:ns+1,j,k,l]*bq[1,nl:ns+1,j,k,l])
                                rhoBch[2,j,k,l]=vol0[2]*1.0/Q[2]*ds*simps(fq[2,0:ns+1,j,k,l]*bq[2,0:ns+1,j,k,l])
                                
                                    
                                
            elif(nblocks==3):
                for j in range(0,nm[0]):
                    for k in range(0,nm[1]):
                        for l in range(0,nm[2]):
                            rhoAch[0,j,k,l]=vol0[0]*1.0/Q[0]*ds*simps(fq[0,0:ns+1,j,k,l]*bq[0,0:ns+1,j,k,l])
                            rhoAch[1,j,k,l]=vol0[1]*1.0/Q[1]*ds*simps(fq[1,0:nl+1,j,k,l]*bq[1,0:nl+1,j,k,l])
                            rhoAch[2,j,k,l]=vol0[2]*1.0/Q[2]*ds*(simps(fq[2,2*nl:ns+1,j,k,l]*bq[2,2*nl:ns+1,j,k,l])+\
                                                                simps(fq[2,0:nl+1,j,k,l]*bq[2,0:nl+1,j,k,l]))
                            rhoAch[3,j,k,l]=vol0[3]*1.0/Q[3]*ds*simps(fq[3,nl:2*nl+1,j,k,l]*bq[3,nl:2*nl+1,j,k,l])
                            rhoAch[4,j,k,l]=vol0[4]*1.0/Q[4]*ds*simps(fq[4,nl:ns+1,j,k,l]*bq[4,nl:ns+1,j,k,l])
                            
                            rhoBch[1,j,k,l]=vol0[1]*1.0/Q[1]*ds*simps(fq[1,nl:ns+1,j,k,l]*bq[1,nl:ns+1,j,k,l])
                            rhoBch[2,j,k,l]=vol0[2]*1.0/Q[2]*ds*simps(fq[2,nl:2*nl+1,j,k,l]*bq[2,nl:2*nl+1,j,k,l])
                            rhoBch[3,j,k,l]=vol0[3]*1.0/Q[3]*ds*(simps(fq[3,2*nl:ns+1,j,k,l]*bq[3,2*nl:ns+1,j,k,l])+\
                                                                simps(fq[3,0:nl+1,j,k,l]*bq[3,0:nl+1,j,k,l]))
                            rhoBch[4,j,k,l]=vol0[4]*1.0/Q[4]*ds*simps(fq[4,0:nl+1,j,k,l]*bq[4,0:nl+1,j,k,l]) 
                            rhoBch[5,j,k,l]=vol0[5]*1.0/Q[5]*ds*simps(fq[5,0:ns+1,j,k,l]*bq[5,0:ns+1,j,k,l])
                                                            
                            
                                
        # iterations to solve self-consistent equations
        
        dens['rhoA']=np.sum(rhoAch,axis=0)
        dens['rhoB']=np.sum(rhoBch,axis=0)
        dens['rho']=dens['rhoA']+dens['rhoB']
        dens['rhoAch']=rhoAch
        dens['rhoBch']=rhoBch
        # print(np.sum(dens['rho'])/prod(nm))
        
        if(ens=='grand'):
            fields['dwa']=-fields['wa']+fields['wb']+chi*(2*dens['rhoB']-1)
            fields['dwb']=fields['wa']-fields['wb']-chi*(1-2*dens['rhoA'])
        elif(ens=='canonical'):
            # fields['dwa']=chi*(dens['rhoB']-fB)+1/2*(fields['wb']-fields['wa'])
            # fields['dwb']=chi*(dens['rhoA']-fA)+1/2*(fields['wa']-fields['wb'])
            fields['dwa']=-fields['wa']+fields['wb']+chi*(2*dens['rhoB']-1)
            fields['dwb']=fields['wa']-fields['wb']-chi*(1-2*dens['rhoA'])
            
        
        if(er>5*1.0e-3 or nn<20):
            his=history(nn,fields,his)
            fields=simplemixing(fields,dens)    
            er=error(fields)                      
            print('er',er)
        else:
            his=history(nn,fields,his)
            fields=andersonmixing(fields,dens,his)
            er=error(fields)           
            print('era',er)
        nn=nn+1   
            
    # return rhoA,rhoB,wa,wb,Q,er,fq,bq,fA,fB
    return dens,fields,Q


def floryhuggins(z,chi,fa0):
    vol=np.zeros(nchains)
    x=1.0
    fa=fa0
    fg=0.0
    a=0
    while x>10.0**(-17): 
        a=0
        for i in range(nchains-1):
            a=a+z[i]*exp(-chi*cf[i]*(1.0-2.0*fa))
        vol[nchains-1]=1.0/(1.0+a)
        for i in range(nchains-1):
            vol[i]=vol[nchains-1]*z[i]*exp(-chi*cf[i]*(1.0-2.0*fa))
        fao=fa
        fa=0.0
        for i in range(nchains-1):
            fa=fa+cf[i]*vol[i]
        x=abs(fa-fao)

    
    fg=-1.0+log(vol[nchains-1])+chi*fa**2
    file.write('Fh: '+str(fg)+'\n')
    print('fg',fg)
    return vol,fa,fg


def freeenergy(dens,fields,fq,bq,dif,D,z):
    L=cal_L(D)
    dif=dif_cal(dif,L)        
    dens,fields,Q=selfconsistent(dens,fields,fq,bq,his,dif,L,z)
    
    if(ens=='grand'):
        Fg=-1.0+chi*1.0/prod(nm)*np.sum(dens['rhoA']**2)-1.0/prod(nm)*np.sum(fields['wb'])
        # for i in range(nchains):
        #     Fg=Fg-z[i]*Q[i]
        print(z,Q)
        print(sum(z[:]*Q[:]))
        volm=np.zeros(nchains)
        volm[:]=z[:]*Q[:]
        fa=np.sum(cf[:]*volm[:])
        F=chi*1.0/prod(nm)*np.sum(dens['rhoA']**2)-1.0/prod(nm)*np.sum(fields['wb'])-1.0
        for i in range(nchains):
            F=F+volm[i]*(log(volm[i]))-volm[i]*log(Q[i])
        print('fa',fa)
        print('F,Fg',F,Fg) 
        
    elif(ens=='canonical'):
        F=chi*1.0/prod(nm)*np.sum(dens['rhoA']**2)-1.0/prod(nm)*np.sum(fields['wb'])-1.0
        print('first term',chi*1.0/prod(nm)*np.sum(dens['rhoA']**2))
        print('second term',-1.0/prod(nm)*np.sum(fields['wb']))
        for i in range(nchains):
            F=F+vol0[i]*(log(vol0[i]))-vol0[i]*log(Q[i])
        # Fg=-1.0+chi*1.0/prod(nm)*np.sum((dens['rhoA'])*(dens['rhoA']))-1.0/prod(nm)*np.sum(fields['wb'])-\
        #     -log(Q[nchains-1])+log(vol[nchains-1])
        Fg=-1.0+chi*1.0/prod(nm)*np.sum(dens['rhoA']**2)-1.0/prod(nm)*np.sum(fields['wb'])
        zc=np.zeros(nchains)
        # zc[:]=vol[:]/Q[:]*Q[nchains-1]/vol[nchains-1]
        zc[:]=vol0[:]/Q[:]
        print('Q',Q)
        print(zc)
        print('Fg',Fg)
        filez=open('z-file.txt','w')
        filez.write(str(zc[0])+'\n')
        filez.write(str(zc[1])+'\n')
        filez.write(str(zc[2])+'\n')
        filez.close()
        print('z',zc[0])
        print('z',zc[1])
        print('z',zc[2])
        print('z',zc[3])
        print('z',zc[4])
        print('z',zc[5])
    
    # phiax=np.zeros(nm[0])
    # phiax[:]=dens['rhoB'][:,0,0]
    # plt.plot(phiax[:])
    # plt.show()
    # way=dens['rhoB'][:,:,0]
    # plt.contour(way)   
    # plt.show() 
    
    np.save(wa_savefile, fields['wa'])
    np.save(wb_savefile, fields['wb'])
    if(ens=='grand'):
        a=Fg
    elif(ens=='canonical'):
        a=F 
        
    return a










