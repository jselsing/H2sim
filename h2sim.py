#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import random
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splrep, splev
from PyAstronomy import pyasl

SN = 30. # Signal to noise
z = 2.3538 # Redshift
COMPONENTS = list(np.array([-0.0015, -0.0004, 0.000, 0.0003, 0.0006, 0.0010]) + z)
WEIGHTS = [0.5, 0.3, 1.0, 0.2, 0.8, 0.5]
TEMP = 100 # Temperature of H2
RES = 0.2 # Resolution of spectrum in AA
NTOTH2s = [20.5] # Total column density of H2
#NTOTH2s = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

NROT = [0, 1, 2, 3] # Rotational levels to consider, maximum 3
DOH2, DOH2S = False, False # Add H2, H2* to spectrum
DOEXC = False # Add finstructure and excited lines
WLRANGE = [900., 3000.] # Wavelength range to plot/write out
# MH2Ss = [0.00, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]  #Multiplier of Draine model with NH2* = 6.73E+16
MH2Ss = [0.0]

# Column densities, need to follow the same naming convention as in atom.dat
# N = {'HI' : 21.95, 'SiII' : 14.20}



N = {'HI' : 21.95, 'FeII' : 15.29, 'MnII' : 13.2, 'NV':14.8,
     'SiII': 16.3, 'SII' : 15.2, 'CIV': 12.8, 'OI': 17,
     'CII': 16.5, 'NiII': 14.2 , 'SiIV': 12.3, 'AlII': 14.3,
     'AlIII':13.3, 'CI': 13.41, 'ZnII': 13.47, 'CrII': 13.75,
     'MgII':18.0, 'MgI':16.0}


# Column densities of excited transition, need to be the same name as in
# atom_excited.dat
Nexc = {'FeIIa' : 13.32, 'FeIIb':13.11, 'OIa':15.29,
        'SiIIa': 14.31, 'NiIIa':13.23}



N_tot = {}
for n in N:
    S = [0]*len(WEIGHTS)
    for ii, w in enumerate(WEIGHTS):
        S[ii] = w*10**N[n]
    N_tot[n] = np.log10(np.sum(S))
    S = 0
print(N_tot)
# Broadening parameter in km/s
b = {'ALL': 25, 'H2' : 10}

# Constants
m_e = 9.1095e-28
e = 4.8032e-10
c = 2.998e10 # in cm/s

#==============================================================================
# Voigt profile
#==============================================================================
def H(a,x):
    P = x**2 
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a / np.sqrt(np.pi) /\
    P * ( H0 ** 2 * (4. * P**2 + 7. * P + 4. + Q) - Q - 1.0 )

#==============================================================================
# Add an absorption line
#==============================================================================

def addAbs(wls, nion, lamb, f, gamma, broad, z):
    C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
    a = lamb * 1E-8 * gamma / (4.*np.pi * broad)   
    dl_D = broad/c * lamb   
    x = (wls/(z+1.0) - lamb)/dl_D+0.01

    # Optical depth
    tau = C_a * nion * H(a,x)
    return np.exp(-tau)

#==============================================================================
# Molecular hydrogen
#==============================================================================
# Calculate relative intensities of first three rotational levels
def fNHII(T, J):
    # Para molecular hydrogen
    if J % 2 == 0:
        I = 0
    # Ortho molecular hydrogen
    else:
        I = 1
    # Statistical weights
    gj = (2*J + 1) * (2*I + 1)
    # Energu difference between the different states
    dE0J = {0:0, 1: 170.5, 2:510,  3:1020}
    nj = gj * np.exp(-dE0J[J] / T)
    return nj
#    return NH2

for NTOTH2 in NTOTH2s:
  for MH2S in MH2Ss:
    wls = np.arange(WLRANGE[0]*(1+z), WLRANGE[1]*(1+z), RES)
    spec = np.ones(len(wls))
    dspec = random.normal(0, 1./SN, len(wls))  

    nJ, NH2 = [], {}
    for J in NROT:
        nJ.append(fNHII(TEMP, J))

    for nj in NROT:
        NH2['H2J%i' %nj] = 10**NTOTH2/sum(nJ)*nJ[nj]

    #==============================================================================
    # Excited Molecular hydrogen
    #==============================================================================
    # Read in excited lines
    if DOH2S == True:
        f = open('h2abs_spec_n3_b5_1e4_1000', 'r')
        fcont = [g for g in f.readlines() if not g.startswith('#')]
        f.close()

        h2swl, modspec, tauspec = [], [], []
        for fin in fcont:
            lines = fin.split()
            if lines != []:
                h2swl.append(1/float(lines[0])*1E8*(1+z))
                modspec.append(float(lines[1]))
                tauspec.append(float(lines[2]))

        # Normalize to 1 and return order
        tauspec = np.array(tauspec[::-1])
        h2swl = np.array(h2swl[::-1])
        # Absorption spectrum, with multiplier, above Lyalpha only
#        tauspec[h2swl < 1220] = 1-1.9845E-01
        modscale = np.exp(-1*(tauspec-1.9845E-01)*MH2S)
#        modscale[h2swl < ] = 1
        # Smooth to the requested RESOLUTION
        modsmo = gaussian_filter1d(modscale, 0.07/RES)
        # Interpolate to our wavelength grid
        tckp = splrep(h2swl, modsmo, s = 3, k = 2)     
        modint = splev(wls, tckp) 
        spec *= modint

    #==============================================================================
    # Normal Ions
    #==============================================================================
    af = open ('atom.dat', 'r')
    atom = af.readlines()
    af.close()
    for zz, ww in list(zip(COMPONENTS, WEIGHTS)):
        print(zz, ww)
        for a in atom:
            a = a.split()
            if a[0] in N.keys():
                lamb = float(a[1])
                f = float(a[2])
                gamma = float(a[3])
                if a[0] in b:
                    broad = b[a[0]] * 1E5
                else:
                    broad = b['ALL'] * 1E5
                nion = 10**N[a[0]]
                # print(a[0], 10**N[a[0]]*ww)
                spec *= addAbs(wls, nion*ww, lamb, f, gamma, broad, zz)

        #==============================================================================
        # Excited transitions
        #==============================================================================

        if DOEXC == True:
            af = open ('atom_excited.dat', 'r')
            atomexc = af.readlines()
            af.close()

            for a in atomexc:
                a = a.split()
                if a[0] in Nexc.keys():
                    lamb = float(a[1])
                    f = float(a[2])
                    gamma = float(a[3])
                    if a[0] in b:
                        broad = b[a[0]] * 1E5
                    else:
                        broad = b['ALL'] * 1E5
                    nion = 10**Nexc[a[0]]
                    # print(a[0], Nexc[a[0]]*ww)
                    spec *= addAbs(wls, nion*ww, lamb, f, gamma, broad, zz)

    #==============================================================================
    # Add H2
    #==============================================================================

    if DOH2 == True:
      for a in atom:
        a = a.split()
        if a[0] in NH2.keys():
            lamb = float(a[1])
            f = float(a[2])
            gamma = float(a[3])
            broad = b['H2'] * 1E5
            nion = NH2[a[0]]
            spec *= addAbs(wls, nion, lamb, f, gamma, broad, z)

    spec[wls < 912*(1+z)] = 0
     #==============================================================================
    # Write out
    #==============================================================================
    if DOH2 == True and DOH2S == False:
        f = open('Synspec_H2_%s_z%.2f.txt' %(NTOTH2, z), 'w')
    elif DOH2S == True:
        f = open('Synspec_H2_%s_z%.2f_H2S_%.2f.txt' %(NTOTH2, z, MH2S), 'w')
    else:

        f = open('Synspec_z%s.txt' %(z), 'w')
    f.write('#Simulated, normalized, high-resolution spectrum of a GRB\n')
    f.write('#S/N = %.1f\n' %SN)
    f.write('#NH = %.2f\n' %N['HI'])
    f.write('#NH2 = %.2f at T = %i K\n' %(NTOTH2, TEMP))
    f.write('#Redshift = %.4f\n' %(z))
    spec_conv, fwhm = pyasl.instrBroadGaussFast(wls, spec, 10000,
          edgeHandling="firstlast", fullout=True)
    for wl, s in zip(wls, spec_conv+dspec):
        f.write('%.2f\t%.3f\n' %(wl, s))
    f.close()

    #==============================================================================
    # Plot out
    #==============================================================================
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wls, spec_conv+dspec)

    ax.plot(wls, spec_conv*0, '--', color = 'black')
    ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
    ax.set_ylabel(r'$\rm{Normalized\,Flux}$')
    ax.set_ylim(-2./SN, 1+4./SN)#, yerr=self.onederro[arm])
    if DOH2 == True and DOH2S == False:
        ax.set_xlim(900.*(1+z), 1250.*(1+z))#, yerr=self.onederro[arm])
    else:
        ax.set_xlim(1150.*(1+z), 1650.*(1+z))#, yerr=self.onederro[arm])

    if DOH2 == True and DOH2S == False:
        fig.savefig('Synspec_H2_%s_z%.2f.pdf' %(NTOTH2, z))
    elif DOH2S == True:
        fig.savefig('Synspec_H2_%s_z%.2f_H2S_%.2f.pdf' %(NTOTH2, z, MH2S))
    else:
        fig.savefig('Synspec_z%s.pdf' %(z))
    plt.show()
    plt.close(fig)