#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:40:33 2023

@author: ezequiel
"""

# ------------------------------------------------------------------------------
# Program:
# -------
#
# This program computes the stationary configurations for an electric field
# between a capacitor with a non-trivial paralell plate array.
# The capacitor consist of one very long plate which is parallel to an
# array of small finger of lenght "l" with separations "w" among them.
# 
# The space between is filled with a dielectric material, a PDLC smaple in this 
# case. One then is interested in the electric field in the capacitor, since 
# it could trigger more rapid orientation times for the polar molecules 
# composing the PDLC, producing a more rapid switching between the opaque and
# transparent configurations.
#
# The code also compute the switching times between two different alignmnets
# of the PDL moleclues
#
# ------------------------------------------------------------------------------


host = 'local'

if host == 'local':
    #root_home = '/home/ezequiel/'                                  # Local
    root_home = '/Users/ezequielboero/'                            # PC Local path
    fileProj  = root_home+'Projects/AGN_Auger/'                    # Folder of work for the Article
    data      = fileProj+'Auger_data_I/'                           # Events and Fluxes of Cosmic Rays
    graficos  = fileProj+'graphics/'                               # All the graphics

elif host == 'IATE':
    root_home = '/home/zboero/'                                    # Clemente y IATE
    fileProj  = root_home+'Projects/CMB/'                          # Folder of work for the article
    data      = fileProj+'data/'                                   # Folder with the data
    graficos  = fileProj+'graphics/'                               # Folder with the plots
    #

import numpy as np
import pandas as pd
import numba
from numba import jit, float64, int64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
from matplotlib.gridspec import GridSpec
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special as sp
import time

# ----------------------------------------------------------------------------
# --     Parameters	 --------------------------------------------
# ----------------------------------------------------------------------------

c_luz     = 2.99792458e8          # speed of light in [m/s]
mu_0      = 1.256637e-6           # magnetic permeability of vaccunm in [H/m]
eps_0     = 1.0/(c_luz**2 * mu_0) # Electric permittivity of the vacuum in [F/m]
poise2Newt = 0.1                  #Conversion factor from Poise to [N * s / m^2]
#
# --- Properties of the polymer substrate (hard exposy resine, cured with UV) ...
eps_polym = 3.5                   # Dielectric constant of the polymer at about 25 Celsius degree
d_gap     = 3.0e-6                # Gap between ITO and finger electrodes in [m] (5e-6 is the minimum possible)
#
# --- Properties of the electrodes ...
V_ITO     = 0.0                   # Potential of the ITO electrode (grounded) in [V]
#
V_fingA   = 12.0                  # Potential of the electrodes A in [V]
V_fingB   = -V_fingA              # Potential of the electrodes B in [V]
L_fing    = 1.0e-6               # Width of the electrodes in [m]   (btw 13e-6 m and 25e-6 m )
fing_gap  = 3.0e-6               # Gap between electrodes in [m]    (10e-6m is the minimum possible)
Lb2La     = fing_gap / L_fing     # Gap between electrodes in units of the width of the lenght of the fingers
#
# --- Properties of the glass (generic lime glass) ...
eps_glass = 7.2                  # Dielectric constant of the glass at about 25 Celsius degree
L_glass   = 1.0e-3                # Thickness of the glass in [m] (it is customized as 1mm thickness)
#
# --- Properties of the air (the external medium surronding the setting and providing boundary conditions at infinity) ...
eps_air   = 1.00058986            # Dielectric constant of the air at normal pressure and temperatue
#                                 # Finally not used, since it is not neccesary in this analysis.
#
# --- The quotient of conductivities betwwen the droplet and the polymer...
# This is neccesary to compute the attenuation of the electric field inside the droplet...
#
sigLC_sigPol = 9.0                 # Generic value obtained from ref [14] of paper... (and used for TL213)
#sigLC_sigPol = 14.0?                 # Generic value obtained from ref [14] of paper... (and used for E7)
#
# --- Properties of the Liquid Crystal within the droplets (nematic phase)...
#eps_parall =                      # Dielectric constant of the Liquid Crystal in the parallel direction to the E field
#eps_perp =                        # Dielectric constant of the Liquid Crystal in the perpendicula direction to the E field
#delta_eps = eps_parall - eps_perp #
delta_eps = 5.2                    # Dielectric anisotropy... For LC TL213 is 5.2
#delta_eps = 13.8                   # Dielectric anisotropy... For E7 is 13.8
gamma     = 1.0 * poise2Newt      # Rotational viscosity of the polymer in [N * s / m^2] 0.1 es lo que lleva de Poise to Ns/m^2
K         = 1.27e-11               # Elastic constant of the lyquid crystal in [N]
R         = 1.0e-6                # Radius of the liquid crystal droplet in [m]
l = 1.06                       # elongation parameter that takes into account the elongation of droplets
#A = 100.0 * (l**2 - 1)/(l**2 + 1) # droplets with weak surface anchoring eqn. (2) of ref. below
A = (l**2 - 1)                    # droplets with weak surface anchoring eqn. (2) of ref. below
#B = 6.2                          # droplets with strong surface anchoring eqn. (3) of ref. below
D = 2.0 * R                       # mean diameter of the droplets....
C = 3.0 / (2.0 + sigLC_sigPol)    # attenuation factor
del_eps_eps0_E_th2 = A * K / (R**2) # According to eqn. (1) of "Magneto-and Electrooptical Measurementsof
#                                            # This is the energy minimum needed to activate the LC...
#
# Below is the switching times without FOA.
T_dec0  = 2.0 * gamma / del_eps_eps0_E_th2   # The previous expression 2.0 * gamma / ( (K/R**2) )
#
# Vizualtion configuration:
ms_lim = 1.0                           # Maximun range of T_switch in miliseconds...



# Below we work with a rectangular grid and re-scaled dimensions for simplicity:
# after computation we will restore the dimensions of lenghts...
#
points_x = 200
points_y = 200
#
edge_x = np.linspace(0, 1, points_x)
edge_x = edge_x * (L_fing + fing_gap)
edge_y = np.linspace(-1, 1, points_y)
edge_y[0:int(points_y/2)] = edge_y[0:int(points_y/2)] * L_glass
edge_y[int(points_y/2):] = edge_y[int(points_y/2):] * d_gap
#
# ---- We make up the grid...
xv, yv = np.meshgrid(edge_x, edge_y)

dx = edge_x[1] - edge_x[0]
dy_cell  = edge_y[int(points_y/2)+1] - edge_y[int(points_y/2)]
dy_glass = edge_y[1] - edge_y[0]

# ----------------------------------------------------------------------------
# --     We implement the relaxation method to find the potential     -------
# ----------------------------------------------------------------------------

#
Lx = points_x                    # Points in the x direction
Ly = points_y                    # Points in the y direction
Ly2 = int( Ly/2 )
#
la  = int( Lx / (1 + Lb2La ) )    # position at the end of the finger_A electrode...
lb  = int(la * Lb2La)             # Lenght of the gap between finger electrodes...
la2 = int( la /2 )                # 1/2 of Finger A is at the beggining and the other 1/2 at the end of the grid...
#
@numba.jit(nopython=True)#( float64[:,:]( float64[:,:], float64 ), nopython=True)
def relaxation( Phi, omega, l2_target, maxiter ):
    #
    ite = 0
    l2norm = 1.0
    while l2norm > l2_target and ite < maxiter:
        Phi_n = Phi.copy()
        # Update the solution at interior points.
        for i in range(1, Lx):
            for j in range(1, Ly2):
                Phi[j,i] = ( 1.0 - omega ) * Phi[j,i] + omega * 0.5 * (\
                            ( Phi[j+1,i] + Phi[j-1,i] ) * dx**2 + ( Phi[j,i+1] + Phi[j,i-1] ) * dy_glass**2 ) /\
                            ( dy_glass**2 + dx**2 )

            for j in range(Ly2+1,Ly):
                Phi[j,i] = ( 1.0 - omega ) * Phi[j,i] + omega * 0.5 * (\
                            ( Phi[j+1,i] + Phi[j-1,i] ) * dx**2 + ( Phi[j,i+1] + Phi[j,i-1] ) * dy_cell**2 ) /\
                            ( dy_cell**2 + dx**2 )
        #
        ##ITO electrode at constant potential
        Phi[-1,:]    = V_ITO              # upper_y (ITO) (top of the rectangle)

        ##Finger electrodes at constant potential
        Phi[Ly2,:la2]  = V_fingA                          # lower_y  (fingerA) (bottom of the rectangle)
        Phi[Ly2,(la2+lb):] = V_fingB                          # lower_y  (fingerB) (bottom of the rectangle)

        ##2nd-order Neumann B.C. along finger gap
        Phi[Ly2,la2:(la2+lb)] = dx**2 * ( (eps_polym / dy_cell)  * Phi[Ly2+1,la2:(la2+lb)] +\
                                          (eps_glass / dy_glass) * Phi[Ly2-1,la2:(la2+lb)] ) / \
                                        ( (eps_polym / dy_cell)  * ( dx**2 + dy_cell**2 ) +\
                                          (eps_glass / dy_glass) * ( dx**2 + dy_glass**2 ) ) + \
                                0.5 * ( (eps_polym * dy_cell + eps_glass * dy_glass) * \
                                        ( Phi[Ly2,(la2+1):(la2+lb+1)] + Phi[Ly2,(la2-1):(la2+lb-1)] ) )/ \
                                      ( (eps_polym / dy_cell)  * ( dx**2 + dy_cell**2 ) +\
                                        (eps_glass / dy_glass) * ( dx**2 + dy_glass**2 ) )

        ##2nd-order Neumann B.C. along the y-axis at the middle of the finger electrodes
        Phi[1:Ly2,0]     = (1.0/3.0) * ( 4.0 * Phi[1:Ly2,1] - Phi[1:Ly2,2] )
        Phi[1:Ly2,-1]    = (1.0/3.0) * ( 4.0 * Phi[1:Ly2,-2] - Phi[1:Ly2,-3] )
        Phi[Ly2+1:Ly,0]  = (1.0/3.0) * ( 4.0 * Phi[Ly2+1:Ly,1] - Phi[Ly2+1:Ly,2] )
        Phi[Ly2+1:Ly,-1] = (1.0/3.0) * ( 4.0 * Phi[Ly2+1:Ly,-2] - Phi[Ly2+1:Ly,-3] )

#        Phi[1:Ly2,0]     = 0.5 * ( 2.0 * Phi[1:Ly2,1] * dy_glass**2 + ( Phi[2:(Ly2+1),0] + Phi[0:(Ly2-1),0] ) * dx**2  ) / \
#                                ( dy_glass**2 + dx**2 )                                # lower_x (left of the rectangle)
#        Phi[1:Ly2,-1]    = -Phi[1:Ly2,0]
##        Phi[1:Ly2,-1]    = 0.5 * ( 2.0 * Phi[1:Ly2,-2] * dy_glass**2 + ( Phi[2:(Ly2+1),-1] + Phi[0:(Ly2-1),-1] ) * dx**2  ) / \
##                                ( dy_glass**2 + dx**2 )                                # lower_x (left of the rectangle)

##        Phi[(Ly2+1):-1,0]    = V_fingA * ( 1.0  - yv[(Ly2+1):-1,0] / d_gap )         # This fails to work properly...
#        Phi[(Ly2+1):-1,0]    = 0.5 * ( 2.0 * Phi[(Ly2+1):-1,1] * dy_cell**2 + ( Phi[(Ly2+2):,0] + Phi[Ly2:-2,0] ) * dx**2  ) / \
#                                ( dy_cell**2 + dx**2 )                                # lower_x (left of the rectangle)
#        Phi[(Ly2+1):-1,-1]   = - Phi[(Ly2+1):-1,0]
##        Phi[(Ly2+1):-1,-1]   = 0.5 * ( 2.0 * Phi[(Ly2+1):-1,-2] * dy_cell**2 + ( Phi[(Ly2+2):,-1] + Phi[Ly2:-2,-1] ) * dx**2  ) / \
##                                ( dy_cell**2 + dx**2 )                                # lower_x (left of the rectangle)


        ##2nd-order Neumann B.C. along glass at infinity
#        Phi[0,0]               = ( Phi[1,0] * dx**2 + Phi[0,1] * dy_glass**2 ) / \
#                               ( dy_glass**2 + dx**2 )            # The field vanishes at the bottom of the glass (sim to - infty)
#        Phi[0,-1]              = ( Phi[1,-1] * dx**2 +  Phi[0,-2] * dy_glass**2 ) / \
#                               ( dy_glass**2 + dx**2 )            # The field vanishes at the bottom of the glass (sim to - infty)
        Phi[0,0]               = Phi[1,0]                          # This is first order but works fine
        Phi[0,-1]              = Phi[1,-1]                         # This is first order but works fine
#        Phi[0,1:-1]            = Phi[1,1:-1]
        Phi[0,1:-1]            = 0.5 * ( 2.0 * Phi[1,1:-1] * dx**2 + ( Phi[0,2:] + Phi[0,:-2] ) * dy_glass**2 ) / \
                               ( dy_glass**2 + dx**2 )            # The field vanishes at the bottom of the glass (sim to - infty)


        # Compute the relative L2-norm of the difference.
        l2norm = np.sqrt( np.sum( ( Phi - Phi_n )**2 ) / np.sum( Phi_n**2 ) )
        ite = ite + 1
 
    return Phi, l2norm, ite
    

# ----------------------------------------------------------------------------
# --     We solve for the potential Phi     ---------------------------------
# ----------------------------------------------------------------------------

# --- Initial condition for starting the iterations...
Phi0  = np.ones( (len(edge_x), len(edge_y)) )
Phi0[Ly2:,:(la2+1)]  =  V_fingA * ( 1.0  - yv[Ly2:,:(la2+1)] / d_gap )
Phi0[Ly2:,(la2+lb):] =  V_fingB * ( 1.0  - yv[Ly2:,(la2+lb):] / d_gap )
Phi0[Ly2:,(la2+1):(la2+lb+1)] = V_fingA * ( 1.0  - yv[Ly2:,(la2+1):(la2+lb+1)] / d_gap ) \
                                        * ( (lb+la2) - xv[Ly2:,(la2+1):(la2+lb+1)] ) / lb \
                               + V_fingB * ( 1.0  - yv[Ly2:,(la2+1):(la2+lb+1)] / d_gap ) \
                                         * ( la2 - xv[Ly2:,(la2+1):(la2+lb+1)] ) / lb
#
start = time.time()
l2_target = 1.0e-9
omega = 1.97
Phi, l2norm, ite = relaxation(Phi0, omega, l2_target, maxiter=50000)


# ---- Contours plot and levels set of the potential...
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
#
gs  = GridSpec(1, 10, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:5 ])
ax2 = fig.add_subplot(gs[0, 5:10 ])
#
ax1.set_title('Levels set of the potential $\Phi$ (Case E_perpendicular)')
cs1 = ax1.contour(1e6 * xv, 1e6 * yv, Phi, levels=25)
ax1.clabel(cs1, cs1.levels, inline=True, fontsize=10)
ax1.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax1.set_ylabel('y-axis [$\mu$m]', fontsize='large')
ax1.set_xlim(0,(L_fing + fing_gap)*1e6)
ax1.set_ylim( 0,d_gap*1e6)
#
ax2.set_title('Contour plot of the potential $\Phi$ (Case E_perpendicular)')
cs2 = ax2.contourf( 1e6 * xv, 1e6 * yv, Phi, 25, cmap=cm.viridis)
cbar = fig.colorbar(cs2)
ax2.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax2.set_ylabel('y-axis [$\mu$m]', fontsize='large')
ax2.set_xlim(0,(L_fing + fing_gap)*1e6)
ax2.set_ylim(0,d_gap*1e6)
#
plt.savefig(output+'ContourPlot_Phi_Eperp.png')
#plt.show()

# ----------------------------------------------------------------------------
# --     We compute the Electric field E     -------------------------------
# ----------------------------------------------------------------------------

Ey, Ex = np.gradient(-Phi)                  # This compute the electric field from the potential
# Below we take into account the attenuation inside the droplet...
Ex, Ey = Ex * C, Ey * C

#Below we re-scale to the appropriated dimensions of the problem...
Ex     = Ex / dx
Ey[0:int(points_y/2),:]    = Ey[0:int(points_y/2),:] / dy_glass
Ey[int(points_y/2):,:]     = Ey[int(points_y/2):,:] / dy_cell

E_mod  = np.sqrt( Ex**2 + Ey**2 )             # Modulus of the elecrtic field...


# ---- Contours plot and levels set of the potential...
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
#
gs  = GridSpec(1, 10, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:5 ])
ax2 = fig.add_subplot(gs[0, 5:10 ])
#
ax1.set_title('Vector plot of the electric field (Case E_perpendicular)')
ax1.streamplot( 1e6 * xv[int(points_y/2):,:], 1e6 * yv[int(points_y/2):,:],\
                Ex[int(points_y/2):,:], Ey[int(points_y/2):,:])
ax1.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax1.set_ylabel('y-axis [$\mu$m]', fontsize='large')
ax1.set_xlim(0,(L_fing + fing_gap)*1e6)
ax1.set_ylim( 0,d_gap*1e6)
#
ax2.set_title('Contour plot of the potential $\Phi$ (Case E_perpendicular)')
cs2 = ax2.contourf( 1e6 * xv, 1e6 * yv, Phi, 25, cmap=cm.viridis)
cbar = fig.colorbar(cs2)
ax2.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax2.set_ylabel('y-axis [$\mu$m]', fontsize='large')
ax2.set_xlim(0,(L_fing + fing_gap)*1e6)
ax2.set_ylim(0,d_gap*1e6)
#
plt.savefig(output+'VectorPlot_ElectField_Eperp.png')
#plt.show()

#
#
#

# ---- Contours plot and levels set of the Electric field...
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
#
gs  = GridSpec(1, 10, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:5 ])
ax2 = fig.add_subplot(gs[0, 5:10 ])
#
ax1.set_title('Levels set of the electric field (Case E_perpendicular)')
cs1 = ax1.contour(1e6 *xv, 1e6 *yv, E_mod, levels=75)
ax1.clabel(cs1, cs1.levels, inline=True, fontsize=10)
ax1.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax1.set_ylabel('y-axis [$\mu$m]', fontsize='large')
ax1.set_xlim(0,(L_fing + fing_gap)*1e6)
ax1.set_ylim( 0,d_gap*1e6)
#
ax2.set_title('Contour plot of the electric field (Case E_perpendicular)')
cs2 = ax2.contourf( 1e6 *xv, 1e6 *yv, E_mod, 75, cmap=cm.viridis)
cbar = fig.colorbar(cs2)
ax2.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax2.set_ylabel('y-axis [$\mu$m]', fontsize='large')
ax2.set_xlim(0,(L_fing + fing_gap)*1e6)
ax2.set_ylim(0,d_gap*1e6)
#
plt.savefig(output+'ContourPlot_ElectricField_Eperp.png')
#plt.show()

#
#
#

# ---- Switching times...
#l = 1.5
#A = 100.0 * (l**2 - 1)/(l**2 + 1)                       # droplets with weak surface anchoring eqn. (2) of ref. below
#B = 6.2                                                # droplets with strong surface anchoring eqn. (3) of ref. below
#D = 2.0 * R
#C = 3.0 / (2.0 + sigLC_sigPol)
#del_eps_eps0_E_th2 = A * C**2 * K / D**2                # According to eqn. (1) of "Magneto-and Electrooptical Measurementsof
#
#T_dec0  = 2.0 * gamma / del_eps_eps0_E_th2              # The previous expression 2.0 * gamma / ( (K/R**2) )
T_decay = gamma / ( eps_0 * delta_eps * E_mod**2 - del_eps_eps0_E_th2 )
#T_decay = gamma / ( eps_0 * delta_eps * E_mod**2 - (K/R**2) ) # The previous expression
RatioT  = T_decay / T_dec0


fig = plt.figure(figsize=(7, 5))
#
ax = fig.add_subplot()
#
Activated = T_decay.copy()
Activated[Activated < 0] = np.nan
#Activated[Activated > 0] = T_decay[T_decay > 0]
#Disabled  = T_decay.copy()
#Disabled[Disabled < 0] = np.nan
#Disabled  = np.nan
#
ax.set_title('Distribution of $T_{decay}$ in the cell (Case E_perp)')
#
levels = np.linspace(0,ms_lim,50)
xx = 1e6 *xv
yy = 1e6 *yv
zz = Activated*1e3
cs = ax.contourf( xx, yy, zz, levels=levels, cmap=cm.viridis)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
cbar = plt.colorbar(cs, format='%4.2f', cax=cax)
cbar.set_label('$T_{decay}$ [miliseconds]', rotation=270, labelpad=16)
#
ax.set_xlim(0,(L_fing + fing_gap)*1e6)
ax.set_ylim( 0,d_gap*1e6)
#
ax.set_xlabel('x-axis [$\mu$m]', fontsize='large')
ax.set_ylabel('y-axis [$\mu$m]', fontsize='large')
#
plt.tight_layout()
#
plt.savefig(output+'Switching_Time_decay_Eperp.png')
#plt.show()
plt.close()

end = time.time()
print(end-start)

