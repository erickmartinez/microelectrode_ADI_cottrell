# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:35:24 2017

@author: Erick Martinez Loran
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('classic')
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
import os,sys
sys.path.append(r'C:\Users\Erick\python\my-modules\echem')
from echem import *
from scipy.sparse import diags
from scipy.sparse import bmat
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as SPSLA
#from numpy.linalg import linalg as LA
import matplotlib.animation as manimation
from scipy.integrate import simps
clear = lambda: os.system('cls')
def getRdHCP(d): return np.sqrt(np.sqrt(3.)/(2.*np.pi))*d

rpill = 5           # The radius of the pillar
zpill = 125.0        # The height of the pillar
d  = 10/np.sqrt(np.sqrt(3.)/(2.*np.pi))           # The hcp nearest neighbour distance between pillars
tmax = 1            # The time length of the simulation
rd = getRdHCP(d)    # The radius of the circle of equivalent area
Do = 7.6e-6         # The diffusion coefficient of the oxidized species cm2/s
Dr = 6.5e-6         # The diffusion coefficient of the reduced species cm2/s
lsim = zpill*1.5    # The max z to show in the plots
o_bulk = 1.e-5      # C_O bulk concentration in mol/cm3 
o_boundary = 0.0    # C_O at the electrode surface for the initial state  
r_bulk = 0.0        # C_R at the bulk 
r_boundary = 0.0    # C_R at the electrode surface for the initial state
alpha = 0.5         # The transfer coefficient
TEMP = 300          # Temperature in Kelvin
k0 = 0.01           # cm/s
E0 = 0.361          # Formal potential in V
E = E0              # The potential for the current simulation
CPOINTS = 100       # The number of points in the Cottrell plot
ne = 1
outputBaseFileName = r'ADI-2D'  # The prefix of the output files
#==============================================================================
# Normalization
# C = c/c*
# R = r/a
# Z = z/a
# tau = Dt/a^2
# K = sqrt(tau/D) = sqrt(1/(D*tnorm))
#==============================================================================
DO = Do*1.0e8       # The diffusion coefficient in um2/s
DR = Dr*1.0e8       # The diffusion coefficient in um2/s
DNORM =  Dr/Do
O_bulk = 1.0
O_boundary = 0.
R_bulk = r_bulk/o_bulk
R_boundary = r_boundary/o_bulk
RPILL = 1
ZPILL = zpill/rpill
RD = rd/rpill
tnorm = DO/(rpill*rpill)
TMAX = tmax*tnorm
ZMAX = 6*np.sqrt(TMAX)
RMAX = RD
lsim_r = RMAX*rpill
lsim = lsim if lsim < ZMAX*rpill else ZMAX*rpill
Apill = np.pi*(RD**2 + 2.0*(RPILL*ZPILL))
Aproj = np.pi*(RD**2)
Area = Aproj*rpill*rpill
K0NORM = k0*(rpill*1e-4/Do) #


#==============================================================================
# Mesh density = 
#==============================================================================
T = 50*50               # The number of time steps
N = int(np.sqrt(T)*6)  # The number of points in the z direction
M = int(1.4*N*RMAX/ZMAX)    # The number of points in the r direction

hr = RMAX/M             # The delta r
hz = ZMAX/N             # The delta z
dt = TMAX/T             # The delta t
DOR = Do/Dr
DRO = Dr/Do
FO_R = Do/hr
FR_R = Dr/hr
FO_Z = Do/hz
FR_Z = Dr/hz

#==============================================================================
# The normalized FD coefficients
#==============================================================================
uza = dt/(2*hz*hz)
uzb = DRO*uza
ura = dt/(2*hr*hr)
urb = DRO*ura

# The error in the concentration in normalized units
c_error = (hr*rpill)**2 + (dt/tnorm)**2
# The error in the concentration in mol/cm^3
dc = c_error*o_bulk

#==============================================================================
# Animation configuration
#==============================================================================
fps_ = 30                       # The number of frames per second for the animation
mod_frame = int(T/fps_/tnorm)   # The step at which the frames will be sampled
mod_frame = 1 if mod_frame == 0 else mod_frame


#==============================================================================
# The filesystem output configuration
#==============================================================================
# Get the current path
cwd = os.path.dirname(os.path.abspath(__file__))
pathToFigure = os.path.join(cwd,outputBaseFileName)


print('rd = %g' % rd)
print('Lsim_r = %g' % lsim_r)
print( 'Lsim_z = %g' % lsim)
print( 'Tmax = %g' % (tmax))
print( 'Do = %g' % Do)
print( 'Dr = %g' % Dr)
print('M = %g' % M)
print('N = %g' % N)
print('T = %g' % T)
print( 'hr = %g' % hr)
print( 'hz = %g' % hz)
print( 'dt = %g' % dt)
print( 'dC = %g' % dc)
print( 'tnorm = %g' % tnorm)
print('Total frames = %d' % (T/mod_frame))
print( 'ura = %g' % ura)
print( 'urb = %g' % urb)
print( 'uza = %g' % uza)
print( 'uzb = %g' % uzb)
print()

#==============================================================================
# The number of simulation snapshots taken
#==============================================================================
sinter = dt/tnorm
snapshot_number = [0]
tn =  dt
print ('snapshot_0 at t = 0')
snum = 1
time = 0
scnt = 1
while time < tmax:
    snum = int(tn/dt)
    time = snum*dt/tnorm
    if time < tmax:
        snapshot_number.append(snum)
        print ('snapshot_%d at t = %.4f' % (scnt, time))
    scnt += 1
    tn = tn*2
print()
snapshot_number = np.array(snapshot_number,dtype=np.int)

# A fix in case the number of sample points is greater than the simulation
# steps
if CPOINTS >= TMAX:
    mod_cpoints = 1
else:
    mod_cpoints = int(TMAX/CPOINTS)


#==============================================================================
# Get the index for the pillar radius and the pillar height
#==============================================================================
RPILL_idx = 0
ZPILL_idx = 0
RD_IDX = 0
for i in range(0,M):
    if i*hr <= RPILL:
        RPILL_idx = i
    if i*hr <= RD:
        RD_IDX = i

for i in range(0,N):
    if i*hz <= ZPILL:
        ZPILL_idx = i

"""
@comment: Tells if the pair of indexes idr,idz lay inside the electrode geometry
"""
def inside_inclusive(idr,idz):
    inside = False
    if idr <= RPILL_idx and (0 <= idz and idz <= ZPILL_idx ):
        inside = True
    return inside

#==============================================================================
# Fill with array of concentrations with boundary concentration
#==============================================================================
def fill_inside(CONC,cbnd):
    for i in range(M+1):
        for j in range(N+1):
            if inside_inclusive(i,j):
                CONC[i][j] = cbnd
                
fpRT = fpR/TEMP
def ENorm(E,temp,n=1):
    f = fpR/temp
    global E0
    return n*f*(E-E0)
   
def zero_flux_r(A,B,idr,idz,k0_,E,sp=1):
    global Do
    global Dr
    global E0
    global alpha
    global TEMP
    global DOR
    global DRO
    global FO_R
    global FR_R
    kf = get_kf(E,E0,k0,TEMP,alpha)
    kb = get_kb(E,E0,k0,TEMP,alpha)
    if sp == 1:
        ka1 = (FO_R + DOR*kb)/(FO_R+kf+DOR*kb)
        ka2 = (kb)/(FO_R+kf+DOR*kb)
        u0 = ka1*A[idr+1][idz] + ka2*B[idr+1][idz]
    else:
        kb1 = (FR_R + DRO*kf)/(FR_R+kb+DRO*kf)
        kb2 = (kf)/(FR_R+kb+DRO*kf)
        u0 = kb1*B[idr+1][idz] + kb2*A[idr+1][idz]
    return u0

def zero_flux_z(A,B,idr,idz,k0_,E,sp=1):
    global Do
    global Dr
    global E0
    global alpha
    global TEMP
    global FO_Z
    global FR_Z
    kf = get_kf(E,E0,k0,TEMP,alpha)
    kb = get_kb(E,E0,k0,TEMP,alpha)
    DOR = Do/Dr
    DRO = Dr/Do
    if sp == 1:
        ka1 = (FO_Z + DOR*kb)/(FO_Z + kf+ DOR*kb)
        ka2 = (kb)/(FO_Z + kf+ DOR*kb)
        u0 = ka1*A[idr][idz+1] + ka2*B[idr][idz+1]
    else:
        kb1 = (FR_Z + DRO*kf)/(FR_Z + kb +DRO*kf)
        kb2 = (kf)/(FR_Z + kb +DRO*kf)
        u0 = kb1*B[idr][idz+1] + kb2*A[idr][idz+1]
    return u0
    
def getCurrent(U,V):
    global E0
    global K0NORM
    KF = get_kf(E,E0,K0NORM)
    KB = get_kb(E,E0,K0NORM)

    dcz_top = (KF*U[0:RPILL_idx,ZPILL_idx+1]-KB*V[0:RPILL_idx,ZPILL_idx+1])
    dcz_base = (KF*U[RPILL_idx+1:RD_IDX+1,0]-KB*V[RPILL_idx+1:RD_IDX+1,0])
    dcx_side = (KF*U[RPILL_idx+1,0:ZPILL_idx+1]-KB*V[RPILL_idx+1,0:ZPILL_idx+1])

    rho = np.array([(i+1)*hr for i in range(0,RPILL_idx)],dtype=np.float32)
    integrand = dcz_top*rho
    i_top = simps(integrand,dx=hz)
    
    rho = np.array([i*hr for i in range(RPILL_idx+1,RD_IDX+1)],dtype=np.float32)
    integrand = dcz_base*rho
    i_base = simps(integrand,dx=hz)
    
    i_side = (hr*(RPILL_idx+1))*simps(dcx_side,dx=hr)
    
    cd = 2*np.pi*ne*F_CONST*(Do*1e+4/rpill)*o_bulk*(i_top + i_base + i_side)*Apill/Aproj
    
    return cd#2*np.pi*ne*F_CONST*o_bulk*Do*(i_top + i_base + i_side)*1e+4/(Aproj*rpill)
             
"""
Gets the matrix that advances the system in the r direction
"""    
def getA(row,E):
    global E0
    global K0
    global DOR
    global DRO
    global FO_R
    global FR_R
    global ura
    global urb
    global TEMP
    global alpha
    kf = get_kf(E,E0,k0,TEMP,alpha)
    kb = get_kb(E,E0,k0,TEMP,alpha)

    # Use the zero-flux boundary condition for the electrode surface
    # a(i-1) = ka1*a(i+1) + ka2*b(i+1)
    # b(i-1) = kb1*a(i+1) + kb2*b(i+1)
    ka1 = (FO_R + DOR*kb)/(FO_R+kf+DOR*kb)
    ka2 = (kb)/(FO_R+kf+DOR*kb)
    kb1 = (FR_R + DRO*kf)/(FR_R + kb +DRO*kf)
    kb2 = (kf)/(FR_R + kb +DRO*kf)
    
    # Two different matrices, one for the region to the right of the electrode
    # and the other for the region above it
    if row <= ZPILL_idx:
        SIZE = M - RPILL_idx
        starti = RPILL_idx + 1
    else:
        SIZE = M
        starti = 1
    aka1 = -ura*ka1
    aka2 = -ura*ka2
    bkb1 = -urb*kb1
    bkb2 = -urb*kb2
    # The diagonal elements for the matrix to the right of the electrode
    a1_diags = [[-ura*(1-1/(2*(i+starti+1))) for i in range(SIZE-1)],[1.+2.0*ura for i in range(SIZE)],[-ura*(1+1/(2*(i+starti))) for i in range(SIZE-1)]]
    b1_diags = [[-urb*(1-1/(2*(i+starti+1))) for i in range(SIZE-1)],[1.+2.0*urb for i in range(SIZE)],[-urb*(1+1/(2*(i+starti))) for i in range(SIZE-1)]]
    
    # At r = 0, (1/R)dC/dR -> 2(d^2C/dR^2):
    a1_diags[1][0] = (1.0+6.0*ura)
    b1_diags[1][0] = (1.0+6.0*urb)
    a1_diags[2][0] = -3.0*ura
    b1_diags[2][0] = -3.0*urb


    if row <= ZPILL_idx:
        # R Boundary conditions for the side of the pillar
        a1_diags[1][0] += aka1*3
        b1_diags[1][0] += bkb1*3
        # boundary conditions at rd (dC/dr = 0)
        a1_diags[1][SIZE-1] -= ura*(1+1/(2*(SIZE+starti)))
        b1_diags[1][SIZE-1] -= urb*(1+1/(2*(SIZE+starti)))
        # The submatrix for the reduced species
        a2_diags = [aka2*3 if i == 0 else 0 for i in range(SIZE)]
        b2_diags = [bkb2*3 if i == 0 else 0 for i in range(SIZE)]
    else:
        # The diagonal elements for the matrix to the right of the electrode.
        # R Boundary conditions for the side of the pillar
        a1_diags[1][0] -= 3*ura
        b1_diags[1][0] -= 3*urb
        # boundary conditions at the rd (dC/dr = 0)
        a1_diags[1][SIZE-1] -= ura*(1+1/(2*(SIZE+starti)))
        b1_diags[1][SIZE-1] -= urb*(1+1/(2*(SIZE+starti)))
                
    
    A1 = diags(a1_diags,[-1,0,1])
    A2 = diags(a2_diags,0) if row <= ZPILL_idx else csr_matrix((SIZE,SIZE))
    B2 = diags(b1_diags,[-1,0,1])
    B1 = diags(b2_diags,0) if row <= ZPILL_idx else csr_matrix((SIZE,SIZE))
    # Create the block matrix
    A = bmat([[A1,A2],[B1,B2]])
    # return the matrix as a compressed sparse array.
    return A.tocsr()

"""
Gets the matrix that advances the system in the z direction
""" 
def getD(col,E):
    global E0
    global k0
    global DOR
    global DRO
    global FO_Z
    global FR_Z
    global ura
    global urb
    global K0NORM
    global TEMP
    global alpha
    kf = get_kf(E,E0,k0,TEMP,alpha)
    kb = get_kb(E,E0,k0,TEMP,alpha)
    
    ka1 = (FO_Z + DOR*kb)/(FO_Z+kf+DOR*kb)
    ka2 = (kb)/(FO_Z+kf+DOR*kb)
    kb1 = (FR_Z + DRO*kf)/(FR_Z + kb +DRO*kf)
    kb2 = (kf)/(FR_Z + kb +DRO*kf)
    aka1 = -uza*ka1
    aka2 = -uza*ka2
    bkb1 = -uzb*kb1
    bkb2 = -uzb*kb2

    if col <= RPILL_idx:
        SIZE = N - ZPILL_idx
    else:
        SIZE = N
    a1_diags = [[-uza for i in range(SIZE-1)],[1.+2.0*uza for i in range(SIZE)],[-uza for i in range(SIZE-1)]]
    b1_diags = [[-uzb for i in range(SIZE-1)],[1.+2.0*uzb for i in range(SIZE)],[-uzb for i in range(SIZE-1)]]
    # Z boundary conditions on the top surface of the pillar and the bottom of the pillar
    a1_diags[1][0] += aka1
    b1_diags[1][0] += bkb1
    # The boundary conditions at the top of the z domain do not multiply concentrations
    # and must be defined in the column vectors
    a2_diags = [aka2 if i == 0 else 0 for i in range(SIZE)]
    b2_diags = [bkb2 if i == 0 else 0 for i in range(SIZE)]
    A1 = diags(a1_diags,[-1,0,1])
    A2 = diags(a2_diags,0)
    B2 = diags(b1_diags,[-1,0,1])
    B1 = diags(b2_diags,0)
    C = bmat([[A1,A2],[B1,B2]])
    return C.tocsr()
    
def bj(A,B, current_j,E):
    global ura
    global urb
    global uza
    global uzb
    global k0
    if current_j <= ZPILL_idx:
        SIZE = (M - RPILL_idx)
        start = RPILL_idx + 1
    else:
        SIZE = (M)
        start = 0
    bo = np.zeros(SIZE,dtype=np.float32)
    br = np.zeros(SIZE,dtype=np.float32)
    u12a = (1.0 - 2.0*uza)
    u12b = (1.0 - 2.0*uzb)
    for i in range (0,SIZE):
        ii = i + start
        if current_j == 0 and ii > RPILL_idx:
            u1a = uza*zero_flux_z(A,B,ii,current_j-1,k0,E,1)
            u1b = uzb*zero_flux_z(A,B,ii,current_j-1,k0,E,2)
        elif current_j == ZPILL_idx + 1 and ii <= RPILL_idx:
            u1a = uza*zero_flux_z(A,B,ii,ZPILL_idx,k0,E,1)
            u1b = uzb*zero_flux_z(A,B,ii,ZPILL_idx,k0,E,2)
        else:
            u1a = uza*A[ii][current_j-1] 
            u1b = uzb*B[ii][current_j-1]
        u2a = u12a*A[ii][current_j]
        u2b = u12b*B[ii][current_j]
        u3a = uza*A[ii][current_j+1] if current_j < N else uza*O_bulk
        u3b = uzb*B[ii][current_j+1] if current_j < N else uzb*R_bulk
        bo[i] = u1a+u2a+u3a
        br[i] = u1b+u2b+u3b
    return np.concatenate((bo,br))


def di(A,B, current_i,E):
    global ura
    global urb
    global uza
    global uzb
    global k0
    if current_i <= RPILL_idx:
        SIZE = N - ZPILL_idx
        start = ZPILL_idx + 1
    else:
        SIZE = N
        start = 0
    do = np.zeros(SIZE,dtype=np.float32)
    dr = np.zeros(SIZE,dtype=np.float32)
    u12a = (1.0 - 6.0*ura) if current_i == 0 else (1.0 - 2.0*ura)
    u12b = (1.0 - 6.0*urb) if current_i == 0 else (1.0 - 2.0*urb)
    p1 = 3 if current_i ==0 else 1 - 1/(2*(current_i + 1))
    p2 = 3 if current_i ==0 else 1 + 1/(2*(current_i + 1))
    
    for j in range (0,SIZE):
        jj = j + start
        if current_i == RPILL_idx + 1 and jj <= ZPILL_idx:
            u1a = ura*p1*zero_flux_r(A,B,RPILL_idx,ZPILL_idx,k0,E,1)
            u1b = urb*p1*zero_flux_r(A,B,RPILL_idx,ZPILL_idx,k0,E,2)
        elif current_i == 0 and jj > ZPILL_idx:
            u1a = ura*p1*A[current_i][jj]
            u1b = urb*p1*B[current_i][jj]
        else:
            u1a = ura*p1*A[current_i-1][jj]
            u1b = urb*p1*B[current_i-1][jj]
        u2a = u12a*A[current_i][jj]
        u2b = u12b*B[current_i][jj]
        u3a = ura*p2*A[current_i+1][jj] if current_i + 1 < M else ura*p2*A[current_i][jj]
        u3b = urb*p2*B[current_i+1][jj] if current_i + 1 < M else urb*p2*B[current_i][jj]
        do[j] = u1a+u2a+u3a
        dr[j] = u1b+u2b+u3b
        if j == SIZE - 1:
            do[j] += uza*O_bulk
            dr[j] += uzb*R_bulk
    return np.concatenate((do,dr))

snap = 1
def contourSnapshot(U,time):
    global snap
    x_grid = np.array([i*hr*rpill for i in range(M+1)],dtype=np.float32)
    y_grid = np.array([j*hz*rpill for j in range(N+1)],dtype=np.float32)
    plt.close('all')
    # Plot style parameters
    plotStyle = {'font.size': 12,
                'legend.fontsize': 10,
                'figure.dpi': 100,
                'font.family': 'Arial',
                'mathtext.rm': 'Arial'}
    mpl.rc('axes', linewidth=2.5)
    mpl.rcParams.update(plotStyle)

    fig = plt.figure()
    # optional kwarg forward=True will cause the canvas size to be automatically
    # updated; e.g., you can resize the figure window from the shell
    fig.set_size_inches(5.4,4.2,forward=True)
    ax = plt.subplot2grid((1,1), (0,0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5,prune='lower'))
    ax.autoscale_view(True,True,True)
    ax.set_xlim([0.0,lsim_r])
    ax.set_ylim([0.0,lsim])
                
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3,3))
    timelbl = 't = $\mathregular{%s}$ s' % latex_float(time)
        
    ax.set_xlabel(r'x ($\mathregular{\mu}$m)',fontweight='bold',fontsize=16)
    ax.set_ylabel(r'y ($\mathregular{\mu}$m)',fontweight='bold',fontsize=16)
    ax.set_title(timelbl, loc='left')
    ax.set_xticks(np.arange(0,lsim_r,rpill/2))
    ax.set_yticks(np.arange(0,lsim,zpill/2))
    
    cp = ax.pcolormesh(x_grid,y_grid,U)
#    CS = ax.contour(X,Y,U,
#                 origin='lower',
#                 linewidths=2,
#                 extent=(-3, 3, -2, 2))

    cb = plt.colorbar(cp,fraction=0.0455, pad=0.04)
    cb.set_label('Concentration ( $\mathregular{C/C^*}$)')
    
    x1,x2 = ax.axes.get_xlim()
    y1,y2 = ax.axes.get_ylim()
    ratio = (x2 - x1) / (y2 - y1)
    ax.set_aspect(ratio)

    
    plt.tight_layout()
    plt.show()
    
    fig_name = '%s_contour_%s' % (pathToFigure,snap)
    fig.savefig(fig_name + '.png', dpi=600)
    fig.savefig(fig_name + '.eps', format='eps', dpi=600)
    fig.savefig(fig_name + '.pdf', format='pdf', dpi=600)
    snap += 1
    

#==============================================================================
# Define the initial concentration
#==============================================================================
Ui = np.full((M+1,N+1),O_bulk,dtype=np.float32)
Ri = np.full((M+1,N+1),R_bulk,dtype=np.float32)

fill_inside(Ui,O_boundary)

x_grid = np.array([i*hr*rpill for i in range(M+1)],dtype=np.float32)
y_grid = np.array([j*hz*rpill for j in range(N+1)],dtype=np.float32)
t_grid = np.array([k*dt/tnorm for k in range(1,T+1)],dtype=np.float32)

U1 = np.copy(Ui)
V1 = np.copy(Ri)
U2 = np.copy(U1)
V2 = np.copy(V1)

time_j = []
current = []
charge = []
charge_cot = []
snapshot_c = []
snapshot_t = []

plt.close('all')
# Plot style parameters
plotStyle = {'font.size': 12,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'font.family': 'Arial',
            'mathtext.rm': 'Arial'}
mpl.rc('axes', linewidth=2)
mpl.rcParams.update(plotStyle)
linew = 2.5
formatter = mticker.ScalarFormatter(useMathText=True)
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3,3))

fig = plt.figure()
# optional kwarg forward=True will cause the canvas size to be automatically
# updated; e.g., you can resize the figure window from the shell
fig.set_size_inches(5.4,4.2,forward=True)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.yaxis.set_major_locator(mticker.MaxNLocator(5,prune='lower'))
ax1.autoscale_view(True,True,True)
ax1.set_xlim([0.0,lsim_r])
ax1.set_ylim([0.0,lsim])

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Concentration animation', artist='Matplotlib',
                comment='Time dependent concentration')
writer = FFMpegWriter(fps=fps_, metadata=metadata)

ax1.set_xlabel(r'x ($\mathregular{\mu}$m)',fontweight='bold',fontsize=16)
ax1.set_ylabel(r'y ($\mathregular{\mu}$m)',fontweight='bold',fontsize=16)
ax1.set_yticks(np.arange(0,lsim,zpill/2))
ax1.set_xticks(np.arange(0,lsim_r,rpill/2))
ax1.yaxis.set_major_formatter(xfmt)
ax1.xaxis.set_major_formatter(xfmt)


ims = []

heatmap = ax1.pcolor(x_grid, y_grid, np.transpose(U1*o_bulk))
ims.append((heatmap,))

contour_n = 1
Qsim = 0
Qcott = 0
## SOLVE THE SYSTEM FOR T=1 to TMAX
for k in range(1,T+1):
#    if k*dt/tnorm >= 0.05: break
    # Work the r direction for fixed z
    for j in range (0,N+1):
        uxj = SPSLA.spsolve(getA(j,E),bj(U1,V1,j,E), use_umfpack=False)
        if j <= ZPILL_idx:
            SIZE = M - RPILL_idx
            start = RPILL_idx+1
            for i in range(0,SIZE):
                ii = start + i
                U2[ii][j] = uxj[i]
                V2[ii][j] = uxj[i+SIZE]
        else:
            SIZE = M 
            for i in range(0,M):
                U2[i][j] = uxj[i]
                V2[i][j] = uxj[i+SIZE]
        
                
    for i in range (0,M+1):
        uyi = SPSLA.spsolve(getD(i,E),di(U2,V2,i,E), use_umfpack=False)
        if i <= RPILL_idx:
            SIZE = N - ZPILL_idx
            start = ZPILL_idx + 1
            for j in range(0,SIZE):
                jj = start + j
                U1[i][jj] = uyi[j]
                V1[i][jj] = uyi[j+SIZE]
        else:
            SIZE = N 
            for j in range(N):
                U1[i][j] = uyi[j]
                V1[i][j] = uyi[j+SIZE]

    
    
    if (k)%mod_cpoints ==0:
        t_ = dt*k/tnorm
        time_j.append(t_)
        cot_ = ne*F_CONST*o_bulk*np.sqrt(Do/(np.pi*t_))
        cur = getCurrent(U1,V1)
        qisim = cur*dt*mod_cpoints*Apill/tnorm
        Qsim += qisim
        qicott = cot_*dt*mod_cpoints*Aproj/tnorm
        Qcott += qicott
        charge.append(Qsim)
        charge_cot.append(Qcott)
        current.append(cur)
        ratio = cur/cot_
        message = 't=%.4f (s), i_pill = %.3e (A), i_cot = %.3e (A), Isim/Icott = %.3f'
        print (message % (t_, cur,cot_,ratio))
    

    if (k) % mod_frame == 0:
        t_ = dt*k/tnorm
        heatmap = ax1.pcolormesh(x_grid, y_grid, np.transpose(o_bulk*U1))
#        ax1.set_title('t = $\mathregular{%s}$ s' % latex_float(t_),loc='left')
        ims.append((heatmap,))
    
    if (k == snapshot_number[contour_n]):
        t_ = dt*k/tnorm
#        print ('snapshot_%d at t = %.4f' % (contour_n, t_))
        snapshot_c.append(np.transpose(np.copy(U1)))
        snapshot_t.append(t_)
        contour_n = contour_n + 1 if contour_n < len(snapshot_number)-1 else contour_n
    
        


x1,x2 = ax1.axes.get_xlim()
y1,y2 = ax1.axes.get_ylim()
ratio = (x2 - x1) / (y2 - y1)
ax1.set_aspect(ratio)

colorbar = plt.colorbar(heatmap,fraction=0.0455, pad=0.04, format=xfmt)
colorbar.set_label('Concentration (mol $\mathregular{cm^{-3}}$)')

plt.tight_layout()
plt.show()

im_ani = manimation.ArtistAnimation(fig, ims, interval=fps_, repeat_delay=0,
                                   blit=False)


### To save this animation with some metadata, use the following command:
writer = manimation.writers['ffmpeg'](fps=fps_)
### Change the video bitrate as you like and add some metadata.
writer = FFMpegWriter(fps=fps_, bitrate=1000, metadata={'artist':'Erick'})
im_ani.save(pathToFigure+'-animation.mp4',writer=writer, dpi=600)

snapshot_c = np.array(snapshot_c,dtype=np.float32)
snapshot_t = np.array(snapshot_t,dtype=np.float32)

for i in range(0,snapshot_t.size):
    contourSnapshot(snapshot_c[i], snapshot_t[i])

# Plot style parameters
plotStyle = {'font.size': 12,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'font.family': 'Arial'}
mpl.rc('font',family='Arial')
mpl.rc('axes', linewidth=2)
mpl.rcParams.update(plotStyle)
linew = 2.5

fig3 = plt.figure()
fig3.set_size_inches(5.4,4.2,forward=True)
ax3 = plt.subplot2grid((1,1), (0,0))
#ax2 = plt.subplot2grid((1,2), (0,1))
ax3.yaxis.set_major_locator(mticker.MaxNLocator(5,prune='lower'))
ax3.autoscale_view(True,True,True)


current = 1000*np.array(current,dtype=np.float32)
time_j = np.array(time_j,dtype=np.float32)

iv_plot = ax3.loglog(time_j,np.fabs(current),label='Simulated')

cottrell = 1000*ne*F_CONST*o_bulk*np.sqrt(Do/(np.pi*time_j))

iv_plot = ax3.loglog(time_j,cottrell,label='Cottrell')

ax3.set_xlabel(r'Time (s)',fontweight='bold',fontsize=16)
ax3.set_ylabel(r'I (mA $\mathregular{cm^{-2}}$)',fontweight='bold',fontsize=16)
ax3.yaxis.set_major_formatter(xfmt)
ax3.xaxis.set_major_formatter(xfmt)
leg = ax3.legend(loc='upper right',prop={'family':'Arial','size':10},frameon=False)


plt.tight_layout()
plt.show()

fig3.savefig(pathToFigure + '_cottrell.png' , dpi=600)
fig3.savefig(pathToFigure + '_cottrell.eps', format='eps', dpi=600)
fig3.savefig(pathToFigure + '_cottrell.pdf', format='pdf', dpi=600)


charge = np.array(charge,dtype=np.float32)
charge_cot = np.array(charge_cot,dtype=np.float32)


fig4 = plt.figure()
fig4.set_size_inches(5.4,4.2,forward=True)
ax4 = plt.subplot2grid((1,1), (0,0))
ax4.yaxis.set_major_locator(mticker.MaxNLocator(5,prune='lower'))
ax4.autoscale_view(True,True,True)

chargeplot_sim = ax4.plot(time_j,charge,label='Simulated')
chargeplot_cot = ax4.plot(time_j,charge_cot,label='Cottrell')

ax4.set_xlabel(r'Time (s)',fontweight='bold',fontsize=16)
ax4.set_ylabel(r'Charge ($\mathregular{C/cm^{2}}$)',fontweight='bold',fontsize=16)
ax4.set_xticks(np.arange(0,max(time_j),max(time_j)/5))
ax4.set_xticks(np.arange(0,max(time_j),max(time_j)/5))
ax4.yaxis.set_major_formatter(xfmt)
ax4.xaxis.set_major_formatter(xfmt)


leg = ax4.legend(loc='upper left',prop={'family':'Arial','size':10},frameon=False)


x1,x2 = ax4.axes.get_xlim()
y1,y2 = ax4.axes.get_ylim()
ratio = (x2 - x1) / (y2 - y1)
ax4.set_aspect(ratio)

plt.tight_layout()
plt.show()

fig4.savefig(pathToFigure + '_charge.png' , dpi=600)
fig4.savefig(pathToFigure + '_charge.eps', format='eps', dpi=600)
fig4.savefig(pathToFigure + '_charge.pdf', format='pdf', dpi=600)


CottrellHD = 'Time (s), Simulated (mA / cm2), Cottrell (mA/cm2)'
ChargeHD = 'Time (s), Simulated (C/cm2), Cottrell (C/cm2)'

data = np.transpose(np.array([time_j,current,cottrell]))
np.savetxt(pathToFigure + '_cottrell.csv',
            data,
            delimiter=',',
            fmt=(
                    '%.5e',
                    '%.5e',
                    '%.5e'),
            header=CottrellHD,
            comments='',)

data = np.transpose(np.array([time_j,charge,charge_cot]))
np.savetxt(pathToFigure + '_charge.csv',
            data,
            delimiter=',',
            fmt=(
                    '%.5e',
                    '%.5e',
                    '%.5e'),
            header=ChargeHD,
            comments='',)

clear()
