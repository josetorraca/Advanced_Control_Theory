
from VdV4x2 import *
from CasadiTools import *
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
import time

# Process
process = ODEModel(dt=dt, x=x, y=y, u=u, dx=dx, d=d, p=p) #process object
process.get_equations(intg='idas')

#SS optimizer
opts = {
    'warn_initial_bounds': False, 'print_time': False, 
    'ipopt': {'print_level': 1}
    }

# Initial guesses
Caguess = 1.7949
Cbguess = 1.0787
Tguess = 144.2363
fguess = 100
Tkguess = 150
Qkguess = 50000
xguess = [Caguess, Cbguess, Tguess, Tkguess]
uguess = [fguess, Qkguess]

# Bounds
lbCa = 0.1
ubCa = 5
lbCb = 0.1
ubCb = 2
lbT = 30
ubT = 200
lbTk = 30
ubTk = 200
lbf = 10
ubf = 400
lbQk = 0
ubQk = 85000
lbCain = 0.1
ubCain = 6
lbTin = 30
ubTin = 200
lbx = [lbCa, lbCb, lbT, lbTk]
ubx = [ubCa, ubCb, ubT, ubTk]
lbu = [lbf, lbQk]
ubu = [ubf, ubQk]
lbd = [lbCain, lbTin]
ubd = [ubCain, ubTin]

process.buildnlp_steady(xguess=xguess, uguess=uguess, lbx=lbx, ubx=ubx, lbu=lbu, 
                        ubu=ubu, opts=opts) 

# AEKF
#P0 = np.diag([1e3, 1e3, 1e3, 1e3, 1e3])
#Q = np.diag([3e-2, 5e-2, 1e1, 1e1, 1e2])*1e-4
#R = np.diag([3e-2, 5e-3, 8e-1])

#aekf = AEKF(dt=dt, P0=P0, Q=Q, R=R, x=x, u=u, y=y, dx=dx, theta=d)

# NMPC
N = 40
M = 10
Q = np.diag([1, 1e-3])
R = np.diag([0, 0])
W = np.diag([1e-5, 1e-8])
lbdf = -50
ubdf = 50
lbdQk = -10000
ubdQk = 10000
lbdu = [lbdf, lbdQk]
ubdu = [ubdf, ubdQk]
opts ={'disc': 'single_shooting'}

nmpc = NMPC(dt=dt, N=N, M=M, Q=Q, R=R, W=W, x=x, u=u, c=c, d=d, p=p, dx=dx,
       xguess=xguess, uguess=uguess, lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu, 
       lbdu=lbdu, ubdu=ubdu, tgt=True, opts=opts)

# Initialization
t = 1 #counter
tsim = 2 #h
niter = math.ceil(tsim/dt)

xf = [3.08275401, 0.52532486, 122.27127671, 77.75680223]
uf = [120.04167236,  50000]
dist = [4, 130]
par_model = [1.287e12, 2]
par_plant = [1.287e12*.95, 2*0.8]
xhat = copy.deepcopy(xf)
dhat = copy.deepcopy(dist[0])
phat = copy.deepcopy(par_model)
ysim = np.zeros([niter, 4])
usim = np.zeros([niter, 2])
spsim = np.zeros([niter, 2])
#targetsim = np.zeros([niter, 2])
dsim = np.zeros([niter, 2])
psim = np.zeros([niter, 2])
xest = np.zeros([niter, 4])
dest = np.zeros([niter, 1])
pest = np.zeros([niter, 2])
cpu_time = []
bar = IncrementalBar('Simulation in progress', max=niter) #progress bar

# Simulation 
for ksim in range(0, niter):
    start = time.time() #comp time
    n = ksim/niter

    # Disturbances
    if n > 1/4 and n < 2/4:
        dist = [5.1, 130]
    elif n >= 2/4:
        dist = [5.1, 130*1.1]
    else:
        dist = [4, 130]

    # Plant
    sim = process.simulate_step(xf=xf, uf=uf, df=dist, pf=par_plant)
    ymeas = sim['x']*(1 + 0.001*np.random.normal(0, 1))
    d1meas = sim['d'][1]*(1 + 0.001*np.random.normal(0, 1))
    xf = sim['x'].ravel()
    ysim[t-1, :] = xf
    usim[t-1, :] = sim['u']
    dsim[t-1, :] = sim['d']
    psim[t-1, :] = sim['p']
    spsim[t-1, :] = spsim

    # AEKF
    #est = aekf.updatestate(xkhat=vertcat(xhat, dhat), uf=uf, ymeas=ymeas)
    #xhat = est['x']
    #dhat = est['theta']
    #xest[t-1, :] = xhat
    #dest[t-1, :] = dhat

    # NMPC
    ctrl = nmpc.calc_actions(ksim=ksim+1, x0=xhat, u0=uf, sp=spsim,
                             d0=dhat, p0=phat)
    uf = ctrl['U']

    t += 1
    end = time.time()
    cpu_time += [end - start]
    bar.next() 

bar.finish()
avg_time = np.mean(cpu_time) # avg time spent at each opt cycle
    
# Plot 
time = np.linspace(0, tsim, niter)

fig1, ax1 = plt.subplots(2, 2, frameon=False) #x 
ax1[0, 0].plot(time, ysim[:, 0], label='Plant') #Ca
ax1[0, 0].plot(time, xest[:, 0], linestyle=None, marker='o', label='MHE')
ax1[0, 0].set_ylabel('C_A [mol/L]')
ax1[0, 0].legend()
ax1[1, 0].plot(time, ysim[:, 1], label='Plant') #Cb 
ax1[1, 0].plot(time, spsim[:, 0], linestyle='--', label='SP') 
ax1[1, 0].plot(time, xest[:, 1], linestyle=None, marker='o', label='MHE')
ax1[1, 0].set_ylabel('C_B [mol/L]')
ax1[1, 0].set_xlabel('time [h]')
ax1[1, 0].legend()
ax1[0, 1].plot(time, ysim[:, 2], label='Plant') #T 
ax1[0, 1].plot(time, spsim[:, 1], linestyle='--', label='SP') 
ax1[0, 1].plot(time, xest[:, 2], linestyle=None, marker='o', label='MHE')
ax1[0, 1].set_ylabel('T [\xb0C]')
ax1[0, 1].legend()
ax1[1, 1].plot(time, ysim[:, 3], label='Plant') #Tk 
ax1[1, 1].plot(time, xest[:, 3], linestyle=None, marker='o', label='MHE')
ax1[1, 1].set_ylabel('T_k [\xb0C]')
ax1[1, 1].legend()

fig2, ax2 = plt.subplots(2, 1, frameon=False) #u
ax2[0].step(time, usim[:, 0]) #f
ax2[0].set_ylabel('f [h^{-1}]')
ax2[0].legend()
ax2[1].step(time, usim[:, 1]) #Qk
ax2[1].set_ylabel('Q_k/(Kw Ar) [kJ/h]')
ax2[1].set_xlabel('time [h]')
ax2[1].legend()
#ax2.legend()

fig3, ax3 = plt.subplots(2, 1, frameon=False) #d
ax3[0].step(time, dsim[:, 0], label='Plant') #Cain
ax3[0].plot(time, dest[:, 0], linestyle=None, marker='o', label='MHE')
ax3[0].set_ylabel('C_{Ain} [mol/h]')
ax3[0].legend()
ax3[1].step(time, dsim[:, 1], label='Plant') #Tin
ax3[1].plot(time, dest[:, 1], linestyle=None, marker='o', label='MHE')
ax3[1].set_ylabel('T_{in} [\xb0C]')
ax3[1].set_xlabel('time [h]')
ax3[1].legend()

fig4, ax4 = plt.subplots(2, 1, frameon=False) #p
ax4[0].step(time, psim[:, 0], label='Plant') #k1
ax4[0].plot(time, pest[:, 0], linestyle=None, marker='o', label='MHE')
ax4[0].set_ylabel('k_1 [h^{-1}]')
ax4[0].legend()
ax4[1].step(time, psim[:, 1], label='Plant') #cp
ax4[1].plot(time, pest[:, 1], linestyle=None, marker='o', label='MHE')
ax4[1].set_ylabel('c_p [kJ/kg K]')
ax4[1].set_xlabel('time [h]')
ax4[1].legend()

plt.show()

