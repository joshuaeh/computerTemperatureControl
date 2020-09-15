import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import sys
import warnings

# Import data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
data_fan = pd.read_csv('a_Fan_step.csv', index_col=0)
t = data_fan['Time'].to_numpy()
u = data_fan['Fan'].to_numpy()
y = data_fan['Tcpu'].to_numpy()
q = data_fan['Q'].to_numpy()
Ta = data_fan['T_air'].to_numpy()

# wait until steady state
tsleep = 60
t = t[tsleep:]
u = u[tsleep:]
y = y[tsleep:]
q = q[tsleep:]
Ta = Ta[tsleep:]
u0 = u[0]
y0 = y[0]
xp0 = [y0, 0.0]
yp = y


# specify number of steps
ns = len(t)
delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
uf = interp1d(t, u)

# define first-order plus dead-time approximation
def fopdt(y,t,uf,Km,taum,thetam):
    # arguments
    #  y      = output
    #  t      = time
    #  uf     = input linear function (for time shift)
    #  Km     = model gain
    #  taum   = model time constant
    #  thetam = model time constant
    # time-shift u
    try:
        if (t-thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t-thetam)
    except:
        #print('Error with time extrapolation: ' + str(t))
        um = u0
    # calculate derivative
    dydt = (-(y-yp[0]) + Km * (um-u0))/taum
    return dydt

# simulate FOPDT model with x=[Km,taum,thetam]
def sim_model(x):
    # input arguments
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    # storage for model values
    ym = np.zeros(ns)  # model
    # initial condition
    ym[0] = yp[0]
    # loop through time steps
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
        ym[i+1] = y1[-1]
    return ym

# define objective
def objective(x):
    # simulate model
    ym = sim_model(x)
    # calculate objective
    obj = 0.0
    for k in range(len(ym)):
        obj = obj + (ym[k]-yp[k])**2
    # return result
    return obj

# initial guesses
x0 = np.zeros(3)
x0[0] = -801.4  # Km
x0[1] = .24  # taum
x0[2] = 1  # thetam

#show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# theta = np.linspace(0, 100, 100)
# results = []
# j = 1
# for b in theta:
#     x0 = [-802, 1, b]
#     results.append(objective(x0))
#     sys.stdout.write("\rDoing Test %d" % j)
#     sys.stdout.flush()
#     j += 1
#
# plt.figure(figsize=(10, 7))
# plt.plot(theta, results)
# plt.xlabel('Theta Value')
# plt.ylabel('SSE Error')
# min_err = min(results)
# Kp_best = theta[results.index(min_err)]
# plt.show()
# x = int(input('x Coordinate of text'))
# y = int(input('Y coordinate of string'))
#
#
# plt.figure(figsize=(10, 7))
# plt.plot(theta, results)
# plt.xlabel('Tau Value')
# plt.ylabel('SSE Error')
# min_err = min(results)
# Kp_best = theta[results.index(min_err)]
# plt.text(x,y, f'Min Err: {min_err}\nBest Theta: {Kp_best}')
# plt.savefig(input('What should this be named?'))
# plt.show()


# optimize Km, taum, thetam
solution = minimize(objective,x0)

# Another way to solve: with bounds on variables
#bnds = ((0.4, 0.6), (1.0, 10.0), (0.0, 30.0))
#solution = minimize(objective,x0,bounds=bnds,method='SLSQP')
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

print('Kp: ' + str(x[0]))
print('taup: ' + str(x[1]))
print('thetap: ' + str(x[2]))

# calculate model with updated parameters
ym1 = sim_model(x0)
ym2 = sim_model(x)
# plot results
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
plt.ylabel('Temperature')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(t,u,'bx-',linewidth=2)
plt.plot(t,uf(t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.text(3100, .025, f'SSE: {round(objective(x), 2)}\nKp: {round(x[0])}\nTauP: {round(x[1],2)}\n\
ThetaP: {round(x[2],2)}')
plt.ylabel('Fan Rate (m^3/sec)')
plt.savefig('FanOptParam_step.png')
plt.show()