# Model parameters for the Van de Vusse CSTR: 4 states and 2 inputs

from casadi import *

# Sampling time 
dt = 0.1/40

# Parameters 
#k10 = 1.287e12 #1st reaction frequency factor (h-1)
k20 = 1.287e12 #2nd reaction frequency factor (h-1)
k30 = 9.043e9 #3rd reaction frequency factor (L/mol L)
E1 = 9758.3  #1st reaction activation energy /R (K)
E2 = 9758.3 #2nd reaction activation energy /R (K)
E3 = 8560 #3rd reaction activation energy /R (K)
deltaH1 = 4.2  #1st reaction enthalpy (kJ/mol)
deltaH2 = -11 #2nd reaction enthalpy (kJ/mol)
deltaH3 = -41.85 #3rd reaction enthalpy (kJ/mol)
rho = 0.9342 #density (kg/L)
cp = 3.01 #heat capacit (kJ/kg K)
Ar = 0.215 #jacket area (m2)
Kw = 4032 #jacket heat transfer coefficient (kJ/h m2 K)
V = 10 #reactor volume (L)
mk = 5;
#cpk = 2;

# States
Ca = MX.sym('C_A',1) #yield of A (mol/L) 
Cb = MX.sym('C_B',1) #yield of B (mol/L)
T = MX.sym('T',1) #system temperature (C)
Tk = MX. sym('T_k',1) #jacket temperature (C)
x = vertcat(Ca, Cb, T, Tk)
c = vertcat(Cb, T) #controlled variables

# Outputs
y = vertcat(Ca, Cb, T, Tk)

# Inputs
f = MX.sym('F/V', 1) #spacial velocity (h-1)
Qk = MX. sym('Q_k', 1) #jacket heat (kJ/h)
u = vertcat(f, Qk)

# Disturances
Cain = MX.sym('C_Ain',1) #inlet yield of A (unmeasured)
Tin = MX.sym('T_in',1)  #inlet temperature (measured)
d = vertcat(Cain, Tin)

# Uncertain parameters
k10 = MX.sym('k10', 1)
cp = MX.sym('cp', 1)
p = vertcat(k10, cp)

# ODE system
dCadt = f*(Cain - Ca) - Ca*k10*exp(-E1/(T + 273.15)) - Ca**2*k30*exp(-E3/(T + 273.15))
dCbdt = -f*Cb + Ca*k10*exp(-E1/(T + 273.15)) - Cb*k20*exp(-E2/(T + 273.15))
dTdt = f*(Tin - T) + (Kw*Ar*(Tk - T)/V + (Ca*(-deltaH1)*k10*exp(-E1/(T + 273.15))) + \
       (Cb*(-deltaH2)*k20*exp(-E2/(T + 273.15))) + (Ca**2*(-deltaH3)*k30*exp(-E3/(T + 273.15))))/(rho*cp)
dTkdt = (Qk + Kw*Ar*(T - Tk))/mk/cpk;
dx = vertcat(dCadt, dCbdt, dTdt, dTkdt)
