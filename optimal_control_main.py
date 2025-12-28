
import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass


class Vehicle():

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

vehicle_attributes = {'g': 9.807, 'vehicleMass': 1300, 'trackwidth': 2.016,
                      'wheelbase': 2.9, 'b': 1.535, 'a': 1.365, 'h': 0.33, 'rho_air': 1.2,
                      'CdA': 0.65, 'CLfA': 0.15, 'CLrA': 0.35, 'gamma': 57/43,
                      'roll_stiffness': 0.53, 'P_max': 415, 'V': 30}

tire_coefs = {'ref_load': 3500, 'pCx1': 1.6935, 'pDx1': 1.8757, 'pDx2': -0.127,
              'pEx1': 0.07708, 'pKx1': 30.5, 'pKx3': 0.2766, 'lambda_mux': 0.93,
              'pCy1': 1.733, 'pDy1': 1.8217, 'pDy2': -0.4388, 'pEy1': 0.29446,
              'pKy1': 44.2, 'pKy2': 2.5977, 'lambda_muy': 0.84}

vehicle = Vehicle(**attributes)

# function to calculate fx fy
def mf_fx_fy(kappa, alpha, Fz):
    global ref_load

    error_eps = 1e-6
    # calculate the coefs
    dfz = (Fz - ref_load)/ref_load
    Kx = Fz*pKx1*ca.exp(pKx3*dfz)
    Ex = pEx1
    Dx = (pDx1 + pDx2*dfz)*lambda_mux
    Cx = pCx1
    Bx = Kx/(Cx*Dx*Fz)
    
    Ky = ref_load*pKy1*ca.sin(2*ca.atan(Fz/(pKy2*ref_load)))
    Ey = pEy1
    Dy = (pDy1 + pDy2*dfz)*lambda_muy
    Cy = pCy1
    By = Ky/(Cy*Dy*Fz)

    # magic formula
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = ca.sqrt((sig_x**2) + (sig_y**2))

    Fx = Fz*(sig_x/(sig + error_eps))*Dx*ca.sin(Cx * ca.atan(Bx*sig - Ex*(Bx*sig - ca.atan(Bx*sig))))
    Fy = Fz*(sig_y/(sig + error_eps))*Dy*ca.sin(Cy*ca.atan(By*sig - Ey*(By*sig - ca.atan(By*sig))))


    return Fx, Fy

# function to calculate fz
def normal_loads(ax, ay, u):
    FLf = 0.5*CLfA*(u**2)
FLr = 0.5*CLrA*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr        
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = ca.fmax(Nfl, 1e-3)
    Nfr = ca.fmax(Nfr, 1e-3)
    Nrl = ca.fmax(Nrl, 1e-3)
    Nrr = ca.fmax(Nrr, 1e-3)


    return Nfl, Nfr, Nrl, Nrr

def main():
    # use main
    
    # discretization in alpha
    N = 180
    alpha = np.linspace(-np.pi/2, np.pi/2, N)
    dalpha = alpha[1] - alpha[0]
    
    # decision variables
    opti = ca.Opti()
    
    # states
    
    rho = opti.variable(N)
    delta = opti.variable(N)
    beta = opti.variable(N)
    kappa_fl = opti.variable(N)
    kappa_fr = opti.variable(N)
    kappa_rl = opti.variable(N)
    kappa_rr = opti.variable(N)
    
    # controls = derivatives wrt alpha
    u_rho = opti.variable(N-1)
    u_delta = opti.variable(N-1)
    u_beta = opti.variable(N-1)
    u_kappa_fl = opti.variable(N-1)
    u_kappa_fr = opti.variable(N-1)
    u_kappa_rl = opti.variable(N-1)
    u_kappa_rr = opti.variable(N-1)

    # set initial guesses
    opti.set_initial(rho, 1.0) 
    opti.set_initial(delta, 0.0) 
    opti.set_initial(beta, 0.0) 
    opti.set_initial(kappa_fl, 0.0) 
    opti.set_initial(kappa_fr, 0.0) 
    opti.set_initial(kappa_rl, 0.0) 
    opti.set_initial(kappa_rr, 0.0)
    
    opti.set_initial(u_rho, 0.0)
    opti.set_initial(u_delta, 0.0)
    opti.set_initial(u_beta, 0.0)
    opti.set_initial(u_kappa_fl, 0.0)
    opti.set_initial(u_kappa_fr, 0.0)
    opti.set_initial(u_kappa_rl, 0.0)
    opti.set_initial(u_kappa_rr, 0.0)
    
    # state dynamics
    for k in range(N-1):
        opti.subject_to(rho[k+1] == rho[k] + dalpha*u_rho[k])
        opti.subject_to(delta[k+1] == delta[k] + dalpha*u_delta[k])
        opti.subject_to(beta[k+1] == beta[k] + dalpha*u_beta[k])
        opti.subject_to(kappa_fl[k+1] == kappa_fl[k] + dalpha*u_kappa_fl[k])
        opti.subject_to(kappa_fr[k+1] == kappa_fr[k] + dalpha*u_kappa_fr[k])
        opti.subject_to(kappa_rl[k+1] == kappa_rl[k] + dalpha*u_kappa_rl[k])
        opti.subject_to(kappa_rr[k+1] == kappa_rr[k] + dalpha*u_kappa_rr[k])
    
    # added to try to resolve errors
    for k in range(N):
        opti.subject_to(opti.bounded(-np.pi/4, beta[k], np.pi/4))
        opti.subject_to(opti.bounded(-0.3, kappa_fl[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, kappa_fr[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, kappa_rl[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, kappa_rr[k], 0.3))
    
    # qss constraints - EOMs and other equations
    for k in range(N):
        ax = rho[k]*g*ca.sin(alpha[k] - beta[k])
        ay = rho[k]*g*ca.cos(alpha[k] - beta[k])
    
        omega = ay / (V*ca.cos(beta[k]))
    
        v = V*ca.tan(beta[k])
        u = V*ca.cos(beta[k])
    
        # slip angles
        lambda_fl = delta[k] - ((v + omega*a)/(u + omega*(trackwidth/2) + 1e-6))
        lambda_fr = delta[k] - ((v + omega*a)/(u - omega*(trackwidth/2) + 1e-6))
        lambda_rl = -((v - omega*b)/(u + omega*(trackwidth/2) + 1e-6))
        lambda_rr = -((v - omega*b)/(u - omega*(trackwidth/2) + 1e-6))
    
        # normal loads
        Nfl, Nfr, Nrl, Nrr = normal_loads(ax, ay, u)
    
        fx_fl, fy_fl = mf_fx_fy(kappa_fl[k], lambda_fl, Nfl)
        fx_fr, fy_fr = mf_fx_fy(kappa_fr[k], lambda_fr, Nfr)
        fx_rl, fy_rl = mf_fx_fy(kappa_rl[k], lambda_rl, Nrl)
        fx_rr, fy_rr = mf_fx_fy(kappa_rr[k], lambda_rr, Nrr)
    
    
        FD = 0.5*CdA*rho_air*(u**2)
    
        # EOMs
        opti.subject_to(vehicleMass*ax == (fx_fl + fx_fr + fx_rl + fx_rr) - (fy_fl + fy_fr)*delta[k] - FD)
        opti.subject_to(vehicleMass*ay == (fy_fl + fy_fr + fy_rl + fy_rr) + (fx_fl + fx_fr)*delta[k])
    
        # yaw moment balance
        opti.subject_to(
            a*(fy_fl + fy_fr + (fx_fl + fx_fr)*delta[k]) - b*(fy_rl + fy_rr) + (trackwidth/2)*(fx_fl - fx_fr + fx_rl - fx_rr) - (trackwidth/2)*(fy_fl - fy_fr)*delta[k] == 0
        )
    
        # TODO: include brake ratio equilibrium?
        # forces on axle are assumed equal
        opti.subject_to(fx_fl == fx_fr)
        opti.subject_to(fx_rl == fx_rr)
    
    # contraints for power and smoothness
    u_max = 10
    
    opti.subject_to(opti.bounded(-u_max, u_rho, u_max))
    opti.subject_to(opti.bounded(-u_max, u_delta, u_max))
    opti.subject_to(opti.bounded(-u_max, u_beta, u_max))
    opti.subject_to(opti.bounded(-u_max, u_kappa_fl, u_max))
    opti.subject_to(opti.bounded(-u_max, u_kappa_fr, u_max))
    opti.subject_to(opti.bounded(-u_max, u_kappa_rl, u_max))
    opti.subject_to(opti.bounded(-u_max, u_kappa_rr, u_max))
    
    # J to minimize
    eps = 1e-37
    J = 0
    for k in range(N-1):
        J += rho[k]**2 + eps*(u_rho[k]**2 + u_delta[k]**2 + u_beta[k]**2 + u_kappa_fl[k]**2 + u_kappa_fr[k]**2 + u_kappa_rl[k]**2 + u_kappa_rr[k]**2)
    
    opti.minimize(-J)
    
    # solver
    opti.solver("ipopt", {
        'expand': True,
        'print_time': False
    }, {'max_iter': 5000, 'tol': 1e-6})
    sol = opti.solve()
    
    
    rho_sol = sol.value(rho)
    ax_hat = rho_sol*g*np.sin(alpha)
    ay_hat = rho_sol*g*np.cos(alpha)

    return rho_sol

rho_sol = main()
ax_hat = rho_sol*g*np.sin(alpha)
ay_hat = rho_sol*g*np.cos(alpha)
