import numpy as np
import casadi as ca

alpha_min = -np.pi/2
alpha_max = np.pi/2
N = 100
d_alpha = (alpha_max - alpha_min)/N

nx = 7 # num states
nu = 7 # num controls

# state
rho = ca.MX.sym('rho')
delta = ca.MX.sym('delta')
beta = ca.MX.sym('beta')
kappa_fl = ca.MX.sym('kappa_fl')
kappa_fr = ca.MX.sym("kappa_fr")
kappa_rl = ca.MX.sym('kappa_rl')
kappa_rr = ca.MX.sym('kappa_rr')

x = ca.vertcat(rho, delta, beta, kappa_fl, kappa_fr, kappa_rl, kappa_rr)

u_rho = ca.MX.sym('u_rho')
u_delta = ca.MX.sym('u_delta')
u_beta = ca.MX.sym('u_beta')
u_kappa_fl = ca.MX.sym('u_kappa_fl')
u_kappa_fr = ca.MX.sym('u_kappa_fr')
u_kappa_rl = ca.MX.sym('u_kappa_rl')
u_kappa_rr = ca.MX.sym('u_kappa_rr')

u = ca.vertcat(u_rho, u_delta, u_beta, u_kappa_fl, u_kappa_fr, u_kappa_rl, u_kappa_rr)

# differential equations
xdot = u
f = ca.Function('f', [x, u], [xdot])

# rk4 integrator
def rk4(f, x, u, h):
    k1 = f(x,u)
    k2 = f(x + h/2 * k1, u)
    k3 = f(x + h/2 * k2, u)
    k4 = f(x+h * k3, u)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

# optimization
opti = ca.Opti()
X = opti.Variable(nx, N+1)
U = opti.Variable(nu, N)

# encode vehicle and aero parameters
V = opti.Parameter()
vehicleMass = opti.Parameter()
a = opti.Parameter()
b = opti.Parameter()
trackwidth = opti.Parameter()
Cd = opti.Parameter()
A = opti.Paramter()
rho_a = opti.paramter()
roll_stiffness = opti.Parameter()
P_max = opti.Parameter()
Cl_f = opti.Parameter()
Cl_r = opti.Parameter()
g = 9.81

# tire force function
def calc_fz(ax, ay, beta, V):
    global vehicleMass, g, a, b, h
    global Cl_f, Cl_r
    
    u = V*ca.cos(beta)

    # front and rear downforces
    FL_f = 0.5*Cl_f*A*(u**2)
    FL_r = 0.5*Cl_r*A*(u**2)

    Fz_fl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + m*ay*(h/trackwidth)*roll_stiffness + 0.5*FL_f

    Fz_fr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*Fl_f

    Fz_rl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FL_r

    Fz_rr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FL_r

    return [Fz_fl, Fz_fr, Fz_fr, Fz_rl]

# dynamics constraints
for k in range(N):
    x_next = rk4(f, X[:, k], U[:, k], dalpha)
    opti.subject_to(X[:, k+1] == x_next)

# equality constraints
for k in range(N):
    # calculate Fz
    fz_mat = calc_fz(

    # k = alpha
    fx_fl = fx_calc(X[3,k]) # kappa_fl(alpha_k)
    fx_fr = fx_calc(X[4, k])
    fx_rl = fx_calc(X[5, k])
    fx_rr = fx_calc(X[6, k])

    fy_fl = fy_calc(
