import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

mu = 1.0
g = 9.81
Fz_total = 4000 # TODO: where is this coming from?
N = 36 # number of points around GG envelope

def compute_R(alpha, ax_offset=0.0, ay_scale=1.0):
    # compute the max acceleration in a given polar dir alpha
    # decision variables
    ax = ca.MX.sym("ax")
    ay = ca.MX.sym("ay")
    a_dir = ax*ca.cos(alpha) + ay*ca.sin(alpha)

    # friction circle
    g_constr = ax**2 + (ay/ay_scale)**2 - (mu*g)**2

    nlp = {'x': ca.vertcat(ax, ay),
           'f': -a_dir,
           'g': g_constr}

    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(lbx = -mu*g, ubx = mu*g, lbg=-1e-8, ubg=0)
    ax_val, ay_val = float(sol['x'][0]), float(sol['x'][1])
    return np.sqrt(ax_val**2 + ay_val**2)

alpha_range = np.linspace(0, 2*np.pi, N)

velocities = [20]
gg_velocity = []

for velocity in velocities:
    ax_offset = 0.0
    ay_scale = 1.0
    R = [compute_R(a, ax_offset, ay_scale) for a in alpha_range]
    ax = [R[i]*np.cos(alpha_range[i]) for i in range(N)]
    ay = [R[i]*np.sin(alpha_range[i]) for i in range(N)]
    gg_velocity.append((ax, ay))

fig, axs = plt.subplots(1, 1, figsize=(18,6))

for i, (ax_, ay_) in enumerate(gg_velocity):
    axs.plot(ax_, ay_, label=f"V = {velocities[i]} m/s")

axs.set_title("GGV")
axs.set_xlabel("Lateral acceleration")
axs.set_ylabel("Longitudinal acceleration")
axs.axis('equal')
axs.grid(True)
axs.legend()

plt.show()
