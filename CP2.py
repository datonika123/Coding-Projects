import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
BETA, GAMMA = 0.4, 0.1
T_MAX, DT = 50, 0.5
TOL, MAX_ITER = 1e-8, 100


def solve_fixed_point(un, h):
    u = un.copy()
    for k in range(MAX_ITER):
        u_old = u.copy()
        # Fixed point update: x = F(x)
        # S = Sn / (1 + h*beta*I)
        u[0] = un[0] / (1 + h * BETA * u[1])
        # I = (In + h*beta*S*I) / (1 + h*gamma)
        u[1] = (un[1] + h * BETA * u[0] * u[1]) / (1 + h * GAMMA)
        # R = Rn + h*gamma*I
        u[2] = un[2] + h * GAMMA * u[1]

        if np.linalg.norm(u - u_old, np.inf) < TOL:
            return u, k + 1
    return u, MAX_ITER


def solve_newton_gs(un, h):
    u = un.copy()
    for k in range(MAX_ITER):
        S, I, R = u
        # Residual Vector F(u)
        F = np.array([
            S - un[0] + h * BETA * S * I,
            I - un[1] - h * (BETA * S * I - GAMMA * I),
            R - un[2] - h * GAMMA * I
        ])

        if np.linalg.norm(F, np.inf) < TOL:
            return u, k

        # 3x3 Jacobian Matrix
        J = np.array([
            [1 + h * BETA * I, h * BETA * S, 0],
            [-h * BETA * I, 1 - h * (BETA * S - GAMMA), 0],
            [0, -h * GAMMA, 1]
        ])

        # Solving J*delta = -F using Gauss-Seidel for the update vector delta
        b = -F
        delta = np.zeros(3)
        for _ in range(15):
            delta[0] = (b[0] - (J[0, 1] * delta[1] + J[0, 2] * delta[2])) / J[0, 0]
            delta[1] = (b[1] - (J[1, 0] * delta[0] + J[1, 2] * delta[2])) / J[1, 1]
            delta[2] = (b[2] - (J[2, 0] * delta[0] + J[2, 1] * delta[1])) / J[2, 2]

        u += delta
    return u, MAX_ITER


# --- Simulation Execution ---
t_steps = np.arange(0, T_MAX + DT, DT)
N = len(t_steps)
U_fp = np.zeros((N, 3))
U_nt = np.zeros((N, 3))

# Initial Conditions [S0, I0, R0]
U_fp[0] = U_nt[0] = [0.99, 0.01, 0.0]

for n in range(N - 1):
    U_fp[n + 1], _ = solve_fixed_point(U_fp[n], DT)
    U_nt[n + 1], _ = solve_newton_gs(U_nt[n], DT)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Newton-Gauss-Seidel
ax1.plot(t_steps, U_nt[:, 0], 'b-', label='Susceptible (S)')
ax1.plot(t_steps, U_nt[:, 1], 'r-', label='Infected (I)')
ax1.plot(t_steps, U_nt[:, 2], 'g-', label='Recovered (R)')
ax1.set_title("Epidemic Modeling: Newton-Gauss-Seidel Solver")
ax1.set_ylabel("Population Fraction")
ax1.legend()
ax1.grid(True)

# Plot 2: Fixed Point Iteration
ax2.plot(t_steps, U_fp[:, 0], 'b--', label='Susceptible (S)')
ax2.plot(t_steps, U_fp[:, 1], 'r--', label='Infected (I)')
ax2.plot(t_steps, U_fp[:, 2], 'g--', label='Recovered (R)')
ax2.set_title("Epidemic Modeling: Fixed Point Iteration Solver")
ax2.set_xlabel("Time (Days)")
ax2.set_ylabel("Population Fraction")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()