import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

# todo:
# - test new model
# - remove function integrate_with_coupling


class L96:
    """
    Class for two time-scale Lorenz 1996 model
    """

    X = "Current X state or initial conditions"
    Y = "Current Y state or initial conditions"
    F = "Forcing"
    h = "Coupling coefficient"
    b = "Ratio of timescales"
    c = "Ratio of amplitudes"
    dt = "Time step"

    def __init__(self, K, J, F=18, h=1, b=10, c=10, t=0, dt=0.001):
        """Construct a two time-scale model with parameters:
        K  : Number of X values
        J  : Number of Y values per X value
        F  : Forcing term (default 18.)
        h  : coupling coefficient (default 1.)
        b  : ratio of amplitudes (default 10.)
        c  : time-scale ratio (default 10.)
        t  : Initial time (default 0.)
        dt : Time step (default 0.001)
        """
        self.F, self.h, self.b, self.c, self.dt = F, h, b, c, dt
        # Initialize system state
        self.X, self.Y, self.t = b * \
            np.random.randn(K), np.random.randn(J * K), t
        self.K, self.J = K, J

        self.JK = J * K  # For convenience
        self.k, self.j = np.arange(self.K), np.arange(self.JK)  # For plotting

    def set_param(self, dt=None, F=None, h=None, b=None, c=None, t=None):
        """Set a model parameter, e.g. .set_param(dt=0.002)"""
        if dt is not None:
            self.dt = dt
        if F is not None:
            self.F = F
        if h is not None:
            self.h = h
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if t is not None:
            self.t = t
        return self

    def set_state(self, X, Y, t=None):
        """Set initial conditions (or current state), e.g. .set_state(X,Y)"""
        # Set state
        self.X, self.Y = X, Y
        if t is not None:
            self.t = t

        # Compute K and J from state
        self.K, self.JK = self.X.size, self.Y.size
        self.J = self.JK // self.K

        # For plotting
        self.k, self.j = np.arange(self.K), np.arange(self.JK)

        return self

    def randomize_IC(self):
        """Randomize the initial conditions (or current state)"""
        X, Y = self.b * \
            np.random.rand(self.X.size), np.random.rand(self.Y.size)
        return self.set_state(X, Y)

    def run(self, si, t_total, store=False, return_coupling=False):
        """Run model for a total time of T, sampling at intervals of si.
        If store=True, then stores the final state as the initial conditions for the next segment.
        If return_coupling=True, returns C in addition to X,Y,t.
        Returns sampled history: X[:,:],Y[:,:],t[:],C[:,:]."""

        # Number of time steps
        nt = int(t_total / si)

        x_0 = self.X
        y_0 = self.Y
        t_0 = self.t

        # Result arrays
        time, x_hist, y_hist, xy_tend_hist = (
            t_0 + np.zeros((nt + 1)),
            np.zeros((nt + 1, len(x_0))),
            np.zeros((nt + 1, len(y_0))),
            np.zeros((nt + 1, len(x_0))),
        )

        x_hist[0, :] = x_0.copy()
        y_hist[0, :] = y_0.copy()
        xy_tend_hist[0, :] = 0

        # Evaluation interval with si size steps
        t_eval = np.arange(0, si * nt, si)

        ode_result = solve_ivp(L96_2t_x_dot_y_dot, (0, nt),
                               y0=np.concatenate((x_0, y_0, np.zeros(self.K,))), method='RK45',
                               eval=t_eval, args=(self.K, self.J, self.F, self.h, self.b, self.c))

        x_hist[1:] = np.swapaxes(ode_result.y[:self.K], 0, 1)
        y_hist[1:] = np.swapaxes(ode_result.y[self.K:self.K+self.JK], 0, 1)
        xy_tend_hist[1:] = np.swapaxes(ode_result.y[self.K+self.JK:], 0, 1)
        time[1:] = ode_result.t

        if store:
            self.X, self.Y, self.t = x_hist[-1], y_hist[-1], time[-1]
        if return_coupling:
            return x_hist, y_hist, time, xy_tend_hist
        else:
            return x_hist, y_hist, time

    def __repr__(self):
        return f"L96: K={self.K} J={self.J} F={self.F} h={self.h} b={self.b} c={self.c} dt={self.dt}"

    def __str__(self):
        return (self.__repr__() + f"\n X={self.X} \nY={self.Y} \nt={self.t}")

    def copy(self):
        copy = L96(self.K, self.J, F=self.F, h=self.h,
                   b=self.b, c=self.c, dt=self.dt)
        copy.set_state(self.X, self.Y, t=self.t)
        return copy

    def print(self):
        print(self)


# @njit
def L96_2t_x_dot_y_dot(t, xy_vars, K, J, F, h, b, c):
    """
    Calculate the time rate of change for the X and Y variables for the Lorenz '96, two time-scale
    model, equations 2 and 3:
        d/dt X[k] =     -X[k-1] ( X[k-2] - X[k+1] )   - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] ( Y[j+2] - Y[j-1] ) - c Y[j]     + h.c/b X[k]

    Args:
        X : Values of X variables at the current time step
        Y : Values of Y variables at the current time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio
    Returns:
        dXdt, dYdt, C : Arrays of X and Y time tendencies, and the coupling term -hc/b*sum(Y,j)
    """
    x, y = xy_vars[:K], xy_vars[K:K+K*J]

    # # Compute K and J from the data
    # JK, K = len(Y), len(X)
    # J = JK // K
    # assert JK == J * K, "X and Y have incompatible shapes"

    # Helper variables
    hcb = (h * c) / b
    y_summed = y.reshape((K, J)).sum(axis=-1)

    # L96 slow equation
    x_dot = -np.roll(x, 1) * (np.roll(x, 2) - np.roll(x, -1)) - \
        x + F - hcb * y_summed

    # L96 fast equation
    y_dot = (
        -c * b * np.roll(y, -1) * (np.roll(y, -2) - np.roll(y, 1))
        - c * y + hcb * np.repeat(x, J)
    )

    return np.concatenate((x_dot, y_dot, -hcb * y_summed))
