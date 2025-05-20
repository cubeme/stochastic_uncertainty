
"""
Provides numerical integration methods for solving ordinary differential 
equations (ODEs) in time.
"""


def euler_forward(fn, dt, x, *params):
    """
    Perform a single Euler forward time-stepping step for d/dt X = fn(X, t, ...).

    Args:
        fn (callable): The function returning the time rate of change of the variables X.
        dt (float): The time step size.
        x (np.ndarray): Current state of the variables at time t.
        params (tuple): Additional arguments to pass to `fn`.

    Returns:
        np.ndarray: Updated state of the variables at time t + dt.
    """

    return x + dt * fn(x, *params)


def RK2(fn, dt, x, *params):
    """
    Perform a single second-order Runge-Kutta (RK2) time-stepping step for d/dt X = fn(X, t, ...).

    Args:
        fn (callable): The function returning the time rate of change of the variables X.
        dt (float): The time step size.
        x (np.ndarray): Current state of the variables at time t.
        params (tuple): Additional arguments to pass to `fn`.

    Returns:
        np.ndarray: Updated state of the variables at time t + dt.
    """

    x_1 = x + 0.5 * dt * fn(x, *params)
    return x + dt * fn(x_1, *params)


def RK4(fn, dt, X, *params):
    """
    Perform a single second-order Runge-Kutta (RK2) time-stepping step for d/dt X = fn(X, t, ...).

    Args:
        fn (callable): The function returning the time rate of change of the variables X.
        dt (float): The time step size.
        x (np.ndarray): Current state of the variables at time t.
        params (tuple): Additional arguments to pass to `fn`.

    Returns:
        np.ndarray: Updated state of the variables at time t + dt.
    """

    x_dot1 = fn(X, *params)
    x_dot2 = fn(X + 0.5 * dt * x_dot1, *params)
    x_dot3 = fn(X + 0.5 * dt * x_dot2, *params)
    x_dot4 = fn(X + dt * x_dot3, *params)
    return X + (dt / 6.0) * ((x_dot1 + x_dot4) + 2.0 * (x_dot2 + x_dot3))


def RK4_two_variables(fn, dt, x, y, *params):
    """
    Perform a single fourth-order Runge-Kutta (RK4) time-stepping step for two coupled variables.

    Args:
        fn (callable): The function returning the time rate of change of the variables (x, y).
        dt (float): The time step size.
        x (np.ndarray): Current state of the first variable at time t.
        y (np.ndarray): Current state of the second variable at time t.
        params (tuple): Additional arguments to pass to `fn`.

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): Updated state of the first variable at time t + dt.
            - y (np.ndarray): Updated state of the second variable at time t + dt.
    """
    x_dot1, y_dot1 = fn(x, y, *params)
    x_dot2, y_dot2 = fn(
        x + 0.5 * dt * x_dot1, y + 0.5 * dt * y_dot1, *params)
    x_dot3, y_dot3 = fn(
        x + 0.5 * dt * x_dot2, y + 0.5 * dt * y_dot2, *params)
    x_dot4, y_dot4 = fn(
        x + dt * x_dot3, y + dt * y_dot3, *params)

    x = x + (dt / 6.0) * ((x_dot1 + x_dot4) + 2.0 * (x_dot2 + x_dot3))
    y = y + (dt / 6.0) * ((y_dot1 + y_dot4) + 2.0 * (y_dot2 + y_dot3))

    return x, y
