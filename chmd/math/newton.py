def newton(fun, x0, tol):
    x = Variable(x0)
    fx = fun(x)
    mu = 1.0
    while np.linalg.norm(fx.data) > tol:
        jac = jacobian(fx, x)
        delta_x = -np.linalg.solve(jac, fx.data)
        x_old = x.data
        x.data = x.data + delta_x.data
        assert x.grad is None
        fx_old = fx
        fx = fun(x)
        if np.linalg.norm(fx.data) > (1 - mu / 4) * np.linalg.norm(fx_old.data):
            mu *= 0.5
            x.data = x_old
    return x.data