import copy
from casadi import *


class ODEModel:
    """
    This class creates an ODE model using casadi symbolic framework
    """

    def __init__(self, dt, x, dx, J=None, y=None, u=None, d=None, p=None):
        self.dt = dt  # sampling
        self.x = x  # states (sym)
        self.y = MX.sym('y', 0) if y is None else y  # outputs (sym)
        self.u = MX.sym('u', 0) if u is None else u  # inputs (sym)
        self.d = MX.sym('d', 0) if d is None else d  # disturbances (sym)
        self.p = MX.sym('p', 0) if p is None else p  # parameters (sym)
        self.dx = dx  # model equations
        self.J = J  # cost function
        #self.theta = vertcat(self.d, self.p)  # parameters to be estimated vector (sym)

    def get_equations(self, intg='idas'):
        """
        Gets equations and integrator
        """

        self.ode = {
            'x': self.x,
            'p': vertcat(self.u, self.d, self.p),
            'ode': self.dx, 
            'quad': self.J
        }  # ODE model

        self.F = Function('F', [self.x, self.u, self.d, self.p], [self.dx, self.J, self.y],
                          ['x', 'u', 'd', 'p'], ['dx', 'J', 'y'])  # model function
        self.rfsolver = rootfinder('rfsolver', 'newton', self.F)  # rootfinder
        self.opts = {'tf': self.dt}  # sampling time
        self.Plant = integrator('F', intg, self.ode, self.opts)  # integrator

    def steady(self, xguess=None, uf=None, df=None, pf=None):
        """
        Calculates root
        """

        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uf = [] if uf is None else uf
        df = [] if df is None else df
        pf = [] if pf is None else pf
        sol = self.rfsolver(x=xguess, u=uf, d=df, p=pf)
        return {
            'x': sol['y'].full(),
            'J': sol['J'].full()
        }

    def simulate_step(self, xf, uf=None, df=None, pf=None):
        """
        Simulates 1 step
        """

        uf = [] if uf is None else uf
        df = [] if df is None else df
        pf = [] if pf is None else pf
        Fk = self.Plant(x0=xf, p=vertcat(uf, df, pf))  # integration

        return {
            'x': Fk['xf'].full().reshape(-1),
            'u': uf, 
            'd': df,
            'p': pf
        }

    def check_steady(self, nss, t, cov, ysim):
        """
        Steady-state identification
        """

        s2 = [0] * len(cov)
        M = np.mean(ysim[t - nss:t, :], axis=0)
        for i in range(0, nss):
            s2 += np.power(ysim[t - i - 1, :] - M, 2)
        S2 = s2 / (nss - 1)
        if np.all(S2 <= cov):
            flag = True
        else:
            flag = False
        return {
            'Status': flag,
            'S2': S2
        }

    def build_nlp_steady(self, xguess=None, uguess=None, lbx=None, ubx=None,
                         lbu=None, ubu=None, opts={}):
        """
        Builds steady-state optimization NLP
        """
        # Guesses and bounds
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []

        # Start NLP
        self.w += [self.x, self.u]
        self.w0 += list(xguess)
        self.w0 += list(uguess)
        self.lbw += list(lbx)
        self.lbw += list(lbu)
        self.ubw += list(ubx)
        self.ubw += list(ubu)
        self.g += [vertcat(self.dx)]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        nlp = {
            'x': vertcat(*self.w),
            'p': vertcat(self.d, self.p),
            'f': self.J,
            'g': vertcat(*self.g)
        }

        # Solver
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

    def optimize_steady(self, ksim=None, df=[], pf=[]):
        """
        Performs 1 optimization step (df and pf must be lists)
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(df+pf),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step ' + str(ksim) + ': Solver did not converge.')
            else:
                print('Optimization step ' + str(ksim) + ': Optimal Solution Found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step: Solver did not converge.')
            else:
                print('Optimization step: Optimal Solution Found.')

                # Solution
        wopt = sol['x'].full()  # solution
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        return {
            'x': wopt[:self.x.shape[0]],
            'u': wopt[-self.u.shape[0]:]
        }

    def build_nlp_dyn(self, N, M, xguess, uguess, lbx=None, ubx=None, lbu=None,
                      ubu=None, m=3, pol='legendre', opts={}):
        """
        Build dynamic optimization NLP
        """

        self.m = m
        self.N = N
        self.M = M
        self.pol = pol

        # Guesses and bounds
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf*np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf*np.ones(self.u.shape[0]) if lbu is None else lbu
        ubx = +inf*np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf*np.ones(self.u.shape[0]) if ubu is None else ubu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])

        # Polynomials
        self.tau = np.array([0] + collocation_points(self.m, self.pol))
        self.L = np.zeros((self.m + 1, 1))
        self.Ldot = np.zeros((self.m + 1, self.m + 1))
        self.Lint = self.L
        for i in range(0, self.m + 1):
            coeff = 1
            for j in range(0, self.m + 1):
                if j != i:
                    coeff = np.convolve(coeff, [1, -self.tau[j]]) / \
                            (self.tau[i] - self.tau[j])
            self.L[i] = np.polyval(coeff, 1)
            ldot = np.polyder(coeff)
            for j in range(0, self.m + 1):
                self.Ldot[i, j] = np.polyval(ldot, self.tau[j])
            lint = np.polyint(coeff)
            self.Lint[i] = np.polyval(lint, 1)

        # "Lift" initial conditions
        xk = MX.sym('x0', self.x.shape[0])  # first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        uk_prev = uguess

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # Start NLP
        self.w += [xk]
        self.w0 += list(xguess)
        self.lbw += list(lbx)
        self.ubw += list(ubx)
        self.g += [xk - x0_sym]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        # NLP build
        for k in range(0, self.N):
            xki = []  # state at collocation points
            for i in range(0, self.m):
                xki.append(MX.sym('x_' + str(k + 1) + '_' + str(i + 1), self.x.shape[0]))
                self.w += [xki[i]]
                self.lbw += list(lbx)
                self.ubw += list(ubx)
                self.w0 += list(xguess)

            # uk as decision variable
            uk = MX.sym('u_' + str(k + 1), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)

            if k >= self.M:
                self.g += [uk - uk_prev]  # delta_u
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))

            uk_prev = uk

            # Loop over collocation points
            xk_end = self.L[0] * xk
            for i in range(0, self.m):
                xk_end += self.L[i + 1] * xki[i]  # add contribution to the end state
                xc = self.Ldot[0, i + 1] * xk  # expression for the state derivative at the collocation poin
                for j in range(0, m):
                    xc += self.Ldot[j + 1, i + 1] * xki[j]
                fi = self.F(xki[i], uk, self.d, self.p)  # model and cost function
                self.g += [self.dt * fi[0] - xc]  # model equality contraints reformulated
                self.lbg += list(np.zeros(self.x.shape[0]))
                self.ubg += list(np.zeros(self.x.shape[0]))
                # self.J += self.dt*fi[1]*self.Lint[i+1] #add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            xk = MX.sym('x_' + str(k + 2), self.x.shape[0])
            self.w += [xk]
            self.lbw += list(lbx)
            self.ubw += list(ubx)
            self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk - xk_end]
            self.lbg += list(np.zeros(self.x.shape[0]))
            self.ubg += list(np.zeros(self.x.shape[0]))

        self.J = fi[1]

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, self.d, self.p)
        }

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def optimize_dyn(self, xf, df=[], pf=[], ksim=None):
        """
        Performs 1 optimization step 
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(xf, df, pf),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step ' + str(ksim) + ': Solver did not converge.')
            else:
                print('Optimization step ' + str(ksim) + ': Optimal Solution Found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step: Solver did not converge.')
            else:
                print('Optimization step: Optimal Solution Found.')

        # Solution
        wopt = sol['x'].full()  # solution
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        for i in range(0, self.x.shape[0]):
            xopt[:, i] = wopt[i::self.x.shape[0] + self.u.shape[0] +
                                 self.x.shape[0] * self.m].reshape(-1)  # optimal state
        for i in range(0, self.u.shape[0]):
            uopt[:, i] = wopt[self.x.shape[0] + self.x.shape[0] * self.m + i::self.x.shape[0] +
                                                                              self.x.shape[0] * self.m + self.u.shape[
                                                                                  0]].reshape(-1)  # optimal inputs
        return {
            'x': xopt,
            'u': uopt
        }


class AEKF:
    """
    This class creates an Adaptative Extended Kalman Filter using casadi 
    symbolic framework (regular EKF if there's no theta) 
    """

    def __init__(self, dt, P0, Q, R, x, u, y, dx, theta):
        self.x = x
        self.u = u
        self.y = y
        self.theta = theta
        self.x_ = vertcat(self.x, self.theta)  # extended state vector
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement nosie covariance matrix
        self.Pk = copy.deepcopy(P0)  # estimation error covariance matrix

        # Model equations
        dx_ = []
        for i in range(0, self.x.shape[0]):
            dx_.append(x[i] + dt * dx[i])
        for j in range(0, self.theta.shape[0]):
            dx_.append(theta[j])
        self.dx_ = vertcat(*dx_)
        self.F_ = Function('F_EKF', [self.x_, self.u], [self.dx_])  # state equation
        self.JacFx_ = Function('JacFx_EKF', [self.x_, self.u],
                               [jacobian(self.dx_, self.x_)])  # jacobian of F respective to x
        self.H_ = Function('H_EKF', [self.x_, self.u], [self.y])  # output equation
        self.JacHx_ = Function('JacHx_EKF', [self.x_, self.u],
                               [jacobian(self.y, self.x_)])  # jacobian of H respective to x

    def update_state(self, xkhat, uf, ymeas):
        """
        Performs 1 model update step
        """

        Fk = self.JacFx_(xkhat, uf).full()
        xkhat_pri = self.F_(xkhat, uf).full()  # priori estimate of xk
        Pk_pri = Fk @ self.Pk @ Fk.transpose() + self.Q  # priori estimate of Pk
        Hk = self.JacHx_(xkhat_pri, uf).full()
        Kk = (Pk_pri @ Hk.T) @ (np.linalg.inv(Hk @ Pk_pri @ Hk.T + self.R))  # Kalman gain
        xkhat_pos = xkhat_pri + Kk @ ((ymeas - self.H_(xkhat_pri, uf)).full())  # posteriori estimate of xk
        self.Pk_pos = (np.eye(self.x.shape[0] + self.theta.shape[0]) - Kk @ Hk) @ Pk_pri  # posteriori estimate of Pk
        self.Pk = copy.deepcopy(self.Pk_pos)

        # Estimations
        return {
            'x': xkhat_pos[:self.x.shape[0]],
            'theta': xkhat_pos[-self.theta.shape[0]:]
        }


class LSE:
    """
    This class creates a steady-state Least-Squares parameter estimator using 
    casadi symbolic framework 
    """

    def __init__(self, F, R, x, y, u, theta, thetaguess=None, lbtheta=None,
                 ubtheta=None, rootfinder=None, opts={}):
        self.x = x
        self.y = y
        self.u = u
        self.theta = theta
        self.x0 = MX.sym('x0', self.x.shape[0])  # guess for rootfinder
        self.rfsolver = rootfinder('F_SS', 'newton', F) if rootfinder is None \
            else rootfinder  # steady-state model
        J = (self.y - vcat(self.rfsolver(self.x0, self.u, self.theta))
        [:self.x.shape[0]]).T @ R @ (self.y - vcat(self.rfsolver(self.x0,
                                                                 self.u, self.theta))[
                                              :self.x.shape[0]])  # quadratic error cost function

        # Guesses and bounds
        thetaguess = np.zeros(self.theta.shape[0]) if thetaguess is None else thetaguess
        lbtheta = -inf * np.ones(self.theta.shape[0]) if lbtheta is None else lbtheta
        ubtheta = -inf * np.ones(self.theta.shape[0]) if ubtheta is None else ubtheta

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []

        # Start NLP        
        self.w += [self.theta]  # theta as decision variable
        self.w0 += list(thetaguess)
        self.lbw += list(lbtheta)
        self.ubw += list(ubtheta)

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': J,
            'p': vertcat(self.x0, self.u, self.y)
        }

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def update_par(self, xf=None, uf=None, ymeas=None, ksim=None):
        """
        Performs 1 model update step
        """

        xf = np.zeros(self.x.shape[0]) if xf is None else xf
        uf = np.zeros(self.u.shape[0]) if uf is None else uf
        ymeas = np.zeros(self.y.shape[0]) if ymeas is None else ymeas

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(xf, uf, ymeas),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Estimation step' + str(ksim) + ': Solver did not converge.')
            else:
                print('Estimation step ' + str(ksim) + ': Optimal Solution Found.')

        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Estimation step: Solver did not converge.')
            else:
                print('Estimation step: Optimal Solution Found.')

        # Solution
        wopt = sol['x'].full()  # estimated parameters
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        return {
            'thetahat': wopt
        }


class NMPC:
    """
    This class creates an NMPC using casadi symbolic framework
    """

    def __init__(self, dt, N, M, Q, R, W, x, u, c, d, p, dx, xguess=None,
                 uguess=None, lbx=None, ubx=None, lbu=None, ubu=None, lbdu=None,
                 ubdu=None, m=3, pol='legendre', tgt=False, DRTO=False, opts={'disc': 'collocation'}):
        self.opts = opts
        self.dt = dt
        self.dx = dx
        self.x = x
        self.c = c
        self.u = u
        self.d = d
        self.p = p
        self.m = m
        self.pol = pol
        self.N = N
        self.M = M
        if tgt:  # evaluates tracking target inputs
            self.R = R
        else:
            self.R = np.zeros((self.u.shape[0], self.u.shape[0]))
        self.Q = Q
        self.W = W

        # Guesses
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        lbdu = -inf * np.ones(self.u.shape[0]) if lbdu is None else lbdu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu
        ubdu = -inf * np.ones(self.u.shape[0]) if ubdu is None else ubdu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in lbdu: lbdu = np.array([-inf if v is None else v for v in lbdu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])
        if None in ubdu: ubdu = np.array([-inf if v is None else v for v in ubdu])

        # Quadratic cost function
        self.sp = MX.sym('SP', self.c.shape[0])
        self.target = MX.sym('Target', self.u.shape[0])
        self.uprev = MX.sym('u_prev', self.u.shape[0])
        J = (self.c - self.sp).T @ Q @ (self.c - self.sp) + (self.u - self.target).T \
            @ R @ (self.u - self.target) + (self.u - self.uprev).T @ W @ (self.u - self.uprev)
        self.F = Function('F', [self.x, self.u, self.d, self.p, self.sp, self.target,
                                self.uprev], [self.dx, J], ['x', 'u', 'd', 'p',
                                'sp', 'target', 'u_prev'], ['dx', 'J'])  # NMPC model function

        # Check if the setpoints and targets are trajectories
        if not DRTO:
            spk = self.sp
            targetk = self.target
        else:
            spk = MX.sym('SP_k', 2 * (N + 1))
            targetk = MX.sym('Target_k', 2 * N)

        # "Lift" initial conditions
        xk = MX.sym('x0', self.x.shape[0])  # first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        u0_sym = MX.sym('u0_par', self.u.shape[0])
        uk_prev = u0_sym

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # Discretization
        if self.opts['disc'] == 'collocation':
            # NLP
            self.w += [xk]
            self.w0 += list(xguess)
            self.lbw += list(lbx)
            self.ubw = list(ubx)
            self.g += [xk - x0_sym]
            self.lbg += list(np.zeros(self.dx.shape[0]))
            self.ubg += list(np.zeros(self.dx.shape[0]))

            # Polynomials
            self.tau = np.array([0] + collocation_points(self.m, self.pol))
            self.L = np.zeros((self.m + 1, 1))
            self.Ldot = np.zeros((self.m + 1, self.m + 1))
            self.Lint = self.L
            for i in range(0, self.m + 1):
                coeff = 1
                for j in range(0, self.m + 1):
                    if j != i:
                        coeff = np.convolve(coeff, [1, -self.tau[j]]) / (self.tau[i] - self.tau[j])
                self.L[i] = np.polyval(coeff, 1)
                ldot = np.polyder(coeff)
                for j in range(0, self.m + 1):
                    self.Ldot[i, j] = np.polyval(ldot, self.tau[j])
                lint = np.polyint(coeff)
                self.Lint[i] = np.polyval(lint, 1)

            # NLP build
            for k in range(0, self.N):
                # State at collocation points
                xki = []
                for i in range(0, self.m):
                    xki.append(MX.sym('x_' + str(k + 1) + '_' + str(i + 1), self.x.shape[0]))
                    self.w += [xki[i]]
                    self.lbw += [lbx]
                    self.ubw += [ubx]
                    self.w0 += [xguess]

                # uk as decision variable
                uk = MX.sym('u_' + str(k + 1), self.u.shape[0])
                self.w += [uk]
                self.lbw += list(lbu)
                self.ubw += list(ubu)
                self.w0 += list(uguess)
                self.g += [uk - uk_prev]  # delta_u

                # Control horizon
                if k >= self.M:
                    self.lbg += list(np.zeros(self.u.shape[0]))
                    self.ubg += list(np.zeros(self.u.shape[0]))
                else:
                    self.lbg += list(lbdu)
                    self.ubg += list(ubdu)

                # Loop over collocation points
                xk_end = self.L[0] * xk
                for i in range(0, self.m):
                    xk_end += self.L[i + 1] * xki[i]  # add contribution to the end state
                    xc = self.Ldot[0, i + 1] * xk  # expression for the state derivative at the collocation point
                    for j in range(0, m):
                        xc += self.Ldot[j + 1, i + 1] * xki[j]
                    if not DRTO:  # check if the setpoints and targets are trajectories
                        fi = self.F(xki[i], uk, self.d,self.p, spk, targetk, uk_prev)
                    else:
                        fi = self.F(xki[i], uk, self.d, self.p, vertcat(spk[k], spk[k + N + 1]),
                                    vertcat(targetk[k], targetk[k + N]), uk_prev)
                    self.g += [self.dt * fi[0] - xc]  # model equality contraints reformulated
                    self.lbg += [np.zeros(self.x.shape[0])]
                    self.ubg += [np.zeros(self.x.shape[0])]
                    self.J += self.dt * fi[1] * self.Lint[i + 1]  # add contribution to obj. quadrature function

                # New NLP variable for state at end of interval
                xk = MX.sym('x_' + str(k + 2), self.x.shape[0])
                self.w += [xk]
                self.lbw += list(lbx)
                self.ubw += list(ubx)
                self.w0 += list(xguess)

                # No shooting-gap constraint
                self.g += [xk - xk_end]
                self.lbg += list(np.zeros(self.x.shape[0]))
                self.ubg += list(np.zeros(self.x.shape[0]))

                # u(k-1)
                uk_prev = copy.deepcopy(uk)

        elif self.opts['disc'] == 'single_shooting':
            # NLP build
            xi = x0_sym
            for k in range(0, self.N):
                uk = MX.sym('u_' + str(k + 1), self.u.shape[0])
                self.w += [uk]
                self.lbw += list(lbu)
                self.ubw += list(ubu)
                self.w0 += list(uguess)
                self.g += [uk - uk_prev]  # delta_u

                # Control horizon
                if k >= self.M:
                    self.lbg += list(np.zeros(self.u.shape[0]))
                    self.ubg += list(np.zeros(self.u.shape[0]))
                else:
                    self.lbg += list(lbdu)
                    self.ubg += list(ubdu)

                # Integrate till the end of the interval
                fi = self.F(xi, uk, self.d, self.p, spk, targetk, uk_prev)
                xi = fi['xf']
                self.J += fi['qf']

                # Inequality constraint
                self.g += [xi]
                self.lbg += list(lbx)
                self.ubw += list(ubx)

                # u(k-1)
                uk_prev = copy.deepcopy(uk)

        # NLP 
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, u0_sym, self.d, self.p, spk, targetk)
        }  # nlp construction

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def calc_actions(self, x0, u0, sp, target=[], d0=[], p0=[], ksim=None):
        """
        Performs 1 optimization step for the NMPC 
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(x0, u0, d0, p0, sp, target),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step ' + str(ksim) + ': NMPC solver did not converge.')
            else:
                print('Time step ' + str(ksim) + ': NMPC optimal solution found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step: NMPC solver did not converge.')
            else:
                print('Time step: NMPC optimal solution found.')

        # Solution
        wopt = sol['x'].full()
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step

        if self.opts['disc'] == 'collocation':
            xopt = np.zeros((self.N + 1, self.x.shape[0]))
            uopt = np.zeros((self.N, self.u.shape[0]))
            for i in range(0, self.x.shape[0]):
                xopt[:, i] = wopt[i::self.x.shape[0] + self.u.shape[0] +
                                     self.x.shape[0] * self.m].reshape(-1)  # optimal state
            for i in range(0, self.u.shape[0]):
                uopt[:, i] = wopt[self.x.shape[0] + self.x.shape[0] * self.m + i::self.x.shape[0] +
                                                                                  self.x.shape[0] * self.m +
                                                                                  self.u.shape[0]].reshape(
                    -1)  # optimal inputs

            # First control action
            uin = uopt[0, :]
            return {
                'x': xopt,
                'u': uopt,
                'U': uin
            }
        elif self.opts['disc'] == 'single_shooting':
            uopt = wopt

            # First control action
            uin = uopt[0, :]
            return {
                'u': uopt,
                'U': uin
            }
