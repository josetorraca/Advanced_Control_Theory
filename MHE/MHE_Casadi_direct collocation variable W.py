%    This file is part of CasADi.
%
%     CasADi -- A symbolic framework for dynamic optimization.
%     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
%                             K.U. Leuven. All rights reserved.
%     Copyright (C) 2011-2014 Greg Horn
%
%     CasADi is free software; you can redistribute it and/or
%     modify it under the terms of the GNU Lesser General Public
%     License as published by the Free Software Foundation; either
%     version 3 of the License, or (at your option) any later version.
%
%     CasADi is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%     Lesser General Public License for more details.
%
%     You should have received a copy of the GNU Lesser General Public
%     License along with CasADi; if not, write to the Free Software
%     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

% A function initializing the optimal control problem
% Based on an implementation of direct collocation by Joel Andersson

function [MHEprop, solver] = moving_horizon_estimation_ATU_var_Q(MHEmodel, MHEprops, x0, W)

import casadi.*

% Degree of interpolating polynomial
intPolDeg = MHEprops.d;

% Get collocation points
tau_root = [0 collocation_points(intPolDeg, 'legendre')];

% Coefficients of the collocation equation
C = zeros(intPolDeg+1,intPolDeg+1);

% Coefficients of the continuity equation
D = zeros(intPolDeg+1, 1);

% Coefficients of the quadrature function
B = zeros(intPolDeg+1, 1);

% Construct polynomial basis
for j=1:intPolDeg+1
  % Construct Lagrange polynomials to get the polynomial basis at the collocation point
  coeff = 1;
  for r=1:intPolDeg+1
    if r ~= j
      coeff = conv(coeff, [1, -tau_root(r)]);
      coeff = coeff / (tau_root(j)-tau_root(r));
    end
  end
  % Evaluate the polynomial at the final time to get the coefficients of the continuity equation
  D(j) = polyval(coeff, 1.0);

  % Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
  pder = polyder(coeff);
  for r=1:intPolDeg+1
    C(j,r) = polyval(pder, tau_root(r));
  end

  % Evaluate the integral of the polynomial to get the coefficients of the quadrature function
  pint = polyint(coeff);
  B(j) = polyval(pint, 1.0);
end
% Time horizon
T = MHEprops.T;

% Control discretization
N = MHEprops.N;
h = T/N;

% Declare model variables
y = MHEmodel.y;
x = MHEmodel.x;
u = MHEmodel.u;
% W = MHEmodel.W;
nx = length(x);
ny = length(y);
nu = length(u);

% Model equations
xdot = MHEmodel.x_dot;

% Noise term, inherited from the model structure
L = MHEmodel.costfun;

% Whether a sample is active or not

% Continuous time dynamics
f = Function('f', {x, y, u}, {xdot, L});

% Constraints
lbx = MHEmodel.x_lower;
ubx = MHEmodel.x_upper;

lbw = ones(nx,1)*-inf;
ubw = ones(nx,1)*inf;

% Make x0 a column matrix
x0 = reshape(x0,nx,1);

%% Start with an empty NLP

decVar = {}; % Total, final decision variable vector
decVar_Xk = {};
decVar_Wk = {};
decVar_Xkj = {};

decVar0 = []; % Total, final initial guess vector
decVar_Xk_0 = [];
decVar_Wk_0 = [];
decVar_Xkj_0 = [];

lbDecVar = [];
ubDecVar = [];

lbXk = [];
ubXk = [];
lbXkj = [];
ubXkj = [];
lbWk = [];
ubWk = [];

J = [];
g={};
lbg = [];
ubg = [];

% Variable inputs to MHE

Y_in = MX.sym('Y', N*ny);
Y = reshape(Y_in,[N,ny]);

U_in = MX.sym('U', N, nu);
U = reshape(U_in,[N,nu]);

P_mat_in = MX.sym('P_mat',(nx)^2);
P_mat = reshape(P_mat_in,[nx,nx']);

% W_in = MX.sym('W',(nx)^2);
% W = reshape(W_in,[nx,nx']);

% Initial conditions
X0 = MX.sym('X0', nx);
decVar_Xk = {decVar_Xk{:}, X0};
lbXk = [lbXk; lbx];
ubXk = [ubXk; ubx];
decVar_Xk_0 = [decVar_Xk_0; x0];

Xk = X0;

% Variable initial value
X0_var = MX.sym('X0_var',nx);

J = ([X0 - X0_var])'*P_mat*([X0 - X0_var]);

% Formulate the NLP
for k=0:N-1
    
    % New NLP variable for the state noise (USALLY DENOTED BY W)
    Wk = MX.sym(['W_' num2str(k)], nx); % Creating uncertainty number k
    decVar_Wk = {decVar_Wk{:}, Wk};
    lbWk = [lbWk; lbw];
    ubWk = [ubWk; ubw];
    decVar_Wk_0 = [decVar_Wk_0; zeros(nx,1)];

    % State at collocation points
    Xkj = {};
    for j=1:intPolDeg   % For all collocation points d
        Xkj{j} = MX.sym(['X_' num2str(k) '_' num2str(j)], nx);   % Creating state number j interpolating between k and k+1: Two variables for two states
        decVar_Xkj = {decVar_Xkj{:}, Xkj{j}};
        lbXkj = [lbXkj; lbx];
        ubXkj = [ubXkj; ubx];
        decVar_Xkj_0 = [decVar_Xkj_0; x0];
    end

    % Loop over collocation points
    Xk_end = D(1)*Xk;
    
    for j=1:intPolDeg
       % Expression for the state derivative at the collocation point
       xp = C(1,j+1)*Xk;
       for r=1:intPolDeg
           xp = xp + C(r+1,j+1)*Xkj{r};
       end

       Yk = Y(k+1, :);
       Uk = U(k+1, :);
       
       % Append collocation equations
       [fj, qj] = f(Xkj{j}, Yk, Uk);
       g = {g{:}, h*fj - xp};
       lbg = [lbg; zeros(nx,1)];
       ubg = [ubg; zeros(nx,1)];

       % Add contribution to the end state
       Xk_end = Xk_end + D(j+1)*Xkj{j};

       % Add contribution to quadrature function
       J = J + B(j+1)*qj*h;
    end

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], nx);
    decVar_Xk = {decVar_Xk{:}, Xk};
    lbXk = [lbXk; lbx];
    ubXk = [ubXk; ubx];
    decVar_Xk_0 = [decVar_Xk_0; x0];

    % Add equality constraint
    g = {g{:}, (Xk_end + Wk) - Xk};
    lbg = [lbg; zeros(nx,1)];
    ubg = [ubg; zeros(nx,1)];

    % Add contribution for state noise
    J = J + Wk.^2'*W;
end

decVar = {decVar_Xk{:}, decVar_Wk{:}, decVar_Xkj{:}};
decVar0 = [decVar_Xk_0; decVar_Wk_0; decVar_Xkj_0];

lbDecVar = [lbXk; lbWk; lbXkj];
ubDecVar = [ubXk; ubWk; ubXkj];

MHEprop = struct;
MHEprop.J = J;
MHEprop.w = decVar;
MHEprop.g = g;
MHEprop.w0 = decVar0;
MHEprop.lbw = lbDecVar;
MHEprop.ubw = ubDecVar;
MHEprop.lbg = lbg;
MHEprop.ubg = ubg;

% Calculating corresponding times to each collocation point
repColPoints = [0, repmat([tau_root(2:end) 1]*h,1,N)]';
baseTimes = [0; reshape((0:h:T-h).*ones(1+intPolDeg,1),[N*(intPolDeg+1),1])];

MHEprop.timeseries = repColPoints+baseTimes;    % Timeseries


opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'eval_errors_fatal',true, ...
    'ipopt',struct('print_level',1) ...
    );

MHE_NLP = struct('f', J, 'x', vertcat(decVar{:}), 'g', vertcat(g{:}), 'p', vertcat(Y_in, U_in, X0_var, P_mat_in));
solver = nlpsol('solver', 'ipopt', MHE_NLP);
