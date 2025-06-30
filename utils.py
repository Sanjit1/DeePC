# Imports
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as nla
from scipy import linalg as sla
from scipy import sparse
from scipy.signal import cont2discrete
from scipy.stats import qmc
from scipy.linalg import svd
from cvxopt.solvers import qp
import cvxpy as cp
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
import matplotlib.pyplot as plt
from matplotlib import  rcParams
# Background color settings to match the plots with the presentation theme
# plt.rcParams['figure.facecolor'] = '#F6F6F8'
# plt.rcParams['axes.facecolor'] = '#F6F6F8'
# rcParams['figure.facecolor'] = '#F6F6F8'
# rcParams['axes.facecolor'] = '#F6F6F8'
from matplotlib import rc
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
import time
import importlib
import systems  # Replace with the module you are working on
import utils
from tqdm import tqdm

def block_hankel(data, L):
	data = data.T
	nmp = data.shape[0] # nmp means the number of n/m/p, aka how many "state"/"input" variables we have
	T = data.shape[1] # T means the number of time steps we have available
	# L is the length of the block hankel matrix, aka context length
	# we need to create:
	# [data[0], data[1], data[2], ... data[T-L]]
	# [data[1], data[2], data[3], ... data[T-L+1]]
	# ...
	# [data[L-1], data[L], data[L+1], ... data[T-1]]
	# we will start by making a 3D tensor of shape (L, T-L+1, nmp)
	block = np.zeros((L, T-L+1, nmp))
	for i in range(L):
		block[i, :, :] = data[:, i:T-L+1+i].T
	# we need to make it such each state var is in the columns, so it should end up being (L*nmp, T-L+1)
	block = block.transpose(0, 2, 1) # this will make it (L, nmp, T-L+1)
	block = block.reshape((L*nmp, T-L+1)) # this will make it (L*nmp, T-L+1)
	return block


class DeePC:
	def __init__(self, system, T=100, order_estimate=2):
		"""
		DeePC class for Data Efficient Predictive Control (DeePC) implementation.
		Args:
			system: The system to be controlled, should have methods step() and get_state().
			T: Length of the input sequence for offline data collection.
			order_estimate: How much context we believe the system requires
		"""
		self.system = system
		self.n_input = system.n_input  # number of input variables
		self.n_state = system.n_state # number of state variables
		self.n_y = system.n_y  # number of output variables # this is 2 for a 2D system
		self.default_y_box_constraints = self.system.y_box_constraints
		self.default_input_box_constraints = self.system.input_box_constraints
  
		self.T = T  # Length of Offline data collection sequence
		self.order_estimate = order_estimate  # length of the input sequence for the initial condition

		# offline data collection storage
		self.U_off_list, self.Y_off_list = [], []  # Lists to store offline data
		self.U_off_data, self.Y_off_data = None, None  # Numpy arrays to store offline data

		# online data list
		self.U_on_list, self.Y_on_list = [], []  # Lists to store online data
		self.u_ini, self.y_ini = None, None  # Initial condition/context for the online data

		# Hankel matrices
		self.Hu, self.Hy, self.Up, self.Uf, self.Yp, self.Yf = None, None, None, None, None, None
		self.last_horizon = None  # To store the last horizon used for solving

	def disable_box_constraints(self):
		# disable box constraints
		self.system.y_box_constraints = None
		self.system.input_box_constraints = None
  
	def enable_box_constraints(self):
		self.system.y_box_constraints = self.default_y_box_constraints
		self.system.input_box_constraints = self.default_input_box_constraints

  
	def collect_offline_data(self, U=None):
		# first, disable the box constraints so that we can collect data
		self.disable_box_constraints()
		# Collect data for DeePC by creating a random input sequence
		if U is None:
			U = np.random.uniform(-1, 1, (self.T, self.n_input))
		U = np.array(U)

		for u in U:
			# set the state
			self.Y_off_list.append(self.system.y)
			self.system.step(u)
			self.U_off_list.append(u)
		
		self.U_off_data = np.array(self.U_off_list)
		self.Y_off_data = np.array(self.Y_off_list)
  
		# now, we can restore the box constraints
		self.enable_box_constraints()
  
	def step_and_collect(self, u):
		# this will step and collect online data

		self.U_on_list.append(u)
		self.system.step(u)
		self.Y_on_list.append(self.system.y)
  
  
		# we only need the last order_estimate steps of data to get the initial condition/context
		self.U_on_list = self.U_on_list[-self.order_estimate:]
		self.Y_on_list = self.Y_on_list[-self.order_estimate:]
		self.u_ini = np.array(self.U_on_list).flatten()
		self.y_ini = np.array(self.Y_on_list).flatten()

	def build_hankel(self, max_horizon):
		# build the hankel matrices from the offline data
		L = max_horizon + self.order_estimate
		self.Hu = block_hankel(self.U_off_data, L)
		self.Hy = block_hankel(self.Y_off_data, L)

		n_input, n_y = self.n_input, self.n_y
		self.Up = self.Hu[:n_input * self.order_estimate, :]
		self.Uf = self.Hu[n_input * self.order_estimate:, :]
		self.Yp = self.Hy[:n_y * self.order_estimate, :]
		self.Yf = self.Hy[n_y * self.order_estimate:, :]
  
  
	def solve(self, setpoint, horizon, Q=None, Qf=None, R=None, Lg=1e-6, Ly = 1e-6):
		n_input, n_y = self.n_input, self.n_y

		setpoint = np.tile(setpoint, horizon) # this is unstacked: [setpoint, setpoint, ..., setpoint], where setpoint = [px, py, vx, vy]

		# setup a cp variable for x0 rolled N times
		u = cp.Variable((n_input*(horizon)))
		y = cp.Variable((n_y*(horizon)))
		g = cp.Variable((self.Up.shape[1]))
		sigma_y = cp.Variable((self.Yp.shape[0]))

		# First, lets setup the cost function
		Q_stacked = sparse.kron(np.eye(horizon-1), Q).toarray()
		Q_stacked = sla.block_diag(Q_stacked, Qf)
		R_stacked = sparse.kron(sparse.eye(horizon), R)

		cost = cp.quad_form(y - setpoint, Q_stacked) + cp.quad_form(u, R_stacked) + Lg * cp.norm1(g) + Ly * cp.norm1(sigma_y)

		# setup the DeePC constraints
		constraints = [
			self.Up @ g == cp.vec(self.u_ini, order='F'),
			self.Yp @ g + sigma_y == cp.vec(self.y_ini, order='F'),
			self.Uf @ g == u,
			self.Yf @ g == y,
		]

		# add box constraints
		if self.system.y_box_constraints is not None:
			for i in range(self.n_y):
				constraints.append(y[i] >= self.system.y_box_constraints[i, 0])
				constraints.append(y[i] <= self.system.y_box_constraints[i, 1])
		# add box constraints for the input
		if self.system.input_box_constraints is not None:
			for i in range(self.n_input):
				constraints.append(u[i] >= self.system.input_box_constraints[i, 0])
				constraints.append(u[i] <= self.system.input_box_constraints[i, 1])
  
		prob = cp.Problem(cp.Minimize(cost), constraints)
		# prob.solve(solver=cp.SCS)
		prob.solve()

		if prob.status not in ["optimal", "optimal_inaccurate"]:
			print("Problem not solved optimally:", prob.status)
			return None, None


		# Reshape and return
		u_opt = u.value.reshape(horizon, n_input)
		y_opt = y.value.reshape(horizon, n_y)
		return u_opt, y_opt

	def build_and_solve(self, setpoint, horizon=10, Q=None, Qf=None, R=None, Lg=1e-6, Ly = 1e-6):
		# if the horizon is different from the last one, we need to rebuild the hankel matrices
		if self.last_horizon != horizon:
			self.last_horizon = horizon
			self.build_hankel(horizon)
		return self.solve(setpoint, horizon, Q, Qf, R, Lg, Ly)


class DeePC_LRA(DeePC):
	def __init__(self, system, T=100, order_estimate=2):
		"""
		DeePC class for Data Efficient Predictive Control (DeePC) implementation.
		Args:
			system: The system to be controlled, should have methods step() and get_state().
			horizon: Prediction/Control horizon for the control inputs.
			T: Length of the input sequence for offline data collection.
			order_estimate: How much context we believe the system requires
		"""
		super().__init__(system, T=T, order_estimate=order_estimate)

	def apply_lra(self, verbose=False):
		# Apply a low-rank approximation to the Hankel matrices
		H_stacked = np.vstack((self.Up, self.Yp, self.Uf, self.Yf))
		U, s, Vh = svd(H_stacked, full_matrices=False)
		min_rank = self.Up.shape[0] + self.Yp.shape[0] + 1
  
		V_l = Vh.T
		# print pre rank shapes
		if verbose:
			print("Pre rank shapes:", self.Up.shape, self.Yp.shape, self.Uf.shape, self.Yf.shape)
		self.Up = self.Up @ V_l
		self.Yp = self.Yp @ V_l
		self.Uf = self.Uf @ V_l	
		self.Yf = self.Yf @ V_l
		# print post rank shapes
		if verbose:
			print("Post rank shapes:", self.Up.shape, self.Yp.shape, self.Uf.shape, self.Yf.shape)
  
	def build_and_solve(self, setpoint, horizon=10, Q=None, Qf=None, R=None, Lg=0.000001, Ly=0.000001):
		# if the horizon is different from the last one, we need to rebuild the hankel matrices
		if self.last_horizon != horizon:
			self.last_horizon = horizon
			self.build_hankel(horizon)
			self.apply_lra()
		return self.solve(setpoint, horizon, Q, Qf, R, Lg, Ly)