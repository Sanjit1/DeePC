# We will create a general system class that can be used to create different systems
import numpy as np
from numpy import linalg as nla

from scipy import linalg as sla
from scipy import sparse
from scipy.signal import cont2discrete
from scipy.stats import qmc
from numpy import hstack, inf, ones
from scipy.sparse import vstack

from cvxopt import spmatrix, matrix, solvers
from cvxopt.solvers import qp
import cvxpy as cp
from osqp import OSQP
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
import subprocess
import os



from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint


# * General System Class
# At the bare minimum a system should have the following variables:
# * Variables:
# - Name: Every system deserves to have a name
# - n_input: the number of inputs to the system, this is an integer
# - n_state: the number of state variables in the system, this is an integer
# - n_y: the number of outputs from the system, this is an integer
# - inital_state: the initial state of the system, this is an array of size n_state
# - current_state: the current state of the system, this is an array of size n_state
# - y: the current output of the system
# - input_box_constraints: the box constraints of the input to the system, this is an array [min, max] for each axis
# - y_box_constraints: the box constraints of the system, this is an array [min, max] for each axis
# - input_history: a history of the inputs to the system, this is a list of arrays
# - state_history: a history of the states of the system, this is a list of arrays
# - y_history: a history of the inputs to the system, this is a list of arrays
# - noise_profile: a tuple of (mu, sigma): mu.shape is n_y and sigma.shape is (n_y, n_y) # Noise will be applied to the output of the system
# * Methods:
# - __init__: to initialize the system
# - set_state & get_state: to set & get the state of the system
# - reset: to reset the system to its initial state and clear the history
# - step: advance the system
# - plot_trajectory: Plot the trajectory of the system for 2D systems
class System:
	def __init__(self, n_input, n_state, n_y, initial_state, input_box_constraints, y_box_constraints, noise_profile, name="Black Box System"):
		"""
		Initialize the system with the given parameters.
		Parameters:
		- n_input: int, number of inputs to the system
		- n_state: int, number of state variables in the system
		- n_y: int, number of outputs from the system. This should be 2 for a 2D system.
		- initial_state: array-like, initial state of the system, shape (n_state,)
		- input_box_constraints: array-like, box constraints for the input, shape (n_input, 2)
		- y_box_constraints: array-like, box constraints for the output, shape (n_y, 2)
		- noise_profile: tuple (mu, sigma), where mu is the mean (shape (n_y,)) and sigma is the covariance matrix (shape (n_y, n_y))
		"""
		self.name = name
		self.n_input, self.n_state, self.n_y = n_input, n_state, n_y # number of inputs, states and outputs
		self.mu, self.sigma = np.zeros(n_y), np.zeros((n_y, n_y)) # initialize as None, will be set if noise_profile is provided
		# validate the dimensions of the initial state, constraints and noise profile
		initial_state = np.array(initial_state)
		if initial_state.shape != (n_state,):
			raise ValueError(f"Initial state must be of shape ({n_state},), got {initial_state.shape}")

	
		if input_box_constraints is not None:
			input_box_constraints = np.array(input_box_constraints)	
			if input_box_constraints.shape != (n_input, 2):
				raise ValueError(f"Input box constraints must be of shape ({n_input}, 2), got {input_box_constraints.shape}")
		if y_box_constraints is not None:
			y_box_constraints = np.array(y_box_constraints)
			if y_box_constraints.shape != (n_y, 2):
				raise ValueError(f"Output box constraints must be of shape ({n_y}, 2), got {y_box_constraints.shape}")
		self.tol = 1e-3 # Box constraints tolerance for numerical stability

		if noise_profile is not None:
			self.mu, self.sigma = noise_profile
			if self.mu.shape != (n_y,):
				raise ValueError(f"Noise profile Mean must be of shape ({n_y},), got {self.mu.shape}")
			if self.sigma.shape != (n_y, n_y):
				raise ValueError(f"Noise profile Covariance must be of shape ({n_y}, {n_y}), got {self.sigma.shape}")

		# set initial and current states
		self.initial_state = np.array(initial_state)
		self.current_state = np.copy(self.initial_state)
		
		# Initialize the output of the system and history
		self.y = None # Current output of the system, to be defined in subclasses
		self.input_box_constraints = input_box_constraints
		self.y_box_constraints = y_box_constraints
		self.input_history = []
		self.state_history = [ np.copy(self.initial_state) ]
		self.y_history = None # Current output history, to be defined in subclasses
  
	def set_noise_profile(self, noise_profile):
		"""
		Set the noise profile of the system.
		Parameters:
		- noise_profile: tuple (mu, sigma), where mu is the mean (shape (n_y,)) and sigma is the covariance matrix (shape (n_y, n_y))
		"""
		self.mu, self.sigma = noise_profile
		if self.mu.shape != (self.n_y,):
			raise ValueError(f"Noise profile Mean must be of shape ({self.n_y},), got {self.mu.shape}")
		if self.sigma.shape != (self.n_y, self.n_y):
			raise ValueError(f"Noise profile Covariance must be of shape ({self.n_y}, {self.n_y}), got {self.sigma.shape}")
  
	def set_state(self, state):
		"""Set the current state of the system."""
		if not isinstance(state, (list, np.ndarray)):
			raise ValueError("State must be a list or numpy array.")
		state = np.array(state)
		if state.shape != (self.n_state,):
			raise ValueError(f"State must be of shape ({self.n_state},), got {state.shape}")
		self.current_state = np.array(state)
  
	def get_state(self):
		"""Get the current state of the system."""
		return np.copy(self.current_state)

	def get_y(self):
		"""Get the current output of the system."""
		if self.y is None:
			raise ValueError("Output y has not been defined yet. Please call step() first.")
		return np.copy(self.y)

	def reset(self):
		"""Reset the system to its initial state and clear the history."""
		self.current_state = np.copy(self.initial_state)
		self.input_history = []
		self.state_history = [np.copy(self.initial_state)]
		self.y = None # Reset the output of the system, to be defined in subclasses
		self.y_history = []
  
	def delete_history(self):
		"""Delete the history of inputs, states, and outputs."""
		self.input_history = []
		self.state_history = []
		self.y_history = []

	def step(self, input_signal):
		"""
		Advance the system by one step with the given input signal.
		This method should be implemented in subclasses to define how the system evolves.
		Parameters:
		- input_signal: array-like, input to the system, shape (n_input,)
		"""
		raise NotImplementedError("The step method must be implemented in subclasses.")

	def plot_trajectory(self, fig=None, ax=None, limit_chart=True):
		"""
		Plot the trajectory of the system for 2D systems.
		This method should be implemented in subclasses to define how the trajectory is plotted.
		Parameters:
		- fig: matplotlib figure object, optional
		- ax: matplotlib axes object, optional
		- limit_chart: bool, whether to limit the chart to the box constraints
		"""
		# validate that the system is 2D
		if self.n_y != 2:
			raise ValueError("Trajectory plotting is only supported for 2D systems.")

		if ax is None or fig is None:
			fig, ax = plt.subplots()

		points = np.array(self.y_history)
		# assert the points are 2D
		if points.shape[1] != 2:
			raise ValueError(f"Output history must be 2D, got shape {points.shape}")

		# Create the segments for the line collection
		segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
		t = np.linspace(0, 1, len(segments))
		colors = cm.viridis(t)[::-1]  # Reverse the color map for the trajectory so that it goes from yellow to blue
		lc = LineCollection(segments, colors=colors)
		ax.add_collection(lc)
  
		# Mark the start and end points
		for i in range(1, len(points)-1):
			ax.plot(*points[i], 'x', color=colors[i], markersize=2)
		ax.plot(*points[0], 'o', color=colors[0], label='Start')
		ax.plot(*points[-1], 'o', color=colors[-1], label='End')
		ax.legend()
		ax.legend(loc='upper right')
  
		# Set the limits of the chart
		if self.y_box_constraints is not None and limit_chart:
			# make the axis 1.2x the size of the box constraints
			x_min = np.min(self.y_box_constraints[:, 0])
			x_max = np.max(self.y_box_constraints[:, 1]) 
			y_min = np.min(self.y_box_constraints[:, 0]) 
			y_max = np.max(self.y_box_constraints[:, 1]) 
			ax.set_xlim(x_min*1.2, x_max*1.2)
			ax.set_ylim(y_min*1.2, y_max*1.2)
			ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'r--')	  
		ax.set_xlabel('x position')
		ax.set_ylabel('y position')
		ax.set_title(f'Trajectory of the {self.name}')
		ax.grid()


# Linear System Class
# A linear system is a special case of the general system, which is represented by x_{t+1} = Ax_t + Bu_t + w_t || y_t = Cx_t + Du_t
# A Linear System should have the following additional variables:
# * Variables:
# - dt: the time step for discretization, if applicable
# - Ad: the state transition matrix, shape (n_state, n_state)
# - Bd: the input matrix, shape (n_state, n_input) where n_input is the number of inputs
# - Cd: the output matrix, shape (n_y, n_state) where n_y is the number of outputs
# - Dd: the feedthrough matrix, shape (n_y, n_input)
class LinearSystem(System):
	def __init__(self, n_state, n_input, n_y, dt, A, B, C, D, initial_state, input_box_constraints, y_box_constraints, discretized=False, noise_profile=None, name="Linear System"):
		"""
		Initialize the linear system with the given parameters.
		Parameters:
		- dt: float, time step for discretization
		- A: array-like, state transition matrix, shape (n_state, n_state)
		- B: array-like, input matrix, shape (n_state, n_input)
		- C: array-like, output matrix, shape (n_y, n_state)
		- D: array-like, feedthrough matrix, shape (n_y, n_input)
		- initial_state: array-like, initial state of the system, shape (n_state,)
		- input_box_constraints: array-like, box constraints for the input, shape (n_input, 2)
		- y_box_constraints: array-like, box constraints for the output, shape (n_y, 2)
		- noise_profile: tuple (mu, sigma), where mu is the mean (shape (n_y,)) and sigma is the covariance matrix (shape (n_y, n_y))
		- discretized: bool, whether the system is already discretized
		- name: str, name of the system
		"""
		# Validate the dimensions of the matrices and vectors
		A, B, C, D = map(np.array, (A, B, C, D))
		if not isinstance(n_state, int) or n_state <= 0 or not isinstance(n_input, int) or n_input <= 0 or not isinstance(n_y, int) or n_y <= 0:
			raise ValueError("n_state, n_input, and n_y must be positive integers.")
		if A.shape != (n_state, n_state):
			raise ValueError(f"State transition matrix A must be of shape ({n_state}, {n_state}), got {A.shape}")
		if B.shape != (n_state, n_input):
			raise ValueError(f"Input matrix B must be of shape ({n_state}, {n_input}), got {B.shape}")	
		if C.shape != (n_y, n_state):
			raise ValueError(f"Output matrix C must be of shape ({n_y}, {n_state}), got {C.shape}")
		if D.shape != (n_y, n_input):
			raise ValueError(f"Feedthrough matrix D must be of shape ({n_y}, {n_input}), got {D.shape}")
		if dt <= 0:
			raise ValueError("Time step dt must be a positive number.")
		self.n_state, self.n_input, self.n_y = n_state, n_input, n_y
		self.dt = dt
		if not discretized:
			# Discretize the system if not already done
			self.Ad, self.Bd, self.Cd, self.Dd, _ = cont2discrete((A, B, C, D), dt=dt)
		else:
			self.Ad, self.Bd, self.Cd, self.Dd = A, B, C, D
		super().__init__(n_input, n_state, n_y, initial_state, input_box_constraints, y_box_constraints, noise_profile, name)
		# set the first output
		self.y = self.Cd @ self.current_state + self.Dd @ np.zeros(self.n_input)
		self.y_history = [np.copy(self.y)]


	def step(self, input_signal):
		"""
		Advance the system by one step with the given input signal.
		Parameters:
		- input_signal: array-like, input to the system, shape (n_input,)
		"""

		if not isinstance(input_signal, (list, np.ndarray)):
			raise ValueError("Input signal must be a list or numpy array.")
		input_signal = np.array(input_signal)
		if input_signal.shape != (self.n_input,):
			raise ValueError(f"Input signal must be of shape ({self.n_input},), got {input_signal.shape}")
		# Ensure the input is within the box constraints
		if self.input_box_constraints is not None:
			for i in range(self.n_input):
				if input_signal[i] < self.input_box_constraints[i, 0] - self.tol or input_signal[i] > self.input_box_constraints[i, 1]+self.tol:
					print(f"Input Constraints violated for input {i}: {input_signal[i]} not in {self.input_box_constraints[i]}")

		noise = np.random.multivariate_normal(self.mu, self.sigma)
		self.current_state = self.Ad @ self.current_state + self.Bd @ input_signal
		self.y = self.Cd @ self.current_state + self.Dd @ input_signal + noise
		self.input_history.append(np.copy(input_signal))
		self.state_history.append(np.copy(self.current_state))
		self.y_history.append(np.copy(self.y))
		# Ensure the output is within the box constraints
		no_violation = True
		if self.y_box_constraints is not None:
			for i in range(self.n_y):
				if self.y[i] < self.y_box_constraints[i, 0] - self.tol or self.y[i] > self.y_box_constraints[i, 1] + self.tol:
					no_violation = False
					print(f"Output Constraints violated for output {i}: {self.y[i]} not in {self.y_box_constraints[i]}")
		return self.y.copy(), no_violation

	def reset(self):
		"""Reset the system to its initial state and clear the history."""
		super().reset()
		self.y = self.Cd @ self.current_state + self.Dd @ np.zeros(self.n_input)
		self.y_history = [np.copy(self.y)]


# Now make a DoubleIntegrator System
# The only thing that should change is the matrices A, B, C, D
class DoubleIntegrator(LinearSystem):
	def __init__(self, n=2, dt=1, initial_state=None, input_box_constraints=None, y_box_constraints=None, noise_profile=None, name="Double Integrator"):
		"""
		Initialize the Double Integrator system.
		Parameters:
		- n: int, number of states (default is 2 for a 2D double integrator)
		- dt: float, time step for discretization
		- initial_state: array-like, initial state of the system, shape (n,)
		- input_box_constraints: array-like, box constraints for the input, shape (n, 2)
		- y_box_constraints: array-like, box constraints for the output, shape (n, 2)
		- noise_profile: tuple (mu, sigma), where mu is the mean (shape (n,)) and sigma is the covariance matrix (shape (n, n))
		- name: str, name of the system
		"""

		# This is just a linear system with the following matrices:
		# A = [[0 eye(n)] [0 0]]
		# B = [[0] [eye(n)]]
  		# C = [[eye(n)] [0]]
		# D = [[0] [0]]
		# we can then discretize it with scipy
		A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
		B = np.vstack([np.zeros((n, n)), np.eye(n)])
		C = np.hstack([np.eye(n), np.zeros((n, n))])
		D = np.zeros((n, n))
		if initial_state is None:
			initial_state = np.zeros(n*2)
		super().__init__(n_state=n*2, n_input=n, n_y=n, dt=dt, A=A, B=B, C=C, D=D, initial_state=initial_state,
			input_box_constraints=input_box_constraints, y_box_constraints=y_box_constraints, noise_profile=noise_profile, name=name, discretized=False)



class InvertedPendulum(System):
	def __init__(self, dt=0.01, g=9.81, l=1.0, m=1.0, initial_state=None,
				 input_box_constraints=None, y_box_constraints=None, noise_profile=None, name="Inverted Pendulum"):
		"""
		Initialize the Inverted Pendulum system.

		- dt: time step
		- g: gravitational constant
		- l: length of pendulum
		- m: mass of pendulum
		- initial_state: [theta, omega]
		- input_box_constraints: box constraints on torque input u
		- y_box_constraints: box constraints on [theta, omega] output
		- noise_profile: (mu, sigma) for output noise
		"""
		self.dt = dt
		self.g = g
		self.l = l
		self.m = m

		n_state = 2  # [theta, omega]
		n_input = 1  # torque input u
		n_y = 1	  # outputs are [theta]

		if initial_state is None:
			initial_state = np.zeros(n_state)

		super().__init__(n_input, n_state, n_y, initial_state,
						 input_box_constraints, y_box_constraints, noise_profile, name)

		self.y = np.array([np.copy(self.current_state)[0]])
		self.y_history = [np.copy(self.y)]

	def step(self, input_signal):
		"""Advance the inverted pendulum by one step using Euler integration."""

		if not isinstance(input_signal, (list, np.ndarray)):
			raise ValueError("Input signal must be a list or numpy array.")
		input_signal = np.array(input_signal)
		if input_signal.shape != (self.n_input,):
			raise ValueError(f"Input signal must be of shape ({self.n_input},), got {input_signal.shape}")

		# Apply input constraints
		if self.input_box_constraints is not None:
			for i in range(self.n_input):
				if input_signal[i] < self.input_box_constraints[i, 0] - self.tol or input_signal[i] > self.input_box_constraints[i, 1] + self.tol:
					print(f"Input Constraints violated for input {i}: {input_signal[i]} not in {self.input_box_constraints[i]}")

		# Current state
		theta, omega = self.current_state
		u = input_signal[0]

		# Nonlinear dynamics
		theta_dot = omega
		omega_dot = (self.g / self.l) * np.sin(theta) + (1 / (self.m * self.l**2)) * u

		# Euler integration
		theta_next = theta + self.dt * theta_dot
		omega_next = omega + self.dt * omega_dot

		# Update state
		self.current_state = np.array([theta_next, omega_next])

		# Compute output with noise
		noise = np.random.normal(self.mu, self.sigma)[0, 0]
		self.y = np.array([self.current_state[0] + noise])
	

		# Update histories
		self.input_history.append(np.copy(input_signal))
		self.state_history.append(np.copy(self.current_state))
		self.y_history.append(np.copy(self.y))

		# Check output constraints
		no_violation = True
		if self.y_box_constraints is not None:
			for i in range(self.n_y):
				if self.y[i] < self.y_box_constraints[i, 0] - self.tol or self.y[i] > self.y_box_constraints[i, 1] + self.tol:
					no_violation = False
					print(f"Output Constraints violated for output {i}: {self.y[i]} not in {self.y_box_constraints[i]}")

		return self.y.copy(), no_violation

	def plot_trajectory(self, fig=None, ax=None):
		"""
		Plot the trajectory of the Inverted Pendulum system.
		Parameters:
		- fig: matplotlib figure object, optional
		- ax: matplotlib axes object, optional
		"""
		if ax is None or fig is None:
			fig, ax = plt.subplots()

		# plot the y_history against time(using dt)
		time = np.arange(len(self.y_history)) * self.dt
		theta_history = np.array(self.y_history).flatten()
		ax.plot(time, theta_history, label='Theta (rad)', color='blue')
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Theta (rad)')
		ax.set_title(f'Trajectory of the {self.name}')
		# if there arae box constraints, plot them
		if self.y_box_constraints is not None:
			y_min = self.y_box_constraints[0, 0]
			y_max = self.y_box_constraints[0, 1]
			ax.axhline(y_min, color='red', linestyle='--', label='Box Constraint Min')
			ax.axhline(y_max, color='red', linestyle='--', label='Box Constraint Max')

     

	def animate_trajectory(self, fig=None, ax=None, show_theta=[], save_folder='./Pendulum_frames', save_file='./figures/pendulum-trajectory.gif', fps=120, ffmpeg_path = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
		"""
		Plot the trajectory of the Inverted Pendulum system.
		Parameters:
		- fig: matplotlib figure object, optional
		- ax: matplotlib axes object, optional
		- limit_chart: bool, whether to limit the chart to the box constraints
		"""
		if ax is None or fig is None:
			fig, ax = plt.subplots()

		l = self.l  # length of the pendulum
		if save_folder:
			if not os.path.exists(save_folder):
				os.makedirs(save_folder)
			else:
				for file in os.listdir(save_folder):
					file_path = os.path.join(save_folder, file)
					try:
						if os.path.isfile(file_path) or os.path.islink(file_path):
							os.unlink(file_path)
					except Exception as e:
						print(f"Failed to delete {file_path}. Reason: {e}")
		for i, theta in enumerate(self.y_history):
			theta = theta[0]  # convert to scalar
			ax.set_xlim(-1.1*l, 1.1*l)
			ax.set_ylim(0, 1.5*l)
			ax.set_aspect('equal')
			ax.set_title(f"Frame {i}")
   
			# Draw cart as a small rectangle
			cart_width = 0.2
			cart_height = 0.1
			cart = plt.Rectangle(( - cart_width/2, 0), cart_width, cart_height, color='black')
			ax.add_patch(cart)

			# Calculate pendulum end position
			
			pendulum_x = l * np.sin(theta)
			pendulum_y = cart_height + l * np.cos(theta)
			# Draw pendulum line
			ax.plot([0, pendulum_x], [cart_height, pendulum_y], 'k-', linewidth=2)
			# show some angles that the user specified
			for theta_show in show_theta:
				pendulum_x_show = l * np.sin(theta_show)
				pendulum_y_show = cart_height + l * np.cos(theta_show)
				ax.plot([0, pendulum_x_show], [cart_height, pendulum_y_show], 'r--', linewidth=1)
			# Draw pivot point
			ax.plot(0, cart_height, 'ko')

			if save_folder:
				plt.savefig(f"{save_folder}/frame_{i:03d}.png")
			else:
				plt.show()
			plt.cla()
		plt.clf() # Close the figure
  
		# Update if needed
		  # Update if needed
			
		if save_folder:
			# C:\Program Files\ffmpeg
			subprocess.run([
				ffmpeg_path,
				"-i", os.path.join(save_folder, "frame_%03d.png"),
				"-vf", "palettegen=max_colors=5",
				"-y", os.path.join(save_folder, "palette.png")
			], check=True)
			subprocess.run([
				ffmpeg_path,
				"-r", str(fps),
				"-i", os.path.join(save_folder, "frame_%03d.png"),
				"-i", os.path.join(save_folder, "palette.png"),
				"-filter_complex", f"fps={fps},scale=600:-1:flags=lanczos[x];[x][1:v]paletteuse",
				"-y", save_file
			], check=True)