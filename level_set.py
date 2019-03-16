#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
phi = np.random.randn(20, 20)
F = 0.2
dt = 1
it = 100

for i in range(it):
	dphi = np.gradient(phi)
	dphi = np.array(dphi)
	dphi_norm = np.sqrt(np.sum(dphi**2, axis=0))

	phi = phi + dt * F * dphi_norm

	plt.contour(phi, 0)
	plt.show()