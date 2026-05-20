#
# 7/2023: allowing an optional weight column in the input data file + encapsulation of private functions
# 3/2019: adding G0 support stored as last gi in transport
#
#
# 12/15/2018: 
# (*) Introducing NNLS optimization of previous optimal solution
#     - can make deltaBaseWeightDist : 0.2 or 0.25 [coarser from 0.05]
#     - wrote new routines: FineTuneSolution, res_tG (for vector of residuals)
# 

from common import *
np.set_printoptions(precision=2)

def initializeDiscSpec(par):
	"""Returns:
		(*)	the experimental data: t, Gexp, wexp
		(*) the continuous spectrum: s, H (from output/H.dat)
		(*) Number of modes range: Nv
		(*) Error Weight estimate from Continuous Curve (AIC criterion)
	"""
	
	# read input; initialize parameters
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data Files: ... {}...'.format(par['GexpFile']))

	# Read experimental data
	t, Gexp, wexp = GetExpData(par['GexpFile'])

	# Read the continuous spectrum
	fNameH  = 'output/H.dat'
	s, H    = np.loadtxt(fNameH, unpack=True)

	n    = len(t);
	
	# range of N scanned
	Nmin  = max(np.floor(0.5 * np.log10(max(t)/min(t))),2);   # minimum Nopt
	Nmax  = min(np.floor(3.0 * np.log10(max(t)/min(t))),n/4); # maximum Nopt

	if(par['MaxNumModes'] > 0):
		Nmax  = min(Nmax, par['MaxNumModes'])
	
	Nv    = np.arange(Nmin, Nmax + 1).astype(int)

		
	# Estimate Error Weight from Continuous Curve Fit
	kernMat = getKernMat(s,t)
	
	if par['plateau']:

		try:
			with open(fNameH) as f:
				first_line = f.readline().split()
				if len(first_line) > 2:
					G0 = float(first_line[-1])
		except OSError:
			print('Problem reading G0 from H.dat; Plateau = True')
			quit()

		Gc      = kernel_prestore(H, kernMat, G0);
	else:
		Gc      = kernel_prestore(H, kernMat);
	
	Cerror  = 1./(np.std(wexp*(Gc/Gexp - 1.)))  #	Cerror = 1.?
	
	return t, Gexp, wexp, s, H, Nv, Gc, Cerror

def MaxwellModes(z, t, Gexp, wexp, isPlateau):
	"""
	
	 Function: MaxwellModes(input)
	
	 Solves the linear least squares problem to obtain the DRS

	 Input: z    = points distributed according to the density, [z = log(tau)]
	        t    = n*1 vector contains times,
	        Gexp = n*1 vector contains experimental data
		    wexp = weight vector of experimental data
	        isPlateau = True if G0 \neq 0
	
	 Output: g, tau = spectrum  (array)
	         error = relative error between the input data and the G(t) inferred from the DRS
	         condKp = condition number
	"""
	tau    = np.exp(z)

	#
	# Prune small and -ve weights g(i)
	#
	g, error, condKp = nnLLS(t, tau, Gexp, wexp, isPlateau)


	# search for small 
	if isPlateau:
		izero = np.where(g[:-1]/max(g[:-1]) < 1e-7)
	else:
		izero = np.where(g/max(g) < 1e-7)
	
	tau   = np.delete(tau, izero)
	g     = np.delete(g, izero)

	return g, tau, error, condKp

def nnLLS(t, tau, Gexp, wexp, isPlateau):
	"""
	#
	# Helper subfunction which does the actual LLS problem
	# helps MaxwellModes; relies on nnls
	#
	"""
	S, T    = np.meshgrid(tau, t)
	K		= np.exp(-T/S)		# n * nmodes
	
	# K is n*ns [or ns+1]
	if isPlateau:
		K = np.hstack(( K, np.ones((len(Gexp), 1)) ))
		
	#
	# gets wE*(Gt/GtE - 1)^2, instead of  (Gt -  GtE)^2
	# 7/27/2023: note with wexp; RHS becomes wexp rather than ones [wLLS is X'WX = X'Wy]
	Kp      = np.dot(np.diag((wexp/Gexp)), K)
	condKp  = np.linalg.cond(Kp)
	g       = nnls(Kp, wexp)[0]	

	GtM   	= np.dot(K, g)
	error 	= np.sum((wexp*(GtM/Gexp - 1.))**2)

	return g, error, condKp

def GetWeights(H, t, s, wb):
	"""
	%
	% Function: GetWeights(input)
	%
	% Finds the weight of "each" mode by taking a weighted average of its contribution
	% to G(t)
	%
	% Input: H = CRS (ns * 1)
	%        t = n*1 vector contains times
	%        s = relaxation modes (ns * 1)
	%       wb = weightBaseDist
	%
	% Output: wt = weight of each mode
	%
	"""
	ns         = len(s)
	n          = len(t)

	hs         = np.zeros(ns)
	wt         = hs
	
	hs[0]      = 0.5 * np.log(s[1]/s[0])
	hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))

	S, T    = np.meshgrid(s, t)
	kern    = np.exp(-T/S)		# n * ns
	wij     =  np.dot(kern, np.diag(hs * np.exp(H)))  # n * ns
	K       = np.dot(kern, hs * np.exp(H))         # n * 1, comparable with Gexp

	for i in np.arange(n):
		wij[i,:] = wij[i,:]/K[i]

	for j in np.arange(ns):
		wt[j] = np.sum(wij[:,j])

	wt  = wt/np.trapezoid(wt, np.log(s))
	wt  = (1. - wb) * wt + (wb * np.mean(wt)) * np.ones(len(wt))

	return wt

def GridDensity(x, px, N):

	"""#
	#  PROGRAM: GridDensity(input)
	#
	#	Takes in a PDF or density function, and spits out a bunch of points in
	#       accordance with the PDF
	#
	#  Input:
	#       x  = vector of points. It need *not* be equispaced,
	#       px = vector of same size as x: probability distribution or
	#            density function. It need not be normalized but has to be positive.
	#  	    N  = Number of points >= 3. The end points of "x" are included
	#  	     necessarily,
	# 
	#  Output:
	#       z  = Points distributed according to the density
	#       hz = width of the "intervals" - useful to apportion domain to points
	#            if you are doing quadrature with the results, for example.
	#
	#  (c) Sachin Shanbhag, November 11, 2015
	#"""

	npts = 100;                              # can potentially change
	xi   = np.linspace(min(x),max(x),npts)   # reinterpolate on equi-spaced axis
	fint = interp1d(x,px,'cubic')	         # smoothen using cubic splines
	pint = fint(xi)        					 # interpolation
	ci   = cumulative_trapezoid(pint, xi, initial=0)                
	pint = pint/ci[npts-1]
	ci   = ci/ci[npts-1]                     # normalize ci

	alfa = 1./(N-1)                          # alfa/2 + (N-1)*alfa + alfa/2
	zij  = np.zeros(N+1)                     # quadrature interval end marker
	z    = np.zeros(N)                       # quadrature point

	z[0]    = min(x);  
	z[N-1]  = max(x); 

	# ci(Z_j,j+1) = (j - 0.5) * alfa
	beta       = np.arange(0.5, N-0.5) * alfa
	zij[0]     = z[0]
	zij[N]     = z[N-1]
	fint       = interp1d(ci, xi, 'cubic')
	zij[1:N]   = fint(beta)
	h          = np.diff(zij)

	# Quadrature points are not the centroids, but rather the center of masses
	# of the quadrature intervals
	beta     = np.arange(1, N-1) * alfa
	z[1:N-1] = fint(beta)

	return z, h

def mergeModes_magic(g, tau, imode):
	"""merge modes imode and imode+1 into a single mode
	   return gp and taup corresponding to this new mode;
	   12/2018 - also tries finetuning before returning
	   
	   uses helper functions which are now made private to this:
	   - costFcn_magic()
	   	- normKern_magic()
	"""

	# Making private functions private: costFcn_magic is only used by mergeModes_magic
	# Furthermore normKern_magic() is only used by costFcn_magic - so
	def costFcn_magic(par, g, tau, imode):
		""""helper function for mergeModes; establishes cost function to minimize"""
		gn   = par[0]
		taun = par[1]

		g1   = g[imode]
		g2   = g[imode+1]
		tau1 = tau[imode]
		tau2 = tau[imode+1]

		tmin = min(tau1, tau2)/10.
		tmax = max(tau1, tau2)*10.

		def normKern_magic(t, gn, taun, g1, tau1, g2, tau2):
			"""helper function: for costFcn and mergeModes"""
			Gn = gn * np.exp(-t/taun)
			Go = g1 * np.exp(-t/tau1) + g2 * np.exp(-t/tau2)
			return (Gn/Go - 1.)**2


		return quad(normKern_magic, tmin, tmax, args=(gn, taun, g1, tau1, g2, tau2))[0]

	####

	iniGuess = [g[imode] + g[imode+1], 0.5*(tau[imode] + tau[imode+1])]
	res      = minimize(costFcn_magic, iniGuess, args=(g, tau, imode))

	newtau   = np.delete(tau, imode+1)
	newtau[imode] = res.x[1]
		
	return newtau

def FineTuneSolution(tau, t, Gexp, wexp, isPlateau, estimateError=False):
	"""Given a spacing of modes tau, tries to do NLLS to fine tune it further
	   If it fails, then it returns the old tau back
	   
	   Uses helper function: res_tG which computes residuals
	   """
	success = False

	# Encapsulating res_tG within FineTuneSolution
	# private since no other method needs access to this 
	def res_tG(tau, texp, Gexp, wexp, isPlateau):
		"""
			Helper function for final optimization problem
		"""
		g, _, _ = nnLLS(texp, tau, Gexp, wexp, isPlateau)
		Gmodel  = np.zeros(len(texp))

		for j in range(len(tau)):
			Gmodel += g[j] * np.exp(-texp/tau[j])
		
		# add G0
		if isPlateau:
			Gmodel += g[-1]
			
		residual = wexp * (Gmodel/Gexp - 1.)
			
		return residual

	#######
	try:
		res  = least_squares(res_tG, tau, bounds=(0., np.inf),	args=(t, Gexp, wexp, isPlateau))
		tau  = res.x
		tau0 = tau.copy()

		# Error Estimate	
		if estimateError:
			J = res.jac
			cov = np.linalg.pinv(J.T.dot(J)) * (res.fun**2).mean()
			dtau = np.sqrt(np.diag(cov))

		success = True			
	except:	
		pass
	
	g, tau, _, _ = MaxwellModes(np.log(tau), t, Gexp, wexp, isPlateau)   # Get g_i, taui

	#
	# if mode has dropped out, then need to delete corresponding dtau mode
	#
	if estimateError and success:
		if len(tau) < len(tau0):		
			nkill = 0
			for i in range(len(tau0)):
				if np.min(np.abs(tau0[i] - tau)) > 1e-12 * tau0[i]:
					dtau = np.delete(dtau, i-nkill)
					nkill += 1
		return g, tau, dtau
	elif estimateError:
		return g, tau, -1*np.ones(len(tau))
	else:
		return g, tau

def getDiscSpecMagic(par):
	"""
	# Function: getDiscSpecMagic(par)
	#
	# Uses the continuous relaxation spectrum extracted using getContSpec()
	# to determine an approximate discrete approximation.
	#
	# Input: Communicated by the datastructure "par"
	#
	# Output: Nopt    = optimum number of discrete modes
	#         [g tau] = spectrum
	#         error   = error norm of the discrete fit
	#        
	#         dmodes.dat : Prints the [g tau] for the particular Nopt
	#         aic.dat    : [N error aic]
	#         Gfitd.dat  : The discrete G(t) for Nopt [t Gt]"""

	t, Gexp, wexp, s, H, Nv, Gc, Cerror = initializeDiscSpec(par)
	
	n    = len(t);
	ns   = len(s);
	npts = len(Nv)

	# range of wtBaseDist scanned
	wtBase = par['deltaBaseWeightDist'] * np.arange(1, 1./par['deltaBaseWeightDist'])
	AICbst = np.zeros(len(wtBase))
	Nbst   = np.zeros(len(wtBase))
	nzNbst = np.zeros(len(wtBase))  # number of nonzeros
	

	# main loop over wtBaseDist
	for ib, wb in enumerate(wtBase):
				
		# Find the distribution of nodes you need
		wt  = GetWeights(H, t, s, wb)

		# Scan the range of number of Maxwell modes N = (Nmin, Nmax) 
		ev    = np.zeros(npts)
		nzNv  = np.zeros(npts)  # number of nonzero modes 

		for i, N in enumerate(Nv):

			z, hz  = GridDensity(np.log(s), wt, N)     # Select "tau" Points
			g, tau, ev[i], _ = MaxwellModes(z, t, Gexp, wexp, par['plateau'])
			nzNv[i]          = len(g)


		# store the best solution for this particular wb
		AIC        = 2. * Nv + 2. * Cerror * ev

		AICbst[ib] = min(AIC)
		Nbst[ib]   = Nv[np.argmin(AIC)]
		nzNbst[ib] = nzNv[np.argmin(AIC)]

	# global best settings of wb and Nopt; note this is nominal Nopt (!= len(g) due to NNLS)
	
	Nopt  = int(Nbst[np.argmin(AICbst)])
	wbopt = wtBase[np.argmin(AICbst)]

	#
	# Recompute the best data-set stats, and fine tune it
	#
	wt             = GetWeights(H, t, s, wbopt)	
	z, _           = GridDensity(np.log(s), wt, Nopt)           # Select "tau" Points
	g, tau, error, cKp = MaxwellModes(z, t, Gexp, wexp, par['plateau'])   # Get g_i, taui
	g, tau, dtau   = FineTuneSolution(tau, t, Gexp, wexp, par['plateau'], estimateError=True)

	#
	# Check if modes are close enough to merge
	#
	if len(tau) > 1:
		indx       = np.argsort(tau)
		tau        = tau[indx]
		tauSpacing = tau[1:]/tau[:-1]
		itry       = 0

		if par['plateau']:
			g[:-1] = g[indx]
		else:
			g      = g[indx]

		while min(tauSpacing) < par['minTauSpacing'] and itry < 3:
			print("\tTau Spacing < minTauSpacing")

			imode   = np.argmin(tauSpacing)      # merge modes imode and imode + 1	
			tau     = mergeModes_magic(g, tau, imode)

			g, tau, dtau  = FineTuneSolution(tau, t, Gexp, wexp, par['plateau'], estimateError=True)

			tauSpacing = tau[1:]/tau[:-1]
			itry      += 1


		if par['plateau']:
			G0 = g[-1]
			g  = g[:-1]

	if par['verbose']:
		print('(*) Number of optimum nodes = {0:d}'.format(len(g)))

	#
	# Some Plotting
	#

	if par['plotting']:

		plt.clf()
		plt.plot(wtBase, AICbst, label='AIC')
		plt.plot(wtBase, nzNbst, label='Nbst')
		#~ plt.scatter(wbopt, len(g), color='k')
		plt.axvline(x=wbopt, color='gray')
		plt.yscale('log')
		plt.xlabel('baseDistWt')
		plt.legend()
		plt.tight_layout()
		plt.savefig('output/AIC.pdf')		


		plt.clf()
		plt.loglog(tau,g,'o-', label='disc')
		plt.loglog(s, np.exp(H), label='cont')
		plt.xlabel(r'$\tau$')
		plt.ylabel(r'$g$')
		plt.legend(loc='lower right')
		plt.tight_layout()
		plt.savefig('output/dmodes.pdf')		


		plt.clf()
		S, T    = np.meshgrid(tau, t)
		K		= np.exp(-T/S)		# n * nmodes			
		GtM   	= np.dot(K, g)

		if par['plateau']:
			GtM += G0

		plt.loglog(t,Gexp,'o')
		plt.loglog(t,GtM, label='disc')	

		plt.loglog(t, Gc, '--', label='cont')
		plt.xlabel('t')
		plt.ylabel('G(t)')
		plt.legend()
		plt.tight_layout()
		plt.savefig('output/Gfitd.pdf')
  
	#
	# Some Printing
	#

	if par['verbose']:

		print('(*) log10(Condition number) of matrix equation: {0:.2f}'.format(np.log10(cKp)))

		if par['plateau']:
			print('(*) Plateau Modulus: {0:.3e}'.format(G0))
			np.savetxt('output/dmodes.dat', np.c_[g, tau, dtau], fmt='%e', header='G0 = {0:0.6e}'.format(G0))
		else:
			np.savetxt('output/dmodes.dat', np.c_[g, tau, dtau], fmt='%e')

		print('\n\t\tModes\n\t\t-----\n\n')
		print('  i \t    g(i) \t    tau(i)\t    dtau(i)\n')
		print('-----------------------------------------------------\n')
		
		for i in range(len(g)):
			print('{0:3d} \t {1:.5e} \t {2:.5e} \t {3:.5e}'.format(i+1,g[i],tau[i], dtau[i]))
		print("\n")

		np.savetxt('output/aic.dat', np.c_[wtBase, nzNbst, AICbst], fmt='%f\t%i\t%e')

		S, T    = np.meshgrid(tau, t)
		K		= np.exp(-T/S)		# n * nmodes			
		GtM   	= np.dot(K, g)

		if par['plateau']:
			GtM += G0		
		
		np.savetxt('output/Gfitd.dat', np.c_[t, GtM], fmt='%e')


	return Nopt, g, tau, error

#############################
#
# M A I N  P R O G R A M
#
#############################

if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = readInput('inp.dat')
	_ = getDiscSpecMagic(par)	
