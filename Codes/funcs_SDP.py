import numpy as np
from numpy import linalg as LA
import picos as pic
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import SM as sm
import Measurements as me

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------PLOTTING POLYHEDRON AND OPTIMAL INSCRIBED SPHERE------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_all(vertices):

    hull = ConvexHull(vertices)
    eta = np.min(np.abs(hull.equations[:, -1]))

    theta, phi = np.linspace(0.0, np.pi, 100), np.linspace(0.0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi) #malha de variação angular
    m, n = np.size(THETA,0), np.size(THETA,1)
    X, Y, Z = np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)
    m, n = np.shape(X)
    X = X.reshape(1,m*n)
    Y = Y.reshape(1,m*n)
    Z = Z.reshape(1,m*n)

    XCIR, YCIR, ZCIR = np.empty(m*n), np.empty(m*n), np.empty(m*n)
    for i in range(m*n):
        x, y, z = X[0,i], Y[0,i], Z[0,i]
        cir = np.array([[x], [y], [z]])
        XCIR[i] = eta*cir[0]
        YCIR[i] = eta*cir[1]
        ZCIR[i] = eta*cir[2]

    XCIR = XCIR.reshape(m,n)
    YCIR = YCIR.reshape(m,n)
    ZCIR = ZCIR.reshape(m,n)

    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

    polys.set_edgecolor('deeppink')
    polys.set_linewidth(.8)
    polys.set_facecolor('hotpink')
    polys.set_alpha(.25)

    ax.add_collection3d(polys)
    ax.plot_surface(XCIR, YCIR, ZCIR, alpha = 0.9)

    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------GENERATING DETERMINISTIC STRATEGIES-------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Third block: Functions to create the deterministic strategies
# Creating the strategies_LHS(m,k) function
def strategies_LHS(m,k):
    #INPUT: m = number of measurements; k = number of results
    #k**m = number of strategies = n_lambdas

    n_lambdas = k**m
    
    #Creating the strategies
    all_est = [np.base_repr(el+n_lambdas,base=k)[-m:] for el in range(n_lambdas)]
    
    all_est = np.array([[int(digit) for digit in el] for el in all_est])

    detp = np.zeros((n_lambdas,k*m))

    for i in range(n_lambdas):
        for j in range(m):
            aux = np.zeros(k)
            aux[all_est[i][j]] = 1
            detp[i][j*k:j*k+k] = np.array(aux)    
            
    #Return the deterministic strategies
    return detp

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------SDP---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Forth block: Functions to create the SDP
# Creating the SDP_LHS(m,k,rho,rho_sep,eta,detp,medicoes) function
def SDP_LHS(rho,rho_sep,vertices, plot = False):

    hull = ConvexHull(vertices)
    eta = np.min(np.abs(hull.equations[:, -1]))

    if plot == True:
        plot_all(vertices)

    medicoes = me.measurements(vertices)
    m_k = medicoes.shape
    k = 2
    m = int(m_k[0]/k)
    detp = strategies_LHS(m,k)

    #Creating the problem
    P = pic.Problem()

    #Creating the optimization variables
    q = pic.RealVariable('q')

    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(2,2)) for i in range(k**m)]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta*chi+(1-eta)*(pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2)))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2)))

    est_det = [pic.sum([sigma[j]*detp[j,i] for j in range(k**m)]) for i in range(k*m)]

    est = [(np.kron(medicoes[i],np.eye(2)))*chi for i in range(k*m)]

    #Creating the constraints
    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)]) 

    P.add_constraint(rho_q == rho_eta)

    P.add_list_of_constraints([pic.partial_trace(est[i],subsystems=0,dimensions=(2,2))==est_det[i] for i in range(k*m)])

    #Setting the objective
    P.set_objective('max',q)

    #Finding the solution
    solution = P.solve(solver='mosek', primals = None)

    #Return the problem created, the solution found, the value of q
    return P, solution, q, solution.value