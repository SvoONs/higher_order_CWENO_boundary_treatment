# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:39:52 2018

Script providing classes of Runge-Kutta time integrators of order 3,5,6 and 7.

@author: Sven
"""

import numpy as np
import scipy as sp
from numerical_helpers import jacobian


class Runge_Kutta:
    """ Base class for one step Runge Kutta time integrator methods. """

    def __init__(self, u0, T, tau, function, A=None, b=None):
        """ The constructor. """
        self.u0 = u0
        self.T = T
        self.tau = tau
        self.f = function
        self.A = A
        self.b = b
        self.d = 1 if len(u0.shape) == 1 else u0.shape[0]
        self.N = u0.shape[0] if len(u0.shape) == 1 else u0.shape[1]
        self.s = int(0 if b is None else len(self.b))
        
    def step(self,u):
        """ Method executing time stepping. """
        if self.d == 1:
            k = np.zeros([self.N,self.s])
            for i in range(self.s):
                k[:,i] = self.f(u + self.tau*np.dot(k,self.A[i,:])) 
        else:
            k = np.zeros([self.d,self.N,len(self.b)])
            for i in range(len(self.b)):
                k[:,:,i] = self.f(u + self.tau*np.dot(k,self.A[i,:]))    
                
        return u + self.tau*np.dot(k,self.b)      
        
    def solve(self):
        """ Method solving problem up to desired time horizon. """
        if self.d == 1:
            solution = np.zeros([self.N,int(self.T/self.tau)+1])
            solution[:,0] = self.u0
            for t in range(1,int(self.T/self.tau)+1):
                solution[:,t] = self.step(solution[:,t-1])        
        else:
            solution = np.zeros([self.d,self.N,int(self.T/self.tau)+1])
            solution[:,:,0] = self.u0
            for t in range(1,int(self.T/self.tau)+1):
                solution[:,:,t] = self.step(solution[:,:,t-1])

        return solution
        
        
class Explicit_RK3(Runge_Kutta):
    """ Subclass implementing the Strong Stability Preserving Runge Kutta method
    of order 3 (SSP RK3). """
    
    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function) 
        self.A = np.array([[0,0,0], 
                           [1,0,0], 
                           [1/4,1/4,0]])
        self.b = np.array([1/6, 1/6, 2/3])
        self.s = len(self.b)
        
        
class Explicit_RK5(Runge_Kutta):
    """ Subclass implementing the Runge Kutta method of order 5. """
    
    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array([[0,0,0,0,0,0], 
                           [1/3,0,0,0,0,0], 
                           [4/25,6/25,0,0,0,0], 
                           [1/4,-3,15/4,0,0,0], 
                           [2/27,10/9,-50/81,8/81,0,0], 
                           [2/25,12/25,2/15,8/75,0,0]])
        self.b = np.array([23/192, 0, 125/192, 0, -27/64, 125/192])
        self.s = len(self.b)

    def apply_alternative_solver(self):
        """ Alternative Butcher tableau for explicit RK method of order 5. """
        self.A = np.array([[0,0,0,0,0,0], 
                           [1/4,0,0,0,0,0], 
                           [1/8,1/8,0,0,0,0], 
                           [0,0,1/2,0,0,0], 
                           [3/16,-3/8,3/8,9/16,0,0], 
                           [-3/7,8/7,6/7,-12/7,8/7,0]])
        self.b = np.array([7/90, 0, 16/45, 2/15, 16/45, 7/90])
        self.s = len(self.b) 
        
        
class Explicit_RK7(Runge_Kutta):
    """ Subclass implementing the Runge Kutta method of order 7. """
    
    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array([[0,0,0,0,0,0,0,0,0], [1/6,0,0,0,0,0,0,0,0], [0,1/3,0,0,0,0,0,0,0],
                     [1/8,0,3/8,0,0,0,0,0,0], [148/1331,0,150/1331,-56/1331,0,0,0,0,0],
                     [-404/243,0,-170/27,4024/1701,10648/1701,0,0,0,0], 
                     [2466/2401,0,1242/343,-19176/16807,-51909/16807,1053/2401,0,0,0],
                     [5/154,0,0,96/539,-1815/20384,-405/2464,49/1144,0,0],
                     [-113/32,0,-195/22,32/7,29403/3584,-729/512,1029/1408,21/16,0]])
        self.b = np.array([0,0,0,32/105,1771561/6289920,243/2560,16807/74880,77/1440,11/270])
        self.s = len(self.b)  
        
        
class Explicit_Dormand_Prince(Runge_Kutta):
    """ Subclass implementing the embedded Runge Kutta method of order 5 
    by Dormand and Prince. This is the standard solver used in Matlabs 'ode45'. """
    
    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array([[0,0,0,0,0,0,0], 
                           [1/5,0,0,0,0,0,0], 
                           [3/40,9/40,0,0,0,0,0], 
                           [44/45,-56/15,32/9,0,0,0,0], 
                           [19372/6561,-25360/2187,64448/6561,-212/729,0,0,0], 
                           [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
                           [35/384,0,500/1113,125/192,-2187/6784,11/84,0]])
        self.b = np.array([35/384,0,500/1113,125/192,-2187/6784,11/84,0])
        self.bHat = np.array([5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40])
        self.s = len(self.b)
        self.T = T
        self.t = 0
        
    def solve(self, TOL, rho, q):
        """ Main routine solving the problem up to the desired time horizon. """
        solution = []
        times = []
        solution.append(self.u0)
        times.append(self.t)
        i = 1
        while self.t < self.T:
            k = np.zeros([self.N,self.s])
            # build k's
            for j in range(0,self.s):
                k[:,j] = self.f(solution[i-1] + self.tau*np.dot(k,self.A[j,:]))
            u1 = solution[i-1] + self.tau*np.dot(k,self.b)
            u2 = solution[i-1] + self.tau*np.dot(k,self.bHat)
            epshat = np.max(np.abs(u1-u2))
            tau_opt = self.tau*min(q,(rho*TOL/epshat)**(1/5))
            if epshat <= TOL:
                solution.append(u1)
                i = i+1
                self.t += self.tau
                self.tau = min(self.tau,self.T-self.t)
                times.append(self.t)
            else:
                self.tau = tau_opt
        shape = (solution[0].shape[0],len(solution))
        solution = np.concatenate(solution).reshape(shape)
        
        return solution, times
        

class Implicit_Runge_Kutta(Runge_Kutta):
    """ Subclass implementing the routines for implicit Runge Kutta methods. """

    def __init__(self, u0, T, tau, function, A=None, b=None):
        """ The constructor. """
        super().__init__(u0, T, tau, function, A, b)
        self.tol = 1e-04
            
    def step(self, u):
        """ Method performing time stepping. """
        return u + self.tau*self.phi(u)
        
    def phi(self, u):
        """ Calculates the summation of b_j*u_j in one step of the Runge Kutta method. """
        M = 20 # max number of newton iterations
        stage_der = np.array(self.s*[self.f(u)]) # initial value: u’_0
        J = jacobian(self.f, u)
        stage_val = self.phi_solve(u, stage_der, J, M)
        return np.array([np.dot(self.b, 
                                stage_val.reshape(self.s,self.N)[:,j]) for j in range(self.N)])
                                
    def phi_solve(self, u, init_val, J, M):
        """ This function solves the sm x sm system
        F(u_i)=0
        by Newton’s method with an initial guess init_val. """
        JJ = np.eye(self.s*self.N)-self.tau*np.kron(self.A, J)
        lu_factor = sp.linalg.lu_factor(JJ)
        for i in range(M):
            init_val, norm_d = self.phi_newtonstep(u, init_val, lu_factor)
            if norm_d < self.tol:
                # print('Newton converged in {} steps'.format(i))
                break
            elif i == M-1:
                raise ValueError('The Newton iteration did not converge.')
        return init_val
        
    def phi_newtonstep(self, u, init_val, lu_factor):
        """ Takes one Newton step by solvning
        G’(u_i)(u^(n+1)_i-u^(n)_i)=-G(u_i)
        where
        G(u_i) = u_i - y_n - h*sum(a_{ij}*u’_j) for j=1,...,s. """
        d = sp.linalg.lu_solve(lu_factor, - self.F(init_val.flatten(), u))
        return init_val.flatten() + d, np.linalg.norm(d)
        
    def F(self, stage_der, u):
        """ Returns the subtraction u’_{i}-f(t_{n}+c_{i}*h, u_{i}),
        where u are the stage values, u’ the stage derivatives and f the
        function of the IVP y’=f(t,y) that should be solved by the RK-method. """
        stage_der_new = np.empty((self.s,self.N)) # the i:th stage_der is on the i:th row
        for i in range(self.s): #iterate over all stage_der
            stage_val = u + np.array([self.tau*np.dot(self.A[i,:], stage_der.reshape(self.s,self.N)[:, j]) 
                                                      for j in range(self.N)])
            stage_der_new[i, :] = self.f(stage_val)
        return stage_der - stage_der_new.reshape(-1)
        

class Gauss_Legendre(Implicit_Runge_Kutta):
    """ Gauss Legendre quadrature used for time integration of order 6. """

    def __init__(self, u0, T, tau, function, TOL=1e-04):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array([[5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
                          [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
                          [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]])
        self.b = [5/18,4/9,5/18]
        self.c = [1/2-np.sqrt(15)/10,1/2,1/2+np.sqrt(15)/10]
        self.s = len(self.b)
        self.tol = TOL
    

class Explicit_RK3_2D_Systems:
    """ Base class for one step SSP Runge Kutta time integrator method of
    order 3 for systems. """
    
    A = np.array([[0,0,0], [1,0,0], [1/4,1/4,0]])
    b = np.array([1/6, 1/6, 2/3])
    
    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        self.u0 = u0
        self.T = T
        self.tau = tau
        self.f = function
        self.d = u0.shape[0]
        self.Nx = u0.shape[1]
        self.Ny = u0.shape[2]
        self.s = int(0 if self.b is None else len(self.b))
        
    def step(self,u):
        """ Method executing time stepping. """
        k = np.zeros([self.d,self.Nx,self.Ny,len(self.b)])
        for i in range(len(self.b)):
            k[:,:,:,i] = self.f(u + self.tau*np.dot(k,self.A[i,:]))    
                
        return u + self.tau*np.dot(k,self.b)      
        
    def solve(self):
        """ Method solving problem up to desired time horizon. """
        solution = np.zeros([self.d,self.Nx,self.Ny,int(self.T/self.tau)+1])
        solution[:,:,:,0] = self.u0
        for t in range(1,int(self.T/self.tau)+1):
            solution[:,:,:,t] = self.step(solution[:,:,:,t-1])

        return solution
        

class Explicit_RK3_1D_SWE:
    """ Base class for one step SSP Runge Kutta time integrator method of
    order 3 for systems. The class is modified in order to integrate the 
    SWE with adaptive time step size. """
    
    A = np.array([[0,0,0], [1,0,0], [1/4,1/4,0]])
    b = np.array([1/6, 1/6, 2/3])
    
    def __init__(self, u0, T, h, c_safety, function):
        """ The constructor. """
        self.u0 = u0
        self.T = T
        self.t = 0
        self.h = h
        self.c_safety = c_safety
        self.f = function
        self.d = u0.shape[0]
        self.Nx = u0.shape[1]
        self.s = int(0 if self.b is None else len(self.b))
        
    def solve(self):
        """ Method solving problem up to desired time horizon. """
        solution = []
        times = []
        solution.append(self.u0)
        times.append(self.t)
        i = 1
        while self.t < self.T:
            lam = max(np.abs(solution[i-1][1,:]/solution[i-1][0,:]) + np.sqrt(solution[i-1][0,:]))
            tau = self.c_safety*self.h/lam
            tau = min(tau,self.T-self.t)
            k = np.zeros([self.d,self.Nx,len(self.b)])
            for j in range(len(self.b)):
                k[:,:,j] = self.f(solution[i-1] + tau*np.dot(k,self.A[j,:]))
                
            solution.append(solution[i-1]+tau*np.dot(k,self.b))
            self.t+=tau
            times.append(self.t)
            i+=1
        
        return solution, times
        
        
class Explicit_RK3_2D_SWE:
    """ Base class for one step SSP Runge Kutta time integrator method of
    order 3 for systems. The class is modified in order to integrate the 
    SWE with adaptive time step size. """
    
    A = np.array([[0,0,0], [1,0,0], [1/4,1/4,0]])
    b = np.array([1/6, 1/6, 2/3])
    
    def __init__(self, u0, T, h, c_safety, function):
        """ The constructor. """
        self.u0 = u0
        self.T = T
        self.t = 0
        self.h = h
        self.c_safety = c_safety
        self.f = function
        self.d = u0.shape[0]
        self.Nx = u0.shape[1]
        self.Ny = u0.shape[2]
        self.s = int(0 if self.b is None else len(self.b))
        
    def solve(self):
        """ Method solving problem up to desired time horizon. """
        solution = []
        times = []
        solution.append(self.u0)
        times.append(self.t)
        i = 1
        while self.t < self.T:
            lam_x = np.amax(np.abs(solution[i-1][1,:,:]/solution[i-1][0,:,:]) 
                            + np.sqrt(solution[i-1][0,:,:]))
            lam_y = np.amax(np.abs(solution[i-1][2,:,:]/solution[i-1][0,:,:]) 
                            + np.sqrt(solution[i-1][0,:,:]))
            tau = self.c_safety*self.h/max(lam_x,lam_y)
#            tau = self.c_safety*self.h/(max(np.amax(np.abs(solution[i-1][1,:,:]/solution[i-1][0,:,:])), 
#                                            np.amax(np.abs(solution[i-1][2,:,:]/solution[i-1][0,:,:]))) 
#                                        + np.sqrt(np.amax(solution[i-1][0,:,:])))
            tau = min(tau,self.T-self.t)
            k = np.zeros([self.d,self.Nx,self.Ny,len(self.b)])
            for j in range(len(self.b)):
                k[:,:,:,j] = self.f(solution[i-1] + tau*np.dot(k,self.A[j,:]))
                
            solution.append(solution[i-1]+tau*np.dot(k,self.b))
            self.t+=tau
            times.append(self.t)
            i+=1
        
        return solution, times