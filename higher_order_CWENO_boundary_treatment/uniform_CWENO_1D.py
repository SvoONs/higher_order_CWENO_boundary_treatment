# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import factorial

from higher_order_CWENO_boundary_treatment.numerical_helpers import undivided_differences


class Uniform_CWENO_1D:
    """ Base class for CWENO reconstructions on uniform grids. """

    Gamma = [np.array([[1,0,0,0],        # coefficents tables for 
                    [0,2,0,0],           # reconstructions according to 
                    [-1/4,-3,3,0],       # Cravero et. al (2017)
                    [1,7,-12,4]]),
            np.array([[1,0,0,0], 
                    [2,2,0,0], 
                    [-1/4,3,3,0], 
                    [0,-5,0,4]]),
            np.array([[1,0,0,0,0], 
                    [4,2,0,0,0], 
                    [23/4,9,3,0,0], 
                    [-1,7,12,4,0], 
                    [9/16,-25/2,-15/2,10,5]]),
            np.array([[1,0,0,0,0,0,0], 
                    [6,2,0,0,0,0,0], 
                    [71/4,15,3,0,0,0,0], 
                    [22,43,24,4,0,0,0],
                    [-71/16,45/2,105/2,30,5,0,0], 
                    [27/8,-341/8,-45,25,30,6,0], 
                    [-225/64,1813/16,777/16,-245/2,-175/4,21,7]])]
    
    def __init__(self, order, avgs, h, eps, p, d0, params=None):
        """ The constructor. """
        
        self.G = order-1        # degree of optimal polynomial
        self.g = int(order/2)   # degree of substencil polynomials
        self.h = h              # cell size of the uniform grid
        self.eps = eps          # epsilon used in computation of nonlinear weights
        self.p = p              # p used in computation of nonlinear weights
        self.d0 = d0            # ideal linear  weight of zero polynomial
        self.params = params    # parameters for boundary treatment; default: None
        self.preprocess_differences(avgs, params)
        
    def compute_reconstruction_polynomial(self):
        """ Main routine to compute the coefficients of 
        reconstruction polynomials. """
        
        # compute interpolants 
        polynomials = self.compute_interpolants()
        
        # set initial centered weights according to order of reconstruction
        if self.G==2:
            d = np.array([self.d0, (1-self.d0)/2, (1-self.d0)/2])
        elif self.G==4:
            d = np.array([self.d0, (1-self.d0)/4, (1-self.d0)/2, (1-self.d0)/4])
        elif self.G==6:
            d = np.array([self.d0, (1-self.d0)/6, (1-self.d0)/3, 
                          (1-self.d0)/3, (1-self.d0)/6])
        
        # compute the polynomial P0            
        polynomials[0] = self.compute_P0(polynomials, d)
        
        # compute the smoothness indicators
        indicators = np.zeros([self.diffs.shape[0]-self.G, len(polynomials)])
        for i in range(len(polynomials)):
            indicators[:,i] = self.jiang_shu_indicator(self.g, polynomials[i], self.h)
            
        # compute the nonlinear weights
        nonlin_weights = self.nonlinear_weights(d, indicators)
        
        # compute final reconstruction polynomial
        temp_coeffs = [np.multiply(nonlin_weights[:,i:i+1],polynomials[i]) for 
                                   i in range(len(polynomials))]
        temp_coeffs[0][:,:self.g+1] += sum(temp_coeffs[1:])
        rec_coeffs = temp_coeffs[0]/np.power(self.h,range(self.G+1))
        
        return rec_coeffs  

    def compute_interpolants(self):
        """ Routine to compute interpolant polynomials. """
        
        coeffs = []
        coeffs.append(np.dot(self.diffs[:-self.G,:],          # optimal polynomial
                                   self.Gamma[self.g][:self.G+1,:self.G+1]))
        coeffs.append(np.dot(self.diffs[:-self.G,:self.g+1],  # P1 with same offset as optimal
                                   self.Gamma[self.g][:self.g+1,:self.g+1]))
        incrementer = 1
        for idx in range(self.g-1,-1,-1):                     # remaining polynomials
            coeffs.append(np.dot(self.diffs[incrementer:-self.G+incrementer,:self.g+1], 
                                    self.Gamma[idx][:self.g+1,:self.g+1]))
            incrementer += 1
            
        return coeffs

    def compute_P0(self, coeffs, d):
        """ Computing the polynomial P0. """
        
        temp_coeffs = [d[i]*coeffs[i] for i in range(1,len(coeffs))]
        coeffs[0][:,:self.g+1] -= sum(temp_coeffs)
        
        return coeffs[0]/d[0]

    def nonlinear_weights(self, d, indicators):
        """ Computing nonlinear weights in typical WENO fashion. """
        
        alphas = d/np.apply_along_axis(lambda x: np.power(x,self.p),
                                       1,indicators+self.eps)
        omegas = alphas/np.sum(alphas,axis=1)[:,None] 
        
        return omegas

    def preprocess_differences(self,avgs,params):
        """ Preprocessing step computing and setting undivided differences. """
        
        if params is None:
            # Assuming periodic boundaries:
            avgs = np.insert(avgs, 0, avgs[-self.g:])   # extend beginning
            avgs = np.append(avgs,avgs[self.g:self.G])  # extend end            
        self.diffs = undivided_differences(avgs,self.G+1)
    
    def reconstruct_cell_interfaces(self):
        """ Computes the values of the underlying at all cell interfaces. """
        
        D_left = np.power(-self.h/2, range(self.G+1))
        D_right = np.power(self.h/2, range(self.G+1))
        rec_coeffs = self.compute_reconstruction_polynomial()
        
        if  self.params is None:
            # assuming periodic boundaries
            left = np.dot(rec_coeffs,D_left)
            right = np.dot(rec_coeffs,D_right)
        else:
            # using WENO-type extrapolation
            left = np.zeros(self.diffs.shape[0])
            right = np.zeros(self.diffs.shape[0])
            left[self.g:-self.g] = np.dot(rec_coeffs[self.g:-self.g,:],D_left)
            right[self.g:-self.g] = np.dot(rec_coeffs[self.g:-self.g,:],D_right)
            # reconstruction polynomials at borders are centered around corresponding inner 
            # neighbor cell
            for i in range(self.g):
                left[i] = np.dot(rec_coeffs[i,:],
                                 np.power((-(self.G+1)+2*i)/2*self.h, range(self.G+1)))
                left[-(i+1)] = np.dot(rec_coeffs[-(i+1),:],
                                      np.power((self.G+1-2*(i+1))/2*self.h, range(self.G+1)))
                right[i] = np.dot(rec_coeffs[i,:],
                                  np.power((-(self.G+1)+2*(i+1))/2*self.h, range(self.G+1)))
                right[-(i+1)] = np.dot(rec_coeffs[-(i+1),:],
                                       np.power((self.G+1-2*i)/2*self.h, range(self.G+1)))
                                              
        return left, right
        
    @staticmethod
    def jiang_shu_indicator(g, coeffs, h):
        """ Computes the Jiang-Shu regularity indicator for a given (multidimensional)
        array of polynomial coefficients, sorted from low to high degree monomials. 
        Assumption for the use of this formula is, that the polynomial is centered around 0. """
        
        n = coeffs.shape[0]    
        indicators = np.zeros([n])
        for l in range(1,g+1):
            for j in range(l,g+1):
                for i in range(j,g+1):
                    if (i+j) % 2 == 0:
                        factorials = (factorial(j)*factorial(i)/(factorial(j-l)*factorial(i-l)))
                        powerTerms = 2**(2*l+1-j-i-int(i==j))/(j+i-2*l+1)
                        rest = coeffs[:,j]*coeffs[:,i]
                        indicators += factorials*powerTerms*rest
                        
        return indicators


class CWENO3_1D(Uniform_CWENO_1D):
    """ Subclass implementing CWENO3 reconstruction for one dimensional underlying function 
    including one-sided boundary treatment. """
    
    def __init__(self, avgs, h, eps, p, d0, params=None):
        """ The constructor. """
        
        super().__init__(3, avgs, h, eps, p, d0, params)
        self.polynomials = self.compute_interpolants()
        self.params = params
        self.avgs = avgs
        
    def compute_reconstruction_polynomial(self):
        """ Main routine computing the coefficients of 
        reconstruction polynomials. """
        
        d = np.array([self.d0, (1-self.d0)/2, (1-self.d0)/2])
        optimal = self.polynomials[0].copy()
        self.polynomials[0] = self.compute_P0(self.polynomials, d)        
        indicators = self.compute_indicators()
        nonlin_weights = self.nonlinear_weights(d, indicators)
        temp_coeffs = [np.multiply(nonlin_weights[:,i:i+1],self.polynomials[i]) 
                      for i in range(len(self.polynomials))]
        temp_coeffs[0][:,:self.g+1] += sum(temp_coeffs[1:])
        if self.params is None:
            rec_coeffs = temp_coeffs[0]/np.power(self.h,range(self.G+1))
        else:
            rec_coeffs = np.zeros([self.avgs.shape[0],self.G+1])        
            rec_coeffs[0,:] = self.boundary('left', optimal[0,:])
            rec_coeffs[self.g:-self.g,:] = temp_coeffs[0]/np.power(self.h,range(self.G+1))
            rec_coeffs[-1,:] = self.boundary('right', optimal[-1,:])
            
        return rec_coeffs
        
    def compute_indicators(self):
        """ Method to compute the smoothness indicators of the candidate polynomials. 
        Supposedly faster than the general formula of the super class. """
        
        indicators = np.zeros([self.diffs.shape[0]-self.G, 3])
        indicators[:,0] = (np.power(self.polynomials[0][:,1],2) + 
                        13/3*np.power(self.polynomials[0][:,2],2))
        indicators[:,1] = np.power(self.polynomials[1][:,1],2)
        indicators[:,2] = np.power(self.polynomials[2][:,1],2)
        return indicators
        
    def boundary(self, side, p_opt):
        """ Method implementing the 3rd order boundary treatment as 
        proposed in Naumann (2017). """
        
        c = self.params  
        if side == 'left':
            p_1 = np.array([self.avgs[0],0,0])
            p_2 = np.append(self.polynomials[1][0,:],[0])
        elif side == 'right':
            p_1 = np.array([self.avgs[-1],0,0])
            p_2 = np.append(self.polynomials[-1][-1,:],[0])
        p_0 = (p_opt - (c[1]*p_1+c[2]*p_2))/c[0]
                            
        IS = np.array([0,0,0], dtype=np.float64)
        if side == 'left':
            IS[0] = (p_0[1]**2 - 4*p_0[1]*p_0[2] + 25*p_0[2]**2/3)
        elif side == 'right':
            IS[0] = (p_0[1]**2 + 4*p_0[1]*p_0[2] + 25*p_0[2]**2/3)
        IS[2] = p_2[1]**2
        
        alphas = c/np.power(IS+self.eps,self.p)
        omegas = alphas/np.sum(alphas)   
        rec_coeffs = (p_0*omegas[0] + p_1*omegas[1] + p_2*omegas[2])
        
        return rec_coeffs/np.power(self.h,range(self.G+1))
        
        
class CWENO5_1D(Uniform_CWENO_1D):
    """ Subclass implementing CWENO5 reconstruction for one dimensional underlying function 
    including one-sided boundary treatment. """
    
    def __init__(self, avgs, h, eps, p, d0, params=None):
        """ The constructor. """
        
        super().__init__(5, avgs, h, eps, p, d0, params)
        self.polynomials = self.compute_interpolants()
        self.params = params
        self.avgs = avgs
        
    def compute_reconstruction_polynomial(self):
        """ Main routine computing the coefficients of reconstruction polynomials. """
        
        d = np.array([self.d0, (1-self.d0)/4, (1-self.d0)/2, (1-self.d0)/4])
        optimal = self.polynomials[0].copy()
        self.polynomials[0] = self.compute_P0(self.polynomials, d)        
        indicators = self.compute_indicators()
        #[CWENO5_1D.jiang_shu_indicator(self.g, polynomial, self.h) for polynomial in self.polynomials]
        nonlin_weights = self.nonlinear_weights(d, indicators)
        temp_coeffs = [np.multiply(nonlin_weights[:,i:i+1],self.polynomials[i]) 
                      for i in range(len(self.polynomials))]
        temp_coeffs[0][:,:self.g+1] += sum(temp_coeffs[1:])
        if self.params is None:
            rec_coeffs = temp_coeffs[0]/np.power(self.h,range(self.G+1))
        else:
            rec_coeffs = np.zeros([self.avgs.shape[0],self.G+1])        
            rec_coeffs[0,:] = self.outer_boundary('left', optimal[0,:])
            rec_coeffs[1,:] = self.inner_boundary('left', optimal[0,:])
            rec_coeffs[self.g:-self.g,:] = temp_coeffs[0]/np.power(self.h,range(self.G+1))
            rec_coeffs[-2,:] = self.inner_boundary('right', optimal[-1,:])
            rec_coeffs[-1,:] = self.outer_boundary('right', optimal[-1,:])
            # rec_coeffs = self.apply_limiter(rec_coeffs,1,-1)  # uncomment to activate limiter
            
        return rec_coeffs
        
    def compute_indicators(self):
        """ Method to compute the smoothness indicators of the candidate polynomials. 
        Supposedly faster than the general formula of the super class. """
        
        indicators = np.zeros([self.diffs.shape[0]-self.G, len(self.polynomials)])
        indicators[:,0] = (1680*np.power(self.polynomials[0][:,1],2) 
                        + 840*np.multiply(self.polynomials[0][:,1],self.polynomials[0][:,3])
                        + 7280*np.power(self.polynomials[0][:,2],2) 
                        + 7056*np.multiply(self.polynomials[0][:,2],self.polynomials[0][:,4])
                        + 65709*np.power(self.polynomials[0][:,3],2) 
                        + 1051404*np.power(self.polynomials[0][:,4],2))/1680
        for i in range(1,len(self.polynomials)):
            indicators[:,i] = np.power(self.polynomials[i][:,1],2) + 13/3*np.power(self.polynomials[i][:,2],2)
            
        return indicators
        
    def outer_boundary(self, side, p_opt):
        """ Method to compute the coefficients of the reconstruction polynomial defined within 
        the outer boundary cells, namely I_{0.5} and I_{N-0.5}. """
        
        c = self.params[1]         
        if side == 'left':
            p_1 = np.array([self.avgs[0],0,0,0,0])
            p_2 = np.append(self.polynomials[1][0,:],[0,0])
        elif side == 'right':
            p_1 = np.array([self.avgs[-1],0,0,0,0])
            p_2 = np.append(self.polynomials[-1][-1,:],[0,0])
        p_0 = (p_opt-(c[1]*p_1+c[2]*p_2))/c[0]        
        IS = np.array([0,0,0], dtype=np.float64)            
        if side == 'left':
            IS[0] = (1680*p_0[1]**2 - 13440*p_0[1]*p_0[2] + 41160*p_0[1]*p_0[3] 
                    - 114240*p_0[1]*p_0[4] + 34160*p_0[2]**2 - 252000*p_0[2]*p_0[3] 
                    + 813456*p_0[2]*p_0[4] + 579789*p_0[3]**2 - 4588080*p_0[3]*p_0[4] 
                    + 11554764*p_0[4]**2)/1680
            IS[2] = (p_2[1]**2 - 8*p_2[1]*p_2[2] + 61*p_2[2]**2/3)
        elif side == 'right':
            IS[0] = (1680*p_0[1]**2 + 13440*p_0[1]*p_0[2] + 41160*p_0[1]*p_0[3] 
                    + 114240*p_0[1]*p_0[4] + 34160*p_0[2]**2 + 252000*p_0[2]*p_0[3] 
                    + 813456*p_0[2]*p_0[4] + 579789*p_0[3]**2 + 4588080*p_0[3]*p_0[4] 
                    + 11554764*p_0[4]**2)/1680
            IS[2] = (p_2[1]**2 + 8*p_2[1]*p_2[2] + 61*p_2[2]**2/3)
    
        if IS.any() < 0:
            raise ValueError('Negative indicator occured!')
        alphas = c/np.power(IS+self.eps,self.p)
        omegas = alphas/np.sum(alphas)   
        rec_coeffs = (p_0*omegas[0] + p_1*omegas[1] + p_2*omegas[2])
        
        return rec_coeffs/np.power(self.h,range(self.G+1))
        
    def inner_boundary(self, side, p_opt):
        """ Method to compute the coefficients of the reconstruction polynomial defined within 
        the inner boundary cells, namely I_{1.5} and I_{N-1.5}. """
        
        c = self.params[0]  
        if side == 'left':
            p_1 = np.array([self.avgs[1],0,0,0,0])
            p_2 = np.append(self.polynomials[1][0,:],[0,0])
            p_3 = np.append(self.polynomials[2][0,:],[0,0])
        elif side == 'right':
            p_1 = np.array([self.avgs[-2],0,0,0,0])
            p_2 = np.append(self.polynomials[-1][-1,:],[0,0])
            p_3 = np.append(self.polynomials[-2][-1,:],[0,0])
        p_0 = (p_opt - (c[1]*p_1+c[2]*p_2+c[3]*p_3))/c[0]
                            
        IS = np.array([0,0,0,0], dtype=np.float64)
        if side == 'left':
            IS[0] = (1680*p_0[1]**2 - 6720*p_0[1]*p_0[2] + 10920*p_0[1]*p_0[3] 
                    - 16800*p_0[1]*p_0[4] + 14000*p_0[2]**2 - 65520*p_0[2]*p_0[3] 
                    + 128016*p_0[2]*p_0[4] + 148869*p_0[3]**2 - 862680*p_0[3]*p_0[4] 
                    + 2447484*p_0[4]**2)/1680
            IS[2] = (p_2[1]**2 - 4*p_2[1]*p_2[2] + 25*p_2[2]**2/3)
            IS[3] = (p_3[1]**2 - 4*p_3[1]*p_3[2] + 25*p_3[2]**2/3)
        elif side == 'right':
            IS[0] = (1680*p_0[1]**2 + 6720*p_0[1]*p_0[2] + 10920*p_0[1]*p_0[3] 
                    + 16800*p_0[1]*p_0[4] + 14000*p_0[2]**2 + 65520*p_0[2]*p_0[3] 
                    + 128016*p_0[2]*p_0[4] + 148869*p_0[3]**2 + 862680*p_0[3]*p_0[4] 
                    + 2447484*p_0[4]**2)/1680
            IS[2] = (p_2[1]**2 + 4*p_2[1]*p_2[2] + 25*p_2[2]**2/3)
            IS[3] = (p_3[1]**2 + 4*p_3[1]*p_3[2] + 25*p_3[2]**2/3)  
        
        if IS.any() < 0:
            raise ValueError('Negative indicator occured!')
        alphas = c/np.power(IS+self.eps,self.p)
        omegas = alphas/np.sum(alphas)   
        rec_coeffs = (p_0*omegas[0] + p_1*omegas[1] + p_2*omegas[2] + p_3*omegas[3])
        
        return rec_coeffs/np.power(self.h,range(self.G+1))


class CWENO7_1D(Uniform_CWENO_1D):
    """ Subclass implementing CWENO7 reconstruction for one dimensional underlying function 
    including one-sided boundary treatment. """

    def __init__(self, avgs, h, eps, p, d0, params=None):
        """ The constructor. """
        
        super().__init__(7, avgs, h, eps, p, d0, params)
        self.polynomials = self.compute_interpolants()
        self.params = params
        self.avgs = avgs

    def compute_reconstruction_polynomial(self):
        """ Main routine computing the coefficients of reconstruction polynomials. """
        
        d = np.array([self.d0, (1-self.d0)/6, (1-self.d0)/3, (1-self.d0)/3, (1-self.d0)/6])
        optimal = self.polynomials[0].copy()
        self.polynomials[0] = self.compute_P0(self.polynomials, d)        
        indicators = self.compute_indicators()
        nonlin_weights = self.nonlinear_weights(d, indicators)
        temp_coeffs = [np.multiply(nonlin_weights[:,i:i+1],self.polynomials[i]) 
                      for i in range(len(self.polynomials))]
        temp_coeffs[0][:,:self.g+1] += sum(temp_coeffs[1:])
        if self.params is None:
            rec_coeffs = temp_coeffs[0]/np.power(self.h,range(self.G+1))
        else:
            rec_coeffs = np.zeros([self.avgs.shape[0],self.G+1])        
            rec_coeffs[0,:] = self.outer_boundary('left', optimal[0,:])
            rec_coeffs[1,:] = self.middle_boundary('left', optimal[0,:])
            rec_coeffs[2,:] = self.inner_boundary('left', optimal[0,:])
            rec_coeffs[self.g:-self.g,:] = temp_coeffs[0]/np.power(self.h,range(self.G+1))
            rec_coeffs[-3,:] = self.inner_boundary('right', optimal[-1,:])
            rec_coeffs[-2,:] = self.middle_boundary('right', optimal[-1,:])
            rec_coeffs[-1,:] = self.outer_boundary('right', optimal[-1,:])
            
        return rec_coeffs
        
    def compute_indicators(self):
        """ Method to compute the smoothness indicators of the candidate polynomials. 
        Supposedly faster than the general formula of the super class. """
        
        indicators = np.zeros([self.diffs.shape[0]-self.G, len(self.polynomials)])
        indicators[:,0] = (887040*np.power(self.polynomials[0][:,1],2) 
                    + 443520*np.multiply(self.polynomials[0][:,1],self.polynomials[0][:,3]) 
                    + 110880*np.multiply(self.polynomials[0][:,1],self.polynomials[0][:,5]) 
                    + 3843840*np.power(self.polynomials[0][:,2],2) 
                    + 3725568*np.multiply(self.polynomials[0][:,2],self.polynomials[0][:,4]) 
                    + 1378080*np.multiply(self.polynomials[0][:,2],self.polynomials[0][:,6])
                    + 34694352*np.power(self.polynomials[0][:,3],2) 
                    + 55942920*np.multiply(self.polynomials[0][:,3],self.polynomials[0][:,5]) 
                    + 555141312*np.power(self.polynomials[0][:,4],2) 
                    + 1342648560*np.multiply(self.polynomials[0][:,4],self.polynomials[0][:,6]) 
                    + 13878542425*np.power(self.polynomials[0][:,5],2) 
                    + 499627530135*np.power(self.polynomials[0][:,6],2))/887040
        for i in range(1,len(self.polynomials)):
            indicators[:,i] = (240*np.power(self.polynomials[i][:,1],2) 
                        + 120*np.multiply(self.polynomials[i][:,1],self.polynomials[i][:,3]) 
                        + 1040*np.power(self.polynomials[i][:,2],2) 
                        + 9387*np.power(self.polynomials[i][:,3],2))/240   
        return indicators
        
    def outer_boundary(self, side, p_opt):
        """ Method to compute the coefficients of the reconstruction polynomial defined within 
        the outer boundary cells, namely I_{0.5} and I_{N-0.5}. """
        
        c = self.params[2]
        if side == 'left':
            p_1 = np.array([self.avgs[0],0,0,0,0,0,0])
            p_2 = np.append(self.polynomials[1][0,:],[0,0,0])
        elif side == 'right':
            p_1 = np.array([self.avgs[-1],0,0,0,0,0,0])
            p_2 = np.append(self.polynomials[-1][-1,:],[0,0,0])
        p_0 = (np.squeeze(p_opt)-(c[1]*p_1+c[2]*p_2))/c[0]        
        IS = np.array([0,0,0], dtype=np.float64)         
        if side == 'left':
            IS[0] = (887040*p_0[1]**2 - 10644480*p_0[1]*p_0[2] + 48343680*p_0[1]*p_0[3] 
                     - 196922880*p_0[1]*p_0[4] + 758530080*p_0[1]*p_0[5] - 2828105280*p_0[1]*p_0[6] 
                     + 35777280*p_0[2]**2 - 359251200*p_0[2]*p_0[3] + 1600397568*p_0[2]*p_0[4] 
                     - 6682737600*p_0[2]*p_0[5] + 26813492640*p_0[2]*p_0[6] + 1004672592*p_0[3]**2 
                     - 9967224960*p_0[3]*p_0[4] + 46144878120*p_0[3]*p_0[5] - 204086116080*p_0[3]*p_0[6] 
                     + 27882182592*p_0[4]**2 - 292452098400*p_0[4]*p_0[5] + 1462297528560*p_0[4]*p_0[6] 
                     + 881856884425*p_0[5]**2 - 10190971222740*p_0[5]*p_0[6] + 34480494630315*p_0[6]**2)/887040
            IS[2] = (240*p_2[1]**2 - 2880*p_2[1]*p_2[2] + 13080*p_2[1]*p_2[3] 
                     + 9680*p_2[2]**2 - 97200*p_2[2]*p_2[3] + 271827*p_2[3]**2)/240
        elif side == 'right':
            IS[0] = (887040*p_0[1]**2 + 10644480*p_0[1]*p_0[2] + 48343680*p_0[1]*p_0[3] 
                     + 196922880*p_0[1]*p_0[4] + 758530080*p_0[1]*p_0[5] + 2828105280*p_0[1]*p_0[6] 
                     + 35777280*p_0[2]**2 + 359251200*p_0[2]*p_0[3] + 1600397568*p_0[2]*p_0[4] 
                     + 6682737600*p_0[2]*p_0[5] + 26813492640*p_0[2]*p_0[6] + 1004672592*p_0[3]**2 
                     + 9967224960*p_0[3]*p_0[4] + 46144878120*p_0[3]*p_0[5] + 204086116080*p_0[3]*p_0[6] 
                     + 27882182592*p_0[4]**2 + 292452098400*p_0[4]*p_0[5] + 1462297528560*p_0[4]*p_0[6] 
                     + 881856884425*p_0[5]**2 + 10190971222740*p_0[5]*p_0[6] + 34480494630315*p_0[6]**2)/887040
            IS[2] = (240*p_2[1]**2 + 2880*p_2[1]*p_2[2] + 13080*p_2[1]*p_2[3] 
                     + 9680*p_2[2]**2 + 97200*p_2[2]*p_2[3] + 271827*p_2[3]**2)/240
        alphas = c/np.power(IS+self.eps, self.p)
        omegas = alphas/np.sum(alphas)   
        rec_coeffs = (p_0*omegas[0] + p_1*omegas[1] + p_2*omegas[2])
        
        return rec_coeffs/np.power(self.h,range(self.G+1))
    
    def middle_boundary(self, side, p_opt):
        """ Method to compute the coefficients of the reconstruction polynomial defined within 
        the middle boundary cells, namely I_{1.5} and I_{N-1.5}. """
        
        c = self.params[1]
        if side == 'left':
            p_1 = np.array([self.avgs[1],0,0,0,0,0,0])
            p_2 = np.append(self.polynomials[1][0,:],[0,0,0])
            p_3 = np.append(self.polynomials[2][0,:],[0,0,0])
        elif side == 'right':
            p_1 = np.array([self.avgs[-2],0,0,0,0,0,0])
            p_2 = np.append(self.polynomials[-1][-1,:],[0,0,0])
            p_3 = np.append(self.polynomials[-2][-1,:],[0,0,0])
        p_0 = (p_opt - (c[1]*p_1+c[2]*p_2+c[3]*p_3))/c[0]
                            
        IS = np.array([0,0,0,0], dtype=np.float64)
        if side == 'left':
            IS[0] = (887040*p_0[1]**2 - 7096320*p_0[1]*p_0[2] + 21732480*p_0[1]*p_0[3] 
                     - 60318720*p_0[1]*p_0[4] + 159778080*p_0[1]*p_0[5] - 412917120*p_0[1]*p_0[6] 
                     + 18036480*p_0[2]**2 - 133056000*p_0[2]*p_0[3] + 429504768*p_0[2]*p_0[4] 
                     - 1291382400*p_0[2]*p_0[5] + 3721623840*p_0[2]*p_0[6] + 306128592*p_0[3]**2 
                     - 2422506240*p_0[3]*p_0[4] + 8697930120*p_0[3]*p_0[5] - 29233401120*p_0[3]*p_0[6] 
                     + 6100915392*p_0[4]**2 - 55104033600*p_0[4]*p_0[5] + 226615326960*p_0[4]*p_0[6] 
                     + 161827574425*p_0[5]**2 - 1703570936760*p_0[5]*p_0[6] + 5894755440615*p_0[6]**2)/887040
            IS[2] = (240*p_2[1]**2 - 1920*p_2[1]*p_2[2] + 5880*p_2[1]*p_2[3] 
                     + 4880*p_2[2]**2 - 36000*p_2[2]*p_2[3] + 82827*p_2[3]**2)/240
            IS[3] = (240*p_3[1]**2 - 1920*p_3[1]*p_3[2] + 5880*p_3[1]*p_3[3] 
                     + 4880*p_3[2]**2 - 36000*p_3[2]*p_3[3] + 82827*p_3[3]**2)/240
        elif side == 'right':
            IS[0] = (887040*p_0[1]**2 + 7096320*p_0[1]*p_0[2] + 21732480*p_0[1]*p_0[3] 
                     + 60318720*p_0[1]*p_0[4] + 159778080*p_0[1]*p_0[5] + 412917120*p_0[1]*p_0[6] 
                     + 18036480*p_0[2]**2 + 133056000*p_0[2]*p_0[3] + 429504768*p_0[2]*p_0[4] 
                     + 1291382400*p_0[2]*p_0[5] + 3721623840*p_0[2]*p_0[6] + 306128592*p_0[3]**2 
                     + 2422506240*p_0[3]*p_0[4] + 8697930120*p_0[3]*p_0[5] + 29233401120*p_0[3]*p_0[6] 
                     + 6100915392*p_0[4]**2 + 55104033600*p_0[4]*p_0[5] + 226615326960*p_0[4]*p_0[6] 
                     + 161827574425*p_0[5]**2 + 1703570936760*p_0[5]*p_0[6] + 5894755440615*p_0[6]**2)/887040
            IS[2] = (240*p_2[1]**2 + 1920*p_2[1]*p_2[2] + 5880*p_2[1]*p_2[3] 
                     + 4880*p_2[2]**2 + 36000*p_2[2]*p_2[3] + 82827*p_2[3]**2)/240
            IS[3] = (240*p_3[1]**2 + 1920*p_3[1]*p_3[2] + 5880*p_3[1]*p_3[3] 
                     + 4880*p_3[2]**2 + 36000*p_3[2]*p_3[3] + 82827*p_3[3]**2)/240
        
        alphas = c/np.power(IS+self.eps, self.p)
        omegas = alphas/np.sum(alphas)   
        rec_coeffs = (p_0*omegas[0] + p_1*omegas[1] + p_2*omegas[2] + p_3*omegas[3])
        
        return rec_coeffs/np.power(self.h,range(self.G+1))

    def inner_boundary(self, side, p_opt):
        """ Method to compute the coefficients of the reconstruction polynomial defined within 
        the inner boundary cells, namely I_{2.5} and I_{N-2.5}. """
        
        c = self.params[0]
        if side == 'left':
            p_1 = np.array([self.avgs[2],0,0,0,0,0,0])
            p_2 = np.append(self.polynomials[1][0,:],[0,0,0])
            p_3 = np.append(self.polynomials[2][0,:],[0,0,0])
            p_4 = np.append(self.polynomials[3][0,:],[0,0,0])
        elif side == 'right':
            p_1 = np.array([self.avgs[-3],0,0,0,0,0,0])
            p_2 = np.append(self.polynomials[-1][-1,:],[0,0,0])
            p_3 = np.append(self.polynomials[-2][-1,:],[0,0,0])
            p_4 = np.append(self.polynomials[-3][-1,:],[0,0,0])
        p_0 = (p_opt - (c[1]*p_1+c[2]*p_2+c[3]*p_3+c[4]*p_4))/c[0]
    
        IS = np.array([0,0,0,0,0], dtype=np.float64)
        if side == 'left':
            IS[0] = (887040*p_0[1]**2 - 3548160*p_0[1]*p_0[2] + 5765760*p_0[1]*p_0[3] 
                     - 8870400*p_0[1]*p_0[4] + 13416480*p_0[1]*p_0[5] - 20180160*p_0[1]*p_0[6] 
                     + 7392000*p_0[2]**2 - 34594560*p_0[2]*p_0[3] + 67592448*p_0[2]*p_0[4] 
                     - 122337600*p_0[2]*p_0[5] + 212937120*p_0[2]*p_0[6] + 78602832*p_0[3]**2 
                     - 455495040*p_0[3]*p_0[4] + 1078810920*p_0[3]*p_0[5] - 2304363600*p_0[3]*p_0[6] 
                     + 1292271552*p_0[4]**2 - 9223552800*p_0[4]*p_0[5] + 26073323760*p_0[4]*p_0[6] 
                     + 32401508425*p_0[5]**2 - 277013485980*p_0[5]*p_0[6] + 1166705407755*p_0[6]**2)/887040
            IS[2] = (240*p_2[1]**2 - 960*p_2[1]*p_2[2] + 1560*p_2[1]*p_2[3] 
                     + 2000*p_2[2]**2 - 9360*p_2[2]*p_2[3] + 21267*p_2[3]**2)/240
            IS[3] = (240*p_3[1]**2 - 960*p_3[1]*p_3[2] + 1560*p_3[1]*p_3[3] 
                     + 2000*p_3[2]**2 - 9360*p_3[2]*p_3[3] + 21267*p_3[3]**2)/240
            IS[4] = (240*p_4[1]**2 - 960*p_4[1]*p_4[2] + 1560*p_4[1]*p_4[3] 
                     + 2000*p_4[2]**2 - 9360*p_4[2]*p_4[3] + 21267*p_4[3]**2)/240
        elif side =='right':
            IS[0] = (887040*p_0[1]**2 + 3548160*p_0[1]*p_0[2] + 5765760*p_0[1]*p_0[3] 
                     + 8870400*p_0[1]*p_0[4] + 13416480*p_0[1]*p_0[5] + 20180160*p_0[1]*p_0[6] 
                     + 7392000*p_0[2]**2 + 34594560*p_0[2]*p_0[3] + 67592448*p_0[2]*p_0[4] 
                     + 122337600*p_0[2]*p_0[5] + 212937120*p_0[2]*p_0[6] + 78602832*p_0[3]**2 
                     + 455495040*p_0[3]*p_0[4] + 1078810920*p_0[3]*p_0[5] + 2304363600*p_0[3]*p_0[6] 
                     + 1292271552*p_0[4]**2 + 9223552800*p_0[4]*p_0[5] + 26073323760*p_0[4]*p_0[6] 
                     + 32401508425*p_0[5]**2 + 277013485980*p_0[5]*p_0[6] + 1166705407755*p_0[6]**2)/887040
            IS[2] = (240*p_2[1]**2 + 960*p_2[1]*p_2[2] + 1560*p_2[1]*p_2[3] 
                     + 2000*p_2[2]**2 + 9360*p_2[2]*p_2[3] + 21267*p_2[3]**2)/240
            IS[3] = (240*p_3[1]**2 + 960*p_3[1]*p_3[2] + 1560*p_3[1]*p_3[3] 
                     + 2000*p_3[2]**2 + 9360*p_3[2]*p_3[3] + 21267*p_3[3]**2)/240
            IS[4] = (240*p_4[1]**2 + 960*p_4[1]*p_4[2] + 1560*p_4[1]*p_4[3] 
                     + 2000*p_4[2]**2 + 9360*p_4[2]*p_4[3] + 21267*p_4[3]**2)/240        
            
        alphas = c/np.power(IS+self.eps, self.p)
        omegas = alphas/np.sum(alphas)   
        rec_coeffs = (p_0*omegas[0] + p_1*omegas[1] + p_2*omegas[2] + p_3*omegas[3] + p_4*omegas[4])
        
        return rec_coeffs/np.power(self.h,range(self.G+1))
