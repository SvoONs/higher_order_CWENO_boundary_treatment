# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 19:32:40 2018

Script containing the CWENO classes for uniform grids in 2D by the dimension-by-dimension
extension of the 1D classes.

@author: Sven
"""

import numpy as np 
from uniform_CWENO_1D import Uniform_CWENO_1D, CWENO3_1D, CWENO5_1D, CWENO7_1D


class Uniform_CWENO_2D:
    
    def __init__(self, order, avgs, hx, hy, eps, p, d0, params=None):
        """ The constructor. """

        self.G = order-1
        self.g = int(order/2)
        self.avgs = avgs
        self.hx = hx
        self.hy = hy
        self.eps = eps
        self.p = p
        self.d0 = d0
        
    def reconstruct_cell_interfaces(self):
        """ Main method applying one dimensional reconstructions dimension by
        dimension. """
        
        funcx = lambda x: Uniform_CWENO_1D(self.G+1, x, self.hx, self.eps, self.p,
                                           self.d0).reconstruct_cell_interfaces()
        
        funcy = lambda y: Uniform_CWENO_1D(self.G+1, y, self.hy, self.eps, self.p,
                                           self.d0).reconstruct_cell_interfaces()
        
        left_and_right = np.apply_along_axis(func1d=funcx, axis=1, arr=self.avgs)
        down, up = np.apply_along_axis(func1d=funcy, axis=0, arr=self.avgs)
        left = left_and_right[:,0,:]
        right = left_and_right[:,1,:]
        
        return left, right, down, up
  
class CWENO3_2D(Uniform_CWENO_2D):
    
    def __init__(self, avgs, hx, hy, eps, p, d0, params=None):
        """ The constructor. """
        
        super().__init__(3, avgs, hx, hy, eps, p, d0)
        self.params = params
        
    def reconstruct_cell_interfaces(self):
        """ Main method applying one dimensional reconstructions dimension by
        dimension. """
        
        funcx = lambda x: CWENO3_1D(x, self.hx, self.eps, self.p, self.d0, 
                                    self.params).reconstruct_cell_interfaces()
        
        funcy = lambda y: CWENO3_1D(y, self.hy, self.eps, self.p, self.d0, 
                                    self.params).reconstruct_cell_interfaces()
        
        left_and_right = np.apply_along_axis(func1d=funcx, axis=1, arr=self.avgs)
        down, up = np.apply_along_axis(func1d=funcy, axis=0, arr=self.avgs)
        left = left_and_right[:,0,:]
        right = left_and_right[:,1,:]
        
        return left, right, down, up

class CWENO5_2D(Uniform_CWENO_2D):
    
    def __init__(self, avgs, hx, hy, eps, p, d0, params=None):
        """ The constructor. """
        
        super().__init__(5, avgs, hx, hy, eps, p, d0)
        self.params = params
        
    def reconstruct_cell_interfaces(self):
        """ Main method applying one dimensional reconstructions dimension by
        dimension. """
        
        funcx = lambda x: CWENO5_1D(x, self.hx, self.eps, self.p, self.d0, 
                                    self.params).reconstruct_cell_interfaces()
        
        funcy = lambda y: CWENO5_1D(y, self.hy, self.eps, self.p, self.d0, 
                                    self.params).reconstruct_cell_interfaces()
        
        left_and_right = np.apply_along_axis(func1d=funcx, axis=1, arr=self.avgs)
        down, up = np.apply_along_axis(func1d=funcy, axis=0, arr=self.avgs)
        left = left_and_right[:,0,:]
        right = left_and_right[:,1,:]
        
        return left, right, down, up
        
class CWENO7_2D(Uniform_CWENO_2D):
    
    def __init__(self, avgs, hx, hy, eps, p, d0, params=None):
        """ The constructor. """
        
        super().__init__(7, avgs, hx, hy, eps, p, d0)
        self.params = params
        
    def reconstruct_cell_interfaces(self):
        """ Main method applying one dimensional reconstructions dimension by
        dimension. """        
        
        funcx = lambda x: CWENO7_1D(x, self.hx, self.eps, self.p, self.d0, 
                                    self.params).reconstruct_cell_interfaces()
        
        funcy = lambda y: CWENO7_1D(y, self.hy, self.eps, self.p, self.d0, 
                                    self.params).reconstruct_cell_interfaces()
        
        left_and_right = np.apply_along_axis(func1d=funcx, axis=1, arr=self.avgs)
        down, up = np.apply_along_axis(func1d=funcy, axis=0, arr=self.avgs)
        left = left_and_right[:,0,:]
        right = left_and_right[:,1,:]
        
        return left, right, down, up