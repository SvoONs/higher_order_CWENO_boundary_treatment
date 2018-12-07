# -*- coding: utf-8 -*-

import numpy as np
from higher_order_CWENO_boundary_treatment.uniform_CWENO_1D import CWENO3_1D, CWENO5_1D, CWENO7_1D
from higher_order_CWENO_boundary_treatment.uniform_CWENO_2D import CWENO3_2D, CWENO5_2D, CWENO7_2D

def right_hand_side(data, h, flux, dflux, params, order):
    """ Method implementing the RHS of the semi discretized system of ODEs
    where periodic boundaries are assumed. """
    
    # CWENO reconstruction
    left, right = reconstructed_interfaces(data,h,params,order)
    
    # periodic boundaries
    left = np.append(left,left[0])
    right = np.insert(right,0,right[-1])
    
    # right border numerical fluxes:
    H_plus = local_lax_friedrichs_flux(left[1:],right[1:],flux,dflux)
    # left border numerical fluxes:
    H_minus = local_lax_friedrichs_flux(left[:-1],right[:-1],flux,dflux)
                                                             
    return -(H_plus-H_minus)/h
    
def reconstructed_interfaces(data, h, params, order):  
    """ Method calling the CWENO operators of corresponding order for 
    reconstruction of cell interfaces. The feasible orders are 3, 5 or 7. """
    
    if order == 3:
        CWENO = CWENO3_1D(data,h,params[0],params[1],params[2],params[-1])
    elif order == 5:
        CWENO = CWENO5_1D(data,h,params[0],params[1],params[2],params[-1])
    elif order == 7:
        CWENO = CWENO7_1D(data,h,params[0],params[1],params[2],params[-1])
    
    left, right = CWENO.reconstruct_cell_interfaces()
    
    return left, right

def RHS_SWE_1D_solid_walls(U, h, flux_f, dflux_f, params, order):
    """ Method implementing the RHS of the SWE simulation with solid wall 
    boundary conditions with desired order. """
    
    d = U.shape[0]
    left, right = np.zeros(U.shape), np.zeros(U.shape)
    # CWENO reconstruction componentwise
    for i in range(d):
        left[i,:], right[i,:] = reconstructed_interfaces(U[i,:],h,params,order)        

    # copy the reconstructed interfaces to the corresponding position
    left = np.asarray([np.append(left[i,:],right[i,-1]) for i in range(left.shape[0])])
    right = np.asarray([np.insert(right[i,:],0,left[i,0]) for i in range(right.shape[0])])
    # negate x-velocity components at left and right boundaries
    left[1,-1], right[1,0] = -left[1,-1], -right[1,0]    
        
    # right border numerical fluxes:
    H_right = local_lax_friedrichs_flux_SWE_1D(left[:,1:],right[:,1:],flux_f,dflux_f)
    # left border numerical fluxes:
    H_left = local_lax_friedrichs_flux_SWE_1D(left[:,:-1],right[:,:-1],flux_f,dflux_f)

    return -(H_right-H_left)/h        
    
def right_hand_side_2D(data, hx, hy, flux_f, dflux_f, flux_g, dflux_g, params, order):
    """ Method implementing the RHS of the semi discretized system of ODEs in 
    two space dimensions where periodic boundaries are assumed. """
    
    # CWENO reconstruction
    left, right, down, up = reconstructed_interfaces_2D(data,hx,hy,params,order)
    
    # periodic boundaries
    left, right = np.c_[left,left[:,0]], np.c_[right[:,-1],right]
    down, up = np.r_['0,2',down,down[0,:]], np.r_['0,2',up[-1,:],up]
    
    # right border numerical fluxes:
    H_right = local_lax_friedrichs_flux(left[:,1:],right[:,1:],flux_f,dflux_f)
    # left border numerical fluxes:
    H_left = local_lax_friedrichs_flux(left[:,:-1],right[:,:-1],flux_f,dflux_f)
    # upper border numerical fluxes:
    H_up = local_lax_friedrichs_flux(down[1:,:],up[1:,:],flux_g,dflux_g)
    # lower border numerical fluxes:
    H_down = local_lax_friedrichs_flux(down[:-1,:],up[:-1,:],flux_g,dflux_g)
                                                             
    return -(H_right-H_left)/hx -(H_up-H_down)/hy    

def reconstructed_interfaces_2D(data, hx, hy, params, order):
    """ Method calling the CWENO operators of corresponding order for 
    reconstruction of cell interfaces in two space dimensions. 
    The feasible orders are 3, 5 or 7. """
    
    if order == 3:
        CWENO_2D = CWENO3_2D(data,hx,hy,params[0],params[1],params[2],params[-1])
    elif order == 5:
        CWENO_2D = CWENO5_2D(data,hx,hy,params[0],params[1],params[2],params[-1])
    elif order == 7:
        CWENO_2D = CWENO7_2D(data,hx,hy,params[0],params[1],params[2],params[-1])
        
    left, right, down, up = CWENO_2D.reconstruct_cell_interfaces()
    
    return left, right, down, up
    
    
def RHS_SWE_2D_solid_walls(U, h, flux_f, dflux_f, flux_g, dflux_g, params, order):
    """ Method implementing the RHS of the SWE simulation with solid wall 
    boundary conditions with desired order. """
    
    d = U.shape[0]
    left, right, down, up = np.zeros(U.shape), np.zeros(U.shape), np.zeros(U.shape), np.zeros(U.shape)
    # CWENO reconstruction componentwise
    for i in range(d):
        left[i,:,:], right[i,:,:], down[i,:,:], up[i,:,:] = reconstructed_interfaces_2D(U[i,:,:],h,h,params,order)        

    # copy the reconstructed interfaces to the corresponding position
    left = np.asarray([np.c_[left[i,:,:],right[i,:,-1]] for i in range(left.shape[0])])
    right = np.asarray([np.c_[left[i,:,0],right[i,:,:]] for i in range(right.shape[0])])
    down = np.asarray([np.r_['0,2',down[i,:,:],up[i,-1,:]] for i in range(down.shape[0])])
    up = np.asarray([np.r_['0,2',down[i,0,:],up[i,:,:]] for i in range(up.shape[0])])    
    # negate x-velocity components at left and right boundaries
    left[1,:,-1], right[1,:,0] = -left[1,:,-1], -right[1,:,0]    
    # negate y-velocity components at upper and lower boundaries
    down[2,-1,:], up[2,0,:] = -down[2,-1,:], -up[2,0,:]
        
    # right border numerical fluxes:
    H_right = local_lax_friedrichs_flux_SWE_2D(left[:,:,1:],right[:,:,1:],flux_f,dflux_f)
    # left border numerical fluxes:
    H_left = local_lax_friedrichs_flux_SWE_2D(left[:,:,:-1],right[:,:,:-1],flux_f,dflux_f)
    # right border numerical fluxes:
    H_up = local_lax_friedrichs_flux_SWE_2D(down[:,1:,:],up[:,1:,:],flux_g,dflux_g)
    # left border numerical fluxes:
    H_down = local_lax_friedrichs_flux_SWE_2D(down[:,:-1,:],up[:,:-1,:],flux_g,dflux_g)

    return -(H_right-H_left+H_up-H_down)/h    
    
def local_lax_friedrichs_flux(u,v,func,spec):
    """ Method implementing the local Lax Friedrichs numerical flux function for the 
    linear degenerate or genuinely nonlinear case. """
    
    sigma = np.maximum(np.abs(spec(u)),np.abs(spec(v)))
    
    return 1/2*(func(u)+func(v)-sigma*(u-v))
    
def local_lax_friedrichs_flux_SWE_1D(u,v,func,spec):
    """ Method implementing the local Lax Friedrichs numerical flux function adapted for SWEs 
    in order to avoid negative water heights. """
    
    h_min = 1e-08
    zero_height = np.where(u[0,:] < h_min)
    u[1,:][zero_height] = 0
    zero_height = np.where(v[0,:] < h_min)
    v[1,:][zero_height] = 0
    
    sigma = np.maximum(np.abs(spec(u)),np.abs(spec(v)))
    
    return 1/2*(func(u)+func(v)-sigma*(u-v))
    
def local_lax_friedrichs_flux_SWE_2D(u,v,func,spec):
    """ Method implementing the local Lax Friedrichs numerical flux function adapted for SWEs 
    in order to avoid negative water heights. """
    
    h_min = 1e-08
    zero_height = np.where(u[0,:,:] < h_min)
    u[1,:,:][zero_height], u[2,:,:][zero_height] = 0, 0
    zero_height = np.where(v[0,:,:] < h_min)
    v[1,:,:][zero_height], v[2,:,:][zero_height] = 0, 0
    
    sigma = np.maximum(np.abs(spec(u)),np.abs(spec(v)))
    
    return 1/2*(func(u)+func(v)-sigma*(u-v))