# higher_order_CWENO_boundary_treatment

![Two dimensional shallow water equation](shallow_water_radial.png)

## Motivation

This repository contains a newly designed one-sided reconstruction algorithm based on so called central weighted essentially non-oscillatory reconstructions.

It is based on *NAUMANN, Alexander; KOLB, Oliver; SEMPLICE, Matteo. On a third order CWENO boundary treatment with application to networks of hyperbolic conservation laws. Applied Mathematics and Computation, 2018, 325. Jg., S. 252-270.* (available on arxiv https://arxiv.org/abs/1703.00321) and takes the presented one dimensional 3rd order approach to higher orders of 5 and 7. It also provides two dimensional implementations by applying the one dimensional approach independently in each dimension.

## Applications

Such reconstruction algorithms can for example be used to construct high-order Finite Volume schemes to solve systems of balance laws or conservation laws. Those arise from a variety of physical phenomena.
