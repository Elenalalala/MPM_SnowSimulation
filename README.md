# 3D Snow Simulation using Material Point Method (MPM)

This project implements a **3D snow simulation** based on the **Material Point Method (MPM)** using **Taichi**.  
It simulates snow, jelly, and fluid materials, supports mesh-based particle loading and USD export.

**Video demo:**
- water+snow+jelly.mp4
- mpm_with_bunny.mp4

**Snow Simulation Feature**
- Extended 2D MPM demo to 3D snow simulation.
- Particle-to-grid (P2G) and grid-to-particle (G2P) updates.
- SVD-based plasticity.
- Stress computation for snow materials.
- Particle and grid collision handling.
- Added `.obj` mesh loading.
- Populated meshes with **randomized particles** to create solid shells.

**Material Point Method Detail**
- particles will carry material properites like position, velocity, mass, deformation gradient and material type. we will calculate the neighbor 3x3x3 grid to enforce momentum conservation and compute forces. Each particles has a mass, which is the product of its volumn and density. So we first calcualte the density and mass for each grid based on distribution of all the particles using B-spline weights so that it can have a smooth interpolation. For each particle, the code computes a base grid index and local fractional position fx, then calculates quadratic weights in each dimension. This ensures that each particle influences a 3x3x3 neighborhood of grid nodes, spreading its mass and momentum based on proximity.
- After we scattered the particles to the grids, we use the grid node to compute velocities and apply any external forces like gravity or attractors. We also need to normalize the velocity by total mass. 
- The deformation gradient F is updated for each particle to track its local deformation. According to the paper, it captures both elastic and plastic deformation. To get the plasiticity, we use Singular Value Decomposition of the deformation gradient F. 
- The singular values Σ represent stretch along principal directions. For snow, we should clamps these singular values to prevent weird compression or stretching.This ensures the particle does not compress or expand beyond physical limits, and the plastic deformation is updated to reflect permanent compression. The stress is computed using a linear elastic constitutive model, where the shear stiffness (mu) and Lamé's first parameter are scaled by a hardening factor(h), which increases as snow is compressed.
- We should do Grid collision at this point using signed distance function.
- After forces are applied on the grid, then we interpolates velocities and affine velocity gradients back to each particle. And then we do another particle collision again here, which makes sure that each particles is not inside of each other, or the boundaries. 


**Sliders for:**
- Snow hardening coefficient ξ, recommended value in the paper is 10
- Collision Shear Stiffness mu, recommended value is around 0.3

**Controls:**
- WASDQE for camera controls
- z to minus hardening coefficient by 0.5, x to add 0.5.
- c to minus collision shear stiffness by 0.5, v to add 0.5.
- press 1 to make the initial material stiffness harder, press 2 to make it softer (resisitance to twisting)
- press 3 to make the initial material to resist volumn change more, press4 to make it soft and squishy
- press o to start/stop recording
- close the program to save to snow_sim.usda (USD format) for external rendering

**How to run**
- pip install taichi
- python mpm.py

**Reference & learning material**
- A Material Point Method for Snow Simulation, SIGGRAPH
- Taichi GPU kernel tutorials and 2D MPM demo
- FLIP and CLIP methods (particle-grid velocity transfer)
- Taichi documentation
- NVIDIA openUSD documentation
- CUDA Quick Start Guide