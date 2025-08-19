import taichi as ti
import trimesh
import numpy as np

ti.init(arch=ti.gpu)

# Simulation constants
quality = 2
n_particles = 9000 * quality**3
n_grid = 64 * quality
dx, inv_dx = 1 / n_grid, float(n_grid) # grid size
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**3, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2
# base material stiffness
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

# position
pos = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
# velocity
vel = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
# affine velocity -> local grid neighbor movement
C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
# deformation gradient: how much the particle is deformed
F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
material = ti.field(dtype=ti.i32, shape=n_particles)
# plastic deform -> permanant deformation level when pressing
Jp = ti.field(dtype=ti.f32, shape=n_particles)

# grid velocity
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
# grid mass
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

# rendering buffers
group_size = n_particles // 3
render_pos_group = ti.Vector.field(3, dtype=ti.f32, shape=group_size)
radius_field = ti.field(dtype=ti.f32, shape=group_size)

gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
attractor_strength = ti.field(dtype=ti.f32, shape=())
attractor_pos = ti.Vector.field(3, dtype=ti.f32, shape=())


cube_vis_points = ti.Vector.field(3, ti.f32, shape=512)  # size of the cursor cube


@ti.kernel
def update_cube_vis(center: ti.types.vector(3, ti.f32), size: float):
    count = 0
    for i, j, k in ti.ndrange(4, 4, 4):
        pos = center + size * (ti.Vector([i, j, k]) / 3.0 - 0.5)
        cube_vis_points[count] = pos
        count += 1

@ti.kernel
def reset():
    for i in range(n_particles):
        mat_id = i // group_size
        material[i] = mat_id
        base_x = 0.2 + 0.2 * mat_id
        base_y = 0.05 + 0.3 * mat_id
        pos[i] = [
            ti.random() * 0.2 + base_x,
            ti.random() * 0.2 + base_y,
            ti.random() * 0.2 + 0.1,
        ]
        vel[i] = [0, 0, 0]
        F[i] = ti.Matrix.identity(ti.f32, 3)
        Jp[i] = 1
        C[i] = ti.Matrix.zero(ti.f32, 3, 3)

def load_obj_vertices(filename, max_particles=n_particles):
    mesh = trimesh.load(filename, force='mesh')
    # mesh = mesh.convex_hull # fill it with simpler shape
    # fill the obj with random points
    # convex_hall_mesh = mesh.convex_hull
    spacing = 1 / 128
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    x_vals = np.arange(min_bound[0], max_bound[0], spacing)
    y_vals = np.arange(min_bound[1], max_bound[1], spacing)
    z_vals = np.arange(min_bound[2], max_bound[2], spacing)
    grid = np.stack(np.meshgrid(x_vals, y_vals, z_vals, indexing='ij'), axis=-1).reshape(-1, 3)

    inside = mesh.contains(grid)
    inside_points = grid[inside]
    
    # points = mesh.vertices
    surface_points = mesh.vertices
    jitter_strength = spacing * 0.1
    num_shell_copies = 5
    
    shell_points = []
    for _ in range(num_shell_copies):
        jitter = (np.random.rand(*surface_points.shape) - 0.5) * 2 * jitter_strength
        shell_points.append(surface_points + jitter)
    shell_points = np.concatenate(shell_points, axis=0)
    
    points = np.concatenate([inside_points, shell_points], axis=0)
    np.random.shuffle(points)
    
    # if len(points) > max_particles:
    #     indices = np.random.choice(len(points), max_particles, replace=False)
    #     points = points[indices]
    # elif len(points) < max_particles:
    #     pad = max_particles - len(points)
    #     extra = np.repeat(points[:1], pad, axis=0)
    #     points = np.concatenate([points, extra], axis=0)

    points = points[:max_particles]
    
    # Normalize and scale
    points -= points.min(axis=0)
    points /= points.max()
    points = points * 0.3 + np.array([0.35, 0.35, 0.35]) 
    
    return points

@ti.kernel
def set_positions_from_numpy(np_points: ti.types.ndarray(), num_valid: int):
    for i in range(n_particles):
        if i < num_valid:
            for j in ti.static(range(3)):
                pos[i][j] = np_points[i, j]
        else:
            pos[i] = ti.Vector([0.0, 0.0, 0.0])
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        F[i] = ti.Matrix.identity(ti.f32, 3)
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(ti.f32, 3, 3)
        material[i] = 2  # snow

@ti.kernel
def fill_radius():
    for i in range(group_size):
        radius_field[i] = 0.003


@ti.kernel
def prepare_render(mat_id: int, actual_count: int):
    start = mat_id * group_size
    for i in range(group_size):
        p = start + i
        if p < actual_count:
            render_pos_group[i] = pos[p]
            radius_field[i] = 0.003
        else:
            render_pos_group[i] = ti.Vector([0.0,0.0,0.0])
            radius_field[i] = 0.0



@ti.kernel
def substep(actual_count: int):
    # Reset grid
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, 3)
        grid_m[I] = 0
    # particle to Grid (P2G)
    for p in range(actual_count):
        base = (pos[p] * inv_dx - 0.5).cast(int)
        fx = pos[p] * inv_dx - base.cast(float)
        # b-spline interpolation weight between neighbor grid
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # update the deformation grdient based on neighbor grid motion C[p]
        F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * C[p]) @ F[p]
        # hardening factor ξ, 10 is a starting point for snow. as the snow get compress, it gets harder
        # h = 10.0
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:
            h = 0.3 # jelly is more soft
        mu, la = mu_0 * h, lambda_0 * h # mu -> shear stiffness, la ->volumetric stiffness
        if material[p] == 0:
            mu = 0.0 # water doesn't have shear stiffness
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = min(max(sig[d, d], 0.95), 1.05)
            # new_sig = sig[d, d]
            if material[p] == 2:
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0: # liquid reset for stability
            F[p] = ti.Matrix.identity(float, 3) * ti.sqrt(J)
        elif material[p] == 2: # snow
            F[p] = U @ sig @ V.transpose()
        # F[p] - R (rotation from SVD)
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                 ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_idx = base + offset
            grid_v[grid_idx] += weight * (p_mass * vel[p] + affine @ dpos)
            grid_m[grid_idx] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = (1 / grid_m[I]) * grid_v[I]
            grid_v[I] += dt * gravity[None] * 30
            dist = attractor_pos[None] - dx * I.cast(float)
            grid_v[I] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            for d in ti.static(range(3)):
                if I[d] < 3 and grid_v[I][d] < 0:
                    grid_v[I][d] = 0
                if I[d] > n_grid - 3 and grid_v[I][d] > 0:
                    grid_v[I][d] = 0
    # Grid back to Particle(G2P)
    for p in range(actual_count):
        base = (pos[p] * inv_dx - 0.5).cast(int)
        fx = pos[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, 3)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            g_v = grid_v[base + offset]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        vel[p], C[p] = new_v, new_C
        pos[p] += dt * vel[p]

@ti.func
def phi(pos):
    # Sphere level set at center c radius r
    c = ti.Vector([0.5, 0.5, 0.5])
    r = 0.15
    return (pos - c).norm() - r


@ti.func
def grad_phi(pos):
    eps = 1e-4
    dx = ti.Vector([eps, 0.0, 0.0])
    dy = ti.Vector([0.0, eps, 0.0])
    dz = ti.Vector([0.0, 0.0, eps])
    nx = phi(pos + dx) - phi(pos - dx)
    ny = phi(pos + dy) - phi(pos - dy)
    nz = phi(pos + dz) - phi(pos - dz)
    n = ti.Vector([nx, ny, nz])
    return n.normalized()

# this should happen after P2G
@ti.kernel
def grid_collision_response(mu: ti.f32):
    for I in ti.grouped(grid_v):
        if grid_m[I] > 0:
            pos = dx * I.cast(ti.f32)
            dist = phi(pos)
            if dist <= 0: 
                n = grad_phi(pos)
                v_rel = grid_v[I]
                vn = v_rel.dot(n)

                if vn < 0: 
                    vt = v_rel - vn * n
                    vt_norm = vt.norm()

                    if vt_norm <= -mu * vn:
                        v_rel = ti.Vector.zero(ti.f32, 3)
                    else:
                        v_rel = vt + mu * (-vn) * vt.normalized()

                    grid_v[I] = v_rel

@ti.kernel
def update_particle_pos(num_obj_particles:int):
    for p in range(num_obj_particles):
        pos[p] += dt * vel[p]

# this should happen after G2p
@ti.kernel
def particle_collision_response(num_particles: int, mu: ti.f32):
    for p in range(num_particles):
        pos = pos[p]
        dist = phi(pos)
        if dist <= 0:
            n = grad_phi(pos)
            v_particle = vel[p]

            v_co = ti.Vector([0.0, 0.0, 0.0])  # collision object velocity
            v_rel = v_particle - v_co

            vn = v_rel.dot(n)
            if vn < 0:
                vt = v_rel - vn * n
                vt_norm = vt.norm()
                v_rel_prime = ti.Vector([0.0, 0.0, 0.0])
                if vt_norm <= -mu * vn:
                    v_rel_prime = ti.Vector([0.0, 0.0, 0.0])
                else:
                    v_rel_prime = vt + mu * (-vn) * (vt / vt_norm)

                v_new = v_rel_prime + v_co
                vel[p] = v_new

                # push particle outside
                pos[p] = pos + n * (-dist + 1e-4)


# GUI & rendering setup
window = ti.ui.Window("3D MPM (.obj)", res=(800,800), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

camera.position(2, 2, 2)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 1, 0)

USE_OBJ = True
OBJ_PATH = "bunny.obj"

if USE_OBJ:
    points = load_obj_vertices(OBJ_PATH)
    num_obj_particles = points.shape[0]
    set_positions_from_numpy(points.astype(np.float32), num_obj_particles)
else:
    reset()
    num_obj_particles = n_particles
    fill_radius()

# reset_from_obj("teapot.obj")
# assert num_obj_particles <= n_particles, "Too many particles from .obj!"

gravity[None] = [0, -1, 0]
attractor_pos[None] = [0.5, 0.5, 0.5]
attractor_strength[None] = 0.0

material_colors = [
    ti.Vector([0.1, 0.6, 0.8]),  # fluid
    ti.Vector([0.8, 0.3, 0.3]),  # jelly
    ti.Vector([0.9, 0.9, 0.9]),  # snow
]

print(f"Loaded {num_obj_particles} snow particles")

while window.running:
    for _ in range(int(2e-3 // dt)):
        substep(num_obj_particles)
        grid_collision_response(mu=0.3)
        particle_collision_response(num_obj_particles, mu=0.3)

        # Now update positions
        update_particle_pos(num_obj_particles)
            
            
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.9,0.9,0.9))
    scene.point_light(pos=(2,2,2), color=(1,1,1))

    prepare_render(0, num_obj_particles)

    scene.particles(
        centers=render_pos_group,
        radius=0.003,
        color=tuple(material_colors[2]),
        index_count=num_obj_particles)
    
    scene.particles(
        centers=cube_vis_points,
        radius=0.004,
        color=(1.0, 0.5, 0.0),
        index_count=64,
    )

    canvas.scene(scene)
    window.show()