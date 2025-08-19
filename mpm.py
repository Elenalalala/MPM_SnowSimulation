import taichi as ti

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



@ti.kernel
def fill_radius():
    for i in range(group_size):
        radius_field[i] = 0.003


@ti.kernel
def prepare_render(mat_id: int):
    start = mat_id * group_size
    for i in range(group_size):
        render_pos_group[i] = pos[start + i]
        radius_field[i] = 0.003



@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, 3)
        grid_m[I] = 0
    # particle to Grid (P2G)
    for p in pos:
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
    for p in pos:
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


# GUI & rendering setup
window = ti.ui.Window("3D MPM", res=(800, 800), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

camera.position(2, 2, 2)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 1, 0)

reset()
fill_radius()

gravity[None] = [0, -1, 0]
attractor_pos[None] = [0.5, 0.5, 0.5]
attractor_strength[None] = 0.0

material_colors = [
    ti.Vector([0.1, 0.6, 0.8]),  # fluid
    ti.Vector([0.8, 0.3, 0.3]),  # jelly
    ti.Vector([0.9, 0.9, 0.9]),  # snow
]

while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            reset()
    # if window.get_event():
    #     if window.event.key == 'r' and window.event.type == ti.GUI.PRESS:
    #         reset()

    for _ in range(int(2e-3 // dt)):
        substep()

    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.9, 0.9, 0.9))
    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

    for mat_id in range(3):
        prepare_render(mat_id)
        scene.particles(
            centers=render_pos_group,
            radius=0.003,
            color=tuple(material_colors[mat_id]),
            index_count=group_size,
        )

    canvas.scene(scene)
    window.show()