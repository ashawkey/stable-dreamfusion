import taichi as ti


@ti.kernel
def composite_test(
    sigmas: ti.types.ndarray(ndim=2), rgbs: ti.types.ndarray(ndim=3),
    deltas: ti.types.ndarray(ndim=2), ts: ti.types.ndarray(ndim=2),
    hits_t: ti.types.ndarray(ndim=2),
    alive_indices: ti.types.ndarray(ndim=1), T_threshold: float,
    N_eff_samples: ti.types.ndarray(ndim=1),
    opacity: ti.types.ndarray(ndim=1),
    depth: ti.types.ndarray(ndim=1), rgb: ti.types.ndarray(ndim=2)):

    for n in alive_indices:
        samples = N_eff_samples[n]
        if samples == 0:
            alive_indices[n] = -1
        else:
            r = alive_indices[n]

            T = 1 - opacity[r]

            rgb_temp_0 = 0.0
            rgb_temp_1 = 0.0
            rgb_temp_2 = 0.0
            depth_temp = 0.0
            opacity_temp = 0.0

            for s in range(samples):
                a = 1.0 - ti.exp(-sigmas[n, s] * deltas[n, s])
                w = a * T

                rgb_temp_0 += w * rgbs[n, s, 0]
                rgb_temp_1 += w * rgbs[n, s, 1]
                rgb_temp_2 += w * rgbs[n, s, 2]
                depth[r] += w * ts[n, s]
                opacity[r] += w
                T *= 1.0 - a

                if T <= T_threshold:
                    alive_indices[n] = -1
                    break

            rgb[r, 0] += rgb_temp_0
            rgb[r, 1] += rgb_temp_1
            rgb[r, 2] += rgb_temp_2
            depth[r] += depth_temp
            opacity[r] += opacity_temp
