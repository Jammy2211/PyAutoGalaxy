# from jax import custom_jvp


r1_s1 = [2.5, 2, 1.5, 1, 0.5]


def reg1(z, _, i_sqrt_pi):
    v = z
    for coef in r1_s1:
        v = z - coef / v
    return i_sqrt_pi / v


r2_s1 = [1.320522, 35.7668, 219.031, 1540.787, 3321.99, 36183.31]
r2_s2 = [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6]


def reg2(z, sqrt_pi, _):
    mz2 = -(z**2)
    f1 = sqrt_pi
    f2 = 1.0
    for s in r2_s1:
        f1 = s - f1 * mz2
    for s in r2_s2:
        f2 = s - f2 * mz2

    return xp.exp(mz2) + 1j * z * f1 / f2


r3_s1 = [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239, 122.60793]
r3_s2 = [10.479857, 53.992907, 170.35400, 348.70392, 457.33448, 352.73063, 122.60793]


def reg3(z, sqrt_pi, _):
    miz = -1j * z
    f1 = sqrt_pi
    f2 = 1
    for s in r3_s1:
        f1 = s + f1 * miz
    for s in r3_s2:
        f2 = s + f2 * miz

    return f1 / f2


#
# @custom_jvp
# def w_f_approx(z):
#     """Compute the Faddeeva function :math:`w_{\\mathrm F}(z)` using the
#     approximation given in Zaghloul (2017).
#
#     :param z: complex number
#     :type z: ``complex`` or ``numpy.array(dtype=complex)``
#     :return: :math:`w_\\mathrm{F}(z)`
#     :rtype: ``complex``
#     """
#     sqrt_pi = 1 / xp.sqrt(xp.pi)
#     i_sqrt_pi = 1j * sqrt_pi
#
#     z_imag2 = z.imag**2
#     abs_z2 = z.real**2 + z_imag2
#
#     # use a single partial fraction approx for all large abs(z)**2
#     # to have better approx of the auto-derivatives
#     r1 = (abs_z2 >= 62.0) | ((abs_z2 >= 30.0) & (abs_z2 < 62.0) & (z_imag2 >= 1e-13))
#     # region bounds for 5 taken directly from Zaghloul (2017)
#     # https://dl.acm.org/doi/pdf/10.1145/3119904
#     r2_1 = (abs_z2 >= 30.0) & (abs_z2 < 62.0) & (z_imag2 < 1e-13)
#     r2_2 = (abs_z2 >= 2.5) & (abs_z2 < 30.0) & (z_imag2 < 0.072)
#     r2 = r2_1 | r2_2
#     r3 = xp.logical_not(r1) & xp.logical_not(r2)
#
#     # exploit symmetry to avoid overflow in some regions
#     r_flip = z.imag < 0
#     z_adjust = xp.where(r_flip, -z, z)
#     two_exp_zz = 2 * xp.exp(-(z_adjust**2))
#
#     args = (z_adjust, sqrt_pi, i_sqrt_pi)
#     wz = xp.empty_like(z)
#     wz = xp.where(r1, reg1(*args), wz)
#     wz = xp.where(r2, reg2(*args), wz)
#     wz = xp.where(r3, reg3(*args), wz)
#
#     # exploit symmetry to avoid overflow in some regions
#     wz = xp.where(r_flip, two_exp_zz - wz, wz)
#
#     return wz


# @w_f_approx.defjvp
# def w_f_approx_jvp(primals, tangents):
#     # define a custom jvp to avoid the issue using `xp.where` with `jax.grad`
#     (z,) = primals
#     (z_dot,) = tangents
#     primal_out = w_f_approx(z)
#     i_sqrt_pi = 1j / xp.sqrt(xp.pi)
#     tangent_out = z_dot * 2 * (i_sqrt_pi - z * primal_out)
#     return primal_out, tangent_out
