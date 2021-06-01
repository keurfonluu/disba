"""
Numba implementation of the Fortran program surf96.

This module is not a one-to-one translation from Fortran to Python.
The code has been adapted and optimized for Numba.

..

    COMPUTER PROGRAMS IN SEISMOLOGY
    VOLUME IV

    COPYRIGHT 1986, 1991
    D. R. Russell, R. B. Herrmann
    Department of Earth and Atmospheric Sciences
    Saint Louis University
    221 North Grand Boulevard
    St. Louis, Missouri 63103
    U. S. A.

"""

import numpy

from .._common import jitted
from .._exception import DispersionError
from ._common import normc

__all__ = [
    "surf96",
]


twopi = 2.0 * numpy.pi


@jitted
def dnka(wvno2, gam, gammk, rho, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz):
    """Dunkin's matrix."""
    ca = numpy.zeros((5, 5), dtype=numpy.float64)

    gamm1 = gam - 1.0
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1

    rho2 = rho * rho
    a0pq = a0 - cpcq
    t = -2.0 * wvno2

    ca[0, 0] = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[0, 1] = (wvno2 * cpy - cqx) / rho
    ca[0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[0, 3] = (cpz - wvno2 * cqw) / rho
    ca[0, 4] = -(2.0 * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2

    ca[1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[1, 1] = cpcq
    ca[1, 2] = gammk * cpz - gamm1 * cqw
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]

    ca[3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[3, 1] = -xy
    ca[3, 2] = gamm1 * cpy - gammk * cqx
    ca[3, 3] = ca[1, 1]
    ca[3, 4] = ca[0, 1]

    ca[4, 0] = (
        -(2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy) * rho2
    )
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = (
        -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy)
        * rho
    )
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]

    ca[2, 0] = t * ca[4, 2]
    ca[2, 1] = t * ca[3, 2]
    ca[2, 2] = a0 + 2.0 * (cpcq - ca[0, 0])
    ca[2, 3] = t * ca[1, 2]
    ca[2, 4] = t * ca[0, 2]

    return ca


@jitted
def var(p, q, ra, rb, wvno, xka, xkb, dpth):
    """Find variables cosP, cosQ, sinP, sinQ..."""
    # Examine P-wave eigenfunctions
    # Checking whether c > vp, c = vp or c < vp
    pex = 0.0
    if wvno < xka:
        sinp = numpy.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = numpy.cos(p)
    elif wvno == xka:
        cosp = 1.0
        w = dpth
        x = 0.0
    elif wvno > xka:
        pex = p
        fac = numpy.exp(-2.0 * p) if p < 16.0 else 0.0
        cosp = (1.0 + fac) * 0.5
        sinp = (1.0 - fac) * 0.5
        w = sinp / ra
        x = ra * sinp

    # Examine S-wave eigenfunctions
    # Checking whether c > vs, c = vs or c < vs
    sex = 0.0
    if wvno < xkb:
        sinq = numpy.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = numpy.cos(q)
    elif wvno == xkb:
        cosq = 1.0
        y = dpth
        z = 0.0
    elif wvno > xkb:
        sex = q
        fac = numpy.exp(-2.0 * q) if q < 16.0 else 0.0
        cosq = (1.0 + fac) * 0.5
        sinq = (1.0 - fac) * 0.5
        y = sinq / rb
        z = rb * sinq

    # Form eigenfunction products for use with compound matrices
    exa = pex + sex
    a0 = numpy.exp(-exa) if exa < 60.0 else 0.0
    cpcq = cosp * cosq
    cpy = cosp * y
    cpz = cosp * z
    cqw = cosq * w
    cqx = cosq * x
    xy = x * y
    xz = x * z
    wy = w * y
    wz = w * z

    qmp = sex - pex
    fac = numpy.exp(qmp) if qmp > -40.0 else 0.0
    cosq *= fac
    y *= fac
    z *= fac

    return w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz


@jitted
def dltar1(wvno, omega, d, a, b, rho, llw):
    """Love-wave period equation."""
    beta1 = b[-1]
    rho1 = rho[-1]
    xkb = omega / beta1
    wvnop = wvno + xkb
    wvnom = numpy.abs(wvno - xkb)
    rb = numpy.sqrt(wvnop * wvnom)
    e1 = rho1 * rb
    e2 = 1.0 / (beta1 * beta1)

    for m in range(len(d) - 2, llw, -1):
        beta1 = b[m]
        rho1 = rho[m]
        xmu = rho1 * beta1 * beta1
        xkb = omega / beta1
        wvnop = wvno + xkb
        wvnom = numpy.abs(wvno - xkb)
        rb = numpy.sqrt(wvnop * wvnom)
        q = d[m] * rb

        if wvno < xkb:
            sinq = numpy.sin(q)
            y = sinq / rb
            z = -rb * sinq
            cosq = numpy.cos(q)
        elif wvno == xkb:
            cosq = 1.0
            y = d[m]
            z = 0.0
        else:
            fac = numpy.exp(-2.0 * q) if q < 16.0 else 0.0
            cosq = (1.0 + fac) * 0.5
            sinq = (1.0 - fac) * 0.5
            y = sinq / rb
            z = rb * sinq

        e10 = e1 * cosq + e2 * xmu * z
        e20 = e1 * y / xmu + e2 * cosq
        xnor = numpy.abs(e10)
        ynor = numpy.abs(e20)
        xnor = max(xnor, ynor)
        if xnor < 1.0e-40:
            xnor = 1.0
        e1 = e10 / xnor
        e2 = e20 / xnor

    return e1


@jitted
def dltar4(wvno, omega, d, a, b, rho, llw):
    """Rayleigh-wave period equation."""
    e = numpy.zeros(5, dtype=numpy.float64)
    ee = numpy.zeros(5, dtype=numpy.float64)

    omega = max(omega, 1.0e-4)
    wvno2 = wvno * wvno
    xka = omega / a[-1]
    xkb = omega / b[-1]
    wvnop = wvno + xka
    wvnom = numpy.abs(wvno - xka)
    ra = numpy.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = numpy.abs(wvno - xkb)
    rb = numpy.sqrt(wvnop * wvnom)
    t = b[-1] / omega

    # E matrix for the bottom half-space
    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = rho[-1]
    e[0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb)
    e[1] = -rho1 * ra
    e[2] = rho1 * (gamm1 - gammk * ra * rb)
    e[3] = rho1 * rb
    e[4] = wvno2 - ra * rb

    # Matrix multiplication from bottom layer upward
    for m in range(len(d) - 2, llw, -1):
        xka = omega / a[m]
        xkb = omega / b[m]
        t = b[m] / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2
        wvnop = wvno + xka
        wvnom = numpy.abs(wvno - xka)
        ra = numpy.sqrt(wvnop * wvnom)
        wvnop = wvno + xkb
        wvnom = numpy.abs(wvno - xkb)
        rb = numpy.sqrt(wvnop * wvnom)

        dpth = d[m]
        rho1 = rho[m]
        p = ra * dpth
        q = rb * dpth

        # Evaluate cosP, cosQ...
        _, _, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = var(
            p, q, ra, rb, wvno, xka, xkb, dpth
        )

        # Evaluate Dunkin's matrix
        ca = dnka(wvno2, gam, gammk, rho1, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz)

        for i in range(5):
            ee[i] = 0.0
            for j in range(5):
                ee[i] += e[j] * ca[j, i]

        ee, _ = normc(ee, 5)
        for i in range(5):
            e[i] = ee[i]

    if llw == 0:
        xka = omega / a[0]
        wvnop = wvno + xka
        wvnom = numpy.abs(wvno - xka)
        ra = numpy.sqrt(wvnop * wvnom)
        dpth = d[0]
        rho1 = rho[0]
        p = ra * dpth
        w, cosp, _, _, _, _, _, _, _, _, _, _ = var(
            p, q, ra, 1.0e-5, wvno, xka, xkb, dpth
        )
        dlt = cosp * e[0] - rho1 * w * e[1]
    else:
        dlt = e[0]

    return dlt


@jitted
def fast_delta(wvno, omega, d, alpha, beta, rho, llw):
    """
    Fast delta matrix.

    After Buchen and Ben-Hador (1996).

    """
    mmax = len(d)

    # Handle water layer
    if llw == 0:
        beta = beta * 1.0
        beta[0] = 1.0e-8

    # Phase velocity
    c = omega / wvno
    c2 = c * c

    # Rayleigh-wave fast delta matrix
    gam0 = beta[0] ** 2.0 / c2
    mu0 = rho[0] * beta[0] ** 2.0
    t0 = 2.0 - c2 / beta[0] ** 2.0

    X = numpy.zeros(5, dtype=numpy.complex_)
    X[0] = 2.0 * t0
    X[1] = -t0 * t0
    X[4] = -4.0
    X *= mu0 * mu0
    X, _ = normc(X, 5)

    for i in range(mmax - 1):
        scale = 0
        gam1 = beta[i + 1] ** 2.0 / c2
        eps = rho[i + 1] / rho[i]
        eta = 2.0 * (gam0 - eps * gam1)
        a = eps + eta
        ap = a - 1.0
        b = 1.0 - eta
        bp = b - 1.0
        gam0 = gam1

        r = (
            numpy.sqrt(1.0 - c2 / alpha[i] ** 2.0)
            if c < alpha[i]
            else numpy.sqrt(c2 / alpha[i] ** 2.0 - 1.0) * 1j
            if c > alpha[i]
            else 0.0
        )
        s = (
            numpy.sqrt(1.0 - c2 / beta[i] ** 2.0)
            if c < beta[i]
            else numpy.sqrt(c2 / beta[i] ** 2.0 - 1.0) * 1j
            if c > beta[i]
            else 0.0
        )

        if c < alpha[i]:
            Ca = numpy.cosh(wvno * r * d[i])
            Sa = numpy.sinh(wvno * r * d[i])
        elif c > alpha[i]:
            Ca = numpy.cos(wvno * r.imag * d[i])
            Sa = numpy.sin(wvno * r.imag * d[i]) * 1j
        else:
            Ca = 1.0
            Sa = 0.0

        if c < beta[i]:
            Cb = numpy.cosh(wvno * s * d[i])
            Sb = numpy.sinh(wvno * s * d[i])

            # Handle hyperbolic overflow
            if wvno * s.real * d[i] > 80.0:
                scale = 1
                Cb /= Ca
                Sb /= Ca

        elif c > beta[i]:
            Cb = numpy.cos(wvno * s.imag * d[i])
            Sb = numpy.sin(wvno * s.imag * d[i]) * 1j

        else:
            Cb = 1.0
            Sb = 0.0

        if i == 0 and llw == 0:
            Cb = 1.0
            Sb = 0.0

        p1 = Cb * X[1] + s * Sb * X[2]
        p2 = Cb * X[3] + s * Sb * X[4]
        p3 = Cb * X[2]
        p4 = Cb * X[4]
        if c == beta[i]:
            p3 += wvno * d[i] * X[1]
            p4 += wvno * d[i] * X[3]
        else:
            p3 += Sb * X[1] / s
            p4 += Sb * X[3] / s

        if scale == 1:
            q1 = p1 - r * p2
            q3 = p3 - r * p4
            q2 = p4
            q4 = p2
            if c != alpha[i]:
                q2 -= p3 / r
                q4 -= p1 / r
        else:
            q1 = Ca * p1 - r * Sa * p2
            q3 = Ca * p3 - r * Sa * p4
            q2 = Ca * p4
            q4 = Ca * p2
            if c == alpha[i]:
                q2 -= wvno * d[i] * p3
                q4 -= wvno * d[i] * p1
            else:
                q2 -= Sa * p3 / r
                q4 -= Sa * p1 / r

        y1 = a * q1
        y2 = ap * q2
        z1 = bp * q1
        z2 = b * q2
        if scale == 0:
            y1 += ap * X[0]
            y2 += a * X[0]
            z1 += b * X[0]
            z2 += bp * X[0]

        X[0] = bp * y1 + b * y2
        X[1] = a * y1 + ap * y2
        X[2] = eps * q3
        X[3] = eps * q4
        X[4] = bp * z1 + b * z2
        X, _ = normc(X, 5)

    r = (
        numpy.sqrt(1.0 - c2 / alpha[-1] ** 2)
        if c < alpha[-1]
        else numpy.sqrt(c2 / alpha[-1] ** 2 - 1.0) * 1j
        if c > alpha[-1]
        else 0.0
    )
    s = (
        numpy.sqrt(1.0 - c2 / beta[-1] ** 2)
        if c < beta[-1]
        else numpy.sqrt(c2 / beta[-1] ** 2 - 1.0) * 1j
        if c > beta[-1]
        else 0.0
    )

    return numpy.real(X[1] + s * X[3] - r * (X[3] + s * X[4]))


@jitted
def dltar(wvno, omega, d, a, b, rho, ifunc, llw):
    """Select Rayleigh or Love wave period equation."""
    if ifunc == 1:
        return dltar1(wvno, omega, d, a, b, rho, llw)
    elif ifunc == 2:
        return dltar4(wvno, omega, d, a, b, rho, llw)
    else:
        return fast_delta(wvno, omega, d, a, b, rho, llw)


@jitted
def nevill(t, c1, c2, del1, del2, d, a, b, rho, ifunc, llw):
    """Hybrid method for refining root once it has been bracketted."""
    x = numpy.zeros(20, dtype=numpy.float64)
    y = numpy.zeros(20, dtype=numpy.float64)

    # Initial guess
    omega = twopi / t
    c3 = 0.5 * (c1 + c2)
    del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw)
    nev = 1
    nctrl = 1

    while True:
        nctrl += 1
        if nctrl >= 100:
            break

        # Make sure new estimate is inside the previous values
        # If not, perform interval halving
        if c3 < min(c1, c2) or c3 > max(c1, c2):
            nev = 0
            c3 = 0.5 * (c1 + c2)
            del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw)

        s13 = del1 - del3
        s32 = del3 - del2

        # Define new bounds according to the sign of the period equation
        if numpy.sign(del3) * numpy.sign(del1) < 0.0:
            c2 = c3
            del2 = del3
        else:
            c1 = c3
            del1 = del3

        # Check for convergence
        if numpy.abs(c1 - c2) <= 1.0e-6 * c1:
            break

        # If the slopes are not the same between c1, c2 and c3
        # Do not use Neville iteration
        if numpy.sign(s13) != numpy.sign(s32):
            nev = 0

        # If the period equation differs by more than a factor of 10
        # Use interval halving to avoid poor behavior of polynomial fit
        ss1 = numpy.abs(del1)
        s1 = 0.01 * ss1
        ss2 = numpy.abs(del2)
        s2 = 0.01 * ss2
        if s1 > ss2 or s2 > ss1 or nev == 0:
            c3 = 0.5 * (c1 + c2)
            del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw)
            nev = 1
            m = 1
        else:
            if nev == 2:
                x[m - 1] = c3
                y[m - 1] = del3
            else:
                x[0] = c1
                y[0] = del1
                x[1] = c2
                y[1] = del2
                m = 1

            # Perform Neville iteration
            flag = 1
            for kk in range(m):
                j = m - kk
                denom = y[m] - y[j]
                if numpy.abs(denom) < 1.0e-10 * numpy.abs(y[m]):
                    flag = 0
                    break
                else:
                    x[j - 1] = (-y[j - 1] * x[j] + y[m] * x[j - 1]) / denom

            if flag:
                c3 = x[0]
                del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw)
                nev = 2
                m += 1
                m = min(m, 10)
            else:
                c3 = 0.5 * (c1 + c2)
                del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw)
                nev = 1
                m = 1

    return c3


@jitted
def getsol(t1, c1, clow, dc, cm, betmx, ifirst, del1st, d, a, b, rho, ifunc, llw):
    """Bracket dispersion curve and then refine it."""
    # Bracket solution
    omega = twopi / t1
    del1 = dltar(omega / c1, omega, d, a, b, rho, ifunc, llw)
    del1st = del1 if ifirst else del1st
    idir = -1.0 if not ifirst and numpy.sign(del1st) * numpy.sign(del1) < 0.0 else 1.0

    # idir indicates the direction of the search for the true phase velocity from the initial estimate
    while True:
        c2 = c1 + idir * dc

        if c2 <= clow:
            idir = 1.0
            c1 = clow
        else:
            omega = twopi / t1
            del2 = dltar(omega / c2, omega, d, a, b, rho, ifunc, llw)

            if numpy.sign(del1) != numpy.sign(del2):
                c1 = nevill(t1, c1, c2, del1, del2, d, a, b, rho, ifunc, llw)
                iret = c1 > betmx
                break

            c1 = c2
            del1 = del2

            iret = c1 < cm or c1 >= betmx + dc
            if iret:
                break

    return c1, del1st, iret


@jitted
def gtsolh(a, b):
    """Starting solution."""
    c = 0.95 * b

    for _ in range(5):
        gamma = b / a
        kappa = c / b
        k2 = kappa ** 2
        gk2 = (gamma * kappa) ** 2
        fac1 = numpy.sqrt(1.0 - gk2)
        fac2 = numpy.sqrt(1.0 - k2)
        fr = (2.0 - k2) ** 2 - 4.0 * fac1 * fac2
        frp = -4.0 * (2.0 - k2) * kappa
        frp += 4.0 * fac2 * gamma * gamma * kappa / fac1
        frp += 4.0 * fac1 * kappa / fac2
        frp /= b
        c -= fr / frp

    return c


@jitted
def getc(t, d, a, b, rho, mode, ifunc, dc):
    """Get phase velocity dispersion curve."""
    # Initialize arrays
    kmax = len(t)
    c = numpy.zeros(kmax, dtype=numpy.float64)
    cg = numpy.zeros(kmax, dtype=numpy.float64)

    # Check for water layer
    llw = 0 if b[0] <= 0.0 else -1

    # Find the extremal velocities to assist in starting search
    betmx = -1.0e20
    betmn = 1.0e20
    mmax = len(d)
    for i in range(mmax):
        if b[i] > 0.01 and b[i] < betmn:
            betmn = b[i]
            jmn = i
            jsol = False
        elif b[i] < 0.01 and a[i] < betmn:
            betmn = a[i]
            jmn = i
            jsol = True
        betmx = max(betmx, b[i])

    # Solid layer solve halfspace period equation
    cc = betmn if jsol else gtsolh(a[jmn], b[jmn])

    # Back off a bit to get a starting value at a lower phase velocity
    cc *= 0.9
    c1 = cc
    cm = cc

    one = 1.0e-2
    onea = 1.5
    for iq in range(mode + 1):
        ibeg = 0
        iend = kmax

        del1st = 0.0
        for k in range(ibeg, iend):
            # Get initial phase velocity estimate to begin search
            ifirst = k == ibeg
            if ifirst and iq == 0:
                clow = cc
                c1 = cc
            elif ifirst and iq > 0:
                clow = c1
                c1 = c[ibeg] + one * dc
            elif not ifirst and iq > 0:
                clow = c[k] + one * dc
                c1 = max(c[k - 1], clow)
            elif not ifirst and iq == 0:
                clow = cm
                c1 = c[k - 1] - onea * dc

            # Bracket root and refine it
            c1, del1st, iret = getsol(
                t[k],
                c1,
                clow,
                dc,
                cm,
                betmx,
                ifirst,
                del1st,
                d,
                a,
                b,
                rho,
                ifunc,
                llw,
            )

            if iret:
                if iq > 0:
                    for i in range(k, kmax):
                        cg[i] = 0.0

                    if iq == mode:
                        return cg
                    else:
                        c1 = 0.0
                        break
                else:
                    raise DispersionError("failed to find root for fundamental mode")

            c[k] = c1
            cg[k] = c[k]
            c1 = 0.0

    return cg


@jitted
def surf96(t, d, a, b, rho, mode=0, itype=0, ifunc=2, dc=0.005, dt=0.025):
    """
    Get phase or group velocity dispersion curve.

    Parameters
    ----------
    t : array_like
        Periods (in s).
    d : array_like
        Layer thickness (in km).
    a : array_like
        Layer P-wave velocity (in km/s).
    b : array_like
        Layer S-wave velocity (in km/s).
    rho : array_like
        Layer density (in g/cm3).
    mode : int, optional, default 0
        Mode number (0 if fundamental).
    itype : int, optional, default 0
        Velocity type:
         - 0: phase velocity,
         - 1: group velocity.
    ifunc : int, optional, default 2
        Select wave type and algorithm for period equation:
         - 1: Love-wave (Thomson-Haskell method),
         - 2: Rayleigh-wave (Dunkin's matrix),
         - 3: Rayleigh-wave (fast delta matrix).
    dc : scalar, optional, default 0.005
        Phase velocity increment for root finding.
    dt : scalar, optional, default 0.025
        Frequency increment (%) for calculating group velocity.

    Returns
    -------
    array_like
        Phase or group dispersion velocity.

    """
    nt = len(t)
    t1 = numpy.empty(nt, dtype=numpy.float64)

    if itype == 1:
        fac = 1.0 + dt
        for i in range(nt):
            t1[i] = t[i] / fac
    else:
        for i in range(nt):
            t1[i] = t[i]

    c1 = getc(t1, d, a, b, rho, mode, ifunc, dc)

    if itype == 1:
        t2 = numpy.empty(nt, dtype=numpy.float64)
        fac = 1.0 - dt
        for i in range(nt):
            t2[i] = t[i] / fac

        c2 = getc(t2, d, a, b, rho, mode, ifunc, dc)

        for i in range(nt):
            if c2[i] > 0.0:
                t1i = 1.0 / t1[i]
                t2i = 1.0 / t2[i]
                c1[i] = (t1i - t2i) / (t1i / c1[i] - t2i / c2[i])
            else:
                c1[i] = 0.0

    return c1
