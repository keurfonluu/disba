"""
Numba implementation of the Fortran programs sregn96 and slegn96.

This module is not a one-to-one translation from Fortran to Python.
The code has been adapted and optimized for Numba.

Only the computation of eigenfunctions is implemented.

..

    COMPUTER PROGRAMS IN SEISMOLOGY
    VOLUME III

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
from ._common import normc
from ._surf96 import surf96

__all__ = [
    "swegn96",
]


@jitted
def evalg(m, d, a, b, rho, wvno, om):
    """Layered half space problem for Rayleigh-wave."""
    gbr = numpy.zeros(5, dtype=numpy.complex_)
    wvno2 = wvno * wvno
    om2 = om * om

    # Set up halfspace conditions
    xka = om / a[m]
    xkb = om / b[m] if b[m] > 0.01 else 0.0
    ra = numpy.sqrt(wvno2 - xka * xka + 0.0 * 1j)
    rb = numpy.sqrt(wvno2 - xkb * xkb + 0.0 * 1j)
    gam = b[m] * wvno / om
    gam = 2.0 * gam * gam
    gamm1 = gam - (1.0 + 0.0 * 1j)

    # Half space
    if b[m] > 0.01:
        gbr[0] = (
            rho[m] * rho[m] * om2 * om2 * (-gam * gam * ra * rb + wvno2 * gamm1 * gamm1)
        )
        gbr[1] = -rho[m] * wvno2 * ra * om2
        gbr[2] = -rho[m] * (-gam * ra * rb + wvno2 * gamm1) * om2 * wvno
        gbr[3] = rho[m] * wvno2 * rb * om2
        gbr[4] = wvno2 * (wvno2 - ra * rb)

        fac = 0.25 / (-rho[m] * rho[m] * om2 * om2 * wvno2 * ra * rb)
        for i in range(5):
            gbr[i] *= fac

    else:
        # All fluid
        if numpy.all(b < 0.01):
            gbr[0] = 0.5 / ra
            gbr[1] = (0.5 + 0.0 * 1j) / (-rho[m] * om2)
            gbr[2] = 0.0 + 0.0 * 1j
            gbr[3] = 0.0 + 0.0 * 1j
            gbr[4] = 0.0 + 0.0 * 1j
        else:
            gbr[0] = 0.0 + 0.0 * 1j
            gbr[1] = 0.0 + 0.0 * 1j
            gbr[2] = 0.0 + 0.0 * 1j
            gbr[3] = 0.5 * rho[m] * om2 / ra
            gbr[4] = -0.5 + 0.0 * 1j

    return numpy.real(gbr)


@jitted
def varl(m, omega, wvno, dpth, b, rho):
    """
    Find variables cosQ, sinQ for Love-wave.

    """
    # Define the horizontal wavenumber for the S-wave
    xkb = omega / b[m]

    # Define the vertical wavenumber for the given wvno
    wvnop = wvno + xkb
    wvnom = numpy.abs(wvno - xkb)
    fac = wvnop * wvnom
    rb = numpy.sqrt(wvnop * wvnom)
    q = rb * dpth
    mu = rho[m] * b[m] * b[m]

    # Examine S-wave eigenfunctions
    # Checking whether c > vs, c = vs or c < vs
    eexl = 0.0
    if wvno < xkb:
        sinq = numpy.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = numpy.cos(q)
    elif wvno == xkb:
        cosq = 1.0
        y = dpth
        z = 0.0
    else:
        eexl = q
        fac = numpy.exp(-2.0 * q) if q < 18.0 else 0.0
        cosq = (1.0 + fac) * 0.5
        sinq = (1.0 - fac) * 0.5
        y = sinq / rb
        z = rb * sinq

    return rb, mu, cosq, sinq, y, z, eexl


@jitted
def varsv(p, q, rp, rsv, d, iwat):
    """
    Find variables cosP, cosQ, sinP, sinQ for Rayleigh-wave.

    """
    pr = numpy.real(p)
    pi = numpy.imag(p)
    qr = numpy.real(q)
    qi = numpy.imag(q)

    pex = pr
    svex = 0.0

    epp = 0.5 * (numpy.cos(pi) + numpy.sin(pi) * 1j)
    epm = numpy.conj(epp)
    pfac = numpy.exp(-2.0 * pr) if pr < 30.0 else 0.0
    cosp = numpy.real(epp + pfac * epm)
    sinp = epp - pfac * epm
    rsinp = numpy.real(rp * sinp)
    sinpr = d if numpy.abs(pr) < 1.0e-5 and numpy.abs(rp) < 1.0e-5 else sinp / rp

    # Fluid layer
    if iwat == 1:
        cosq = 1.0
        rsinq = 0.0
        sinqr = 0.0

    # Elastic layer
    else:
        svex = qr
        eqp = 0.5 * (numpy.cos(qi) + numpy.sin(qi) * 1j)
        eqm = numpy.conj(eqp)
        svfac = numpy.exp(-2.0 * qr) if qr < 30.0 else 0.0
        cosq = numpy.real(eqp + svfac * eqm)
        sinq = eqp - svfac * eqm
        rsinq = numpy.real(rsv * sinq)
        sinqr = d if numpy.abs(qr) < 1.0e-5 and numpy.abs(rsv) < 1.0e-5 else sinq / rsv

    return cosp, cosq, rsinp, rsinq, sinpr, sinqr, pex, svex


@jitted
def hskl(m, b, mu, cosq, y, z):
    """Thomson-Haskell's matrix for Love-wave."""
    hl = numpy.zeros((2, 2), dtype=numpy.float64)

    if b[m] > 0.01:
        hl[0, 0] = cosq
        hl[0, 1] = y / mu
        hl[1, 0] = z * mu
        hl[1, 1] = cosq
    else:
        hl[0, 0] = 1.0
        hl[0, 1] = 0.0
        hl[1, 0] = 0.0
        hl[1, 1] = 1.0

    return hl


@jitted
def hska(
    omega, wvno, b, rho, cosp, rsinp, sinpr, tcossv, trsinsv, tsinsvr, pex, svex, iwat
):
    """Thomson-Haskell's matrix for Rayleigh-wave."""
    aa = numpy.zeros((4, 4), dtype=numpy.complex_)
    wvno2 = wvno * wvno
    om2 = omega * omega

    if iwat == 1:
        # Fluid layer
        dfac = numpy.exp(-pex) if pex < 35.0 else 0.0
        aa[0, 0] = dfac
        aa[3, 3] = dfac
        aa[1, 1] = cosp
        aa[2, 2] = cosp
        aa[1, 2] = -rsinp / rho / om2
        aa[2, 1] = -rho * om2 * sinpr
    else:
        # Elastic layer
        dfac = numpy.exp(svex - pex) if pex - svex < 70.0 else 0.0
        cossv = dfac * tcossv
        rsinsv = dfac * trsinsv
        sinsvr = dfac * tsinsvr
        gam = 2.0 * b * b * wvno2 / om2
        gamm1 = gam - 1.0

        aa[0, 0] = cossv + gam * (cosp - cossv)
        aa[0, 1] = -wvno * gamm1 * sinpr + gam * rsinsv / wvno
        aa[0, 2] = -wvno * (cosp - cossv) / rho / om2
        aa[0, 3] = (wvno2 * sinpr - rsinsv) / rho / om2
        aa[1, 0] = gam * rsinp / wvno - wvno * gamm1 * sinsvr
        aa[1, 1] = cosp - gam * (cosp - cossv)
        aa[1, 2] = (-rsinp + wvno2 * sinsvr) / rho / om2
        aa[1, 3] = -aa[0, 2]
        aa[2, 0] = rho * om2 * gam * gamm1 * (cosp - cossv) / wvno
        aa[2, 1] = rho * om2 * (-gamm1 * gamm1 * sinpr + gam * gam * rsinsv / wvno2)
        aa[2, 2] = aa[1, 1]
        aa[2, 3] = -aa[0, 1]
        aa[3, 0] = rho * om2 * (gam * gam * rsinp / wvno2 - gamm1 * gamm1 * sinsvr)
        aa[3, 1] = -aa[2, 0]
        aa[3, 2] = -aa[1, 0]
        aa[3, 3] = aa[0, 0]

    return numpy.real(aa)


@jitted
def dnka(omega, wvno, b, rho, cosp, rsinp, sinpr, cossv, rsinsv, sinsvr, ex, exa, iwat):
    """Dunkin's matrix for Rayleigh-wave."""
    ca = numpy.zeros((5, 5), dtype=numpy.complex_)
    wvno2 = wvno * wvno
    om2 = omega * omega

    if iwat == 1:
        # Fluid layer
        dfac = numpy.exp(-ex) if ex < 35.0 else 0.0
        ca[2, 2] = dfac
        ca[0, 0] = cosp
        ca[4, 4] = cosp
        ca[0, 1] = -rsinp / rho / om2
        ca[1, 0] = -rho * sinpr * om2
        ca[1, 1] = cosp
        ca[3, 3] = cosp
        ca[3, 4] = ca[0, 1]
        ca[4, 3] = ca[1, 0]
    else:
        a0 = numpy.exp(-exa) if exa < 60.0 else 0.0
        cpcq = cosp * cossv
        cpy = cosp * sinsvr
        cpz = cosp * rsinsv
        cqw = cossv * sinpr
        cqx = cossv * rsinp
        xy = rsinp * sinsvr
        xz = rsinp * rsinsv
        wy = sinpr * sinsvr
        wz = sinpr * rsinsv

        # Elastic layer
        rho2 = rho * rho
        gam = 2.0 * b * b * wvno2 / om2
        gam2 = gam * gam
        gamm1 = gam - 1.0
        gamm2 = gamm1 * gamm1
        cqww2 = cqw * wvno2
        cqxw2 = cqx / wvno2
        gg1 = gam * gamm1
        a0c = (2.0 + 0.0 * 1j) * (a0 + 0.0 * 1j - cpcq)
        xz2 = xz / wvno2
        gxz2 = gam * xz2
        g2xz2 = gam2 * xz2
        a0cgg1 = a0c * (gam + gamm1)
        wy2 = wy * wvno2
        g2wy2 = gamm2 * wy2
        g1wy2 = gamm1 * wy2

        # OK by symmetry
        temp = a0c * gg1 + g2xz2 + g2wy2
        ca[2, 2] = a0 + temp + temp
        ca[0, 0] = cpcq - temp
        ca[0, 1] = (-cqx + wvno2 * cpy) / rho / om2

        temp = (0.5 + 0.0 * 1j) * a0cgg1 + gxz2 + g1wy2
        ca[0, 2] = wvno * temp / rho / om2
        ca[0, 3] = (-cqww2 + cpz) / rho / om2

        temp = wvno2 * (a0c + wy2) + xz
        ca[0, 4] = -temp / rho2 / om2 / om2

        ca[1, 0] = (-gamm2 * cqw + gam2 * cpz / wvno2) * rho * om2
        ca[1, 1] = cpcq
        ca[1, 2] = (gamm1 * cqww2 - gam * cpz) / wvno
        ca[1, 3] = -wz
        ca[1, 4] = ca[0, 3]

        temp = (0.5 + 0.0 * 1j) * a0cgg1 * gg1 + gam2 * gxz2 + gamm2 * g1wy2
        ca[2, 0] = -(2.0 + 0.0 * 1j) * temp * rho * om2 / wvno
        ca[2, 1] = -wvno * (gam * cqxw2 - gamm1 * cpy) * (2.0 + 0.0 * 1j)
        ca[2, 3] = -2.0 * ca[1, 2]
        ca[2, 4] = -2.0 * ca[0, 2]

        ca[3, 0] = (-gam2 * cqxw2 + gamm2 * cpy) * rho * om2
        ca[3, 1] = -xy
        ca[3, 2] = -0.5 * ca[2, 1]
        ca[3, 3] = ca[1, 1]
        ca[3, 4] = ca[0, 1]

        temp = gamm2 * (a0c * gam2 + g2wy2) + gam2 * g2xz2
        ca[4, 0] = -rho2 * om2 * om2 * temp / wvno2
        ca[4, 1] = ca[3, 0]
        ca[4, 2] = -0.5 * ca[2, 0]
        ca[4, 3] = ca[1, 0]
        ca[4, 4] = ca[0, 0]

    return numpy.real(ca)


@jitted
def shup(omega, wvno, d, a, b, rho):
    """Find the elements of the Haskell matrix for Love-wave."""
    mmax = len(d)

    uu = numpy.zeros(mmax, dtype=numpy.float64)
    tt = numpy.zeros(mmax, dtype=numpy.float64)
    exl = numpy.zeros(mmax, dtype=numpy.float64)

    # Kludge for fluid core
    if b[-1] > 0.01:
        dpth = 0.0
        rb, mu, cosq, _, _, _, eexl = varl(mmax - 1, omega, wvno, dpth, b, rho)
        uu[-1] = 0.0
        tt[-1] = -mu * rb
    else:
        uu[-1] = 0.0
        tt[-1] = 1.0

    for i in range(mmax - 2, -1, -1):
        if b[i] > 0.01:
            dpth = d[i]
            rb, mu, cosq, _, y, z, eexl = varl(i, omega, wvno, dpth, b, rho)
            hl = hskl(i, b, mu, cosq, y, z)

            # We actually use A^-1 since we go from the bottom to top
            i1 = i + 1
            amp0 = hl[0, 0] * uu[i1] - hl[0, 1] * tt[i1]
            str0 = -hl[1, 0] * uu[i1] + hl[1, 1] * tt[i1]

            # Normalize
            rr = numpy.abs(amp0)
            ss = numpy.abs(str0)
            rr = max(rr, ss)
            if rr < 1.0e-30:
                rr = 1.0
            exl[i] = numpy.log(rr) + eexl
            uu[i] = amp0 / rr
            tt[i] = str0 / rr

    return uu, tt, exl


@jitted
def svup(omega, wvno, d, a, b, rho):
    """
    Find the values of the Dunkin vectors for Rayleigh-wave.

    Values are calculated at each layer boundaries from bottom layer upward.

    """
    mmax = len(d)

    ee = numpy.zeros(5, dtype=numpy.float64)
    cd = numpy.zeros((mmax, 5), dtype=numpy.float64)
    exe = numpy.zeros(mmax, dtype=numpy.float64)

    # Set up starting values for bottom halfspace
    gbr = evalg(mmax - 1, d, a, b, rho, wvno, omega)
    for i in range(5):
        cd[-1, i] = gbr[i]

    # Matrix multiplication from bottom layer upward
    wvno2 = wvno * wvno

    exsum = 0.0
    for m in range(mmax - 2, -1, -1):
        xka = omega / a[m]
        xkb = omega / b[m] if b[m] > 0.01 else 0.0
        rp = numpy.sqrt(wvno2 - xka * xka + 0.0 * 1j)
        rsv = numpy.sqrt(wvno2 - xkb * xkb + 0.0 * 1j)
        p = rp * d[m]
        q = rsv * d[m]

        iwat = int(b[m] < 0.01)
        cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex = varsv(
            p, q, rp, rsv, d[m], iwat
        )
        ca = dnka(
            omega,
            wvno,
            b[m],
            rho[m],
            cosp,
            rsinp,
            sinpr,
            cossv,
            rsinsv,
            sinsvr,
            pex,
            pex + svex,
            iwat,
        )

        for i in range(5):
            cr = 0.0
            for j in range(5):
                cr += cd[m + 1, j] * ca[j, i]
            ee[i] = cr

        ee, exn = normc(ee, 5)
        exsum += pex + svex + exn
        exe[m] = exsum
        for i in range(5):
            cd[m, i] = ee[i]

    return cd, exe


@jitted
def svdown(omega, wvno, d, a, b, rho):
    """
    Find the values of the Haskell vectors for Rayleigh-wave.

    Values are calculated at each layer boundaries from top layer downward.

    """
    mmax = len(d)

    aa0 = numpy.zeros(4, dtype=numpy.float64)
    vv = numpy.zeros((mmax, 4), dtype=numpy.float64)
    exa = numpy.zeros(mmax, dtype=numpy.float64)
    wvno2 = wvno * wvno

    # Initialize the top surface for the first column of the Haskell propagator
    vv[0, 0] = 1.0
    exa[0] = 0.0

    # Multiplication from top layer downward
    exsum = 0.0
    for m in range(mmax - 1):
        xka = omega / a[m]
        xkb = omega / b[m] if b[m] > 0.01 else 0.0
        rp = numpy.sqrt(wvno2 - xka * xka + 0.0 * 1j)
        rsv = numpy.sqrt(wvno2 - xkb * xkb + 0.0 * 1j)
        p = rp * d[m]
        q = rsv * d[m]

        iwat = int(b[m] < 0.01)
        cosp, cossv, rsinp, rsinsv, sinpr, sinsvr, pex, svex = varsv(
            p, q, rp, rsv, d[m], iwat
        )
        aa = hska(
            omega,
            wvno,
            b[m],
            rho[m],
            cosp,
            rsinp,
            sinpr,
            cossv,
            rsinsv,
            sinsvr,
            pex,
            svex,
            iwat,
        )

        for i in range(4):
            cc = 0.0
            for j in range(4):
                cc += aa[i, j] * vv[m, j]
            aa0[i] = cc

        aa0, ex2 = normc(aa0, 4)
        exsum += pex + ex2
        exa[m + 1] = exsum

        for i in range(4):
            vv[m + 1, i] = aa0[i]

    return vv, exa


@jitted
def shfunc(omega, wvno, d, a, b, rho):
    """
    Compute eigenfunctions for Love-wave.

    This evaluates the eigenfunctions by calling sub up.

    """
    mmax = len(d)

    uu, tt, exl = shup(omega, wvno, d, a, b, rho)

    ext = 0.0
    umax = uu[0]
    tt[0] = 0.0
    for i in range(1, mmax):
        if b[i] > 0.01:
            ext += exl[i - 1]
            fac = numpy.exp(ext) if ext < 80.0 else 0.0
            uu[i] /= fac
            tt[i] /= fac
        else:
            uu[i] = 0.0
            tt[i] = 0.0

        if numpy.abs(uu[i]) > numpy.abs(umax):
            umax = uu[i]

    if uu[0] != 0.0:
        umax = uu[0]

    if numpy.abs(umax) > 0.0:
        for i in range(mmax):
            if b[i] > 0.0:
                uu[i] /= umax
                tt[i] /= umax

    return uu, tt


@jitted
def svfunc(omega, wvno, d, a, b, rho):
    """
    Compute eigenfunctions for Rayleigh-wave.

    This combines the Haskell vector from sub down and Dunkin vector from sub up.

    """
    mmax = len(d)

    ur = numpy.zeros(mmax, dtype=numpy.float64)
    uz = numpy.zeros(mmax, dtype=numpy.float64)
    tz = numpy.zeros(mmax, dtype=numpy.float64)
    tr = numpy.zeros(mmax, dtype=numpy.float64)

    # Get compound matrix from bottom to top
    cd, exe = svup(omega, wvno, d, a, b, rho)
    vv, exa = svdown(omega, wvno, d, a, b, rho)

    # Get propagator from top to bottom
    f1213 = -cd[0, 1]
    ur[0] = cd[0, 2] / cd[0, 1]
    uz[0] = 1.0

    ext = 0.0
    for i in range(1, mmax):
        ext = exa[i] + exe[i] - exe[0]

        if -80.0 < ext < 80.0:
            cd1 = cd[i, 0]
            cd2 = cd[i, 1]
            cd3 = cd[i, 2]
            cd4 = -cd[i, 2]
            cd5 = cd[i, 3]
            cd6 = cd[i, 4]
            tz1 = -vv[i, 3]
            tz2 = -vv[i, 2]
            tz3 = vv[i, 1]
            tz4 = vv[i, 0]

            ur[i] = tz2 * cd6 - tz3 * cd5 + tz4 * cd4
            uz[i] = -tz1 * cd6 + tz3 * cd3 - tz4 * cd2
            tz[i] = tz1 * cd5 - tz2 * cd3 + tz4 * cd1
            tr[i] = -tz1 * cd4 + tz2 * cd2 - tz3 * cd1

            fac = numpy.exp(ext) / f1213
            ur[i] *= fac
            uz[i] *= fac
            tz[i] *= fac
            tr[i] *= fac
        else:
            ur[i] = 0.0
            uz[i] = 0.0
            tz[i] = 0.0
            tr[i] = 0.0

    # Correction for fluid layers on top if not all fluid
    if numpy.any(b > 0.01):
        jwat = 0
        for i in range(mmax):
            if b[i] < 0.01:
                jwat += 1
            else:
                break

        if jwat > 0.0:
            for i in range(jwat):
                ur[i] = 0.0
                tr[i] = 0.0

    return ur, uz, tz, tr


@jitted
def swegn96(t, d, a, b, rho, mode, ifunc, dc):
    """
    Get eigenfunctions for a given period and mode.

    Parameters
    ----------
    t : scalar
        Period (in s).
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
    ifunc : int, optional, default 2
        Select wave type and algorithm for period equation:
         - 1: Love-wave (Thomson-Haskell method),
         - 2: Rayleigh-wave (Dunkin's matrix),
         - 3: Rayleigh-wave (fast delta matrix).
    dc : scalar, optional, default 0.005
        Phase velocity increment for root finding.

    Returns
    -------
    array_like
        Eigenfunctions.

    """
    mmax = len(d)

    # Compute eigenvalue (phase velocity)
    period = numpy.empty(1, dtype=numpy.float64)
    period[0] = t
    c = surf96(period, d, a, b, rho, mode, 0, ifunc, dc)
    omega = 2.0 * numpy.pi / t
    wvno = omega / c[0]

    # Compute eigenfunctions
    if ifunc == 1:
        uu, tt = shfunc(omega, wvno, d, a, b, rho)

        egn = numpy.empty((mmax, 2), dtype=numpy.float64)
        for i in range(mmax):
            egn[i, 0] = uu[i]
            egn[i, 1] = tt[i]

    else:
        ur, uz, tz, tr = svfunc(omega, wvno, d, a, b, rho)

        egn = numpy.empty((mmax, 4), dtype=numpy.float64)
        for i in range(mmax):
            egn[i, 0] = ur[i]
            egn[i, 1] = uz[i]
            egn[i, 2] = tz[i]
            egn[i, 3] = tr[i]

    return egn
