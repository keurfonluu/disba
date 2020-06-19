import numpy

from ._common import DispersionCurve, jitted

__all__ = [
    "ThomsonHaskell",
]


twopi = 2.0 * numpy.pi


class ThomsonHaskell:
    def __init__(self, velocity_model):
        """
        Thomson-Haskell propagator.

        Parameters
        ----------
        velocity_model : array_like (num_layers, 4)
            Velocity model where each row corresponds to a layer:
             - P-wave velocity (km/s)
             - S-wave velocity (km/s)
             - Density (g/cm3)
             - Thickness (km)

        """
        self._velocity_model = numpy.asarray(velocity_model, dtype="float64")

    def __call__(self, t, mode=0, velocity_type="phase", dc=0.005, dt=0.005):
        """
        Calculate phase or group velocities for input period axis.

        Parameters
        ----------
        t : array_like
            Periods (s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        velocity_type : str {'phase', 'group'}, optional, default 'phase'
            Velocity type.
        dc : scalar, optional, default 0.005
            Phase velocity increment for searching root.
        dt : scalar, optional, default 0.005
            Frequency increment (%) for calculating group velocity.

        Returns
        -------
        namedtuple
            Dispersion curve as a namedtuple (period, velocity, mode, type).

        Note
        ----
        This function does not perform any check to reduce overhead in case this function is called multiple times (e.g. inversion).

        """
        if velocity_type == "phase":
            t1 = numpy.asarray(t, dtype="float64")
        elif velocity_type == "group":
            t1 = t / (1.0 + dt)
            t1[:] = numpy.asarray(t1, dtype="float64")

        alpha, beta, rho, d = self._velocity_model.T
        mode = numpy.int(mode)
        dc = numpy.float(dc)
        c = get_dispersion_curve(t1, alpha, beta, rho, d, mode + 1, dc)

        idx = c > 0.0
        t = t[idx]
        c = c[idx]

        if velocity_type == "group":
            t1 = t1[idx]
            t2 = t / (1.0 - dt)
            t2[:] = numpy.asarray(t2, dtype="float64")
            c2 = get_dispersion_curve(t2, alpha, beta, rho, d, mode + 1, dc)
            t1[:] = 1.0 / t1
            t2[:] = 1.0 / t2
            c = (t1 - t2) / (t1 / c - t2 / c2)

        return DispersionCurve(t, c, mode, velocity_type)


@jitted("f8(f8, f8, f8[:], f8[:], f8[:], f8[:])")
def fast_delta_matrix(w, k, alpha, beta, rho, d):
    """After Buchen and Ben-Hador (1996)."""
    # Convert to meters
    k = k / 1.0e3
    alpha = alpha * 1.0e3
    beta = beta * 1.0e3
    rho = rho * 1.0e3
    d = d * 1.0e3

    # Initialize arrays
    nl = len(alpha)

    mu = numpy.zeros(nl)
    gam = numpy.zeros(nl)
    t = numpy.zeros(nl)

    r = numpy.zeros(nl, dtype=numpy.complex_)
    s = numpy.zeros(nl, dtype=numpy.complex_)
    Ca = numpy.ones(nl, dtype=numpy.complex_)
    Cb = numpy.ones(nl, dtype=numpy.complex_)
    Sa = numpy.zeros(nl, dtype=numpy.complex_)
    Sb = numpy.zeros(nl, dtype=numpy.complex_)

    eps = numpy.zeros(nl - 1, dtype=numpy.complex_)
    eta = numpy.zeros(nl - 1, dtype=numpy.complex_)
    a = numpy.zeros(nl - 1, dtype=numpy.complex_)
    ap = numpy.zeros(nl - 1, dtype=numpy.complex_)
    b = numpy.zeros(nl - 1, dtype=numpy.complex_)
    bp = numpy.zeros(nl - 1, dtype=numpy.complex_)

    X = numpy.zeros(5, dtype=numpy.complex_)

    # Phase velocity
    c = w / k
    c2 = c * c

    # Other variables
    for i in range(nl):
        mu[i] = rho[i] * beta[i] ** 2
        gam[i] = beta[i] ** 2 / c2
        t[i] = 2.0 - c2 / beta[i] ** 2

    for i in range(nl - 1):
        eps[i] = rho[i + 1] / rho[i]
        eta[i] = 2.0 * (gam[i] - eps[i] * gam[i + 1])
        a[i] = eps[i] + eta[i]
        ap[i] = a[i] - 1.0
        b[i] = 1.0 - eta[i]
        bp[i] = b[i] - 1.0

    # Layer eigenfunctions
    for i in range(nl):
        if c < alpha[i]:
            r[i] = numpy.sqrt(1.0 - c2 / alpha[i] ** 2)
            Ca[i] = numpy.cosh(k * r[i] * d[i])
            Sa[i] = numpy.sinh(k * r[i] * d[i])
        elif c > alpha[i]:
            r[i] = numpy.sqrt(c2 / alpha[i] ** 2 - 1.0) * 1j
            Ca[i] = numpy.cos(k * r[i].imag * d[i])
            Sa[i] = numpy.sin(k * r[i].imag * d[i]) * 1j

        if c < beta[i]:
            s[i] = numpy.sqrt(1.0 - c2 / beta[i] ** 2)
            Cb[i] = numpy.cosh(k * s[i] * d[i])
            Sb[i] = numpy.sinh(k * s[i] * d[i])
        elif c > beta[i]:
            s[i] = numpy.sqrt(c2 / beta[i] ** 2 - 1.0) * 1j
            Cb[i] = numpy.cos(k * s[i].imag * d[i])
            Sb[i] = numpy.sin(k * s[i].imag * d[i]) * 1j

    # Rayleigh-wave fast Delta matrix
    X[0] = 2.0 * t[0]
    X[1] = -t[0] * t[0]
    X[4] = -4.0
    X *= mu[0] * mu[0]

    for i in range(nl - 1):
        p1 = Cb[i] * X[1] + s[i] * Sb[i] * X[2]
        p2 = Cb[i] * X[3] + s[i] * Sb[i] * X[4]
        if c == beta[i]:
            p3 = k * d[i] * X[1] + Cb[i] * X[2]
            p4 = k * d[i] * X[3] + Cb[i] * X[4]
        else:
            p3 = Sb[i] * X[1] / s[i] + Cb[i] * X[2]
            p4 = Sb[i] * X[3] / s[i] + Cb[i] * X[4]

        q1 = Ca[i] * p1 - r[i] * Sa[i] * p2
        q3 = Ca[i] * p3 - r[i] * Sa[i] * p4
        if c == alpha[i]:
            q2 = -k * d[i] * p3 + Ca[i] * p4
            q4 = -k * d[i] * p1 + Ca[i] * p2
        else:
            q2 = -Sa[i] * p3 / r[i] + Ca[i] * p4
            q4 = -Sa[i] * p1 / r[i] + Ca[i] * p2

        y1 = ap[i] * X[0] + a[i] * q1
        y2 = a[i] * X[0] + ap[i] * q2
        z1 = b[i] * X[0] + bp[i] * q1
        z2 = bp[i] * X[0] + b[i] * q2

        X[0] = bp[i] * y1 + b[i] * y2
        X[1] = a[i] * y1 + ap[i] * y2
        X[2] = eps[i] * q3
        X[3] = eps[i] * q4
        X[4] = bp[i] * z1 + b[i] * z2

    return numpy.real(X[1] + s[-1] * X[3] - r[-1] * (X[3] + s[-1] * X[4]))


@jitted("f8(f8, f8, f8, f8, f8, f8[:], f8[:], f8[:], f8[:], i4)")
def nevill(t, c1, c2, del1, del2, a, b, rho, d, ifunc):
    """
    Hybrid method for refining root once it has been bracketted between c1 and c2.

    Note
    ----
    Adapted from Fortran program SRFDIS (COMPUTER PROGRAMS IN SEISMOLOGY).

    """
    x = numpy.zeros(20, dtype=numpy.float64)
    y = numpy.zeros(20, dtype=numpy.float64)

    # Initial guess
    omega = twopi / t
    c3 = 0.5 * (c1 + c2)
    del3 = fast_delta_matrix(omega, omega / c3, a, b, rho, d)
    nev = 1
    nctrl = 1

    while True:
        nctrl += 1
        if nctrl >= 100:
            break

        # Make sure new estimate is inside the previous values.
        # If not, perform interval halving
        if c3 < min(c1, c2) or c3 > max(c1, c2):
            nev = 0
            c3 = 0.5 * (c1 + c2)
            del3 = fast_delta_matrix(omega, omega / c3, a, b, rho, d)

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
        # Do not use neville iteration
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
            del3 = fast_delta_matrix(omega, omega / c3, a, b, rho, d)
            nev = 1
            m = 1
        else:
            if nev == 2:
                x[m-1] = c3
                y[m-1] = del3
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
                    x[j-1] = (-y[j-1] * x[j] + y[m] * x[j-1]) / denom

            if flag:
                c3 = x[0]
                del3 = fast_delta_matrix(omega, omega / c3, a, b, rho, d)
                nev = 2
                m += 1
                m = min(m, 10)
            else:
                c3 = 0.5 * (c1 + c2)
                del3 = fast_delta_matrix(omega, omega / c3, a, b, rho, d)
                nev = 1
                m = 1
    
    return c3


@jitted(
    "Tuple((f8, f8, i4))(f8, f8, f8, f8, f8, f8, i4, f8, f8[:], f8[:], f8[:], f8[:], i4)"
)
def getsol(t1, c1, clow, dc, cm, betmx, ifirst, del1st, a, b, rho, d, ifunc):
    """
    Function to bracket dispersion curve and then refine it.

    Note
    ----
    Adapted from Fortran program SRFDIS (COMPUTER PROGRAMS IN SEISMOLOGY).

    """
    # Bracket solution
    omega = twopi / t1
    wvno = omega / c1
    del1 = fast_delta_matrix(omega, wvno, a, b, rho, d)
    del1st = del1 if ifirst == 1 else del1st
    idir = -1.0 if ifirst != 1 and numpy.sign(del1st) * numpy.sign(del1) < 0.0 else 1.0

    # idir indicates the direction of the search for the true phase velocity from the initial estimate.
    while True:
        c2 = c1 + idir * dc

        if c2 <= clow:
            idir = 1.0
            c1 = clow
        else:
            omega = twopi / t1
            wvno = omega / c2
            del2 = fast_delta_matrix(omega, wvno, a, b, rho, d)

            if numpy.sign(del1) != numpy.sign(del2):
                c1 = nevill(t1, c1, c2, del1, del2, a, b, rho, d, ifunc)
                iret = -1 if c1 > betmx else 1
                break

            c1 = c2
            del1 = del2

            if betmx + dc <= c1 < cm:
                iret = -1
                break

    return c1, del1st, iret


@jitted("f8(f8, f8)")
def getsolh(a, b):
    """
    Starting solution.

    Note
    ----
    Adapted from Fortran program SRFDIS (COMPUTER PROGRAMS IN SEISMOLOGY).

    """
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


@jitted("f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], i4, f8)")
def get_dispersion_curve(t, a, b, rho, d, mode, dc):
    """
    Get phase velocity dispersion curve.

    Note
    ----
    Adapted from Fortran program SRFDIS (COMPUTER PROGRAMS IN SEISMOLOGY).

    """
    # Initialize arrays
    kmax = len(t)
    c = numpy.zeros(kmax, dtype=numpy.float64)
    cg = numpy.zeros(kmax, dtype=numpy.float64)

    # Find the extremal velocities to assist in starting search
    betmx = -1.0e20
    betmn = 1.0e20
    for i in range(len(b)):
        if b[i] > 0.01 and b[i] < betmn:
            betmn = b[i]
            jmn = i
        elif b[i] < 0.01 and a[i] > betmn:
            betmn = a[i]
            jmn = i
        betmx = max(betmx, b[i])

    # Solid layer solve halfspace period equation
    cc = getsolh(a[jmn], b[jmn])

    # Back off a bit to get a starting value at a lower phase velocity
    cc *= 0.9
    c1 = cc
    cm = cc

    one = 1.0e-2
    onea = 1.5
    ifunc = 2
    for iq in range(mode):
        ibeg = 0
        iend = kmax

        del1st = 0.0
        for k in range(ibeg, iend):
            # Get initial phase velocity estimate to begin search
            ifirst = int(k == ibeg)
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
                t[k], c1, clow, dc, cm, betmx, ifirst, del1st, a, b, rho, d, ifunc
            )

            if iret == -1 and iq > 0:
                cg[k:] = 0.0
                if iq == mode - 1:
                    return cg
                else:
                    c1 = 0.0
                    break

            c[k] = c1
            cg[k] = c[k]
            c1 = 0.0

    return cg
