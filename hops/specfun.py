#!/usr/bin/env python
# -*- coding: utf-8 -*-
# original: functions_ger.py

import numpy as np
import numpy.linalg as la
import numpy.fft as fft

#import scipy as sp
import scipy.linalg as sla

import cmath

import sys



def lorentz(w, P=1./np.pi, G=1., W=0.):
  w = np.expand_dims(w, -1)
  return np.sum(P * G / ((w - W)**2 + G**2), axis=-1)


def gauss(w, P=1./np.sqrt(2.*np.pi), s=1., m=0.):
  # Be careful! P is not allowed to depend on w, because if P is an
  # array, it is summed over!
  w = np.expand_dims(w, -1)
  x = (w - m) / s
  return np.sum(P / s * np.exp(-0.5 * x**2), axis=-1)


def J(w, Pt=None, G=None, W=None, P=None, Er=None, H=None, implementation='PFD'):
  """ Calculates the spectral density J(w) as sum of anti-symmetrized
      Lorentzians following

        J(w | Pt, G, W) = sum_k J_k(w | Pt_k, G_k, W_k)

                        = sum_k L(w | P_k, G_k, W_k) - L(w | P_k, G_k, -W_k)

                        = sum_k Pt_k * w / [(w-W_k)² + G_k²][(w+W_k)² + G_k²]

      with   L(w | P, G, W) = P_k * G_k / [(w-W_k)² + G_k²]

      and    Pt_k = 4 * G_k * W_k * P_k.

      Instead of the weight factors for the anti-symmetrized Lorentzians, 'Pt',
      also the ones for the un-symmetrized Lorentzians, 'P', the reorganization
      energies 'Er' with

                   ∞
        Er_k = 1/π ∫ dw J_k(w) / w
                   ⁰

      or the heights 'H' can be used to define J(w).

      Arguments:
       'w'      --  frequency array of points where J(w) should be evaluated
       'Pt'     --  array of weight factors for each anti-symmetrized
                    Lorentz term k
       'G'      --  array of damping constants for each term
       'W'      --  array of central frequencies for each term
       'P'      --  array of weight factors for the corresponding
                    un-symmetrized Lorentz terms
       'Er'     --  array of reorganization energies for each term
       'H'      --  array of heights for each term

       Return values:
                --  array J(w) at the frequencies given by array w

      Remark:
        The connection between the prefactors
          - of the usual, unsymmetrized Lorentzians, 'P',
          - of the antisymmetrized Lorentzians, 'Pt',
          - and of the pole decomposition, 'p',
        is (for anti-symmetrized Lorentzian SPDs) given by

        Pt = 4 G W P = 4 W p.
  """
  W = np.asarray(W); G = np.asarray(G)
  if P is not None: Pt = 4. * G * W * np.asarray(P)
  if Er is not None: Pt = 4. * G * np.asarray(Er)  * (W**2 + G**2)
  if H is not None:
    W2 = W**2; W4 = W2**2; G2 = G**2; G4 = G2**2
    root = np.sqrt(G4 + G2 * W2 + W4)
    Pt = ((8. * np.asarray(H) * (G4 + W4 - W2 * root + G2 * (4. * W2 + root))) /
          (3. * np.sqrt(3. * (W2 - G2 + 2. * root))))
  if Pt is None or G is None or W is None:
    raise TypeError('J() takes at least 4 arguments (central frequency, damping or weight undefined)')
  w = np.expand_dims(w, -1)
  if implementation == 'normal':
    J = np.sum(Pt * w / (((w - W)**2 + G**2) * ((w + W)**2 + G**2)), axis=-1)
  elif implementation == 'factored':
    J = np.sum(Pt * w / ((w - W - 1j*G) * (w - W + 1j*G) *
                            (w + W - 1j*G) * (w + W + 1j*G)), axis=-1)
  elif implementation == 'PFD':
    poles = np.array([W + 1j*G, W - 1j*G, -W + 1j*G, -W - 1j*G])
    J = np.sum(Pt / (8 * G * W) * w * 1j *
                (-1 / (poles[0] * (w - poles[0])) +
                  1 / (poles[1] * (w - poles[1])) +
                  1 / (poles[2] * (w - poles[2])) +
                 -1 / (poles[3] * (w - poles[3])) ), axis=-1)
  if np.iscomplexobj(w):
    return J
  else:
    return J.real


def spd2bcf(Pt, G, W, T, poles, residues=1.):
  """ THIS IS AN OLDER FUNCTION THAT WORKS ONLY FOR OHMIC SPDs!
      IT IS SUPERSEDED BY spd_to_bcf().

      Calculates the parameters for an exponential decomposition of the bath
      correlation function

                   ∞
        α(t) = 1/π ∫ dw J(w) [coth(w / 2T) cos(w t) - i sin(w t)]
                   ⁰

      from the parameters of an anti-symmetrized Lorentz bath spectral density

        J(w) = sum_k Pt_k * w / [(w-W_k)² + G_k²][(w+W_k)² + G_k²]

      such that

        α(t) = sum_j P_j exp[-i W_j t - G_j t]   (t > 0).

      Arguments:
       'Pt'       --  array of weight factors for anti-symmetrized
                      Lorentz spectral density
       'G'        --  array of damping constants for spectral density
       'W'        --  array of central frequencies for spectral density
       'T'        --  temperature
       'poles'    --  array of complex poles of hyperbolic cotangent expansion
                      with positive imaginary part
       'residues' --  array of residues for the coth decomposition (optional)
                      If no residues are given they are assumed to be unity.
      Return values:
               --  tuple (P_bcf, G_bcf, W_bcf) of arrays of coefficients
                   for bath correlation function as sum of exponentials

      Remark:
        The connection between the prefactors
          - of the usual, unsymmetrized Lorentzians, 'P',
          - of the antisymmetrized Lorentzians, 'Pt',
          - and of the pole decomposition, 'p',
        is (for anti-symmetrized Lorentzian SPDs) given by

        Pt = 4 G W P = 4 W p.
  """
  P_bcf = np.concatenate((
      1. / 8 * Pt / (W * G) * (coth(( W + 1j*G) / 2 * Tinv(T)) - 1),
     -1. / 8 * Pt / (W * G) * (coth((-W + 1j*G) / 2 * Tinv(T)) - 1),
      1j * 2 * T * residues * J(2 * T * poles, Pt, G, W) ))
  G_bcf = np.concatenate(( G, G,  np.imag(2 * T * poles)))
  W_bcf = np.concatenate((-W, W, -np.real(2 * T * poles)))
  return (P_bcf, G_bcf, W_bcf)


def coth(x):
  """ coth function for scalar or array argument """
  if np.isscalar(x):
    if(x == 0):
      val = np.inf
    else:
      val = 1. / cmath.tanh(x)
  else:
    shape = x.shape
    x = np.ravel(x)
    val = np.zeros(len(x), dtype=np.complex128)
    for i in np.arange(len(x)):
      tan_hyp = cmath.tanh(x[i])
      if(tan_hyp == 0):
        val[i] = np.inf
      else:
        val[i] = 1. / tan_hyp
    val = val.reshape(shape)
  if np.iscomplexobj(x):
    return val
  else:
    return val.real


def Tinv(T):
  """ Returns inverse temperature 1/T and np.inf if T=0
      input:  np.array of temperatures T
      output: np.array of inverse temperatures
  """
  return np.where(T == 0, np.inf, 1. / T)


def coth_poles_pade(N, use_scipy=False):
  """ Calculates the simple poles of the [N-1, N] Padé approximant to the
      hyperbolic cotangent function and the corresponding residues.
      The implementation follows
        Hu et al. JCP 134, 244106 (2011).
      Remarks: Pole at x = 0 is NOT included!
        Only poles with non-negative imaginary part are returned. The other
        ones can be obtained by either -poles or conj(poles).
      Arguments:
       'N' -- number of expansion terms (integer)
      Return values:
       '(i·xi, eta)'
           -- tuple containing sorted np.array 'i·xi' of the purely imaginary
              poles and a sorted array 'eta' of the corresponding residues
      The hyperbolic cotangent is then approximated by

        coth(x) ≅ 1/x + sum( eta_j / (x - i xi_j) + eta_j / (x + i xi_j) )
  """
  # set-up the symmetric matrices A (2N,2N) and B (2N-1,2N-1)
  i = 1 + np.arange(2*N - 1)
  d = 1. / np.sqrt((2*i + 1) * (2*i + 3))
  A = np.diag(d, 1) + np.diag(d, -1)
  B = A[1:, 1:]
  # find eigenvalues of matrices A and B
  if not use_scipy:
    AZ = la.eigvalsh(A)
    BZ = la.eigvalsh(B)
  else:
    AZ = sla.eigvalsh(A, overwrite_a=True)  # gives no segfault
    BZ = sla.eigvalsh(B, overwrite_a=True)
  BZ = np.delete(-np.sort(-BZ), N - 1)  # kick out the root at zero
  xi = np.sort(1. / AZ[AZ > 0.]); xi2 = np.square(xi)
  zeta = 1. / BZ[BZ > 0.]; zeta2 = np.square(zeta)
  nx = np.newaxis
  eta = (np.hstack([zeta2[nx, :] - xi2[:, nx], np.ones(N)[:, nx]]) /
         ((xi2[nx, :] - xi2[:, nx]) *
          (1. - np.identity(N)) + np.identity(N))).prod(-1)
  # The 'hstack' with 'ones' in the numerator of the product compensates for
  # smaller size of 'zeta' (N-1) compared to 'xi' (N). The 'identity'
  # construction in the denominator sets the main diagonal to unity, because
  # the product xi2_k - xi2_j is supposed to run over all k that are not equal
  # to j.
  eta *= N * (N + 1.5)
  return 1j * xi, eta  # return poles and residues


def coth_approx(x, poles, residues=1.):
  """ Calculates an approximation for the hyperbolic cotangent in terms of a
      rational function using simple poles and their residues. Then the coth
      function is approximated by

        coth(x) = sum (residues / (x - poles_all))
                = 1 / x + sum (2x * residues / (x² - poles²))

      where in the last line only the poles with non-negative imaginary part
      are used and the simple pole at x=0 is already written apart.
  """
  if np.isscalar(poles): poles, residues = coth_poles_pade(poles)
  sq = np.square
  if np.isscalar(x):
    coth_x = 1. / x + 2. * np.sum(residues * x / (sq(x) - sq(poles)))
  else:
    x_ = np.expand_dims(x, -1)
    coth_x = 1. / x + 2. * np.sum(residues * x_ / (sq(x_) - sq(poles)), axis=-1)
  if np.iscomplexobj(x):
    return coth_x
  else:
    return coth_x.real



################################################################################

# several routines for calculation and analysis of spectra


def window(C_t, type=None, width=None, mirror=False):
  """ Filters (half-sided) time series C(t) through one of two different
      window functions.
      The implemented functions are cos(π/2 t/T) and exp[-1/2 (t/λT)²].
      Appropriate window functions must have an area of unity in frequency
      domain, which corresponds to a value of one at time zero in time domain.
      for Gaussian: width λ = σ / T
  """
  len = np.alen(C_t)
  if width == np.inf: type = None
  if type == 'cos':
    if width == None: width = 1.0
    cut = np.floor(len * width)
    C = C_t * np.append(np.cos(0.5 * np.pi * np.linspace(0., 1., cut)),
                        np.zeros(len - cut))
  elif type == 'exp':
    if width is None: width = 0.3
    C = C_t * np.exp(-0.5 * (np.linspace(0., 1., len) / width)**2)
  elif type is None or type == 'None':
    C = C_t
  else:
    sys.exit("\nError: Window type '{0}' unknown. Exit Program!".format(type))
  if mirror:  # explicit mirroring, e.g. [1., 0.5, 0.] -> [0., 0.5, 1., 0.5]
    return np.append(C[::-1].conj(), C[1:-1])  # C(-t) = C*(t)
  else:
    return C


################################################################################

# Routines related to Fourier transformation


def anti_symmetrize(x):
  """ Calculates a two-sided array for a one-sided input array such that the
      output is anti-symmetric.
      When the input array 'x' has the structure [0, dx .. X], the output
      will have [-X .. -dx, 0 .. X-dx] and is calculated using the formula

        np.concatenate((-x[1:][::-1], x[:-1]))

      This function is used, e.g., to construct a two-sided time or frequency
      array from a one-sided one.
      It matches the usual input/output array conventions
      for Fourier transformations.

      See also: fg.take_right_half, fg.fourier_partner, fg.fourier
  """
  return np.concatenate((-x[1:][::-1], x[:-1]))


def take_right_half(x2):
  """ Calculates the one-sided (usually positive) array for a two-sided
      anti-symmetric input array.
      When the input array 'x2' has the structure [-X .. -dx, 0 .. X-dx],
      the output will have [0, dx .. X].

      This function is used, e.g., to construct a one-sided time or frequency
      array from a two-sided one.
      It matches the usual input/output array conventions
      for Fourier transformations.

      See also: fg.anti_symmetrize, fg.fourier_partner, fg.fourier
  """
  N = np.alen(x2) // 2 + 1
  return np.concatenate((x2[N - 1:], [-x2[0]]))


def fourier_partner(t, hermitian=True):
  """ Calculates frequency array from (Fourier-) corresponding time array
      and vice versa.

      The defining relations are:  dw = 2π / T   and   dt = 2π / W

      In the following we are using the nomenclature for the transformation
      from 't' to 'w'. (The transformation is perfectly symmetric.)

      The flag 'hermitian' controls whether or not the Fourier values should
      be considered hermitian in time or not resulting in either a frequency
      array of the same length as the time array or of doubled length (2N-2).
      input:  (one-sided) time array 't'  [0 .. T]
              boolean value 'hermitian' (optional, defaults to 'True')
      output: (two-sided) frequency array 'w'  [-W .. -dw, 0 .. W-dw]

      See also: fg.anti_symmetrize, fg.fourier
  """
  N = np.alen(t)
  if hermitian: N = 2 * (N - 1)  # output of 'hfft' has length 2(N-1), not N
  dt = t[1] - t[0]
  dw = 2 * np.pi / (N * dt)
  return dw * (np.arange(N, dtype=float) - N // 2)
  #return 2. * np.pi * fftshift(fftfreq(N, dt))  # this does the same


# alias functions that are sometimes more descriptive
t_to_t2 = anti_symmetrize
w_to_w2 = anti_symmetrize
t_to_w = fourier_partner
w_to_t = fourier_partner
t2_to_t = take_right_half
w2_to_w = take_right_half

# alias functions for compatibility with older scripts
t2w = t_to_w
t2t2 = t_to_t2
w2t = w_to_t
w2w2 = w_to_w2


def fourier(f_t, dt, n=None, hermitian=True, output_w=False, sigma_w=0.):
  """ Calculates the correctly normalized Fourier transform of (complex-valued)
      input array 'f_t'.
      The Fourier transformed function is ordered with increasing 'w',
      i.e. [-W .. -dw, 0 .. W-dw], and not in 'standard order', which would be
      [0 .. W-dw, -W .. -dw]. This frequency array can be obtained using
      the function 't_to_w'.
      The Fourier convention of this routine is such that it resembles the
      upper one of these integrals:

                   ∞                          ∞
      F{f(t)}(w) = ∫ dt f(t) exp(+iwt) = 2 Re ∫ dt f(t) exp(+iwt)
                  -∞                          ⁰
                          ∞                            ∞
      F⁻¹{f(w)}(t) = 1/2π ∫ dw f(w) exp(-iwt) = 1/π Re ∫ dw f(w) exp(-iwt)
                         -∞                            ⁰

      The rightmost parts are valid, if f(t) has Hermitian symmetry in time,
      i.e. f(-t) = f(t)*.

      f_t:        Numpy array of an equally sampled function in time domain.
      dt:         Size of one time step 'f_t' was discretized with (float).
      n:          Integer number telling the total desired length of the
                  (two-sided) output array. Only used, if n > len(f_t).
      hermitian:  Boolean flag controlling whether 'f_t' is considered to have
                  Hermitian symmetry in the time domain, i.e. f(-t) = f(t)*.
                  If 'hermitian' is 'True' (default), only the values of 'f_t'
                  for non-negative times (t ≥ 0) are allowed to be handed over
                  to this routine.
                  If 'hermitian' is 'False', then time t = 0 is considered to
                  be at index len(t) / 2 of 'f_t', i.e. the corresponding time
                  array has to have the structure [-T .. -dt, 0 .. T-dt].
      output_w:   Boolean flag controlling whether in addition to the Fourier
                  transform 'Ff_w' of 'f_t' the corresponding angular
                  frequency array 'w' shall be returned. Defaults to 'False'.
                  If 'output_w' is 'True' the function returns a list (w, Ff_w)
                  instead of just 'Ff_w' if 'output_w' is False.
      sigma_w:    Desired convolution width in frequency domain (float). If
                  'sigma_w' is a finite number, then the time domain
                  function 'f_t' is multiplied with a Gaussian of width
                  1 / sigma_w resulting in a faster damping in time domain and
                  a convolution with this Gaussian in frequency domain.
                  Thus a controlled broadening in frequency domain with width
                  'sigma_w' can be achieved.

      Returns the Fourier transform Ff_w of f_t if output_w is False and
      returns the tuple (w, Ff_w) if output_w is True.
  """
  f_t_new = np.copy(f_t)  # original f_t should not be modified
  if hermitian:
    len_t = len(f_t)  # length of the one-sided array
    len_t2 = np.max([2 * (len_t - 1), n])  # length of the two-sided arrays
      # if n > len_t2, f_t is zero-padded
    if sigma_w > 0.:
      sigma_t = 1. / sigma_w
      f_t_new *= gauss(dt * np.arange(len_t), P=sigma_t, s=sigma_t, m=0.)
    Ff_w = dt * len_t2 * fft.fftshift(fft.irfft(f_t_new, len_t2))
  else:
    len_t2 = np.max([len(f_t), n])
    shift = len(f_t) // 2
    if sigma_w > 0.:
      sigma_t = 1. / sigma_w
      f_t_new *= gauss(t2t2(dt * np.arange(len_t2 // 2 + 1)),
                       P=sigma_t, s=sigma_t, m=0.)
    Ff_w = dt * len_t2 * fft.fftshift(
        fft.ifft(f_t_new, len_t2) *
        np.exp(-2j * np.pi / len_t2 * shift * np.arange(len_t2)) )
  # The inverse Fourier transform in numpy.fft is normalized by the array
  # length of the two-sided arrays. This has to be compensated here by the
  # factor 'len_t2'. The factor 'dt' then gives the right normalization for
  # the abovementioned integral.
  # 'irfft' goes with +iwt in the exponent and assumes Hermitian symmetry of
  # the input argument.
  # It is necessary to use 'fftshift' and not 'ifftshift on the output of
  # 'irfft', if n is given explicitly as an odd number.

  #Ff_w = dt * len_t2 * fftshift(ifft(fftshift(f_t2)))
    ## version for double-sided time arrays

  #Ff_w = dt * fftshift(hfft(f_t))  # WRONG !!!
    ## inverse order and thus has to be plotted against -w

  #Ff_w = dt * fftshift(hfft(f_t))[::-1]  # WRONG !!!
    ## shifted by one index if plotted against w, because w is even-length array

  if output_w:
    #return (t2w(dt * np.arange(len_t)), Ff_w)
    return (2. * np.pi * fft.fftshift(fft.fftfreq(len_t2, dt)), Ff_w)
  else:
    return Ff_w


def ifourier(f_w, dw, n=None, hermitian=True, output_t = False):
  if hermitian:
    len_w = len(f_w)  # length of the one-sided array
    len_w2 = np.max([2 * (len_w - 1), n])  # length of the two-sided arrays
      # if n > len_w2, f_w is zero-padded
    iFf_t = dw / (2. * np.pi) * fft.fftshift(fft.hfft(f_w, len_w2))
  else:
    len_w2 = np.max([len(f_w), n])
    shift = -len(f_w) / 2
    iFf_t = dw / (2. * np.pi) * fft.fftshift(
        fft.fft(f_w, len_w2) *
        np.exp(-2j * np.pi / len_w2 * shift * np.arange(len_w2)) )

  if output_t:
    #return (t2w(dt * np.arange(len_t)), Ff_w)
    return (2. * np.pi * fft.fftshift(fft.fftfreq(len_w2, dw)), iFf_t)
  else:
    return iFf_t


################################################################################
