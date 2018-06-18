import numpy as np
from astropy.stats import LombScargle
from tqdm import tqdm

def _2d(arr):
    """ Reshape 1d array to 2d array """
    return np.reshape(arr, (len(arr), 1))

def design_matrix_unmod(t, freq, nharms):
    """
    Design matrix for unmodulated signal

    Parameters
    ----------
    t: array_like
        Time coordinates
    freq: float
        Frequency of signal
    nharms: int
        Number of harmonics to use to model unmodulated signal

    Returns
    -------
    X: np.ndarray
        Design matrix, shape = ``(N, 2 * nharms + 1)``, where
        ``N`` is the number of observations, and ``nharms`` is the
        number of harmonics
    """

    omega = 2 * np.pi * freq
    phi = omega * t

    m_0 = np.ones_like(t)

    # mean magnitude
    cols = [m_0]

    # (unmodulated) signal
    for n in range(1, nharms + 1):
        for func in [np.cos, np.sin]:
            cols.append(func(n * phi))

    cols = list(map(_2d, cols))

    return np.hstack(cols)

def design_matrix(t, freq, blazhko_freq, H, M):
    """
    Design matrix for Blazhko model

    Parameters
    ----------
    t: array_like
        Time coordinates
    freq: float
        Frequency of unmodulated signal
    blazhko_freq: float
        Modulation frequency
    H: int
        Number of harmonics to use when modeling
        unmodulated signal
    M: array_like, length = H + 1
        List of integers denoting the number of
        harmonics to use when modeling the modulation
        of the n-th harmonic of the unmodulated signal.

        - M[0] = the number of harmonics to use to model
          the modulation of the mean magnitude
        - M[1] = the number of harmonics to use to model
          the modulation of the fundemental mode
        - M[2] = the number of harmonics to use to model
          the modulation of the first harmonic

        ...

    Returns
    -------
    X: array_like
        Design matrix for the Blazhko model
    """
    omega = 2 * np.pi * freq
    omega_b = 2 * np.pi * blazhko_freq

    phi = omega * t
    phi_b = omega_b * t

    X_0 = design_matrix_unmod(t, freq, H)

    cols = []
    # mean modulation
    for m in range(1, M[0] + 1):
        for func_b in [np.cos, np.sin]:
            cols.append(func_b(m * phi_b))

    # modulation of harmonics
    for n in range(1, H + 1):
        for m in range(1, M[n] + 1):
            for func in [np.cos, np.sin]:
                for func_b in [np.cos, np.sin]:
                    cols.append(func_b(m * phi_b) * func(n * phi))

    cols = list(map(_2d, cols))

    X_m = np.hstack(cols)

    return np.hstack((X_0, X_m))

def regularization(nharms, mod_nharms, prior0, prior_mod):
    """
    Regularization term for imposing Gaussian priors on the
    signal and modulation amplitudes

    Parameters
    ----------
    nharms: int
        The number of harmonics to use to model the unmodulated
        signal.
    mod_nharms: array_like, length = nharms + 1
        The number of harmonics to use to model each of the
        modulation frequencies
    prior0: float
        1/var, where var is the variance of the Gaussian prior
        on the signal amplitudes
    prior_mod: float or array_like
        1/var, where var is the variance of the Gaussian prior
        on the modulation amplitudes. If array_like, must have
        length ``nharms + 1``, with each ``prior_mod[n]`` corresponding
        to ``1/var[n]`` where ``var[n]`` is the variance of
        the Gaussian prior on all modulation amplitudes for
        the ``n``-th harmonic.

    Returns
    -------
    V: array_like
        The regularization matrix (diagonal), with shape ``(m, m)``,
        where ``m = (2H + 1 + sum(mod_nharms))``
    """
    # offset (1)
    regs = [0]

    # unmod lightcurve (2H)
    regs += [prior0] * 2 * nharms

    if isinstance(prior_mod, int):
        regs += [prior_mod] * 2 * mod_nharms[0]
        for n in range(1, nharms + 1):
            regs += [prior_mod] * 4 * mod_nharms[n]

        return np.diag(regs)

    regs += [prior_mod[0]] * 2 * mod_nharms[0]
    for n in range(1, nharms + 1):
        regs += [prior_mod[n]] * 4 * mod_nharms[n]

    return np.diag(regs)

def _parameter_names(nharms, mod_nharms):
    """
    String labels for each of the free parameters

    Parameters
    ----------
    nharms: int
        Number of harmonics used to model the primary signal
    mod_nharms: array_like
        Number of harmonics used to model the n-th harmonic
        of the primary signal, length = ``2H + 1``

    Returns
    -------
    names: list of str
        A list of parameter names for each of the parameters
        in the design matrix
    """

    names = ['y_0']
    for n in range(1, nharms + 1):
        names += ['c_%d'%(n), 's_%d'%(n)]

    for m in range(1, mod_nharms[0] + 1):
        names += ['a_0%d'%(m), 'b_0%d'%(m)]

    for n in range(1, nharms + 1):
        for m in range(1, mod_nharms[n] + 1):
            names += ['ac_%d%d'%(n, m), 'bc_%d%d'%(n, m),
                      'as_%d%d'%(n, m), 'bs_%d%d'%(n, m)]

    return names

class BlazhkoModel(object):
    """
    Blazho model of a lightcurve.

    Accounts for modulation of mean magnitude and the harmonics
    of the primary signal.

    Parameters
    ----------
    nharms: int
        Number of harmonics to use when modeling the primary signal.
    mod_nharms: int
        Number of harmonics to use when modeling the modulation
        amplitudes of all harmonics except the 0th (mean)
    mean_mod_nharms: int
        Number of harmonics to use when modeling the modulation
        of the mean magnitude
    mod_amp_prior: float
        Standard deviation of the Gaussian prior on the modulation
        amplitudes (except 0th (mean)).
    mean_mod_amp_prior: float
        Standard deviation of the Gaussian prior on the
        modulation amplitudes for the mean magnitude
    unmod_amp_prior: float
        Standard deviation of the Gaussian prior on the unmodulated
        amplitudes.
    """
    def __init__(self,
                 nharms=7,
                 mod_nharms=3,
                 mean_mod_nharms=1,
                 mod_amp_prior=3,
                 mean_mod_amp_prior=1,
                 unmod_amp_prior=None):

        self.nharms = nharms
        self.mod_nharms = mod_nharms

        self.mod_nharms = [mean_mod_nharms]
        if isinstance(mod_nharms, int):
            mod_nharms = [mod_nharms] * self.nharms
        self.mod_nharms += mod_nharms

        assert(len(self.mod_nharms) == nharms + 1)

        self.parameter_names = _parameter_names(self.nharms,
                                                self.mod_nharms)
        self.unmod_parameter_names = _parameter_names(self.nharms,
                                                      [0] * (1 + self.nharms))
        self.params = None

        self.unmod_reg = 0
        if unmod_amp_prior is not None:
            self.unmod_reg = 1./(unmod_amp_prior ** 2)

        self.mod_reg = [1./(mean_mod_amp_prior ** 2)]
        if mod_amp_prior is not None:
            self.mod_reg = [1./(mod_amp_prior ** 2)] * self.nharms

        self.V = regularization(self.nharms,
                                self.mod_nharms,
                                self.unmod_reg,
                                self.mod_reg)

    def unmod_params(self):
        """ Retrieve parameters for the unmodulated signal """

        is_unmod = lambda i : self.parameter_names[i] in self.unmod_parameter_names
        inds = list(filter(is_unmod, range(len(self.parameter_names))))
        return self.params[inds]

    def fit(self, t, y, dy, freq, blazhko_freq,
            dy_as_weights=False, return_score=False,
            only_primary=False):
        """
        Fit Blazhko model to lightcurve

        Parameters
        ----------
        t: array_like
            Time coordinates for each observation
        y: array_like
            Mean of the Gaussian posterior at each time coordinate
        dy: array_like
            Standard deviation of the Gaussian posterior
            on each measurement
        freq: float
            Frequency of primary signal
        blazhko_freq: float
            Modulation frequency
        dy_as_weights: bool
            Whether to treat the ``dy`` values as weights or as
            standard deviations
        return_score: bool
            Return the Blazhko score
        only_primary: bool
            Only fit primary signal

        Returns
        -------
        self
        """

        self.t_ = t
        self.y_ = y
        self.dy_ = dy

        self.freq = freq
        self.blazhko_freq = blazhko_freq
        X = design_matrix(t, freq, blazhko_freq,
                          self.nharms, self.mod_nharms)

        # cov^(-1)
        W = None
        if dy_as_weights:
            W = np.diag(dy)
        else:
            W = np.diag(np.power(dy, -2))

        # WLS: (X.T @ W @ X + V) @ theta = (X.T @ W) @ y
        # where V = \Sigma^(-1) is the *prior* covariance matrix
        # of the model parameters
        z = (X.T @ W) @ y

        S = X.T @ W @ X + self.V

        self.params = np.linalg.solve(S, z)
        return self

    def score(self, t=None, y=None, dy=None,
              only_primary=False, dy_as_weights=False):
        """ Equivalent to Lomb-Scargle power """

        if self.params is None:
            raise Exception("Must fit data first before calling")

        t, y, dy = None, None, None
        if t is None:
            t = self.t_
        if y is None:
            y = self.y_
        if dy is None:
            dy = self.dy_

        W = None
        if dy_as_weights:
            W = np.diag(dy)
        else:
            W = np.diag(np.power(dy, -2))

        w = np.copy(np.diag(W))
        w /= sum(w)

        # model values
        yhat = self.unmod(t) if only_primary else self(t)
        yhat_0 = np.dot(w, y) if only_primary else self.unmod(t)

        # residuals
        r = y - yhat
        r0 = y - yhat_0

        # chi2
        chi2 = r.T @ W @ r
        chi2_0 = r0.T @ W @ r0

        return 1 - chi2 / chi2_0


    def __call__(self, t):
        if self.params is None:
            raise Exception("Must fit data first before calling")

        X = design_matrix(t, self.freq, self.blazhko_freq,
                          self.nharms, self.mod_nharms)

        return X @ self.params

    def unmod(self, t):
        if self.params is None:
            raise Exception("Must fit data first before calling unmod")

        X = design_matrix_unmod(t, self.freq, self.nharms)

        return X @ self.unmod_params()

def _autofrequency(baseline, minimum_frequency, maximum_frequency,
                   samples_per_peak):

    delta_f = maximum_frequency - minimum_frequency

    df = 1. / baseline / samples_per_peak

    n_freqs = np.ceil(delta_f / df).astype(int)

    return minimum_frequency + df * np.arange(n_freqs)


class BlazhkoPeriodogram(object):
    def __init__(self, t, y, dy=None,
                 show_progress=False,
                 freq=None, **kwargs):
        self.model = BlazhkoModel(**kwargs)
        self.t = t
        self.y = y
        self.dy = dy
        self.freq = freq
        self.show_progress = show_progress
        self.progress_bar = lambda x: x
        if self.show_progress:
            self.progress_bar = tqdm

        if self.dy is None:
            self.dy = np.ones_like(t)

        if self.freq is None:
            freqs, powers = LombScargle(t, y, dy).autopower(minimum_frequency=0.5,
                                                            maximum_frequency=10,
                                                            samples_per_peak=10)
            self.freq = freqs[np.argmax(powers)]

    def _power_single_freq(self, freq, only_primary=False):
        return self.model.fit(self.t, self.y, self.dy,
                              self.freq, freq).score(only_primary=only_primary)

    def power(self, freqs, only_primary=False):
        if not isinstance(freqs, (np.ndarray, list, tuple)):
            return self._power_single_freq(freqs)

        powers = np.array(list(map(lambda f: self._power_single_freq(f),
                                   self.progress_bar(freqs))))
        return powers

    def autopower(self, minimum_frequency=None, maximum_frequency=0.25,
                  samples_per_peak=5):
        baseline = max(self.t) - min(self.t)
        if minimum_frequency is None:
            minimum_frequency = max([1./baseline, 0.012])

        freqs = _autofrequency(baseline,
                               minimum_frequency,
                               maximum_frequency,
                               samples_per_peak)

        powers = self.power(freqs)

        return freqs, powers

    def get_model(self, freq):
        return self.model.fit(self.t, self.y, self.dy,
                              self.freq, freq)

    def amplitude(self, t, n):
        pass

    def phase(self, t, n):
        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import argparse
    import os
    import warnings
    warnings.filterwarnings('ignore')

    default_mod_amp_prior = 2
    default_mod_nharms = 2
    default_nharms = 6
    default_amp_prior = None
    default_mean_mod_amp_prior = 1
    default_mag_col = 'RM1'

    parser = argparse.ArgumentParser(description="Model Blazhko effect")
    parser.add_argument('lightcurve', type=str, help="Path to lightcurve file")
    parser.add_argument('--nharms',
                        type=int,
                        help="Number of harmonics (unmodulated)",
                        default=default_nharms,
                        metavar='<int: default = %d>'%(default_nharms))
    parser.add_argument('--mod-nharms', type=int,
                        default=default_mod_nharms,
                        metavar='<int: default = %d>'%(default_mod_nharms),
                        help="Number of harmonics (modulated)")
    parser.add_argument('--amp-prior', type=float,
                        default=default_amp_prior,
                        metavar='<float: default = {val}>'.format(val=default_amp_prior),
                        help="Amplitude prior (unmodulated)")
    parser.add_argument('--mod-amp-prior', type=float,
                        default=default_mod_amp_prior,
                        metavar='<float: default = %.1f>'%(default_mod_amp_prior),
                        help="Amplitude prior (modulated)")
    parser.add_argument('--mean-mod-amp-prior', type=float,
                        default=default_mean_mod_amp_prior,
                        metavar='<float: default = {val}>'.format(val=default_mean_mod_amp_prior),
                        help="Amplitude prior on mean modulation")
    parser.add_argument('--mag-col', type=str, default=default_mag_col,
                        metavar='<{IM,RM,EP,TF}{1,2,3}; default = %s>'%(default_mag_col),
                        help="LC column (lctype)(aperture); for labeling purposes only")
    parser.add_argument('--color-by', nargs='+', choices=['station_id', 'ccd'],
                        default=None,
                        help='Color lightcurve points by station and/or ccd')
    parser.add_argument('--debug', action='store_true',
                        help="Just use random values for Blazhko periodogram and make the plot")
    args = parser.parse_args()
    lc = pd.read_csv(args.lightcurve)
    lc.loc[:, 't'] -= 55000
    name = os.path.basename(args.lightcurve).split('.')[0]

    colors = ['g', 'orange', 'darkviolet', 'crimson', 'darkcyan']

    def find_best_start_time(t, dt):
        t0s = np.linspace(min(t), max(t) - dt)
        return max(t0s, key=lambda t0: sum((t > t0) & (t < (t0 + dt))))


    def stringify(value, grpby):
        if isinstance(value, (np.ndarray, list, tuple)):
            return "(" + ','.join(value) + ")"
        return "{val}".format(val=value)

    def group_lc(lc_df, grpby='station_id'):
        grp = lc_df.groupby(grpby)

        results = []
        for i, (name, group) in enumerate(grp):
            label = stringify(name, grpby)
            lcg = (group.t.values, group.y.values, group.dy.values)
            color = colors[i%len(colors)]

            results.append((lcg, label, color))
        return results

    t, y, dy = lc.t.values, lc.y.values, lc.dy.values

    kwargs = dict(dy=dy,
                  nharms=args.nharms,
                  mod_nharms=args.mod_nharms,
                  mod_amp_prior=args.mod_amp_prior,
                  unmod_amp_prior=args.amp_prior,
                  mean_mod_amp_prior=args.mean_mod_amp_prior,
                  show_progress=True)


    solver = BlazhkoPeriodogram(t, y, **kwargs)
    freqs, powers = None, None
    if args.debug:
        freqs = np.linspace(0.012, 0.5, 100)
        powers = np.random.rand(len(freqs))
    else:
        bpsname = "blazhko_periodogram_%s.csv"%(name)
        if os.path.exists(bpsname):
            df = pd.read_csv(bpsname)
            freqs = df.freqs.values
            powers = df.powers.values
        else:
            freqs, powers = solver.autopower()

            pd.DataFrame({'freqs': freqs,
                          'powers': powers}).to_csv(bpsname,
                                                    header=True,
                                                    index=False)

    best_freq = freqs[np.argmax(powers)]
    model = solver.get_model(best_freq)

    ymean = model.params[0]
    amp = max(model.unmod(np.linspace(0, 1, 100)/best_freq) - ymean)


    blazhko_period = 1./best_freq


    f_full = plt.figure(figsize=(12, 6))

    ax_full = f_full.add_subplot(211)
    axls = f_full.add_subplot(234)
    axb = f_full.add_subplot(235)
    ax_unmod = f_full.add_subplot(236)

    # LOMB-SCARGLE
    freqs_ls, powers_ls = LombScargle(t, y, dy).autopower(minimum_frequency=0.5,
                                                          maximum_frequency=10.,
                                                          samples_per_peak=10)

    axls.plot(1./freqs_ls[::-1], powers_ls[::-1], color='k')
    axls.set_xlabel("Primary period (d)")
    axls.set_ylabel("Lomb-Scargle power")
    axls.axvline(1./model.freq, ls=':', color='r', label="%.3f d"%(1./model.freq))
    axls.legend(loc='upper right')

    # BLAZHKO POWERSPECTRUM
    axb.plot(1./freqs[::-1], powers[::-1], color='k')
    axb.set_xlabel('Blazhko period (d)')
    axb.set_ylabel('Blazhko power')
    axb.axvline(blazhko_period, ls=':', color='r', label="%.3f d"%(blazhko_period))
    axb.legend(loc='upper right')

    # SHOW LIGHTCURVE WITH BLAZHKO MODULATIONS
    baseline = max(t) - min(t)
    dt = blazhko_period #min([ 60 / model.freq, 2 * blazhko_period ])
    t0 = find_best_start_time(t, dt)
    tf = t0 + dt


    time = np.linspace(t0, tf, 10000)

    scatters = []
    sc_labels = []
    if args.color_by is not None:
        for (tg, yg, dyg), labelg, colorg in group_lc(lc, grpby=args.color_by):
            sc = ax_full.scatter(tg, yg, c=colorg, s=1, alpha=0.4)
            scatters.append(sc)
            sc_labels.append(labelg)
    else:
        ax_full.scatter(t, y, s=1, c='k', alpha=0.7)

    lunm, = ax_full.plot(time, model.unmod(time), lw=0.5, color='b', alpha=0.8)
    lmod, = ax_full.plot(time, model(time), lw=0.5, color='r', alpha=0.8)

    from matplotlib.legend import Legend
    if args.color_by is not None:
        ngroups = len(sc_labels)
        titles = {'station_id' : 'HAT Station', 'ccd': "CCD No."}
        ncol = int(np.ceil(ngroups / 2.))
        title = ','.join(map(lambda c: titles[c], args.color_by))

        legend_sc = Legend(ax_full, scatters, sc_labels, title=title,
                           loc='upper right', ncol=ncol)
        for handle in legend_sc.legendHandles:
            handle._sizes = [10]

        ax_full.add_artist(legend_sc)

    ax_full.legend([lunm, lmod], ["Primary", "+ Blazhko"], title="Signal (fit)",
                    loc='upper left', ncol=1)

    ax_full.set_xlim(t0, tf)
    ax_full.set_ylim(ymean - 4 * amp, ymean + 2 * amp)
    ax_full.invert_yaxis()
    ax_full.set_title(name)
    ax_full.set_xlabel('BJD - 55000')
    ax_full.set_ylabel('RM1 mag')

    # UNMODULATED COMPONENT PLOT
    ax_unmod.scatter((t * model.freq) % 1.0, y, s=1, c='k', alpha=0.1)
    phase = np.linspace(0, 1, 100)
    ax_unmod.plot(phase, model.unmod(phase / model.freq), lw=2, color='b', alpha=0.8)
    ax_unmod.set_xlim(0, 1)
    ax_unmod.set_xlabel('$\\phi$')
    ax_unmod.set_ylabel('RM1 mag')
    ax_unmod.set_ylim(ymean - 3 * amp, ymean + 3 * amp)

    ax_unmod.invert_yaxis()
    f_full.tight_layout()
    f_full.savefig('blazhko_%s.png'%(name), dpi=400)

    plt.show()


