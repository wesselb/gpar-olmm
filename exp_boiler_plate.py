from gpar import GPARRegressor
from varz import Vars
import lab as B


def _inv(a):
    return B.cholsolve(B.chol(B.reg(a)), B.eye(a))


def _pinv(a):
    return B.cholsolve(B.chol(B.reg(B.matmul(a, a, tr_a=True))), B.transpose(a))

# TODOs:
# - Check the orthogonal noise computations in `predict` and in `sample`
# - Fix the way the updating of the model is happening (i.e. how `vs` gets substituted).
# - Remove side-effects from `logpdf`.
# - Separate `logpdf` into itself plus `objective`. Have the L2 regulariser on the weights
# show up only in the objective.
# - Add plotting back.
# - Add docstrings.
# - Optimise code.

class GPAROLMM:

    def __init__(
        self,
        h,
        gpar,
        noise,
    ):
        self.h = h
        self.noise = noise
        self.gpar = gpar
        self.vs = gpar.vs
        self.p, self.m = B.shape(h)
        self.vs.get(h, name='h') # attach the mixing matrix to the variable vector
        self.vs.pos(noise, name='noise')

    def logpdf(self, x, y, vs):
        self.vs = vs # Side-effect!
        self.gpar.vs = vs # Side-effect! This is ugly but works

        proj_x, proj_y, proj_w, reg = self._project(x, y)
        logpdf = self.gpar.logpdf(proj_x, proj_y, proj_w)
        print('*** logpdf: ', logpdf)
        print('*** reg: ', reg)
        return logpdf - reg
        # return self.gpar.logpdf(proj_x, proj_y, proj_w) - reg

    def _project(self, x, y):
        n = B.shape(x)[0]
        available = ~B.isnan(B.to_numpy(y))

        # Extract patterns.
        patterns = list(set(map(tuple, list(available))))

        if len(patterns) > 30:
            warnings.warn(f'Detected {len(patterns)} patterns, which is more '
                          f'than 30.',
                          category=UserWarning)

        # Per pattern, find data points that belong to it.
        patterns_inds = [[] for _ in range(len(patterns))]
        for i in range(n):
            patterns_inds[patterns.index(tuple(available[i]))].append(i)

        # Per pattern, perform the projection.
        proj_xs = []
        proj_ys = []
        proj_ws = []
        total_reg = 0

        for pattern, pattern_inds in zip(patterns, patterns_inds):
            proj_x, proj_y, proj_w, reg = \
                self._project_pattern(B.take(x, pattern_inds),
                                      B.take(y, pattern_inds),
                                      pattern)

            proj_xs.append(proj_x)
            proj_ys.append(proj_y)
            proj_ws.append(proj_w)
            total_reg = total_reg + reg

        return B.concat(*proj_xs, axis=0), \
               B.concat(*proj_ys, axis=0), \
               B.concat(*proj_ws, axis=0), \
               total_reg

    def _project_pattern(self, x, y, pattern):
        # Filter by the given pattern.
        y = B.take(y, pattern, axis=1)

        # Get number of data points and outputs in this part of the data.
        n = B.shape(x)[0]
        p = sum(pattern)

        # Build mixing matrix and projection.
        h = self.vs['h']
        hm = B.take(h, pattern)
        u = B.svd(h)[0]
        um = B.take(u, pattern)
        proj = B.matmul(_pinv(h), B.matmul(u, _pinv(um)))
        proj_orth = B.matmul(hm, _pinv(hm))

        # Perform projection.
        proj_y = B.matmul(y, proj, tr_b=True)
        proj_y_orth = y - B.matmul(y, proj_orth, tr_b=True)

        # Compute projected noise.
        h_sq = B.matmul(h, h, tr_a=True)
        hm_sq = B.matmul(hm, hm, tr_a=True)
        noise = self.vs['noise']
        proj_noise = noise * (_inv(hm_sq) - _inv(h_sq))

        # Convert projected noise to weights.
        # This requires the self.gpar to have been initialised! That means we need to call
        # self.gpar.sample or something similar. Ugly, again.
        m = self.m
        lat_noises = B.concat(*[self.vs[f'{i}/noise'][None] for i in range(m)])
        weights = lat_noises / (lat_noises + B.diag(proj_noise))
        proj_w = B.ones(self.vs.dtype, n, m) * weights[None, :]

        # Compute regularising term.
        reg = 0.5 * (n * (p - m) * B.log(2 * B.pi * noise) + # lognoise
                    B.sum(proj_y_orth ** 2) / noise + # logfrob
                    n * B.logdet(B.reg(B.matmul(h, h, tr_a=True))) + # logdet1
                    n * B.logdet(B.reg(B.matmul(um, um, tr_a=True))) + # logdet2
                    1e-4 * B.sum((1 / weights) ** 2)) # This is a regularisation term,
                                                      # rigorously, does not belong in here

        return x, proj_y, proj_w, reg

    def condition(self, x, y):
        proj_x, proj_y, proj_w, _ = self._project(x, y)
        self.gpar.condition(proj_x, proj_y, proj_w)


    def predict(self, x, latent=False, num_samples=100):
        means, lowers, uppers = self.gpar.predict(
            x, latent=latent, credible_bounds=true, num_samples=num_samples
        )

        # Pull means and variances through mixing matrix.
        means = B.matmul(means, self.h, tr_b=True)
        lowers = B.matmul(lowers, self.h, tr_b=True)
        uppers = B.matmul(uppers, self.h, tr_b=True)

        if not latent:
            # compute noise on orthogonal complement in a very dumb way.
            # Gotta double-check.
            proj = B.matmul(hm, _pinv(hm)).T
            proj_orth = B.eye(B.shape(proj)[0]) - proj
            orth_noise = self.noise * proj_orth
            lowers = lowers - B.diag(orth_noise)
            uppers = uppers + B.diag(orth_noise)

        return means, lowers, uppers

    def sample(self, x, latent=False, num_samples=100):
        latent_sample = self.gpar.sample(
            x, p=self.m, latent=latent, num_samples=100
        )
        observed_sample = B.matmul(latent_sample, self.h, tr_b=True)
        if not latent:
            # compute noise on orthogonal complement in a very dumb way.
            # Gotta double-check.
            proj = B.matmul(hm, _pinv(hm)).T
            proj_orth = B.eye(B.shape(proj)[0]) - proj
            orth_noise = self.noise * proj_orth
            observed_sample = observed_sample + \
                              B.chol(B.reg(orth_noise)) * B.randn(observed_sample)
        return observed_sample

# def plot_latents(vs, m, x, mean_lat, lower_lat, upper_lat):
#     plt.figure(figsize=(12, 8))
#     num = int(np.ceil(m ** .5))
#     obs_lat = B.to_numpy(build(vs).proj)
#     for i in range(min(m, num * num)):
#         plt.subplot(num, num, i + 1)
#         plt.scatter(x, obs_lat[:, i], label='Observations', c='black')
#         plt.plot(x, mean_lat[:, i], label='Prediction', c='tab:blue')
#         plt.plot(x, lower_lat[:, i], c='tab:blue', ls='--')
#         plt.plot(x, upper_lat[:, i], c='tab:blue', ls='--')
#         wbml.plot.tweak(legend=False)
#     plt.show()
#
# def plot_missing_outputs(p, x, y, inds1_n, inds2_n, inds2_p, mean, lower, upper):
#     plt.figure(figsize=(12, 8))
#     num = int(np.ceil(p ** .5))
#     for i in range(min(p, num * num)):
#         plt.subplot(num, num, i + 1)
#         plt.scatter(x[inds1_n], y[inds1_n, i], label='Observations', c='black')
#         if i in inds2_p:
#             plt.scatter(x[inds2_n], y[inds2_n, i], label='Observations', c='black')
#         else:
#             plt.scatter(x[inds2_n], y[inds2_n, i], c='tab:red')
#         plt.plot(x, mean[:, i], label='Prediction', c='tab:blue')
#         plt.plot(x, lower[:, i], c='tab:blue', ls='--')
#         plt.plot(x, upper[:, i], c='tab:blue', ls='--')
#         wbml.plot.tweak(legend=False)
#     plt.show()
#
# def plot_outputs(p, x, y, mean, lower, upper):
#     plt.figure(figsize=(12, 8))
#     num = int(np.ceil(p ** .5))
#     for i in range(min(p, num * num)):
#         plt.subplot(num, num, i + 1)
#         plt.scatter(x, y[:, i], label='Observations', c='black')
#         plt.plot(x, mean[:, i], label='Prediction', c='tab:blue')
#         plt.plot(x, lower[:, i], c='tab:blue', ls='--')
#         plt.plot(x, upper[:, i], c='tab:blue', ls='--')
#         wbml.plot.tweak(legend=False)
#     plt.show()
