from gpar import GPARRegressor
from varz import Vars
import lab as B

def _pinv(h):
    return B.cholsolve(B.dense(B.chol(Dense(h.T @ h))), B.dense(h.T))


def _inv(a):
    return B.cholsolve(B.dense(B.chol(Dense(a))), B.eye(a))

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
        return self.gpar.logpdf(proj_x, proj_y, proj_w) - reg

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
        hcopy = B.to_numpy(h)
        if np.isnan(hcopy).any():
            exit()
        print('H: ', h)
        hm = B.take(h, pattern)
        u = B.svd(h)[0]
        um = B.take(u, pattern)
        proj = B.matmul(_pinv(h), B.matmul(u, _pinv(um))).T
        proj_orth = B.matmul(hm, _pinv(hm)).T

        # Perform projection.
        proj_y = B.matmul(y, proj)
        proj_y_orth = y - B.matmul(y, proj_orth)

        # Compute projected noise.
        h_sq = B.matmul(h.T, h)
        hm_sq = B.matmul(hm.T, hm)
        noise = self.vs['noise']
        print('noise: ', noise)
        proj_noise = noise * (_inv(hm_sq) - _inv(h_sq))

        # Convert projected noise to weights.
        # This requires the self.gpar to have been initialised! That means we need to call
        # self.gpar.sample or something similar. Ugly, again.
        m = self.m
        lat_noises = B.concat(*[self.vs[f'{i}/noise'][None] for i in range(m)])
        print('latnoises: ', lat_noises)
        weights = lat_noises / (lat_noises + B.diag(proj_noise))
        print('weights: ', weights)
        proj_w = B.ones(self.vs.dtype, n, m) * weights[None, :]

        # Compute regularising term.
        reg = 0.5 * (n * (p - m) * B.log(2 * B.pi * noise) + # lognoise
                    B.sum(proj_y_orth ** 2) / noise + # logfrob
                    n * B.logdet(B.reg(B.matmul(um.T, um))) + # logdet
                    1e-4 * B.sum((1 / weights) ** 2)) # This is a regularisation term,
                                                      # rigorously, does not belong in here
        print('reg: ', reg)
        return x, proj_y, proj_w, reg
