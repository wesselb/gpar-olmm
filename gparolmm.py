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

        # Initialise things
        p = self.p
        m = self.m
        h = vs['h']
        noise = vs['noise']
        u = B.svd(h)[0]
        pinv = _pinv(h)
        x_train = torch.tensor(x)
        y_train = Dense(torch.tensor(y))
        proj = y_train @ pinv.T
        proj_orth = y_train - proj @ h.T
        n = B.shape(x)[0]

        # Construct regulariser.
        logdet = n * B.logdet(Dense(h.T @ h))
        lognoise = (n * (p - m)) * B.log(2 * B.pi * noise)
        logfrob = B.sum(proj_orth ** 2) / noise
        reg = 0.5 * (logdet + lognoise + logfrob)

        return self.gpar.logpdf(x_train, proj, B.ones(n, m)) - reg

    def logpdf_missing(
        self,
        x,
        y,
        inds1_n,
        inds2_n,
        inds2_p,
        vs,
    ):
        self.vs = vs # Side-effect!

        # Initialise things
        p = self.p
        m = self.m
        h = vs['h']
        noise = vs['noise']
        n = B.shape(x)[0]
        n1 = len(inds1_n)
        n2 = len(inds2_n)
        u = B.svd(h)[0]
        pinv = _pinv(h)
        x_train = torch.tensor(x)
        y_train = Dense(torch.tensor(y))
        proj = y_train @ pinv.T
        p1 = p
        p2 = len(inds2_p)

        # Deal with missings
        x_train1 = x_train[inds1_n]
        y_train1 = y_train[inds1_n]
        proj1 = y_train1 @ pinv.T
        proj1_orth = y_train1 - proj1 @ h.T

        x_train2 = x_train[inds2_n]
        y_train2 = y_train[inds2_n][:, inds2_p]
        h2 = h[inds2_p]
        u2 = u[inds2_p]
        proj2 = y_train2 @ (pinv @ u @ _pinv(u2)).T
        proj2_orth = y_train2 - y_train2 @ (h2 @ _pinv(h2)).T

        # Maybe add spectral gap and other debugging stuff later

        # Determine weights.
        cov_missing = noise * (_inv(h2.T @ h2) - _inv(h.T @ h))
        # This requires the self.gpar to have been initialised! That means we need to call
        # self.gpar.sample or something similar.
        lat_noises = B.concat(*[vs[f'{i}/noise'][None] for i in range(m)])
        weights = lat_noises / (lat_noises + B.diag(cov_missing))

        # Convert to weights for all data.
        dtype = B.dtype(weights)
        weights = B.concat(
            B.ones(dtype, n1, m),
            B.ones(dtype, n2, m) * weights[None, :],
            axis=0
        )

        # Construct regulariser.
        logdet = n * B.logdet(Dense(h.T @ h)) + n2 * B.logdet(Dense(u2.T @ u2))
        lognoise = (n1 * (p1 - m) + n2 * (p2 - m)) * B.log(2 * B.pi * noise)
        logfrob = (B.sum(proj1_orth ** 2) + B.sum(proj2_orth ** 2)) / noise
        reg = 0.5 * (logdet + lognoise + logfrob) + 1e-4 * B.sum((1 / weights) ** 2)

        return self.gpar.logpdf(x_train, proj, weights) - reg
