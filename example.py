from collections import namedtuple

import numpy as np
import lab as B
import matplotlib.pyplot as plt
import torch
import wbml.out
import wbml.plot
from gpar import GPARRegressor
from matrix import Dense
from varz import parametrised, Unbounded, Positive
from varz.torch import minimise_l_bfgs_b


def Model(**kw_args):
    return namedtuple('Model', kw_args.keys())(**kw_args)


# Model parameters:
p = 16
m = 4
gpar = GPARRegressor(replace=True, impute=False, noise=0.05,
                     normalise_y=False, linear=False, nonlinear=True)
vs = gpar.vs

m_true = m  # True number of latent processes.

# Sample some test data.
n = 200
x = B.linspace(0, 30, n)
y = gpar.sample(x, w=B.ones(n, m_true), p=m_true) @ B.randn(m_true, p)

# Add noise to the test data.
noise = 0.05
y = y + noise ** .5 * B.randn(*B.shape(y))

# Split data.
n_split = 100
inds1_n = B.range(0, n_split)
inds2_n = B.range(n_split, n)
n1 = len(inds1_n)
n2 = len(inds2_n)
inds2_p = B.range(p - m, p)
p1 = p
p2 = len(inds2_p)


def _pinv(h):
    return B.cholsolve(B.dense(B.chol(Dense(h.T @ h))), B.dense(h.T))


def _inv(a):
    return B.cholsolve(B.dense(B.chol(Dense(a))), B.eye(a))


@parametrised
def build(vs, h: Unbounded(shape=(p, m)), noise: Positive = 0.05):
    """Build model."""
    wbml.out.kv('Noise', noise)

    x_train = torch.tensor(x)
    y_train = Dense(torch.tensor(y))

    u = B.svd(h)[0]
    pinv = _pinv(h)

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

    # Check spectral gap for debugging.
    vals = B.svd(u2.T @ u2)[1]
    wbml.out.kv('Spectral gap', vals[0] - vals[-1])

    # Determine weights.
    cov_missing = noise * (_inv(h2.T @ h2) - _inv(h.T @ h))
    lat_noises = B.concat(*[vs[f'{i}/noise'][None] for i in range(m)])
    weights = B.diag(cov_missing) / lat_noises + 1
    wbml.out.kv('Weights', weights)

    # Convert to weights for all data.
    dtype = B.dtype(weights)
    weights = B.concat(B.ones(dtype, n1, m),
                       B.ones(dtype, n2, m) * weights[None, :], axis=0)

    return Model(x_train=x_train,
                 y_train=y_train,
                 h=h,
                 noise=noise,

                 x_train1=x_train1,
                 y_train1=y_train1,
                 proj1=proj1,
                 proj1_orth=proj1_orth,

                 x_train2=x_train2,
                 y_train2=y_train2,
                 proj2=proj2,
                 proj2_orth=proj2_orth,
                 u2=u2,

                 proj=B.concat(proj1, proj2, axis=0),
                 weights=weights)


def nlml(vs):
    """Compute the negative LML."""
    model = build(vs)

    # Construct regulariser.
    logdet = n * B.logdet(Dense(model.h.T @ model.h)) + \
             n2 * B.logdet(Dense(model.u2.T @ model.u2))
    lognoise = (n1 * (p1 - m) + n2 * (p2 - m)) * B.log(2 * B.pi * model.noise)
    logfrob = (B.sum(model.proj1_orth ** 2) +
               B.sum(model.proj2_orth ** 2)) / model.noise
    reg = 0.5 * (logdet + lognoise + logfrob) + \
          1e-4 * B.sum(model.weights ** 2)

    gpar.vs = vs
    return -gpar.logpdf(model.x_train, model.proj, model.weights) + reg


def predict(vs, x, num_samples=100):
    """Predict by sampling."""
    model = build(vs)
    h = B.to_numpy(model.h)
    weights = B.to_numpy(model.weights)

    # Condition GPAR and sample from the posterior.
    gpar.vs = vs
    gpar.condition(B.to_numpy(model.x_train),
                   B.to_numpy(model.proj),
                   weights)
    samples = gpar.sample(x, weights, num_samples=num_samples, posterior=True)

    # Compute empirical mean and error bars for predictions of latents.
    samples_lat = B.stack(*samples, axis=0)
    mean_lat = B.mean(samples_lat, axis=0)
    err_lat = 2 * B.std(samples_lat, axis=0)

    # Transform to observation space.
    samples = B.stack(*[sample @ h.T for sample in samples], axis=0)

    # Compute empirical mean and error bars for predictions in observation
    # space.
    mean = B.mean(samples, axis=0)
    err = 2 * B.std(samples, axis=0)

    return ((mean_lat, mean_lat - err_lat, mean_lat + err_lat),
            (mean, mean - err, mean + err))


# Perform training.
with wbml.out.Section('Before training'):
    wbml.out.kv('NLML', nlml(vs))
    vs.print()
minimise_l_bfgs_b(nlml, vs, trace=True, iters=1000)
with wbml.out.Section('After training'):
    wbml.out.kv('NLML', nlml(vs))
    vs.print()

# Perform prediction.
with wbml.out.Section('Predicting'):
    ((mean_lat, lower_lat, upper_lat),
     (mean, lower, upper)) = predict(vs, x)

# Plot predictions for latent processes.
plt.figure(figsize=(12, 8))
num = int(np.ceil(m ** .5))
obs_lat = B.to_numpy(build(vs).proj)
for i in range(min(m, num * num)):
    plt.subplot(num, num, i + 1)
    plt.scatter(x, obs_lat[:, i], label='Observations', c='black')
    plt.plot(x, mean_lat[:, i], label='Prediction', c='tab:blue')
    plt.plot(x, lower_lat[:, i], c='tab:blue', ls='--')
    plt.plot(x, upper_lat[:, i], c='tab:blue', ls='--')
    wbml.plot.tweak(legend=False)

# Plot predictions.
plt.figure(figsize=(12, 8))
num = int(np.ceil(p ** .5))
for i in range(min(p, num * num)):
    plt.subplot(num, num, i + 1)
    plt.scatter(x[inds1_n], y[inds1_n, i], label='Observations', c='black')
    if i in inds2_p:
        plt.scatter(x[inds2_n], y[inds2_n, i], label='Observations', c='black')
    else:
        plt.scatter(x[inds2_n], y[inds2_n, i], c='tab:red')
    plt.plot(x, mean[:, i], label='Prediction', c='tab:blue')
    plt.plot(x, lower[:, i], c='tab:blue', ls='--')
    plt.plot(x, upper[:, i], c='tab:blue', ls='--')
    wbml.plot.tweak(legend=False)

plt.show()
