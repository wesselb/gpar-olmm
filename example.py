from collections import namedtuple

import lab as B
import matplotlib.pyplot as plt
import torch
import wbml.out
import wbml.plot
from gpar import GPARRegressor
from matrix import Dense
from varz.torch import minimise_l_bfgs_b


def Model(**kw_args):
    return namedtuple('Model', kw_args.keys())(**kw_args)


# Model parameters:
p = 16
m = 4
gpar = GPARRegressor()

m_true = m  # True number of latent processes.

# Sample some test data.
n = 200
x = B.linspace(0, 20, n)
y = gpar.sample(x, p=m_true) @ B.randn(m_true, p)

# Add noise to the test data.
noise = 0.1
y = y + noise ** .5 * B.randn(*B.shape(y))

# Split data.
inds1_n = B.range(0, 100)
inds2_n = B.range(100, 200)
n1 = len(inds1_n)
n2 = len(inds2_n)
inds2_p = B.range(8, p)
p1 = p
p2 = len(inds2_p)


def _pinv(h):
    return B.cholsolve(B.dense(B.chol(Dense(h.T @ h))), B.dense(h.T))


def build(vs):
    """Build model."""
    x_train = torch.tensor(x)
    y_train = Dense(torch.tensor(y))

    h = vs.unbounded(shape=(p, m), name='h')

    x_train1 = x_train[inds1_n]
    y_train1 = y_train[inds1_n]
    h1 = h
    pinv1 = _pinv(h1)
    proj1 = y_train1 @ pinv1.T
    proj1_orth = y_train1 - proj1 @ h1.T

    x_train2 = x_train[inds2_n]
    y_train2 = y_train[inds2_n][:, inds2_p]
    h2 = h[inds2_p]
    pinv2 = _pinv(h2)
    proj2 = y_train2 @ pinv2.T
    proj2_orth = y_train2 - proj2 @ h2.T

    return Model(x_train=x_train,
                 y_train=y_train,
                 h=h,

                 x_train1=x_train1,
                 y_train1=y_train1,
                 h1=h1,
                 pinv1=pinv1,
                 proj1=proj1,
                 proj1_orth=proj1_orth,

                 x_train2=x_train2,
                 y_train2=y_train2,
                 h2=h2,
                 pinv2=pinv2,
                 proj2=proj2,
                 proj2_orth=proj2_orth,

                 proj=B.concat(proj1, proj2, axis=0))


def nlml(vs):
    """Compute the negative LML."""
    model = build(vs)

    # Construct regulariser.
    logdet = n1 * B.logdet(Dense(model.h1.T @ model.h1)) + \
             n2 * B.logdet(Dense(model.h2.T @ model.h2))
    logfrob = (n1 * (p1 - m) + n2 * (p2 - m)) * \
              B.log(B.sum(model.proj1_orth ** 2) +
                    B.sum(model.proj2_orth ** 2))
    reg = 0.5 * (logdet + logfrob)

    return -gpar.logpdf(model.x_train, model.proj) + reg


def predict(vs, x, num_samples=40):
    """Predict by sampling."""
    model = build(vs)
    h = B.to_numpy(model.h)

    # Condition GPAR and sample from the posterior.
    gpar.condition(B.to_numpy(model.x_train), B.to_numpy(model.proj))
    samples = gpar.sample(x, num_samples=num_samples, posterior=True)
    samples = B.stack(*[sample @ h.T for sample in samples], axis=0)

    # Return empirical mean and error bars.
    mean = B.mean(samples, axis=0)
    err = 2 * B.std(samples, axis=0)
    return mean, mean - err, mean + err


# Perform training.
wbml.out.kv('NLML before', nlml(gpar.vs))
minimise_l_bfgs_b(nlml, gpar.vs, trace=True, iters=400)
wbml.out.kv('NLML after', nlml(gpar.vs))

# Perform prediction.
with wbml.out.Section('Predicting'):
    mean, lower, upper = predict(gpar.vs, x)

# Plot predictions.
plt.figure()
for i in range(min(p, 16)):
    plt.subplot(4, 4, i + 1)
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
