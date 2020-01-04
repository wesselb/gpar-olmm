from collections import namedtuple

import matplotlib.pyplot as plt
import torch
import wbml.out
import wbml.plot
from gpar import GPARRegressor
from lab import B
from matrix import Dense, Diagonal
from varz.torch import minimise_l_bfgs_b


def Model(**kw_args):
    return namedtuple('Model', kw_args.keys())(**kw_args)


# Model parameters:
p = 16
m = 5
gpar = GPARRegressor()
dense_noise = False

m_true = m  # True number of latent processes.

# Sample some test data.
n = 200
x = B.linspace(0, 10, n)
y = gpar.sample(x, p=m_true) @ B.randn(m_true, p)

# Add noise to the test data.
noise = 0.2
y = y + noise ** .5 * B.randn(*B.shape(y))


def build(vs):
    """Build model."""
    x_train = torch.tensor(x)
    y_train = Dense(torch.tensor(y))

    # Construct the orthogonal matrices.
    u_full = vs.orth(shape=(p, p), name='u_full')
    u = Dense(u_full[:, :m])
    u_orth = Dense(u_full[:, m:])

    # Construct the mixing matrix and the projection.
    s_sqrt = Diagonal(vs.pos(B.ones(m), name='s_sqrt'))
    s = s_sqrt @ s_sqrt
    h = u @ s_sqrt
    h_pinv = B.inv(s_sqrt) @ u.T

    # Project the data.
    proj = y_train @ h_pinv.T
    proj_orth = y_train @ u_orth

    return Model(x_train=x_train,
                 y_train=y_train,
                 proj=proj,
                 proj_orth=proj_orth,
                 u=u,
                 u_orth=u_orth,
                 s_sqrt=s_sqrt,
                 s=s,
                 h=h,
                 h_pinv=h_pinv)


def nlml(vs):
    """Compute the negative LML."""
    model = build(vs)

    reg = B.logdet(model.s)
    if dense_noise:
        reg = reg + B.logdet(Dense(model.proj_orth.T @ model.proj_orth))
    else:
        reg = reg + (p - m) * B.log(B.sum(model.proj_orth ** 2))

    return -gpar.logpdf(model.x_train, model.proj) + 0.5 * n * reg


def predict(vs, x, num_samples=100):
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
minimise_l_bfgs_b(nlml, gpar.vs, trace=True)
wbml.out.kv('NLML after', nlml(gpar.vs))

# Perform prediction.
with wbml.out.Section('Predicting'):
    mean, lower, upper = predict(gpar.vs, x)

# Plot predictions.
plt.figure()
for i in range(p):
    plt.subplot(int(p ** .5), int(p ** .5), i + 1)
    plt.scatter(x, y[:, i], label='Observations', c='black')
    plt.plot(x, mean[:, i], label='Prediction', c='tab:blue')
    plt.plot(x, lower[:, i], c='tab:blue', ls='--')
    plt.plot(x, upper[:, i], c='tab:blue', ls='--')
    wbml.plot.tweak()
plt.show()
