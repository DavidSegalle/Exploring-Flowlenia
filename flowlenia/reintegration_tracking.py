import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Optional, Tuple, TypeAlias, Any
from flowlenia.utils import growth


class ReintegrationTracking:

    def __init__(
        self,
        cfg,
        has_hidden=False,
    ):
        self.cfg = cfg
        self.X = cfg.X
        self.Y = cfg.Y
        self.dt = cfg.dt
        self.dd = cfg.dd
        self.sigma = cfg.sigma
        self.has_hidden = has_hidden
        self.border = cfg.border if cfg.border in ["wall", "torus"] else "wall"
        self.mix = cfg.mix_rule
        self.beta_param = cfg.beta_param

    def __call__(self, *args, **kwargs):

        if self.has_hidden:
            return self._apply_with_hidden(*args, **kwargs)
        else:
            return self._apply_without_hidden(*args, **kwargs)

    def _apply_without_hidden(
        self, A: jax.Array, F: jax.Array, *args, **kwargs
    ) -> jax.Array:
        x, y = jnp.arange(self.X), jnp.arange(self.Y)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + 0.5  # (X, Y, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)

        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def step(A, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
            if self.border == "torus":
                dpmu = jnp.min(
                    jnp.stack(
                        [
                            jnp.absolute(
                                pos[..., None]
                                - (mur + jnp.array([di, dj])[None, None, :, None])
                            )
                            for di in (-self.X, 0, self.X)
                            for dj in (-self.Y, 0, self.Y)
                        ]
                    ),
                    axis=0,
                )
            else:
                dpmu = jnp.absolute(pos[..., None] - mur)
            sz = 0.5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2 * self.sigma)), axis=2) / (
                4 * self.sigma**2
            )
            nA = Ar * area
            return nA

        ma = self.dd - self.sigma  # upper bound of the flow magnitude
        mu = pos[..., None] + jnp.clip(
            self.dt * F, -ma, ma
        )  # (x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.X - self.sigma)

        nA = step(A, mu, dxs, dys).sum(0)
        return nA

    def _apply_with_hidden(
        self,
        state,
        F: jax.Array,
        key: jax.random.PRNGKey,
        *args,
        **kwargs,
    ):
        A = state.A
        H = state.P
        Hmix = state.Pmix
        x, y = jnp.arange(self.X), jnp.arange(self.Y)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + 0.5  # (X, Y, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)

        @partial(jax.vmap, in_axes=(None, None, None, None, 0, 0))
        def step_flow(A, H, Hmix, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            Hr = jnp.roll(H, (dx, dy), axis=(0, 1))  # (x, y, k)
            Hmixr = jnp.roll(Hmix, (dx, dy), axis=(0, 1))  # (x, y, k)
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))

            if self.border == "torus":
                dpmu = jnp.min(
                    jnp.stack(
                        [
                            jnp.absolute(
                                pos[..., None]
                                - (mur + jnp.array([di, dj])[None, None, :, None])
                            )
                            for di in (-self.X, 0, self.X)
                            for dj in (-self.Y, 0, self.Y)
                        ]
                    ),
                    axis=0,
                )
            else:
                dpmu = jnp.absolute(pos[..., None] - mur)

            sz = 0.5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2 * self.sigma)), axis=2) / (
                4 * self.sigma**2
            )
            nA = Ar * area
            return nA, Hr, Hmixr

        ma = self.dd - self.sigma  # upper bound of the flow magnitude
        mu = pos[..., None] + jnp.clip(
            self.dt * F, -ma, ma
        )  # (x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.X - self.sigma)
        nA, nH, nHmix = step_flow(A, H, Hmix, mu, dxs, dys)

        if self.mix == "avg":
            nH = jnp.sum(nH * nA.sum(axis=-1, keepdims=True), axis=0)
            nA = jnp.sum(nA, axis=0)
            nH = nH / (nA.sum(axis=-1, keepdims=True) + 1e-10)

        elif self.mix == "softmax":
            expnA = jnp.exp(nA.sum(axis=-1, keepdims=True)) - 1
            nA = jnp.sum(nA, axis=0)
            nH = jnp.sum(nH * expnA, axis=0) / (expnA.sum(axis=0) + 1e-10)  # avg rule

        elif self.mix == "stoch":
            categorical = jax.random.categorical(
                key, jnp.log(nA.sum(axis=-1, keepdims=True)), axis=0
            )
            mask = jax.nn.one_hot(
                categorical, num_classes=(2 * self.dd + 1) ** 2, axis=-1
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)

        elif self.mix == "stoch_gene_wise":
            keys = jax.random.split(key, H.shape[-1])
            mask = jnp.concatenate(
                [
                    jax.nn.one_hot(
                        jax.random.categorical(
                            k,
                            jnp.log(nA.sum(axis=-1, keepdims=True)),
                            axis=0,
                        ),
                        num_classes=(2 * dd + 1) ** 2,
                        axis=-1,
                    )
                    for k in keys
                ],
                axis=2,
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)
        elif self.mix == "stoch_softmax":
            categorical = jax.random.categorical(
                key, nA.sum(axis=-1, keepdims=True) * self.beta_param, axis=0
            )
            mask = jax.nn.one_hot(
                categorical, num_classes=(2 * self.dd + 1) ** 2, axis=-1
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)
        elif self.mix == "stoch_softmax_gene_wise":
            keys = jax.random.split(key, H.shape[-1])
            mask = jnp.concatenate(
                [
                    jax.nn.one_hot(
                        jax.random.categorical(
                            k,
                            nA.sum(axis=-1, keepdims=True) * self.beta_param,
                            axis=0,
                        ),
                        num_classes=(2 * dd + 1) ** 2,
                        axis=-1,
                    )
                    for k in keys
                ],
                axis=2,
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)
        elif self.mix == "dot":
            normalized_nH = (
                nH / (jnp.linalg.norm(nH) + 1e-10) * nA.sum(axis=-1, keepdims=True)
            )
            tmp = jnp.einsum("pxyk,qxyk->pxy", normalized_nH, normalized_nH)
            categorical = jnp.argmax(tmp[:, :, :, None], axis=0)
            mask = jax.nn.one_hot(
                categorical, num_classes=(2 * self.dd + 1) ** 2, axis=-1
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)
        elif self.mix == "best_against_old":
            # TODO: add mass
            normalized_nH = nH / (jnp.linalg.norm(nH) + 1e-10)
            normalized_H = H / (jnp.linalg.norm(H) + 1e-10)
            tmp = jnp.abs(jnp.einsum("pxyk,xyk->pxy", normalized_nH, normalized_H))
            categorical = jnp.argmin(tmp[:, :, :, None], axis=0)
            mask = jax.nn.one_hot(
                categorical, num_classes=(2 * self.dd + 1) ** 2, axis=-1
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)

        elif self.mix == "negotiate":
            pass  # Return unprocessed values

        elif self.mix == "energy":
            v2 = (dxs**2 + dys**2)[:, None, None, None]
            categorical = jax.random.categorical(
                key,
                nA.sum(axis=-1, keepdims=True)
                * self.beta_param
                * v2
                / (self.cfg.dt**2),
                axis=0,
            )
            mask = jax.nn.one_hot(
                categorical, num_classes=(2 * self.dd + 1) ** 2, axis=-1
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)

        else:
            raise ValueError(f"Unknown mixing rule {self.mix}")

        return nA, nH, nHmix

    # -------------------------------------------------------------------
