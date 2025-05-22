import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
import optax
import chex
import os
from tqdm import tqdm
from flowlenia.simutils import load_single_file


@chex.dataclass
class Pred_Config:
    learning_rate: float = 0.1
    momentum: float = 0.9


default_kernel_size = 3
default_hidden_size = 32


class Predictor(nn.Module):

    kernel_size: int = default_kernel_size  # An odd number
    hidden_size: int = default_hidden_size

    def setup(self):
        super().__init__()
        self.mask = jnp.ones((self.kernel_size, self.kernel_size))
        self.mask = self.mask.at[self.kernel_size // 2, self.kernel_size // 2].set(0)

    @nn.compact
    def __call__(self, x):
        # Hide center pixel information
        full_mask = self.mask[..., None, None]
        full_mask = jnp.tile(full_mask, (1, 1, x.shape[-1], self.hidden_size))
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.kernel_size, self.kernel_size),
            mask=full_mask,
        )(x)
        x = nn.relu(x)
        x = nn.Conv(features=3, kernel_size=(1, 1))(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        output = state.apply_fn({"params": params}, images)
        loss = jnp.mean((output - labels) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
    train_loss = np.mean(epoch_loss)
    return state, train_loss


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = Predictor()
    params = cnn.init(rng, jnp.ones([1, 512, 512, 3]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def compute_nncomplexity_online(
    path: str, limit=None, subsampling_rate=1, train_steps=1
):
    files = os.listdir(path)
    ordered_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    ordered_files = ordered_files[::subsampling_rate]
    if limit is not None:
        ordered_files = ordered_files[:limit]

    train_state = None
    dataset = None
    losses = np.zeros((len(ordered_files),))
    test_error = np.zeros((len(ordered_files),))
    t = 0

    for file in tqdm(ordered_files, desc="Processing files"):
        data = load_single_file(f"{path}/{file}")
        assert "s" in data, "Data must contain full state s"
        A = data["s"].A[None, ...]
        if train_state is None:
            rng = jax.random.PRNGKey(0)
            config = Pred_Config()
            train_state = create_train_state(rng, config)
            dataset = A
        else:
            dataset = np.concatenate([dataset, A])[-32:, ...]

        grads, loss = apply_model(train_state, A, A)
        test_error.append(loss)
        train_ds = {
            "image": dataset,
            "label": dataset,
        }
        print(dataset.shape)
        train_state, train_loss = train_epoch(train_state, train_ds, 8, rng)
        # run this train_steps times while summing the loss
        for _ in range(train_steps - 1):
            train_state, l = train_epoch(train_state, train_ds, 8, rng)
            train_loss += l
        print(f"Train loss: {train_loss}")
        losses[t] = train_loss
        t += 1
    return losses, test_error


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import argparse

    seeds = [0, 1, 2, 3, 4]

    train_steps = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    path = parser.parse_args().path
    assert path is not None, "Please provide a path to the data"
    assert os.path.exists(path), f"Path does not exist ({path})"

    all_data = pd.DataFrame(columns=["seed", "t", "Loss"])
    for seed in seeds:
        if not os.path.exists(path + f"/{seed}"):
            print(f"Seed {seed} not found")
            continue
        complexity_data = pd.DataFrame(columns=["seed", "t", "Loss", "Test Error"])
        losses, test_errors = compute_nncomplexity_online(
            path + f"/{seed}", train_steps=train_steps
        )
        complexity_data["t"] = np.arange(len(losses))
        complexity_data["Loss"] = losses
        complexity_data["seed"] = [seed] * len(losses)
        complexity_data["Test Error"] = test_errors
        all_data = pd.concat([all_data, complexity_data])

    # get dir name
    name = path.split("/")[-1]
    all_data.to_csv(
        f"{path}/../{name}_h{default_hidden_size}_k{default_kernel_size}_{train_steps}steps_complexity.csv"
    )
