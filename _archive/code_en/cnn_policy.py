"""A CNN actor-critic for rsl_rl 2.3.3, which ships none.

rsl_rl 2.3.3's `modules/` contains only MLPs (actor_critic, actor_critic_recurrent,
student_teacher, ...) and its `ActorCritic.__init__` swallows unknown kwargs with
a printed warning, so an image observation cannot be fed to the stock classes:
`ActorCritic` would try `Linear(30000, 64)` on a flattened image.

Two things make subclassing clean here:

  1. `ActorCritic` only ever uses `self.actor(...)` and `self.critic(...)` as
     callables (`act`, `act_inference`, `update_distribution`, `evaluate`), plus
     `self.std` / `self.log_std`, `self.noise_std_type` and `self.distribution`.
     Replace those and every inherited method keeps working unchanged.
  2. `OnPolicyRunner` resolves the class with `eval(self.policy_cfg.pop("class_name"))`
     evaluated in its own module globals, so `install()` binding this class into
     that namespace is enough to make `class_name = "ActorCriticCNN"` resolve.

    import rsl_rl.runners.on_policy_runner as _opr
    _opr.ActorCriticCNN = ActorCriticCNN      # what install() does

We call `nn.Module.__init__` directly instead of `super().__init__()` on purpose:
the parent would allocate and immediately discard `Linear(30000, hidden[0])`.

The critic is chosen by the size of its input (see `install`'s docstring for the
asymmetric arrangement): an input that matches the image gets a CNN, anything
else is treated as a privileged state vector and gets an MLP.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class ImageEncoder(nn.Module):
    """Conv stack over an image that arrives FLATTENED.

    Isaac Lab hands the observation over as (N, H*W*C) because a group of one
    term is not concatenated into anything else, and the wrapper sizes the
    network with `group_obs_dim["policy"][0]`. The pixel order is row-major over
    (H, W, C), so this undoes it with view -> permute rather than guessing.
    """

    def __init__(self, image_shape=(100, 100, 3), out_dim: int = 256,
                 channels=(32, 64, 64, 64), activation: nn.Module = None):
        super().__init__()
        h, w, c = image_shape
        self.image_shape = (h, w, c)
        act = activation if activation is not None else nn.ELU()

        layers = []
        in_c = c
        for i, out_c in enumerate(channels):
            kernel, stride, pad = (5, 2, 2) if i == 0 else (3, 2, 1)
            layers += [nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=pad), act]
            in_c = out_c
        self.conv = nn.Sequential(*layers, nn.Flatten())

        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, out_dim), act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H*W*C) -- the shape the observation manager and the rsl_rl
        # wrapper both report and size the network from
        if x.dim() == 2:
            h, w, c = self.image_shape
            x = x.view(x.shape[0], h, w, c).permute(0, 3, 1, 2)
        elif x.dim() == 4:                      # already (N, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 5:                      # (N, T, H, W, C)
            n, t = x.shape[0], x.shape[1]
            h, w, c = self.image_shape
            x = x.reshape(n * t, h, w, c).permute(0, 3, 1, 2)
        return self.head(self.conv(x))


class ActorCriticCNN(ActorCritic):
    """Actor-critic whose actor consumes an image; the critic may too.

    Extra kwargs are accepted and ignored so the class can be named in an
    RslRlPpoActorCriticCfg that still carries MLP fields.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape=(100, 100, 3),
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        nn.Module.__init__(self)
        act = resolve_nn_activation(activation)

        # --- action noise, mirroring ActorCritic exactly so PPO's
        # `self.policy.std` / entropy / log_prob paths are unchanged ---
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

        def mlp(in_dim: int, hidden, out_dim: int) -> nn.Sequential:
            layers = []
            for h in hidden:
                layers += [nn.Linear(in_dim, h), act]
                in_dim = h
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        img_dim = image_shape[0] * image_shape[1] * image_shape[2]

        def encoder(in_dim: int, hidden, out_dim: int) -> nn.Module:
            """Image input gets a conv stem; anything else is treated as state."""
            if in_dim == img_dim:
                return nn.Sequential(
                    ImageEncoder(image_shape, out_dim=hidden[0], activation=act),
                    mlp(hidden[0], tuple(hidden[1:]), out_dim),
                )
            return mlp(in_dim, hidden, out_dim)

        self.actor = encoder(num_actor_obs, actor_hidden_dims, num_actions)
        self.critic = encoder(num_critic_obs, critic_hidden_dims, 1)

        print(f"Actor:  in={num_actor_obs} {'(image)' if num_actor_obs == img_dim else '(state)'}")
        print(f"Critic: in={num_critic_obs} {'(image)' if num_critic_obs == img_dim else '(state)'}")
        print(f"  actor  = {self.actor}")
        print(f"  critic = {self.critic}")


def install() -> None:
    """Make `class_name = "ActorCriticCNN"` resolvable by rsl_rl's OnPolicyRunner.

    OnPolicyRunner does `eval(self.policy_cfg.pop("class_name"))` in its own
    module namespace, so the name has to exist there -- binding it into
    `rsl_rl.runners.on_policy_runner` is the whole mechanism. Call this once
    before constructing the runner.

    Note on the critic: RslRlVecEnvWrapper reports `num_privileged_obs` from the
    environment's "critic" observation group, and rsl_rl's PPO feeds that group
    to the critic when it exists. So

      * with a "critic" group  -> critic input is the 14-dim state  -> MLP critic
        (asymmetric actor-critic: the actor still sees only pixels, and the
        critic's privileged input is never deployed)
      * with no "critic" group -> critic input is the image          -> CNN critic
        (symmetric, fully vision-only end to end)
    """
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticCNN = ActorCriticCNN


def load_checkpoint(runner, path: str, load_optimizer: bool = True):
    """`runner.load`, migrating the action-noise parameter across noise_std_type.

    "scalar" stores the noise as `std`; "log" stores it as `log_std`. Both are
    the same single parameter in the same position, so the migration is exact
    (std = exp(log_std)) and the optimizer's per-parameter state stays valid --
    but `load_state_dict` is strict on names, so an unconverted checkpoint fails
    with "Missing key(s): log_std / Unexpected key(s): std".

    That matters because switching a config must not invalidate checkpoints
    trained before the switch: doing it the hard way once already cost a
    33-checkpoint vision run a failed evaluation.
    """
    import tempfile

    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", {})
    wants_log = any(k.endswith("log_std") for k in runner.alg.policy.state_dict())

    if wants_log and "std" in sd and "log_std" not in sd:
        sd["log_std"] = torch.log(sd.pop("std").clamp_min(1e-8))
    elif (not wants_log) and "log_std" in sd and "std" not in sd:
        sd["std"] = sd.pop("log_std").exp()
    else:
        return runner.load(path, load_optimizer=load_optimizer)

    fd, tmp = tempfile.mkstemp(suffix=".pt", prefix="migrated_")
    os.close(fd)
    torch.save(ckpt, tmp)
    try:
        print(f"  [migrate] noise parameter re-keyed for this config: {os.path.basename(path)}")
        return runner.load(tmp, load_optimizer=load_optimizer)
    finally:
        os.remove(tmp)
