"""给 rsl_rl 2.3.3 用的 CNN actor-critic（该版本不自带）。

rsl_rl 2.3.3 的 `modules/` 里只有 MLP（actor_critic、actor_critic_recurrent、
student_teacher……），而且它的 `ActorCritic.__init__` 会把不认识的 kwargs 吞掉只打一行
警告，所以图像观测喂不进原版类：`ActorCritic` 会对展平的图像尝试 `Linear(30000, 64)`。

这里继承之所以干净，靠两点：

  1. `ActorCritic` 只把 `self.actor(...)` 和 `self.critic(...)` 当可调用对象用
     （`act`、`act_inference`、`update_distribution`、`evaluate`），另外只用到
     `self.std` / `self.log_std`、`self.noise_std_type` 和 `self.distribution`。
     把这些替换掉，所有继承来的方法都能原样工作。
  2. `OnPolicyRunner` 用 `eval(self.policy_cfg.pop("class_name"))` 在**它自己的模块
     globals** 里解析类名，所以 `install()` 只要把这个类绑进那个命名空间，
     `class_name = "ActorCriticCNN"` 就能解析到。

    import rsl_rl.runners.on_policy_runner as _opr
    _opr.ActorCriticCNN = ActorCriticCNN      # install() 做的事

我们**故意**直接调 `nn.Module.__init__` 而不是 `super().__init__()`：
父类会分配一个 `Linear(30000, hidden[0])` 然后立刻丢弃它。

critic 按输入的维度来选择（非对称的排布见 `install` 的 docstring）：输入维度与图像
相符的走 CNN，其余一律当作特权状态向量，走 MLP。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class ImageEncoder(nn.Module):
    """对一张**以展平形式到达**的图像做卷积。

    Isaac Lab 交过来的观测形状是 (N, H*W*C)，因为只含单项的组不会被拼接到任何东西上，
    而 wrapper 用 `group_obs_dim["policy"][0]` 给网络定维度。像素在 (H, W, C) 上是行主序，
    所以这里用 view -> permute 精确还原，而不是靠猜。
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
        # x: (N, H*W*C)——观测管理器和 rsl_rl wrapper 都是按这个形状上报并给网络定维度的
        if x.dim() == 2:
            h, w, c = self.image_shape
            x = x.view(x.shape[0], h, w, c).permute(0, 3, 1, 2)
        elif x.dim() == 4:                      # 已经是 (N, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 5:                      # (N, T, H, W, C)
            n, t = x.shape[0], x.shape[1]
            h, w, c = self.image_shape
            x = x.reshape(n * t, h, w, c).permute(0, 3, 1, 2)
        return self.head(self.conv(x))


class ActorCriticCNN(ActorCritic):
    """actor 吃图像的 actor-critic；critic 也可以吃。

    多出来的 kwargs 会被接受并忽略，好让这个类能写在一个仍然带着 MLP 字段的
    RslRlPpoActorCriticCfg 里。
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

        # --- 动作噪声，与 ActorCritic 完全一致，这样 PPO 里
        # `self.policy.std` / entropy / log_prob 那几条路径都不用改 ---
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
            """输入是图像就走卷积主干；其它一律当作状态。"""
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
    """让 rsl_rl 的 OnPolicyRunner 能解析到 `class_name = "ActorCriticCNN"`。

    OnPolicyRunner 在自己的模块命名空间里执行
    `eval(self.policy_cfg.pop("class_name"))`，所以这个名字必须存在于那里——
    把它绑进 `rsl_rl.runners.on_policy_runner` 就是全部机制。构造 runner 之前调用一次。

    关于 critic：RslRlVecEnvWrapper 会从环境的 "critic" 观测组上报
    `num_privileged_obs`，而 rsl_rl 的 PPO 在该组存在时会把这一组喂给 critic。于是

      * 有 "critic" 组  -> critic 输入是 14 维状态      -> MLP critic
        （非对称 actor-critic：actor 仍然只看像素，critic 的特权输入从不部署）
      * 无 "critic" 组  -> critic 输入是图像            -> CNN critic
        （对称，端到端完全只有视觉）
    """
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticCNN = ActorCriticCNN


def load_checkpoint(runner, path: str, load_optimizer: bool = True):
    """`runner.load`，并让动作噪声参数跨 noise_std_type 迁移。

    "scalar" 把噪声存成 `std`；"log" 存成 `log_std`。两者是同一位置上的同一个参数，
    所以迁移是精确的（std = exp(log_std)），优化器里每个参数的状态也依然有效——
    但 `load_state_dict` 对名字是严格的，所以没转换过的检查点会以
    "Missing key(s): log_std / Unexpected key(s): std" 失败。

    这件事重要，是因为改一个配置**不能**让改动之前训出的检查点作废：
    当初用笨办法硬来一次，已经让一轮 33 个检查点的视觉训练白白评估失败过。
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
