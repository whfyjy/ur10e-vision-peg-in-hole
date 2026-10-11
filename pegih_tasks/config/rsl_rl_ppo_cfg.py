"""UR10e 视觉插销任务的 PPO runner 配置。

algorithm 块来自 genpih，未改动，包括那个固定学习率调度——为什么 rsl_rl 的自适应调度
在这里不可用，见下面 `schedule` 上的注释。

policy 块是唯一不同的地方：rsl_rl 2.3.3 不自带 CNN，所以 `class_name` 指向一个库里
不存在的类。`pegih_tasks.tasks.peg_hole.cnn_policy.install()` 把它绑进 `OnPolicyRunner`
解析类名所用的命名空间；脚本必须在构造 runner 之前调用它，否则 runner 会以
`NameError: ActorCriticCNN` 死掉。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PegIhPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 视觉 rollout 每一步的代价远高于 genpih 的"盲"rollout（每个环境都要渲染一次，
    # 再对 100x100x3 做一遍卷积前向和反向），所以 horizon 比 genpih 的 24 x 8192 更长、
    # batch 更小。
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 25
    experiment_name = "pegih_vision"
    run_name = ""
    resume = False
    # 图像到达网络时已经被 mdp.camera_image 缩放到 [0,1] 了；再叠一个 running normalizer
    # 等于给像素拟合一个移动的分布
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        # 通过 rsl_rl.runners.on_policy_runner 里的 eval() 解析——见 cnn_policy.install()。
        # 下面的 hidden 维度被 ActorCriticCNN 当作卷积主干之后的 MLP 头：
        #   Conv(3->32,k5,s2) -> Conv(32->64,k3,s2) -> Conv(64->64,k3,s2)
        #   -> Conv(64->64,k3,s2) -> flatten(3136) -> Linear(256) -> Linear(128)
        class_name="ActorCriticCNN",
        init_noise_std=1.0,
        # rsl_rl 的默认值，也是唯一那一轮真正学会本任务的训练所用的值。
        # 它一度被改成 "log"，之后又改了回来。
        #
        # 当初改是为了躲开
        #     Assertion `normal expects all elements of std >= 0.0` failed
        # 但那是误判：这个断言是 critic 发散的**症状**（value loss 达到约 1e25，
        # float32 溢出），而发散会把 std 参数推成负数。"log" 只是从构造上掩盖了症状
        # （std = exp(log_std) 不可能为负），同时抹掉了信号。真正的防线是
        # pegih_tasks/train_guard.py；保持它启用（不要传 --no-guard）。
        #
        # 改回来之后的实测："scalar" **救不活**一轮失败训练。两轮其它条件完全相同、
        # 只有这个开关不同的 512 环境视觉训练：
        #
        #             噪声峰值   熵峰值    parse 峰值
        #   "scalar"    1.15      8.97       2.36   从未插入
        #   "log"       2.03     11.98       2.38   从未插入
        #
        # 所以 "log" 确实让噪声失控更严重，但它不是决定结果的因素——两者都卡在同一个
        # parse 上。不要把这个更小的 "scalar" 数字读成"修好了"。
        #
        # 真正决定结果的是**训练间的随机性**。三轮代码相同、超参相同、seed 均为 42 的
        # 视觉训练，在第 0 次迭代就已经不一致了（Episode_Reward/dense 分别为
        # 0.006018 / 0.005987 / 0.005994）：Isaac 的场景与渲染器构建会消耗随机流，
        # 所以 seed 并没有钉住这一轮跑什么。三轮里一轮学会了插入，另外两轮从未离开
        # parse ≈ 2.0 的"对准了但不靠近"平台期。而学会的那一轮也**不是**稳定成功——
        # 见下面 learning_rate 上的注释。
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 这个学习率能训出一个好策略，但守不住它——这是本项目测出来的最具体的一处缺陷。
        # 在唯一学会的那轮视觉训练（10-08，512 环境）上做检查点扫描，用确定性动作、
        # 每环境 1 回合，得到：
        #
        #   迭代    200   250   300   350   400   450   500   550   600+
        #   插入/步  .42   .60   .66   .63   .02   .72   .84   .84   0.00
        #
        # 这是**两次**独立的失败，不是一次。model_400 是一次短暂塌陷
        # （.63 -> .02 -> .72），训练自己恢复了。600 那次是终局的：
        # Loss/value_function 从 4.3（第 550 次迭代）-> 91（600）-> 1.3e7（650），
        # 从 600 起每一个检查点的 插入/步 都是 0.0000，因为 critic 发散了，
        # 并把 actor 一起带走了。两个幸存者都保存在 best_policy/ 里：
        # model_550.pt 是应该用的那个（确定性动作 96 回合 100%，采样 96.9%），
        # model_500.pt 是亚军（64 回合 100%，采样 90.6%）。
        #
        # 所以：将来重训时**先降这个**（3e-4），并保持 pegih_tasks/train_guard.py 里的
        # guard 启用，让爆炸能中止训练，而不是毁掉最新的检查点。
        learning_rate=1.0e-3,
        # "fixed"，**不是** rsl_rl 默认的 "adaptive"。自适应 KL 调度跑在 mini-batch 循环
        # 内部，每个 mini-batch 把学习率乘或除以 1.5——5 个 epoch × 4 个 minibatch = 20 次
        # 调整，即一次迭代内最多 1.5**20 ≈ 3300 倍，会把学习率砸到某个边界上钉死。
        # genpih 实测过一轮训练有 94% 的时间停在 1e-5，等于没在学。desired_kl 保留着，
        # 但在 "fixed" 下不起作用。
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
