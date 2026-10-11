# ur10e-peg-in-hole rl — Isaac Lab 视觉插销（Vision Peg-in-Hole）

在 Isaac Sim 4.5 / Isaac Lab 2.2.1 上搭的 UR10e 视觉插销强化学习环境。

> **可直接使用的策略（已验证）：**
>
> | 用途 | 检查点 | success % |
> |---|---|---|
> | **视觉（本项目的目标）** | `runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/model_550.pt` | **100.0 %**（确定性）／93.8 %（采样） |
> | 状态（对照基线） | `runs/rsl_rl/pegih_vision/2026-10-09_16-00-10_state_long_resume/model_3875.pt` | **100.0 %**（两种动作源都是 128/128） |
>
> **不要用各自 run 目录里编号最大的检查点**——它们都已塌陷（视觉组 `model_799` 的 return 是 0.20，
> 状态组 `model_5150` 是 0.45 且已在崩溃边缘）。详见「训练进展与评估结果」。
>
> 复现评估：`$PY scripts/eval.py --headless --which best --run_dir <上面的 run 目录> --num_envs 32 --episodes 2`
> （状态那条要加 `--no-vision`）

> **文档**：[`docs/学习路线.md`](docs/学习路线.md) —— 按什么顺序读这个项目、每步动手做什么、面试会问什么（含 20 题题库）；
> [`docs/训练手册.md`](docs/训练手册.md) —— 怎么跑、怎么选检查点、怎么排错。

- **机械臂 / 轴销 / 孔板**：直接复用 [`genpih`](../genpih) 的资产（UR10e + Cranfield 轴销 Ø42.4×120 mm、
  孔板 200×200×30 mm、Ø43 通孔 + 5 mm×45° 倒角），USD 已复制进本项目，**不依赖 genpih 目录**。
- **新场景**：孔板放在一块**不可穿透的实心木板**上，木板顶面与**机械臂基座顶面等高**；
  机械臂、木板、孔**全部固定**，零域随机化。
- **观测**：眼在手上相机的图像（纯视觉）；**框架**：Isaac Lab + rsl_rl。

## 环境

```bash
PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python   # conda env `isaac_env`
```
Isaac Sim 4.5.0 / Isaac Lab 2.2.1（editable，源码在 `/home/whf/whf/IsaacLab`）/ rsl-rl-lib 2.3.3 / RTX 4090。

## 目录

```
ur10e-peg-in-hole rl/
├── assets/usd/{robot,hole}/    # 从 genpih 复制的 USD（robot 8.1 MB，hole 32 KB）
├── pegih_tasks/
│   ├── tasks/peg_hole/
│   │   ├── peg_hole_env_cfg.py # 场景 / 奖励 / 终止 / 相机 / 观测组
│   │   ├── mdp.py              # MDP 函数（判据与奖励沿用 genpih）
│   │   └── cnn_policy.py       # ActorCriticCNN + 注入 rsl_rl 的 install()
│   └── config/rsl_rl_ppo_cfg.py
└── scripts/
    ├── kin.py             # URDF 正/逆运动学（脚本化演示用）
    ├── cam_offset.py      # 由"世界机位 + 看向点"反解相机 offset 四元数
    ├── cam_calib.py       # 渲染 home/hover/seated 三张图，人工核验取景
    ├── probe_collision.py # 碰撞探针（孔板 mesh 是否真的挡得住 peg）
    ├── probe_insert.py    # 精细下压探针（成功判据是否可达）
    ├── smoke_test.py      # 冒烟测试（18 项 PASS/FAIL）
    ├── train.py           # PPO 训练
    ├── eval.py            # 评估：success% / insert/step / return，采样与确定性两种动作源
    └── gui_demo.py        # 脚本化插销演示
```

## 场景与几何（数字均已实测，不是估算）

用 URDF 做正运动学 + 阻尼最小二乘 IK 算出来，并用仿真复核过：

| 量 | 值 | 来源 |
|---|---|---|
| `HOME_JOINTS` | `[67.5°, -90°, 90°, -90°, -90°, 0°]` | genpih |
| home 时 peg 尖端 | `(0.1037, 0.7054, 0.5568)` | FK，与 genpih 实测一致 |
| home 时 peg 轴线 | `(0, 0, -1)`，垂直度误差 **0.0000°** | FK |
| 木板顶面 `BOARD_TOP_Z` | **0.1807**（= `shoulder_pan_joint` 高度 = 基座顶面） | URDF |
| 孔板顶面 | 0.2107（板厚 30 mm） | |
| 下压行程 | 0.3761 m | |
| 孔位 `FIXTURE_POS` | `(0.1637, 0.7054, 0.1807)` = home 尖端 xy + **(0.06, 0)** 偏移 | 见下 |
| 孔径间隙 | 0.6 mm 直径（半径 0.3 mm） | genpih 网格 |

**初始不对准**是需求：peg 尖端在 `(0.1037, 0.7054)`，孔在 `(0.1637, 0.7054)`，横向差 **60 mm**，
所以策略必须先把 peg 移过去再下压。偏移大小可调（`START_OFFSET_XY`），受相机视场约束（见下）。

**可达性**（IK 扫过 0.04–0.15 m 的各种偏移）：位置误差 0.000 mm、倾斜 0.0000°，
关节偏移仅 0.60–0.71 rad（动作裁剪是 ±1.5 rad），且所有连杆最低点（wrist_3 = 0.301 m）
都高于孔板顶面 0.2107 m，**下压全程不会撞板**。

## 任务规格

| 项 | 值 | 说明 |
|---|---|---|
| 动作 | 6 个关节绝对位置目标，scale 1.5，裁剪到 home ± 1.5 | 沿用 genpih |
| 观测（policy） | 相机 RGB 100×100×3，展平为 30000 维 | 纯视觉 |
| 观测（critic） | 14 维特权状态（孔位姿在 peg 系 7 + 尖端离板高度 1 + 上一步动作 6） | 非对称 AC |
| 密集奖励 | `1/(1+dp/0.043) · 1/(1+dq/0.05)`，权重 10 | 沿用 genpih |
| 稀疏奖励 | 对准 2.6 / 插入 10 | 沿用 genpih Eq. 7 |
| 成功判据 | `dq < 0.05 rad 且 dp < 0.003 m` | **未改动** |
| 终止 | 仅超时，成功不终止 | 沿用 genpih |
| 步长 / 回合 | dt = 1/120，decimation 2（策略 60 Hz），900 步 / 15 s | |
| 并行环境 | **512**（视觉；实测 21.7 GB，1024 会 OOM） | 状态观测可上 1024+ |


## 视觉

**相机**：`TiledCamera`，挂在 `{ENV_REGEX_NS}/Robot/wrist_3_link/Camera`，`update_period = 1/60`，
100×100 RGB，focal 24 cm、aperture 20.955 cm（HFOV 47.2°），`clipping_range=(0.05, 20)`。

相机 offset 这里用 `scripts/cam_offset.py` 反解：
给定世界机位与看向点，求出 `OffsetCfg` 需要的 (pos, rot, convention="ros")。
选定的机位是沿 wrist 局部 -x 方向 0.09 m、与腕同高，瞄向 peg 尖端与孔的中点：

| 目标 | 离轴角 | 距离 |
|---|---|---|
| peg 尖端 | 12.6° | 0.150 m |
| 孔 | 8.2° | 0.518 m |

两者都在 23.6° 半视场内，余量充足。更远（0.12/0.20 m）会把 peg 顶出画面上方而孔几乎没改善，
实测渲染确认过。

**图像归一化**刻意**不用** `mdp.image` 的 `normalize=True`：它对 rgb 会减去**当前 batch 的均值**，
使同一场景在 batch=1024 与 batch=1 下产生不同观测（训练与单环境确定性评估的差异正来源于此）。
`mdp.camera_image` 改为固定除以 255。

**CNN 策略**：rsl_rl 2.3.3 只有 MLP。`cnn_policy.ActorCriticCNN` 子类化 `ActorCritic`，
直接调 `nn.Module.__init__`（避免父类先建一个 30000→256 又立刻丢弃的层），只替换
`actor`/`critic` 与前向相关的属性；`act` / `evaluate` / `update_distribution` / `entropy` 等
全部继承，已被 rsl_rl 的 PPO 实际调用路径覆盖验证。

注入方式：`OnPolicyRunner` 用 `eval(self.policy_cfg.pop("class_name"))` 在**自己的模块命名空间**里
解析类名，所以 `cnn_policy.install()` 把类绑进 `rsl_rl.runners.on_policy_runner` 即可，
`cfg.policy.class_name = "ActorCriticCNN"`。

网络：`Conv(3→32,k5,s2) → Conv(32→64,k3,s2) → Conv(64→64,k3,s2) → Conv(64→64,k3,s2)`（ELU）
→ flatten 3136 → `Linear(256) → Linear(128)` 分叉到 actor(6) / critic(1)。约 0.97 M 参数。

**critic 是否吃特权状态**：默认吃（非对称 AC，actor 仍只看像素，critic 不参与部署）。
`--symmetric-critic` 可关掉，关掉后 critic 也吃图像，网络端到端只有视觉。

## 运行

```bash
cd "/home/whf/whf/ur10e-peg-in-hole rl"
PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python

# 冒烟测试（默认视觉；--no-vision 走状态观测，快得多）
$PY scripts/smoke_test.py --headless --image_dir /tmp/smoke
$PY scripts/smoke_test.py --headless --no-vision

# 相机取景核验（出 PNG，人工看）
$PY scripts/cam_calib.py --headless --out /tmp/cam
$PY scripts/cam_calib.py --headless --out /tmp/camL --lateral 0.15   # 试别的机位

# 物理探针
$PY _archive/scripts/probe_collision.py --headless
$PY _archive/scripts/probe_insert.py --headless

# 训练
$PY scripts/train.py --headless --num_envs 1024
$PY scripts/train.py --headless --no-vision --num_envs 512    # 状态基线，用来对比
$PY scripts/train.py --headless --num_envs 256 --max_iterations 20   # 快速确认能跑

# GUI 演示（脚本化插销，不看策略）
DISPLAY=:1 $PY scripts/gui_demo.py --gui
```

## 设计取舍与实测结论


1. **成功判据可达。** `scripts/probe_insert.py` 以 1 mm 步进、每步驻留 30 个物理步下压，
   实测 `best dp = 0.194 mm`，尖端停在 `z = 0.1807`（正好木板顶面）。
   genpih 默认执行器参数（k=50000 / d=1500 / effort 87）就够；把 `effort_limit_sim` 提到 330
   只带来边际改善（0.177 mm），因此**未改动**执行器参数。

2. **但粗放的下压会卡死。** 用两个路点一步跳 27 mm 时，peg 只进孔 7.6 mm 就卡住（`dp = 22.2 mm`、
   接触力 188 N、倾斜 1.6°）。原因是 0.3 mm 的半径间隙：负载下伺服变形让 peg 倾斜，
   斜插的柱面在 `L ≈ 间隙/倾角` 处咬住孔壁。genpih 独立记录过同一阈值
   （"dq=0.043 rad 时卡在 dp≈9.5 mm，0.01 rad 时能到 1.2 mm"）。
   **这是插销任务本身的核心难点，不是环境缺陷**——策略必须学会先对准再缓降，
   这正是这个环境要训的东西。

3. **近距离时 peg 必然遮住孔**（几何决定，非取景问题）：peg Ø42.4 vs 孔 Ø43，
   尖端离板 30 mm 以内时任何机位都拍不到完整孔——视差随尖端-板距离缩小而消失。
   可见信号变成 peg 侧面露出的孔的月牙形，其宽度直接编码横向误差，信息没丢只是换了形式。
   （参考项目的相机有同样性质但要宽容得多：它的 peg 是 Ø36、孔是 Ø48。）

4. **成功不终止回合**（沿用 genpih）。稀疏奖励是**每步**支付的，若插入即终止，
   悬停对准整回合反而严格优于插入。代价是插入后策略会抖，且
   `Episode_Termination/success` 不出现在训练日志里——用 `eval.py` 或 `Episode_Reward/parse` 看。

5. **纯视觉这件事有一个口径要注意**：默认配置里 actor 只看图像，但 critic 额外拿到 14 维特权状态。
   这是视觉 RL 的标准做法、critic 不参与部署，但它确实不是"整个网络只有视觉"。
   要完全对称用 `--symmetric-critic`。

6. **训练收敛性已验证（视觉组）。** 视觉策略在 1970 万步内
   达到 `success = 100%`（确定性）/ `93.8%`（采样）。**但这不是终点**：训练轨迹显示它会
   反复塌陷（`model_400`、以及 `model_600` 之后彻底死亡），**最终能拿到什么完全取决于选对检查点**。

## 验证状态

`scripts/smoke_test.py` **18/18 通过**（视觉）与 **16/16 通过**（`--no-vision`），覆盖：
观测/动作维度、home 位姿与 URDF 一致、peg 垂直且初始不对准、
奖励取值范围、**两次 reset 逐位相同**（证明零域随机化）、
**脚本化下压能到 dp = 0.196 mm 且尖端停在木板顶面**、相机图像非常量且随位姿变化、
随机动作 120 步无 NaN。

训练吞吐实测：状态观测 1024 环境约 **10 万 steps/s**；视觉观测 512 环境约 **4500 steps/s**
（GPU 显存 21.7 GB / 24.5 GB，1024 环境会 OOM，故 `num_envs` 默认取 512）。

## 训练进展与评估结果

`Episode_Reward/parse` 在 2.6（只对准）与 10（插入）之间线性对应插入步数占比。

### 最终评估（`eval.py` 对冻结检查点的回放，论文口径）

| 组 | 选中的检查点 | **success %（确定性）** | success %（采样） | insert/step | return |
|---|---|---|---|---|---|
| **视觉** | `..._vision/model_550.pt` | **100.0 %**（64/64 回合） | 93.8 %（60/64） | 0.847 | 219.4 |
| **状态** | `..._state_long_resume/model_3875.pt` | **100.0 %**（128/128 回合） | **100.0 %**（128/128） | 0.982 | 276.9 |

`success %` = 回合中至少满足过一次 `dq<0.05 rad 且 dp<0.003 m` 的环境比例。
状态组两个动作源都是 128/128，`insert/step = 0.98` 表示**约 98% 的步数都处于插入状态**。

### 场景固定，但策略不是"背下来的"

任务没有域随机化，所以有一个必须回答的问题：这个 100% 究竟是"一个闭环视觉策略"，
还是"记住了一条固定轨迹、刚好打中训练时那个孔"？两者在固定场景下表现完全一样。

`_archive/scripts/probe_robustness.py` 把孔沿 x、y 同时平移后重测同一个冻结的检查点：

| 孔位偏移 | 0 mm | 2 mm | 5 mm | 10 mm | 20 mm | 40 mm |
|---|---|---|---|---|---|---|
| **视觉 `model_550`** success % | 93.8 | 96.9 | **100.0** | **100.0** | **100.0** | 93.8 |

**孔挪 2 cm 仍然 100%，挪 4 cm 仍有 93.8%。** 这说明它确实在用图像做闭环修正，
学到的是"把 peg 对准看到的孔"这个关系，而不是"从 home 走一条固定路径"。

反直觉的是：**训练位置（0 mm）反而是最差的一档**，向外挪 5–20 mm 反而升到 100%。

> 复现：`$PY _archive/scripts/probe_robustness.py --headless --checkpoint <视觉检查点> --offsets_mm 0,2,5,10,20,40`

注意这不等于"泛化能力好"：孔位只在 ±40 mm 内变化，且**姿态、高度、光照、场景布局都没变**。
真正的域随机化（genpih 的 Table 1：x±0.2 m、y±0.26 m、z 0–0.16 m、三轴 ±25°）没有测过。

### 塌陷：两组都反复出现，且能持续数百轮

**视觉组**（扫描 17 个候选点）：

```
model_200    0.3556  110.06     <- 开始会插了
model_350    0.6659  186.32
model_400    0.0241    8.93     <- 塌陷
model_450    0.6690  183.49     <- 自行恢复
model_550    0.8352  216.79     <- 最优，被选中
model_600    0.0000    0.20     <- 彻底死亡
model_799    0.0000    0.20     <- 最后一个检查点
```

**状态组**（扫描 14 个候选点）：

```
model_3875   0.9811  276.87     <- 被选中
model_3975   0.9811  270.66
model_4075   0.0000  104.39     <- 塌陷开始
model_4175   0.0000  104.50
model_4275   0.0000  104.14
model_4375   0.0000  110.65
model_4475   0.0000  107.85
model_4575   0.0000   95.40     <- 连续 500 轮完全死亡
model_4675   0.9738  285.75     <- 自行恢复
model_4775   0.9767  286.80     <- return 最高
model_4875   0.9734  279.81
model_4975   0.8515  261.76
model_5075   0.9436  285.42
model_5150   0.4521  184.75     <- critic 发散崩溃点
```

**取最后一个检查点在两组里都会拿到废模型**（视觉组 `return = 0.20`，状态组 `0.45` 且已开始崩），
而各自的最优检查点是 100%。这不是"保险起见"，它直接决定结论。
`eval.py --which best` 按 `insert/step` 选点，就是为这个准备的。

### 平台期是假象，不要过早下结论

状态组在第 988 次迭代时稳在 `parse = 2.5`（≈92% 步数"对准"、一次没插进去），
看起来完全像收敛到局部最优；它在第 **1482** 次迭代才突破，最终到 9.84。
我一度据此判断"卡死"，是错的——**给不够预算时，"学不会"和"还没到时候"长得一模一样。**

### 同代码同 seed，三次重训只有一次学会：重训失败 ≠ 代码回归

2026-10-11 用**完全相同的代码、超参和 `seed 42`** 重训了两次（一次 `noise_std_type="log"`，
一次改回 `"scalar"`），两次都失败：`Episode_Reward/parse` 从第 50 次迭代起就贴在 ≈2.0
（`2.6` = 全程"对准"、一次没插进去的躺平值），到 3000 次迭代结束纹丝不动。
而 2026-10-08 那一轮（同样 512 环境、同样命令、同样 seed）在第 200 次迭代前后就学会了。

**把成功那轮的检查点拿到今天的代码和今天的机器上回放，仍然是 100%：**

| 检查点 | 确定性 | 采样 |
|---|---|---|
| `..._vision/model_550.pt` | **100.0 %**（96/96） | 96.9 %（93/96） |
| `..._vision/model_500.pt` | **100.0 %**（64/64） | 90.6 %（58/64） |

两个检查点已备份到 [`best_policy/`](best_policy/)（被选中的最优是 **550**：确定性同为 100%，
但对带噪声的动作更抗噪 96.9% vs 90.6%；`--which best` 当时按 1 回合的 `insert/step` 排序，
500 与 550 只差 0.001，是**误选**）。

**所以这不是回归**：环境、相机、奖励、接触、网络全部完好。成功那轮之后被改动过的文件已逐个
提取核对（`peg_hole_env_cfg.py` 只改了默认 `num_envs` 1024→512、`cnn_policy.py` 只加了一个
加载辅助函数、`train.py` 只加了 guard 开关和 `safe_close`），**没有一行触及训练动力学**。

真正的解释是**随机性**：三轮从**第 0 次迭代起**数据就不一样（`Episode_Reward/dense` =
0.006018 / 0.005987 / 0.005994）——`seed` 并没有钉住这一轮跑什么，Isaac 的场景与渲染器构建
会消耗随机流（具体哪一处没有继续追）。三次里一次学会，另外两次从未离开平台期。


## 已知问题

- `scripts/smoke_test.py` 的脚本化插入段在视觉模式下很慢（每约 2 个物理步触发一次 tiled 渲染），
  脚本里已在该段临时挂起相机更新来规避。需要更快的验证直接用 `--no-vision`。
- **训练会被价值网络发散打断（根因未定位，与 genpih 记录的是同一现象）。**
  两次长跑都以相同方式死掉，崩溃前 `Mean value_function loss` 呈指数放大：

  | 运行 | 崩溃前的 value_function loss 序列 |
  |---|---|
  | 状态 1024 env × 3958 it | `2.7e6 → 2.25e12 → 1.66e18 → 1.99e24 → 4.6e30` |
  | 状态续训 × 5167 it | `410 → 1.4e14 → 2.0e25` |

  而同期 `Mean reward`、`Episode_Reward/parse`、surrogate loss、entropy、动作噪声、
  回合长度**全程平稳**——**是 critic 自己炸了，不是策略塌陷**。
  崩溃瞬间策略表现很好（`parse = 9.74`，约 99% 步数插入）。

  实际可用的缓解手段是**工程性**的：检查点存密一点（`save_interval`），
  崩溃后用 `--resume` 从最近的检查点接上——因为策略本身在崩溃时是好的。
  想真正压住它需要改动 rsl_rl 的 PPO（例如把 critic 的 MSE 换成 Huber，
  或者对 critic 单独降学习率），那超出了本仓库的范围。
