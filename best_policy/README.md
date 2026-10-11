# best_policy/ — 冻结的视觉策略检查点

这个目录存放**已验证可用**的视觉 peg-in-hole 策略，从训练 run 目录里复制出来单独保存。
run 目录随时可能被归档或清理，而这两个文件是整个项目唯一的成果，所以不要只留在 `runs/` 里。

## 内容

| 文件 | 来源 | 确定性 success | 采样 success | insert/step | return |
|---|---|---|---|---|---|
| `model_550.pt` | `runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/` | **100.0 %**（96/96） | **96.9 %**（93/96） | 0.851 | 220.3 |
| `model_500.pt` | 同上 | **100.0 %**（64/64） | 90.6 %（58/64） | 0.839 | 219.7 |

**用 `model_550.pt`。** 两者的确定性成功率完全一样（都是 100%），但 550 对带噪声的动作
明显更抗噪（96.9% vs 90.6%）。`eval.py --which best` 当初选中的是 500 —— 因为扫描按
1 回合的 `insert/step` 排序，两者只差 0.001（0.8373 vs 0.8363），属于误选。

两个数字的测量口径：`--num_envs 32`，`--episodes 3`（550）/ `2`（500），
`success %` = 回合中至少满足过一次 `dq < 0.05 rad 且 dp < 0.003 m` 的环境比例。
两次评估都在 **2026-10-11** 用当时的代码重跑过，用来确认环境没有回归。

## 如果这两个 .pt 还没被复制过来

```bash
cd "/home/whf/whf/ur10e-peg-in-hole rl"
cp -n runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/model_500.pt best_policy/
cp -n runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/model_550.pt best_policy/
```

## 复现评估

```bash
PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
cd "/home/whf/whf/ur10e-peg-in-hole rl"
$PY scripts/eval.py --headless --checkpoint best_policy/model_550.pt \
    --num_envs 32 --episodes 3
```

## 背景：为什么只留了这两个

2026-10-08 那一轮（512 环境、`seed 42`、`noise_std_type="scalar"`、`learning_rate=1e-3`）
的完整生死曲线，用 `insert/step` 表示（确定性动作，每环境 1 回合）：

```
model_0   ~ 150   0.0000        什么都没学会
model_200         0.4213  ┐
model_250         0.5977  │
model_300         0.6555  │  学会
model_350         0.6259  ┘
model_400         0.0191  ← 中途塌陷一次（能自行恢复）
model_450         0.7242  ┐
model_500         0.8373  │
model_550         0.8363  ┘  最好，本目录保存的就是这一段
model_600  ~ 799  0.0000  ← critic 发散，彻底死亡
```

`model_600` 之后全部归零，不是策略被训练坏了，是 critic 发散把 actor 一起带走了：
`Loss/value_function` 从 4.3（第 550 次迭代）→ 91（600）→ 1.3e7（650）。
现象与 README「已知问题」里记的两次长跑崩溃是同一个，根因仍未定位。

**这也是为什么将来若要从零重训，`learning_rate` 应该先降到 3e-4**：1e-3 能训出好策略，
但守不住它；而且同代码同 seed 的三次重训只有一次学会（详见 README
「同代码同 seed，三次重训只有一次学会」一节），重训失败不一定代表代码有问题。
