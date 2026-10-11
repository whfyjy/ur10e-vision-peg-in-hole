# _archive/code_en/ — 四个核心文件的英文注释原版

2026-10-11，应要求把项目里**四个核心文件**的注释和 docstring 翻成了中文。中文版是现在
生效的版本（在 `pegih_tasks/` 下），这里是改动**之前**的英文原文，只为留个底。

## 为什么留这一份

这个项目**不是 git 仓库**，所以覆盖掉就没有回退了。而英文注释在这两件事上仍然有用：

- **发 GitHub / 投简历**：机器人方向的公司很多代码库是全英文的，中文注释会砍掉一半读者。
- **改主意**：想换回英文、或者要中英对照，直接从这里取。

## 内容

| 文件 | 中文版位置 |
|---|---|
| `mdp.py` | `pegih_tasks/tasks/peg_hole/mdp.py` |
| `peg_hole_env_cfg.py` | `pegih_tasks/tasks/peg_hole/peg_hole_env_cfg.py` |
| `cnn_policy.py` | `pegih_tasks/tasks/peg_hole/cnn_policy.py` |
| `rsl_rl_ppo_cfg.py` | `pegih_tasks/config/rsl_rl_ppo_cfg.py` |

## 翻译的验证方式

改完之后用 AST 比较过（`ast.parse` 之后剥掉 docstring 再 dump）：
**四个文件的可执行代码完全相同，只有注释和 docstring 变了。**

这个验证比文本 diff 强，因为注释根本不出现在 AST 里，docstring 是可以精确剥掉的——
所以"剥完 docstring 后两份 AST 相同"等价于"一行代码都没动"。
另外 `scripts/smoke_test.py --headless --no-vision` 跑过，全部检查通过（退出码 0）。

## 没有翻译的部分

- **其余 17 个文件**（`scripts/` 下的工具脚本等）保持英文。它们是拿来**运行**的，不是拿来读的。
- **运行时报错信息**（`raise RuntimeError(...)`、`ValueError`）保持英文。
  Isaac Lab 自己抛的错全是英文，混着中文更难查，而且英文报错可以搜。
- **训练日志里的 `print`**（如 `Actor: in=30000 (image)`）保持英文，
  README 里有引用这些输出，改了就对不上了。

## 想恢复英文

```bash
cd "/home/whf/whf/ur10e-peg-in-hole rl"
cp _archive/code_en/mdp.py              pegih_tasks/tasks/peg_hole/mdp.py
cp _archive/code_en/peg_hole_env_cfg.py pegih_tasks/tasks/peg_hole/peg_hole_env_cfg.py
cp _archive/code_en/cnn_policy.py       pegih_tasks/tasks/peg_hole/cnn_policy.py
cp _archive/code_en/rsl_rl_ppo_cfg.py   pegih_tasks/config/rsl_rl_ppo_cfg.py
```
