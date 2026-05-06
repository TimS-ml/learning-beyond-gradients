# Heuristic System：软件在代谢中进化

> Jiayi Weng

我们以前常以为 heuristic 输给神经网络，是因为规则不够聪明。做完这组实验后，我更怀疑另一件事：很多时候，它们只是输给了维护成本。

我最早碰到这个问题，是在给 [EnvPool](https://github.com/sail-sg/envpool) 做环境正确性验证。随机策略太弱，很多环境一整局都碰不到关键奖励；失败时只能看到超时或者 0 分，很难判断是环境错了、封装错了，还是策略根本没走到有信息量的状态。给每个环境都训练一个神经网络又太重，训练脚本、依赖、版本、检查点，都会变成测试系统自己的负担。

于是问题变成了：

```text
能不能写一些便宜、可复现、比随机强很多的 heuristic，
专门把环境跑到有信息量的状态？
```

一开始我直接问 Codex：“写一个能解决 Breakout 的策略。”效果一般。低分没有解释力：它不知道是动作语义错了、状态检测错了、评测设置错了，还是策略结构本身不行。后来我把任务改成了另一种形式：别只交一个 `policy.py`，要维护完整闭环。

闭环大概长这样：

```text
探测动作和观测
-> 写状态检测器
-> 写策略
-> 跑完整回合
-> 记录 trials.jsonl 和 summary.csv
-> 生成视频或曲线
-> 看失败模式
-> 改策略
-> 简化代码并做回归
```

当时我还没给这件事起名字，只觉得任务的形状变了：最后产出的东西从一个策略文件，变成了一套还能继续改的实验系统。它有探测器，有记录，有回放，有失败模式，也有下一轮该怎么改的线索。

## 1. Breakout：没有训练，也能满分

Breakout 表面上是几何问题：球在哪里，挡板在哪里，球撞墙以后会落到哪里。真正麻烦的是后半段：策略能一直接到球，却打不到新砖，分数卡在一个稳定循环里。

Codex 第一轮没有急着写最终策略。它先确认动作空间和观测形状，再从 RGB 画面里找挡板、球、砖块颜色，然后用这些图像标签去扫 128 个 RAM 字节。早期实验记录大概长这样：

```text
trial_name                 score   cumulative_env_steps   note
shape_action_probe          -      32                     inspect obs/info/action
ram_byte_corr_probe_v1      -      5,032                  correlate RAM bytes
ram_fit_action_probe_v2     -      9,532                  action 2=right, 3=left
baseline_v0                99      16,303                 initial RAM intercept
tunnel0_v1                387      43,303                 no tunnel offset
```

`387` 是第一个很容易骗过人的局部高分。策略已经能稳定接球，但它只是把球送进一个周期：不会死，也不会继续清砖。如果是人手写，很容易继续调“接球精度”。Codex 看了视频和最后几十步轨迹后发现，真正的问题是球路缺少扰动。

<video controls src="heuristic_breakout_score387_tunnel0_render210x160.mp4" width="360"></video>

第一个关键机制是打破循环：如果连续很久没有奖励，就在预测落点上周期性加偏移，把球从局部循环里打出去。这一改把分数从 `387` 推到 `507`。

```python
if steps_since_reward >= stuck_trigger_steps:
    phase = stuck_offset_index % 4
    if phase == 0:
        offset = +stuck_offset_px
    elif phase == 1:
        offset = -stuck_offset_px
    elif phase == 2:
        offset = +0.5 * stuck_offset_px
    else:
        offset = -0.5 * stuck_offset_px
else:
    offset = 0.0
```

后来又遇到另一个失败模式：高速低位球如果按普通截距追，挡板会被过度前视带偏。Codex 加了 `fast_low_ball_lead_steps=3`，分数从 `507` 跳到 `839`。

```python
if vy > 0.1 and ball_y <= paddle_y:
    steps_to_paddle = max((paddle_y - ball_y) / vy, 0.0)
    intercept_x = reflect_position(ball_x + vx * steps_to_paddle)
    target_x = intercept_x + stuck_offset
elif vy >= fast_ball_min_vy:
    target_x = ball_x + fast_low_ball_lead_steps * vx
else:
    target_x = ball_x + chase_lead_steps * vx
```

从 `839` 到 `864`，最像在照料一个会变复杂的系统。Codex 试了死区、发球偏移、卡住偏移、砖块平衡偏置、前视步数，很多方向都没用。最后起作用的是一个后期条件：分数超过第一面墙以后，卡住偏移只在离挡板还远的时候生效；快接球时把偏移逐步收掉，不然最后几块砖阶段会自己把挡板带偏。同时它加了一个很小的挡板漂移补偿，用来补动作和挡板位置之间的一步延迟。

```python
if score >= 432 and stuck_release_horizon_steps > 0:
    release_ratio = clip(steps_to_paddle / stuck_release_horizon_steps, 0.0, 1.0)
    offset *= release_ratio

if score >= 432 and ball_y >= 170 and last_action == RIGHT:
    control_paddle_x = paddle_x + 2.0
elif score >= 432 and ball_y >= 170 and last_action == LEFT:
    control_paddle_x = paddle_x - 2.0
```

<video controls src="heuristic_breakout_ci3985ae2_score864_render210x160.mp4" width="360"></video>

最终 RAM 默认配置三局验证是 `864 / 864 / 864`。后面 Codex 又把同一套几何控制迁移回纯图像输入：不用 RAM，只用 RGB 分割找挡板、球和砖块平衡。纯图像版本先是 `310`，然后 `428`，最后把后期“卡住偏移逐步收掉”的阈值放低到全程生效，7 个策略本地回合后第一次到 `864`，对应图里的 `14,504` 个策略本地环境步。

![Breakout 样本效率](heuristic_breakout_sample_efficiency.png)

这段不能写成“纯图像从零 14.5K 步到满分”。实际过程是：Codex 先在 RAM 版本里摸出了几何控制、打破循环、后期收偏移这些结构；等结构稳定以后，再把状态读取层从 RAM 换成 RGB 检测器。纯图像的 `14.5K` 是迁移预算。

这件事给出的信号不止分数。启发式策略一旦被写成可维护的软件结构，就能替换输入层、重用控制逻辑、继续回归测试，不必每换一种观测就重新训练。它像一个小器官：输入层可以换，控制逻辑可以留，失败模式可以继续被吸收。

完整策略代码在 [`heuristic_breakout.py`](heuristic_breakout.py)，实验记录在 [`heuristic_breakout_trials_summary.csv`](heuristic_breakout_trials_summary.csv)。

## 2. 软件开始有代谢

Breakout 之后，我才意识到，Codex 维护的对象已经从一个 heuristic program，变成了一个 **heuristic system**。

单独看，“球在左边就把挡板往左移”是一条 heuristic。真正让它变成系统的，是后面那套配套机制：怎样检测球和挡板，怎样确认动作含义，怎样发现球路卡住了，怎样复现某个 `387` 分或 `864` 分，怎样记录一次改动为什么有效，下一轮又该从哪里继续。

分界看规则数量不够，还要看反馈能不能进入下一轮运行。只执行固定规则的系统，仍然是 heuristic program；历史结果一旦能改写之后的状态表示、行动逻辑、评估方式或记忆，它就进入了本文讨论的 heuristic system。

这就是我说“软件开始有代谢”的意思。这句话没有玄学成分，也没有把软件说成生命。它指向的是一个很工程的事实：反馈不再只供人类事后复盘，也可以被 agent 消化成代码、配置、测试和记忆的变化。

我想表达的其实是一个很朴素的关系：

![空气、食物和反馈：三种代谢入口](ig_0c2dd0d2f07176560169faa8a1edd081968d1579bca5cba35f.png)

空气让轮胎充盈，食物让孩子长大。coding agent 把反馈接进软件的更新通路，让它慢慢长成新的结构。

它大概是这样一个闭环：

```text
观测 -> 状态表示 -> 程序策略 -> 动作 -> 反馈 -> 记忆 -> 更新
```

测试失败可以变成回归测试，日志异常可以变成新的状态检测器，实验结果可以变成策略版本，人工经验可以变成 memory 和下一轮 patch。只要这些更新能留下来，影响下一轮运行，系统就开始一边执行规则，一边吸收反馈。

先把这个直觉留住：agent 照料的是一个会变复杂、会留下历史、会继续被改的软件系统。这个直觉后面还会带出另一个问题：agent 能照料的复杂度有上限，而且这个上限受反馈、测试、模块化和工具质量影响。

## 3. Ant 和 HalfCheetah：没有训练，也能走起来

Ant 对我更意外。Breakout 的几何结构还比较直观；Ant 是连续控制，动作是 8 个关节，失败模式也从“球没接到”变成了更复杂的身体动力学问题。

我没有一开始就指定“用 CPG”或“用 MPC”。要求只有几条：别训练神经网络，能本地复现，每轮实验留下记录，继续把分数往上推。Codex 先读 EnvPool/Gymnasium 的 Ant 观测和回报，确认动作顺序、根部速度、躯干朝向、关节位置和关节速度，然后自己提出第一版节律步态。

第一版是四腿相位振荡器：左右腿反相，髋关节和踝关节跟踪正弦目标角，动作由 PD 控制器给出。它不优雅，但一上来就比随机强很多，5 个随机种子的平均分是 `2291`。

```python
leg_phase = warp_phase(phase + LEG_PHASE, stance_duty(vx))
stance = leg_phase < pi

hip_wave = HIP_BIAS + stance_or_swing_scale * (
    HIP_AMP * sin(leg_phase)
    + HIP_H2_AMP * sin(2 * leg_phase + HIP_H2_PHASE)
    + HIP_H3_AMP * sin(3 * leg_phase + HIP_H3_PHASE)
)

action[0::2] = KP * (
    HIP_SIGN * hip_wave
    + HEADING_AXIS * (YAW_GAIN * yaw + YAW_RATE_GAIN * yaw_rate)
    - q[0::2]
)
action[1::2] = KP * (ANKLE_SIGN * (ankle_wave + balance) - q[1::2])
```

后面的早期迭代很像调一个真实控制器：先加偏航反馈到 `2718`，再调相位速度、髋/踝幅度、偏航角速度增益到 `3025`，然后加二阶/三阶谐波到 `3162`。Codex 也试过大范围参数搜索，但结果没有稳定超过当前节律策略，于是停止扩大搜索预算，转向另一种表示。

跃迁来自残差模型预测规划，也就是代码里写的 MPC。粗略讲，MPC 是“边走边想一小段未来”：保留节律步态作为基础反射，每个真实环境步在本地 MuJoCo 模型里采样几十条小的残差动作序列，打分后只执行第一个残差动作；下一步重新看状态、重新规划，并把上一轮没执行完的计划作为热启动。

这样每一步都不用从零规划 8 个关节怎么动。策略先有一个稳定步态，再用短视窗模型规划去修正它。

```python
base = cpg_action(phase, q, dq, roll, pitch, yaw, rates, contacts, vx)

best_plan = previous_plan.copy()
best_obj = rollout_objective(obs, best_plan)
for _ in range(CANDIDATES - 1):
    residuals = clip(
        best_plan + rng.normal(0.0, MPC_SIGMA, size=(HORIZON, 8)),
        -MPC_CLIP,
        MPC_CLIP,
    )
    residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
    obj = rollout_objective(obs, residuals)
    if obj > best_obj:
        best_obj = obj
        best_plan = residuals

plan[:-1] = PLAN_DECAY * best_plan[1:]
return clip(base + best_plan[0], -1.0, 1.0)
```

这些结构主要是在迭代里长出来的。节律策略到 `3162` 后，Codex 自己写了一个基于模型的残差规划器。第一版视窗长度 6、32 个候选、小残差，就从 `3135` 提到 `3635`。接着它继续扩展这套控制器：

```text
trial_name                               score_mean   cumulative_env_steps   note
ant_lr_cpgpd_v1                         2291.9       5,000                  左右腿反相 CPG + PD
ant_yawaxis_grid_v2                     2857.9       20,000                 偏航反馈 + 重调参数
ant_h3_428_v1                           3162.0       50,000                 二阶/三阶谐波
ant_mpc_residual_v1_ep1                 3635.5       62,000                 视窗=6，候选=32
ant_mpc_residual_cfg4_eval5             3964.7       67,000                 视窗=8，候选=48
ant_mpc_residual_cand07_eval5           4647.1       73,000                 围绕 MPC 配置做局部搜索
ant_mpc_residual_narrow04_eval5         4871.3       79,000                 降低 z 目标，增大 kp/候选数
ant_mpc_residual_warm02_eval5           5165.2       85,000                 热启动残差计划
ant_mpc_fast065x060_sigma008_clip012    5759.4       95,000                 更快步态 + 更大残差
ant_mpc_term001_ep1                     6054.5       100,000                终端速度代价
ant_mpc_default_adaptive_ep1            6146.2       106,300                速度自适应相位 + 支撑期
```

![Ant 样本效率](heuristic_ant_sample_efficiency.png)

<video controls src="heuristic_ant_mpc_default_6146_render480.mp4" width="480"></video>

完整实验脚本在 [`heuristic_ant.py`](heuristic_ant.py)，抽出的最小可调用策略在 [`heuristic_ant_min_policy.py`](heuristic_ant_min_policy.py)，实验记录在 [`heuristic_ant_trials_summary.csv`](heuristic_ant_trials_summary.csv)。

Ant 的信号和 Breakout 不一样。Breakout 是发现几何以后，输入规整化带来了很夸张的样本效率；Ant 说明另一件事：启发式系统能一路长复杂，同时仍然保持可检查、可复现、可继续修改。到最后，策略里有振荡器相位、支撑期比例、速度自适应、滚转/俯仰/偏航反馈、脚部接触、短视窗模型内展开、残差平滑、终端速度代价、热启动计划衰减。人类当然能写其中一两个模块，但要在短时间内同时照顾实验记录、代码、视频和失败方向，难度完全不同。

HalfCheetah 是同一类证据的另一个点。我重新跑了 `mpc-staged-tree-asym-pd-cpg` 的 5 局复测，seeds `100..104` 的结果是均值 `11836.7`、最小值 `11735.0`、最大值 `12041.2`。策略靠的是可解释的步态/姿态规则和在线 staged-tree MPC：先用 CPG/PD 形成高分步态，再用短视窗模型评分和 staged swing-amplitude schedule 修正动作。对应脚本是 [`heuristic_halfcheetah_v5.py`](heuristic_halfcheetah_v5.py)，迭代记录在 [`heuristic_halfcheetah_v5_log.md`](heuristic_halfcheetah_v5_log.md)。

这条证据的意思很窄：连续控制当然不用整体回到手写规则。但一旦有 agent 愿意长期维护，复杂程序策略也不一定立刻塌成一团。它可以被拆成模块，被记录，被回归，被继续长。

## 4. Atari57：无人值守，也能长出策略

Breakout 和 Ant 都是单点故事。Atari57 则是规模化检验：同一套 Codex 工作流直接扔到整套 Atari57 上，每个环境同时跑 `ram` 和 `native_obs` 两种输入，每种输入跑 3 个独立重复。总共是：

```text
57 个游戏 x 2 种输入 x 3 次运行 = 342 条 coding-agent 搜索轨迹
```

这组实验没有人在旁边一点点提示。我用 Codex CLI 批量启动 `gpt-5.4 xhigh`，每个 agent 拿到同一个模板和不同的 `ENV_ID / OBS_MODE / REPEAT_INDEX`，然后自己执行到停止。每个 run 都要写 `policy.py`、`trials.jsonl`、`summary.csv`、`sample_efficiency.png` 和 `README.md`。

完整提示词放在 [`atari57_prompt_template.txt`](atari57_prompt_template.txt)。主要约束是：

```text
目标：只针对一个 EnvPool Atari 环境，在给定 OBS_MODE 下自己设计并迭代手写 heuristic policy。

硬约束：
- 不训练神经网络。
- 不读环境源码、测试、ROM 细节或隐藏状态。
- native_obs 模式只能用 reset/step 返回的原生 obs。
- ram 模式可以用 info["ram"]。
- Atari 初始化参数固定，包括 frame_skip=1、reward_clip=False、sticky action=0。
- 所有实际 step 过环境的 probe/debug/trial 都必须计入 cumulative_env_steps。

输出文件：
- policy.py：当前最好且尽量简化的 heuristic。
- trials.jsonl：每次 trial 的分数、环境步、配置、备注。
- summary.csv：从 trials 汇总。
- sample_efficiency.png：按环境步和 episode 画分数曲线。
- README.md：最好分数、复现命令、失败方向、停止原因。
```

这张图里的横轴从 `10^4` 环境步开始，因为前面的部分基本看不出变化；纵轴是 Atari human-normalized score，也就是 HNS。Codex 曲线按 Atari 常见的中位数口径画：每个游戏先把 3 次独立运行取中位数，再在 57 个游戏上取中位数。这种口径不会被少数特别高分的游戏拉爆，主要看覆盖率。

![Atari57 样本效率对比 OpenRL Benchmark](atari57_openrl_sample_efficiency_context.png)

在完全无人工介入的批量运行里，`native_obs` 到 `9.7M` 步附近是 `0.81`，`ram` 是 `0.59`。同一张图里，[OpenRL Benchmark](https://arxiv.org/abs/2402.03046) 保存的 PPO2 / CleanRL EnvPool PPO median HNS 曲线到 `10M` 步大约是 `0.88 / 0.92`。

这里比较的是环境交互效率；coding agent 读日志、写代码和看视频的开销没有折算进总计算成本。

读这张图时要收窄比较对象。它给出的信号很具体：一个还很粗糙的 coding agent 批量流程，在完全不看中途结果的情况下，已经能把 Atari57 的中位数推进到接近这些基线的区间。至于“heuristic 全面超过强化学习”这种说法，这张图支撑不了。

聚合曲线会把差异压到一个中位数里，所以我又把 57 个游戏逐个画出来。Atari 原始回报跨游戏没有可比性，这里仍然用每个游戏自己的 HNS；虚线 `1.0` 表示人类分数。

![Atari57 每个游戏 HNS 对比](atari57_per_game_hns_comparison.png)

这张图能看出两件事。第一，重合是有的：Breakout、Krull、DoubleDunk、Boxing、DemonAttack 这些游戏里，heuristic 和强化学习基线都能拿到明显高于人类基线的分数。第二，差异也很大：heuristic 在 Asterix、Jamesbond、Centipede、Bowling、Skiing、Tennis 这类游戏上相对突出；PPO 在 Atlantis、VideoPinball、UpNDown、Assault、RoadRunner、StarGunner 上明显强很多。

这张分布图比一个中位数有信息。heuristic system 并没有均匀地学会“玩 Atari”。它在某些游戏里很快写出了有效机制，在另一些游戏里还没有找到合适的状态表示或长期策略。

我觉得 Atari57 最有意思的地方，是样本效率的来源变了。传统神经网络 Atari 学习要在每个环境里从高维输入重新学表示、信用分配和动作含义；Codex 做的是把环境拆成可维护的小程序系统：射击游戏的瞄准/躲避，接球游戏的反弹，躲避游戏的位置规则，环境包装器细节，以及每个环境自己的失败实验记录。最后留下来的是一批被生成、验证和修改过的局部启发式策略；通用神经网络没有出现在这个过程里。

到这里，游戏分数反而退到了后面。更有意思的是：这套流程已经能批量生成、验证和修改局部策略，软件开始有了一个足够通用、能把反馈写回代码、测试、日志和 memory 的维护者。

## 5. 反例：Montezuma

有些环境不适合普通反应式启发式策略。Montezuma's Revenge 是典型例子。

之前那轮单独搜 Montezuma 的状态图搜索能把钥匙距离从 `72` 推到 `28`，但奖励仍然是 `0`。后面 Atari57 的纯图像批量实验里，有一条无人值守 Codex run 到了 `400.0` 分：修复后的最佳回放是 `repair_replay_r1_t19734`，seed 是 `10001`，用了 `1769` 个环境步，本质是一条 `86` 个宏动作组成的开环路线。

我把恢复出来的 [policy](heuristic_montezuma_400_policy.py)、[宏动作](heuristic_montezuma_400_macros.json) 和 [视频](montezuma_400_render_seed10001_h264.mp4) 放在了 repo 里。

<video controls src="montezuma_400_render_seed10001_h264.mp4" width="360"></video>

Montezuma 暴露的是表达力问题。普通 `policy.py` 状态机很难装下这类路线：动作必须对齐时机，失败后要能恢复，中间状态还要能重新进入计划。有些环境需要可组合宏动作、可恢复搜索状态，甚至需要一种比普通 `if else` 更适合长期规划的程序结构。

这类失败对 heuristic system 很有价值。它告诉我们边界在哪里，也提示下一层抽象大概该长什么样。有代谢的系统也要看消化结构；有些反馈需要新的表示和新的程序形态，才进得了系统。

## 6. Heuristic System：从策略到系统

前面一直用游戏，是因为反馈清楚、分数好量化。真正离读者更近的，其实是代码仓库。

### 6.1 定义

到了这里，光说“闭环”已经不够了。要把它从游戏搬到别的软件系统里，我需要先说清楚它到底由哪些部分组成：

```text
HS = (O, Z, P, A, R, M, U)
```

`O / observation` 是系统看到的东西：图像、RAM、状态向量、日志、请求特征、代码 diff、监控指标、用户反馈。

`Z / state` 是内部状态表示：球和挡板的位置、机器人姿态、服务负载、PR 风险区域、失败模式、缓存、belief 或风险标签。

`P / policy or program` 是行动逻辑：条件分支、阈值、状态机、宏动作、控制器、路由策略、回退策略、测试选择策略。

`A / action` 是真正落到外部世界里的动作：Atari 按键、机器人关节力矩、一次路由切换、一组被选择的测试、一个代码 patch、一条回复，或者一次 memory 写入。

`R / feedback` 是评价信号：回报、测试结果、延迟、错误率、成本、人工标注、线上指标、回放评分。它不一定是单个标量，但要能给更新提供方向。

`M / memory` 保存历史：策略版本、配置、实验结果、失败原因、回放材料、回滚点。没有这层记忆，agent 跑二十轮以后很容易原地打转。

`U / update` 根据 `R` 和 `M` 修改 `Z`、`P`，有时也修改 `A` 的接口、`R` 的评估方式和 `M` 的组织方式。改动要留下来，写进代码、配置、测试、memory 或策略版本里，影响下一轮运行。在本文里，coding agent 通常就是 `U` 的实现之一；把它接进来以后，HS 的边界会扩展到这条更新通路，执行中的策略代码只是其中一部分。

这不保证系统每轮都会变强。更新过程还需要回归或选择机制，把明显变差的改动筛掉。没有筛选，系统只是漂移。

### 6.2 代码仓库维护

有经验的工程师做代码审查时，会用很多启发式判断：

```text
这个 diff 改了鉴权路径，风险高。
这个测试失败像不稳定测试，先查主干最近是否也失败。
这个函数被多个服务复用，不能轻易改返回语义。
这个改动碰到启动路径，可能影响导入时间。
这个 PR 太大，应该先拆出机械改动。
```

这些判断没有形式化证明，也没有经过端到端模型训练。它们来自工程经验，又会被 CI、线上事故、历史提交、代码审查评论和测试覆盖不断校正。

沿用上面七个部分，代码仓库可以写成：

```text
O / observation = diff、CI 日志、代码索引、历史事故、监控指标
Z / state       = 风险区域、依赖图、owner、失败模式、测试覆盖
P / policy      = review 规则、测试选择、拆 PR 策略、回滚策略
A / action      = review 评论、测试运行、代码 patch、PR 拆分、回滚动作
R / feedback    = CI 结果、线上指标、代码审查反馈、缺陷回归率
M / memory      = PR 历史、失败记录、决策理由、修复路径
U / update      = coding agent 修改检查脚本、测试策略、文档和代码
```

这时 coding agent 做的远不止“帮我写代码”。它在维护一个代码仓库的启发式控制系统：哪些路径危险，哪些测试有信息量，哪些失败像历史问题，哪些规则已经过时，都能被记录、执行和修改。

软件工程已经给这种系统准备了很成熟的材料。单元测试、集成测试、黄金用例、回归测试、lint、类型检查、CI、性能基线，这些东西既是质量检查，也是在给代码库建立协议。协议说清楚一个函数对外承诺什么，旧缺陷不能怎样复发，某个库的边界在哪里，哪些路径改动以后必须多跑测试。

一旦协议够完整，内部实现就能大胆替换。比如一个解析器只要继续通过语法黄金用例和错误恢复测试，它内部是手写递归下降、解析器生成器，还是一套被 agent 重构过的状态机，都没那么要紧。真正要紧的是边界有没有覆盖住下游依赖的行为。

测试、lint、CI 本身还不够。它们开始像本文说的 HS，是在 coding agent 会根据失败记录和历史经验改测试、改检查脚本、改风险标签、改拆 PR 策略以后。agent 最擅长在明确反馈下改代码；测试越像可执行协议，agent 就越能在边界内部搜索实现。反过来，如果测试写得很差，agent 只会更快地利用漏洞，把系统推向错误的局部最优。坏测试就是坏奖励信号，只是穿了软件工程的衣服。

这里真正倒过来的，是判断代码值不值得长期拥有的标准：

```text
agent 改变了写代码的速度，
也改变了哪些代码值得被长期拥有。
```

过去很多规则层“不值得写”，真正卡住它们的常常是后续维护：写完以后没人养。coding agent 出现后，一些原来太细、太烦、太依赖失败记录的局部规则，可能重新值得拥有。测试、日志、patch 和 memory 也有机会从工程配套变成软件的学习器官。

### 6.3 本来就靠 heuristic 活着的软件

更典型的例子，是那些从一开始就没法便宜地求出全局最优的软件系统。

调度系统就是这样。集群调度、作业排队、GPU 分配、Kubernetes 放置策略，都要在优先级、公平性、数据就近性、抢占、冷启动、失败重试和尾延迟之间做取舍。组合优化也一样：车辆路径、装箱、排班、仓储拣货、广告预算分配，解空间很快爆炸，最后一定会长出贪心、局部搜索、beam search、修复启发式和一堆剪枝规则。

流量分流和灰度发布也比把 `1%` 变成 `10%` 复杂得多。真正的策略要同时看延迟、错误率、容量、成本、用户一致性和回滚风险。数据库查询规划器和编译器优化器更是老牌 heuristic 系统：连接顺序、索引选择、内联阈值、循环展开、优化 pass 的顺序，背后都是代价模型、阈值、剪枝和回归基准。

这些系统的共同点是：heuristic 本来就在核心路径上，只是长期维护很贵。一次慢查询修复，可能会变成新的连接代价规则和回归基准；某个阈值在今天的流量上很好，下个月分布变了就开始振荡；一组局部补丁堆久了，没人敢删，也没人敢改。

HS 视角有意思的地方在这里：失败作业、慢查询、灰度事故、基准退化、调度回放和线上日志，都可以变成更新材料。coding agent 可以读失败样本，改打分函数、兜底策略、剪枝规则、测试集和回放脚本，再把有效经验写回策略和文档。过去这些 heuristic 只能靠少数专家长期手养；现在它们有机会变成能持续吸收反馈的软件系统。

### 6.4 机器人

Ant 让人很自然想到机器人，但这里最容易讲过头。

仿真里能让 Codex 几万步、几百万步试错，摔了也只是重置；真实世界里每次试错都有时间、硬件磨损、安全、场地复位和传感器漂移成本。所以这套闭环不能直接搬到真机上，更不能像在仿真里那样盲跑。

我更愿意把它看成一种 **hierarchical HS**，也就是一组层级化的小闭环。低层负责局部安全和低延迟控制，中层负责肢体协调和接触，高层负责任务、恢复和长期记忆。

```text
关节级 HS -> 肢体级 HS -> 全身平衡 HS -> 任务级 HS
```

一个关节可以是 HS；几个关节拼成肢体 HS；肢体再拼成身体和全身 HS。

![从关节级 HS 到全身级 HS](ig_0c2dd0d2f07176560169fab20d5a708196b02a2dd41def1241.png)

关节级 HS 看到编码器、力矩、电流、IMU 和接触传感器，把它们整理成误差、速度、负载、危险标志，然后输出目标位置、目标速度或力矩命令。它的反馈是过载、打滑、能耗、安全违规；更新过程主要改增益、阈值、保护规则和测试。这里的更新必须很保守，不能让 agent 在真机上随便改安全边界。

肢体级和全身级 HS 可以负责步态、接触策略、摔倒恢复、身体姿态和能耗。任务级 HS 再往上，负责“先靠近杯子，再调整夹爪，再失败恢复”这种长一点的流程。这样看，机器人更像很多局部 HS 通过层级关系串起来；一堆“HS 关节”简单堆在一起，还到不了这个层级。

最有想象力的版本大概是这样：一个新机器人上电以后，coding agent 接上仿真、日志、视频、传感器流和回归测试，先让关节级 HS 在安全边界里自己找稳态，再让肢体级 HS 长出协调，再让全身级 HS 长出站立、恢复和平衡，最后才往任务级动作走。agent 像插进系统的更新管线：它持续把电、算力、token、失败视频和测试结果喂进系统，把反馈改写成代码、参数、保护规则和 memory。进化发生在里面：各层 HS 的状态表示、控制策略和恢复动作一点点变形，直到更适合这具身体。

如果这条路走通，过去像婴儿几个月学站、学走、学摔倒恢复的过程，未来也许会被仿真、回放和 agent 维护压缩成一小时的软件代谢过程。这个画面的主角是一套层级化的 heuristic system：它持续被喂反馈，持续筛掉坏变异，持续固化好动作；大模型直接“懂得走路”只占这张图的一小角。这里的“代谢”是反馈被消化成系统改动，“进化”是系统自身在这些改动里长出更合适的形态。

操作任务会麻烦很多。叠衣服、整理线缆、打开软包装这种任务，难点并不限于“机器人关节怎么动”，还在物体本身的状态：布料形变、遮挡、接触历史、摩擦、皱褶、目标形状。光靠调几个关节相位解决不了这类问题；如果没有好的感知表示、可恢复的动作原语和足够真实的仿真，coding agent 也会在错误的状态变量上写出很脆的规则。

现实一点的形态可能是混合系统：仿真和离线数据先用来生成、筛掉候选启发式；真机上只做小步、安全、有护栏的验证；神经网络负责感知、物体状态估计和长程价值；hierarchical HS 负责低延迟反射、安全约束、局部恢复、任务分解和测试判据；coding agent 负责维护接口、检测、失败处理和回归测试。

这样讲，重点就从“用启发式手写叠衣服”转向另一件事：把一些能写成程序的局部规律从端到端训练里拆出来。

### 6.5 持续学习与耦合复杂度

到这里，真正值得研究的问题从“heuristic 能不能写”转向“HS 能被维护到多复杂”。

把 HS 放回机器学习语境里，最接近的概念可能是 continual learning。online learning 更关心数据来了以后是不是立刻更新；continual learning 更关心长期更新里怎样不忘旧能力。HS 关心的是另一层：新反馈进来以后，被改写的对象可以从参数扩展到状态检测器、阈值、宏动作、测试、日志解析、memory 的组织方式，甚至是下一轮怎样收集反馈。只要这些改动能留下来，并改变下一轮运行，软件就在持续学习。

这也把“防遗忘”说得更具体。神经网络的 continual learning 担心 catastrophic forgetting；HS 也会忘，只是忘法更工程：新规则修好一个失败模式，同时破坏旧场景；一段 memory 把 agent 反复带到错误方向；测试没覆盖的边界被一次 patch 悄悄改掉。回归测试、固定回放、实验记录、版本差异和简化阶段，就是 HS 的防遗忘机制。它们让反馈能累积下来，下一轮不用重新靠直觉。

于是问题可以说得更尖一点：如果能刻画 heuristic 的复杂程度，就能问 agent 到底能维护多复杂的 HS。我现在更想把这个量叫做 **耦合复杂度**：一次更新必须同时照顾多少互相牵连的状态、规则、测试、反馈和历史。代码行数和规则数量只是表面；真正吃上下文的是交互面有多大。

朝代码这一侧看，耦合复杂度受模块边界、接口稳定性、测试覆盖、固定回放、日志可观测性、反馈延迟、回滚成本和状态可复现性限制。好的模块化会把全局耦合切成局部耦合；好的测试让 agent 不必每次都在脑子里模拟整个系统。

朝 coding agent 这一侧看，能压住多少耦合复杂度，取决于模型能力、上下文长度、memory 质量、工具质量和实验速度。更强的模型能同时处理更多相互作用；更长的上下文让它少丢线索；memory 把跨轮经验留下；搜索、运行、定位、回放这些工具把一部分认知负担搬到外部。任何一环差，可维护的耦合复杂度上限都会下降。

把这两侧放在一起，可以得到一组更像研究假设的判断：

```text
反馈越清楚，单位 agent 智力能维护的耦合复杂度越高。
同等工具和反馈下，模型能力越强，能处理的耦合复杂度越高。
模块化、测试和回放会把一部分耦合复杂度转移到环境里。
memory 和工具会提高 agent 的有效上下文。
只增长不压缩的 HS，会让耦合复杂度持续上升，直到超过维护能力。
```

这也解释了为什么 HS 和软件工程的老哲学贴得很近。模块化的价值在于降低耦合复杂度。测试从事后验收变成可执行反馈；测试、回放、CI 和基准测试都是反馈通道。日志、链路追踪和回放是软件的感官，没有它们，失败进不了系统，也就没有代谢。重构负责压缩学习历史，把一堆局部补丁折回更简单的表示。技术债则是没有压缩的反馈残留：每次失败都修成一条新规则，短期有效，长期会让耦合复杂度爆炸。

Breakout 能走到 `864`，有规则简单的一面，也有失败可以视频回放、局部复现、回归验证的一面。Ant 复杂得多，但结构分层，反馈密集，策略还能被拆成节律、姿态、接触、残差规划这些模块。Montezuma 则提示另一条边界：长程时序、可恢复状态和宏动作组合会把耦合复杂度一下子推高，普通 `policy.py` 很快不够用。

这可能是 HS 以后最值得研究的量：在给定模型、上下文、memory、工具、测试协议和反馈质量下，系统能稳定承载多少耦合复杂度。如果这条曲线能被测出来，维护成本就会从比喻变成可以比较的对象。

## 7. 限制

第一，结果依赖当前 agent 能力。这里主要讨论 Codex 5.4 `xhigh`。换模型、换推理预算、换工具、换提示词，分布可能会变。本文是现象报告，算不上完整 benchmark。

第二，单点实验有人在环。Breakout 和 Ant 并非全自动基准；我会看结果，决定让 Codex 往哪里继续。Atari57 才是无人值守批量运行。把这两类证据混在一起，会让结论变糊。

第三，比较计算量要小心。Codex 写代码、读日志和看视频的计算量没有按神经网络训练计算量计入；Ant 和 HalfCheetah 里的 MPC 还会做本地模型内展开。这里的环境步数主要说明真实环境交互少，不等于总计算便宜。

第四，能更新不等于自动变好。agent 生成的 heuristic system 可能越写越复杂，最后变成不好维护的代码泥潭。实验记录、回归测试和简化阶段不能省，它们是防止系统塌掉的主要机制。

第五，不要把这条路线讲成强化学习的替代品。神经网络在复杂视觉、跨状态泛化、长程价值估计上仍然有明显优势。合理方向仍然是组合：神经网络负责感知、泛化和长程策略；heuristic system 负责局部反射、安全约束、回退策略、测试准则和可解释调试。

这些边界把这件事放回它真正成立的位置。软件开始有代谢，不代表它可以随便自己改自己；它需要测试、权限、回滚、可解释记录、人工审核和沙箱。没有这些东西，代谢就会变成乱长。

## 8. 结论

这篇文章从一个很小的测试需求开始：能不能写一些便宜、可复现、比随机强很多的策略，把环境跑到有信息量的状态。走到最后，分数退到后面；更重要的是，当 coding agent 能持续读取失败、改代码、补测试、写记录、做回归时，启发式策略会从一段临时规则变成一个可以被照料的软件系统。

Breakout 给了最清楚的起点。`387 -> 507 -> 839 -> 864` 的满分来自实验闭环对失败的持续吸收：动作语义、状态检测、卡住循环、低位高速球、后期偏移释放、漂移补偿，每一步都被视频、日志和回归测试固定下来。Ant 和 HalfCheetah 说明复杂度也能被照料；Atari57 说明这套工作流可以批量跑；Montezuma 则提醒我们，长程时序和可恢复状态会很快推高耦合复杂度。

所以这篇文章真正想保住的判断是：过去很多 heuristic 看起来没有前途，核心问题经常落在维护成本上。没人愿意长期照料几百条局部规则、失败记录、测试边界和状态表示。coding agent 改变的是这条维护成本曲线。规则、测试、日志、memory 和 patch 原来只是散落的工程材料，现在开始可以组成一个会持续更新的 heuristic system。

这也让 continual learning 换了一种软件形态。学习可以发生在参数之外：新反馈进入系统以后，如果它能改写状态检测器、测试、宏动作、memory、日志解析和下一轮策略，软件本身也在学习。软件工程那些老原则也因此有了新含义：模块化降低耦合复杂度，测试提供反馈，日志和回放让失败可见，重构负责压缩历史。

接下来最值得量化的，是在给定模型、上下文、memory、工具、测试协议和反馈质量下，一个 HS 能稳定承载多少耦合复杂度。这个量如果能被测出来，“维护成本”就会从一个直觉变成可以比较的对象，也能解释为什么有些系统适合交给 agent 长期养，有些系统很快会长成代码泥潭。

过去写软件，主要是在写它现在怎么运行。以后越来越多时候，我们也会在写它将来怎样吸收反馈、压缩历史、长出新行为。

## 免责声明

本文仅代表个人观点，不代表公司立场；文中讨论与任何公司具体项目、产品规划或内部工作无关。

## 致谢

感谢 [Costa Huang](https://costa.sh/) 和 [Tairan He](https://tairanhe.com/) 的反馈。

## 引用

如果需要在 LaTeX 里引用这篇文章，可以用下面这个 BibTeX。

```bibtex
@misc{weng2026heuristic_system,
  title = {Heuristic System: Software Evolves Through Metabolism},
  author = {Weng, Jiayi},
  year = {2026},
  month = may,
  howpublished = {\url{https://trinkle23897.github.io/heuristic-system/}},
  note = {Blog post}
}
```

## 附录：复现五个代表性结果

完整 artifact repo 在 [https://github.com/Trinkle23897/heuristic-system](https://github.com/Trinkle23897/heuristic-system)。下面命令默认你已经 clone 了这个 repo，并在仓库根目录运行；GitHub Pages 只展示文章和必要静态文件，完整脚本、CSV、视频和实验材料都在 repo 里。

### A.1 Pong 21

```bash
python heuristic_pong.py \
  --policy ram \
  --episodes 1 \
  --seed 0
```

当前 Gym 版本会先打印一段维护状态 warning；有效输出是：

```text
episode=0 score=21.0
summary: episodes=1 mean=21.000 min=21.0 max=21.0
```

### A.2 Breakout 864

```bash
rm -f /tmp/repro_breakout_864.jsonl /tmp/repro_breakout_864.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 108000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 3 \
  --fast-low-ball-lead-steps 3 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 8 \
  --brick-balance-deadzone 0.01 \
  --brick-balance-bias-min-score 432 \
  --late-game-paddle-lag-px 2 \
  --late-game-lag-ball-y 170 \
  --trial-name repro_breakout_864 \
  --log-path /tmp/repro_breakout_864.jsonl \
  --summary-path /tmp/repro_breakout_864.csv
```

期望输出里应该包含 `score=864.0` 和 `mean=864.000`。

### A.3 Ant 默认 MPC 策略

运行下面命令可以复现最终默认 MPC 策略。它比 Breakout 慢很多，因为每个真实环境步都会在本地 MuJoCo 模型里评估 `96 x 10` 个候选残差动作。

```bash
rm -f /tmp/repro_ant_6146_eval5.jsonl /tmp/repro_ant_6146_eval5.csv
python heuristic_ant.py \
  --policy mpc \
  --episodes 5 \
  --seed 0 \
  --max-steps 1000 \
  --mujoco-xml-path ant_envpool.xml \
  --trial-name repro_ant_6146_eval5 \
  --log-path /tmp/repro_ant_6146_eval5.jsonl \
  --summary-path /tmp/repro_ant_6146_eval5.csv
```

期望输出里应该包含 `mean=6005.521`、`min=5776.805`、`max=6146.208`。我本地重跑时输出是：

```text
episode=0 score=6146.208 x_position=285.434
episode=1 score=5982.507 x_position=277.088
episode=2 score=6028.890 x_position=279.226
episode=3 score=5776.805 x_position=267.084
episode=4 score=6093.194 x_position=282.733
eval_summary: episodes=5 env_steps=5000 mean=6005.521 min=5776.805 max=6146.208 x_mean=278.313 x_max=285.434
```

### A.4 HalfCheetah staged-tree MPC 5 局复测

运行下面命令可以复现当前 checked script path 的 5 局评估。它明显比普通 rollout 慢，因为每个真实环境步都会做在线模型评分。

```bash
python heuristic_halfcheetah_v5.py \
  --policy mpc-staged-tree-asym-pd-cpg \
  --eval-episodes 5 \
  --eval-seed 100
```

我本地重跑时输出是：

```json
{
  "episodes": 5,
  "frames": 5000,
  "max_return": 12041.189857475818,
  "mean_return": 11836.693449819431,
  "min_return": 11735.02927325886,
  "policy": "mpc-staged-tree-asym-pd-cpg",
  "returns": [
    12041.189857475818,
    11735.02927325886,
    11854.710591778263,
    11767.164473961016,
    11785.373052623192
  ],
  "std_return": 109.49617764723155
}
```

### A.5 Montezuma 400 分回放

运行下面命令可以复现 Atari57 批量实验里恢复出来的 `400` 分开环路线。这条路线是 `86` 个宏动作组成的回放，具备的通用反应能力很有限，所以正文里把它当作边界案例。

```bash
python heuristic_montezuma_400_policy.py \
  --metadata-out /tmp/repro_montezuma_400.json
```

当前 Gym 版本会先打印一段维护状态 warning；有效输出是：

```json
{
  "env_id": "MontezumaRevenge-v5",
  "seed": 10001,
  "score": 400.0,
  "env_steps": 1769,
  "done": true,
  "macro_count": 86,
  "scripted_action_steps": 1793,
  "expected_score": 400.0,
  "expected_steps": 1769,
  "record_mp4": null,
  "frame0_png": null
}
```
