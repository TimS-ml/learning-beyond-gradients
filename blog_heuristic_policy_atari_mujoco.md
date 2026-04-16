# Make Heuristics Great Again：让 Codex 从零构建启发式系统

> Jiayi Weng

最近我在 EnvPool 里做了一组实验：不训练神经网络，也不做任何强化学习，只是把环境交给 Codex，让它自己写策略、跑评测、看日志、改代码，目标只有一个：把 Atari 和 MuJoCo 的分数往上推。结果比我预期夸张：Pong 到了 `21`，Breakout 的 RAM 输入和纯图像输入都到了 `864`；MuJoCo 里 Ant-v5 到了 `6146`，HalfCheetah-v5 到了 `12041`。

虽然这个不是“几条 if-else 就能打赢任何强化学习策略”的标题党，但是真正让我在意的是成本结构：以前启发式系统难维护，要写检测，要猜动作含义，要处理跳帧和发球，要维护条件分支，要跑参数扫描，还要记清楚每个分数到底从哪里来。人很难长期把这些东西塞进脑子里全局维护，所以启发式策略很容易变成一次性脚本。

Codex 让我感觉这个取舍变了。它不是只写一个 `policy.py` 就结束，而是继续像维护小型软件项目一样做后面的事：写测试，跑批量实验，看视频，改参数，加日志，画样本效率曲线，删掉坏方向，把好配置固化下来。以前启发式看起来像旧时代的土办法，现在我反而觉得它可能正好适合交给编程智能体维护。

下面先讲怎么跑实验，再按顺序写三个例子：Breakout，因为它最直观，能看到一个 Atari 策略怎么一点点被调出来；Ant，因为连续控制里这个现象更像“条件反射系统自己长出来”；最后是 Atari57，因为它虽然没有单点故事那么干净，但更能说明批量无人工介入时这个方法的分布长什么样。

## 实验怎么跑

模型配置上，这些实验主要用的是 Codex 5.4、`xhigh`。我没有在这里做模型大小或推理强度的系统消融，所以这篇文章讨论的是这个配置下的现象，而不是不同模型之间的比较。

最早的出发点其实很工程：我想给 EnvPool 做环境正确性验证，需要一些比随机策略强很多、但又不需要训练的策略。随机策略太弱，很多环境里一整局都碰不到关键奖励，失败时只能看到超时或者 0 分，很难判断是环境错了、封装层错了，还是策略根本没走到有信息的状态。给每个环境都训练一个神经网络又太重，版本、依赖、训练预算、检查点都会变成测试系统自己的负担。

一开始我试过最直接的办法：问 Codex “写一个能解决 Breakout 的策略”，让它跑一局、看分数、继续改。效果一般，因为低分没有解释力。它不知道是动作理解错了、状态读错了、评测设置错了，还是策略结构本身不行。后来我把任务改成让它先维护一个完整闭环：先写检测，再写策略，再跑完整回合，再写实验记录，最后才把 `policy.py` 固化下来。

这个闭环最早是在 Pong 和 Breakout 上磨出来的。检测一开始只是确认动作空间和观测形状，后来变成动作含义检测、像素阈值检测、RAM 字节和图像位置的相关性检测、短视频回放、最后几十步轨迹记录。再后来才加上 `trials.jsonl` / `summary.csv` 这种实验记录，每次改动都必须记累计环境步、分数、配置和失败原因。最后比较好的设置是先探测可观测状态，再提出一个小改动，完整回合复验，通过以后才固化；每次刷新最高分后还要做一轮简化，把搜索过程中长出来的临时脚本和无效分支删掉。

开发这个闭环最耗时间是把“看见什么、动作是什么意思、这个分数从哪里来”这些弄清楚。Pong 里最耗时的是阈值和 RAM 坐标对齐；Breakout 里最耗时的是区分“会接球”和“会继续清砖”；Ant 里最耗时的是把真实环境步、MuJoCo 模型内规划步和实验记录分开。这个过程很像把一个原本只会给分数的黑盒，改造成一个可以调试的软件系统。

最后固定下来的流程大概是这样：

1. 先写评测框架，确认观测、动作、奖励、终止条件、随机种子和跳帧设置。
2. 每次实验都追加到 `trials.jsonl`，再生成 `summary.csv`，记录 `env_steps`、分数、配置和备注。
3. 写探测脚本找可控状态变量，比如 RAM 字节、RGB 连通域、MuJoCo 关节顺序、根部速度、接触模式。
4. 每轮只改一个方向，比如检测器、目标点、卡住处理、规划器目标、候选动作库，或者短视窗代理目标。
5. 只有完整回合复验通过，才把配置提升成默认。

这看起来只是普通实验纪律，但对 Codex 很关键。没有实验记录，它跑几十轮以后很容易把“300 步变好”和“完整回合变好”混在一起，也会把真实环境步数和 MuJoCo 模型里的内部展开混在一起。有实验记录以后，它就更像在维护一个软件系统：每个改动有来源，每个失败方向有备注，下一轮不会又踩同一个坑。

## Breakout：从只会接球到理论最高分 864

Breakout 最开始看起来就是个几何问题：球在哪里，挡板在哪里，球撞墙以后会落到哪里。后面麻烦的其实是另一件事：策略可以永远接得到球，但再也打不到新砖。

Codex 第一轮没有急着写最终策略。它先确认动作空间和观测形状，再从 RGB 画面里找挡板、球、砖块的颜色，然后用图像标签去扫 128 个 RAM 字节。早期实验记录大概长这样：

```text
trial_name                 score   cumulative_env_steps   note
shape_action_probe          -      32                     inspect obs/info/action
ram_byte_corr_probe_v1      -      5,032                  correlate RAM bytes
ram_fit_action_probe_v2     -      9,532                  action 2=right, 3=left
baseline_v0                99      16,303                 initial RAM intercept
tunnel0_v1                387      43,303                 no tunnel offset
```

`387` 是第一个很容易骗过人的局部高分。策略已经能稳定接球，但它只是把球送进一个周期：不会死，也不会继续清砖。这里如果是人手写启发式策略，很容易以为问题还是“接球不准”，然后继续调截距。Codex 是通过视频和最后几十步轨迹看出来，问题其实是球路缺少扰动。

<video controls src="heuristic_breakout_score387_tunnel0_render210x160.mp4" width="360"></video>

`387` 这段视频里能看到问题：球和挡板已经形成稳定循环，但砖块推进停住了。

<details>
<summary>复现 `387` 分</summary>

下面命令假设当前目录就是这篇文章的 artifact 目录，也就是 `heuristic_breakout.py` 所在目录。它会把临时实验记录写到 `/tmp`，不污染正文用的实验记录。

```bash
rm -f /tmp/repro_breakout_387.jsonl /tmp/repro_breakout_387.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 1000000000 \
  --stuck-trigger-steps 1000000000 \
  --stuck-switch-steps 0 \
  --stuck-offset 0 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_387 \
  --log-path /tmp/repro_breakout_387.jsonl \
  --summary-path /tmp/repro_breakout_387.csv
```

期望输出里应该包含 `score=387.0` 和 `mean=387.000`。

</details>

它加的第一个关键机制是打破循环：如果连续很久没有奖励，就在预测落点上周期性加偏移，把球打出局部循环。这个改动把分数从 `387` 推到 `507`。

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

<video controls src="heuristic_breakout_score507_stuckbreaker_render210x160.mp4" width="360"></video>

`507` 这一步的关键不在接球能力，而在球路控制：策略开始主动把球从局部循环里打出来。

<details>
<summary>复现 `507` 分</summary>

```bash
rm -f /tmp/repro_breakout_507.jsonl /tmp/repro_breakout_507.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 1000000000 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_507 \
  --log-path /tmp/repro_breakout_507.jsonl \
  --summary-path /tmp/repro_breakout_507.csv
```

期望输出里应该包含 `score=507.0` 和 `mean=507.000`。

</details>

后面又遇到另一个失败模式：高速低位球如果按普通截距追，挡板会被过度前视带偏。Codex 加了 `fast_low_ball_lead_steps=3`，分数直接从 `507` 跳到 `839`。

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

<video controls src="heuristic_breakout_score839_fastlead_render210x160.mp4" width="360"></video>

`839` 这段能看出另一个变化：低位高速球不再把挡板一路带偏，策略终于稳定进入清后半面墙的阶段。

<details>
<summary>复现 `839` 分</summary>

```bash
rm -f /tmp/repro_breakout_839.jsonl /tmp/repro_breakout_839.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 3 \
  --fast-low-ball-lead-steps 3 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_839 \
  --log-path /tmp/repro_breakout_839.jsonl \
  --summary-path /tmp/repro_breakout_839.csv
```

期望输出里应该包含 `score=839.0` 和 `mean=839.000`。

</details>

从 `839` 到 `864`，反而是最像维护复杂启发式系统的一段。Codex 试了死区、发球偏移、卡住偏移、砖块平衡偏置、前视步数，很多方向都没用。最后有效的是一个后期条件：分数超过第一面墙以后，卡住偏移只在离挡板还远的时候生效；快接球时把偏移逐步收掉，不然最后几块砖阶段会自己把挡板带偏。同时它又加了一个很小的挡板漂移补偿，补动作和挡板位置之间的一步延迟。

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

<details>
<summary>复现 `864` 分</summary>

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

</details>

最终 RAM 默认配置三局验证是 `864/864/864`。更有意思的是，Codex 后来把同一套几何控制迁移回纯图像输入：不用 RAM，只用 RGB 分割找挡板、球和砖块平衡。纯图像版本先是 `310`，然后 `428`，最后把后期“卡住偏移逐步收掉”的阈值放低到全程生效，7 个策略本地回合后第一次到 `864`，对应图里的 `14,504` 个策略本地环境步数。

![Breakout 样本效率](heuristic_breakout_sample_efficiency.png)


完整策略代码在 [`heuristic_breakout.py`](heuristic_breakout.py)，实验记录在 [`heuristic_breakout_trials_summary.csv`](heuristic_breakout_trials_summary.csv)。

这里最值得看的不是“纯图像从零 `14.5K` 步到满分”。这个说法不准确。更准确的是：Codex 先在 RAM 版本里把几何控制、打破循环、后期收偏移这些结构摸出来；等这些结构稳定以后，再把状态读取层从 RAM 换成 RGB 检测器。纯图像的 `14.5K` 是迁移预算，不是从零预算。它说明的是另一件事：启发式策略一旦被写成可维护的软件结构，后面可以替换输入层、重用控制逻辑、继续回归测试，而不是每换一种观测就重新训练一个系统。

## Ant：Codex 自己把步态和模型预测规划引进来

如果说 Breakout 的几何结构还比较直观，Ant 对我更意外。我没有一开始说“用 CPG”，也没有说“用 MPC”；事实上我现在也不知道 MPC 是什么。我的要求只是延续前面的闭环：写启发式策略，不训练神经网络，提高分数，留下日志和复现命令。Codex 先读 EnvPool/Gymnasium 的 Ant 观测和奖励，确认动作顺序、根部速度、躯干朝向、关节位置和关节速度，然后自己提出第一版节律步态。

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

后面的早期迭代很像调一个真实机器人：先加偏航反馈到 `2718`，再调相位速度、髋/踝幅度、偏航角速度增益到 `3025`，然后加二阶/三阶谐波到 `3162`。Codex 也试过大范围 CEM 搜索，但结果没有稳定超过当前节律策略，于是它没有继续把搜索规模硬怼上去，转向了另一种表示方式。

Ant 的跃迁来自 Codex 引入的残差模型预测规划，也就是代码里写的 MPC。我的理解是：保留节律步态作为“条件反射”，每个真实环境步在本地 MuJoCo 模型里采样几十条小的残差动作序列，打分后只执行第一个残差动作，并把剩下的计划热启动到下一步。这样每一步都不用从零规划 8 个关节怎么动；策略先有一个稳定的基础步态，再用短视模型规划去修正这个步态。

MPC 可以理解成“边走边想一小段未来”。它不是训练一个策略网络，也不是一开始就算出从第 0 步到终点的完整轨迹；它是在当前状态下用模型试几种接下来可能的动作序列，挑一个看起来最好的，只执行第一步，然后下一步重新看状态、重新试。对 Ant 来说，这件事很适合放在启发式策略外面：节律步态负责让机器人像一个能走路的东西，MPC 负责在每一步小幅修正，让它更少摔倒、更少偏航、更快往前走。

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

这件事最让我意外的地方在这里：我没有告诉它“模型预测规划很适合 Ant”。它是在节律策略到 `3162` 后，自己写了一个基于模型的残差规划器。第一版视窗长度 6、32 个候选、小残差，就从 `3135` 提到 `3635`。然后它继续维护这个系统：

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

<details>
<summary>复现默认 Ant 策略</summary>

这个命令会复现最终默认 MPC 策略。它比 Breakout 慢很多，因为每个真实环境步都会在本地 MuJoCo 模型里评估 `96 x 10` 个候选残差动作。下面命令假设当前目录就是这篇文章的 artifact 目录；`ant_envpool.xml` 已经和脚本放在同一个目录里。

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

这条命令一次跑 5 个 episode，比单 episode 复现慢很多，但能看到一点方差。期望输出里应该包含 `mean=6005.521`、`min=5776.805`、`max=6146.208`。我本地重跑这条命令时，输出是：

```text
episode=0 score=6146.208 x_position=285.434
episode=1 score=5982.507 x_position=277.088
episode=2 score=6028.890 x_position=279.226
episode=3 score=5776.805 x_position=267.084
episode=4 score=6093.194 x_position=282.733
eval_summary: episodes=5 env_steps=5000 mean=6005.521 min=5776.805 max=6146.208 x_mean=278.313 x_max=285.434
```

</details>

完整实验脚本在 [`heuristic_ant.py`](heuristic_ant.py)，抽出的最小可调用策略在 [`heuristic_ant_min_policy.py`](heuristic_ant_min_policy.py)，实验记录在 [`heuristic_ant_trials_summary.csv`](heuristic_ant_trials_summary.csv)。

Ant 的例子和 Breakout 不太一样。Breakout 是发现几何以后，输入规整化带来了很夸张的样本效率；Ant 是一个可维护启发式系统不断变复杂，但没有变成一坨不可控脚本。到最后，策略里有振荡器相位、支撑期比例、速度自适应、滚转/俯仰/偏航反馈、脚部接触、短视窗模型内展开、残差平滑、终端速度代价、热启动计划衰减。人类当然可以写其中一两个模块，但要在一天内维护完整实验记录、代码、视频和失败方向，难度完全不一样。

## Atari57：把这个流程直接撒到 57 个游戏上

Breakout 和 Ant 是两个单点故事。Atari57 这一组实验更粗暴：同一套 Codex 工作流直接扔到整套 Atari57 上，每个环境同时跑 `ram` 和 `native_obs` 两种输入，每种输入撒 3 个独立运行。也就是说，每种输入都有 `57 个游戏 x 3 次运行 = 171` 条编程智能体搜索轨迹。

我没有像上面两个例子一样，在 Codex App 里一边看中间结果、一边提示、一边纠偏、一边让它继续调。当时做法很简单：用 Codex CLI 批量启动一堆 `gpt-5.4`、`xhigh` 的智能体，每个智能体拿到同一个提示词模板和不同的 `ENV_ID / OBS_MODE / REPEAT_INDEX`，然后我全程不管，等它们执行完停下来以后再收集结果。为了减少不可见上下文，我没有用 Codex App，也没有让它读长期记忆；这是一次性的 CLI 批量运行。下面是 Atari57 批量提示词原文，折叠起来放在这里，方便复现时判断边界。

<details>
<summary>Atari57 批量提示词原文</summary>

````text
你是一个务实、严谨的 coding agent。现在请只针对 EnvPool 的 {{ENV_ID}}，在 {{OBS_MODE}} 设定下，自己设计并持续迭代一个手写 heuristic policy，把分数尽量推到最高，同时完整记录 sample efficiency。整个过程只在当前机器本地跑，不要联网查资料，不要参考任何现成解法。

任务配置：

ENV_ID = "{{ENV_ID}}"
OBS_MODE = "{{OBS_MODE}}"
REPEAT_INDEX = {{REPEAT_INDEX}}
KNOWN_BEST_SCORE = {{KNOWN_BEST_SCORE}}
ENV_ROOT_DIR = "/tmp/envpool_heuristic/{{ENV_ID}}"
ROOT_DIR = "/tmp/envpool_heuristic/{{ENV_ID}}/{{OBS_MODE}}"
RUN_DIR = "/tmp/envpool_heuristic/{{ENV_ID}}/{{OBS_MODE}}/run_{YYYYMMDD_HHMMSS}_{PID}"

ENVPOOL_VERSION = "1.1.1"
FRAME_BUDGET = 20000000

请先创建 ENV_ROOT_DIR、ROOT_DIR 和 RUN_DIR。本次运行产生的所有脚本、日志、图、说明文档都写到 RUN_DIR 下面，不要写进任何 repo 工作区。

开始实验前，不要创建 venv，也不要重装 envpool。直接在当前 Python 环境里做一次最小 import 验证，确认 `envpool.__version__ == "1.1.1"`。把 Python 版本、envpool 版本、envpool 包文件位置、以及一次最小 import 验证结果写进 RUN_DIR/README.md。如果当前环境不是 `envpool==1.1.1`，就直接报错停止，不要自行切版本。

请把 REPEAT_INDEX 用作本次搜索的初始随机种子偏移、trial 命名后缀或其它去相关机制，保证 3 次 repeat 不是机械地跑成完全同一条搜索轨迹。

不同任务会落在不同的 ROOT_DIR / RUN_DIR 下面，目录天然隔离，不需要为了避免路径冲突而串行化；如果你认为有帮助，可以放心在本任务内部并行执行互不覆盖的探索、评估或分析步骤。

参考分数规则：

这个 prompt 已经直接给出了 `KNOWN_BEST_SCORE`。它只用于参考和结果对比，不用于决定何时停止。

硬约束：

1. 严格把环境当黑盒。不要阅读 envpool 仓库里该环境/wrapper 的实现源码、测试代码、ROM/XML 细节，或者任何会泄漏环境内部实现的文件。只能使用 envpool 对外 API 暴露的信息：make/reset/step/render、action_space、observation_space、obs、reward、done、info 的公开字段。

2. 如果 OBS_MODE = "native_obs"：
   - policy 只能基于 reset/step 返回的原生 obs，以及 policy 自己维护的内部状态来决策。
   - 不允许读取 info["ram"] 或任何 RAM 内容。
   - 不要把 render() 当成额外输入喂给 policy。

3. 如果 OBS_MODE = "ram"：
   - policy 可以读取 info["ram"] 和 policy 自己维护的内部状态。
   - 仍然不要调用未公开的 simulator state 或源码内部实现。

4. 如果 ENV_ID 是 Atari 环境，必须用下面这段环境初始化模板，不要擅自改分辨率、frame stack、frame skip、reward clipping 或 sticky action：

```python
import envpool

env = envpool.make_gym(
    ENV_ID,
    num_envs=1,
    batch_size=1,
    seed=seed,
    img_height=210,
    img_width=160,
    stack_num=1,
    gray_scale=False,
    frame_skip=1,
    noop_max=1,
    use_fire_reset=True,
    episodic_life=False,
    reward_clip=False,
    repeat_action_probability=0.0,
    full_action_space=False,
)
```

5. 如果 ENV_ID 不是 Atari，就直接使用该环境默认 native obs 初始化；除非环境 API 明确要求，否则不要额外包 wrapper。

6. 不限定你怎么迭代、怎么搜索、怎么组织 heuristic。你可以自己决定用目标检测、轨迹预测、状态机、参数搜索、短 horizon planning、controller 或任何别的纯手写方法；唯一要求是不要训练神经网络，不要读环境源码，不要读隐藏状态。

7. 不要停下来讨论方案，不要请求确认，不要输出中途进展汇报。

8. 每当你刷新当前 run 的 `best_score` 时，不要立刻停止。先进入一次“代码简化阶段”：
   - 目标是在不降低当前 `best_score` 的前提下，把 heuristic 尽可能压缩成更短、更直接、更容易复现的实现。
   - 优先删除冗余搜索脚本、重复分支、无效状态、过度参数化和对最终分数没有贡献的辅助逻辑。
   - 简化后必须重新评估，确认 `best_score` 没有下降；如果下降，就回退到上一个不掉分的版本。
   - 最终留下的 `policy.py` 应该是当前 best score 对应的尽可能简单版本，而不是搜索过程中最臃肿的版本。

停止规则：

- 对 Atari 任务，这里的 frame budget 定义为 `FRAME_BUDGET = 20000000`。由于固定使用 `frame_skip=1`，所以这里可以把 `cumulative_env_steps` 直接当成累计 frame 数。
- 在 `cumulative_env_steps < FRAME_BUDGET` 时，不要因为分数高低、短期平台期、暂时找不到更好的策略、或者已经超过/没超过 KNOWN_BEST_SCORE 就停止；必须持续尝试新的 heuristic、结构、搜索或评估。
- 只有当 `cumulative_env_steps >= FRAME_BUDGET` 时，才允许停止并输出最终总结。
- 如果当前环境不是 Atari，也沿用同一个 `FRAME_BUDGET = 20000000`，把它解释为累计 env steps 上限。

输出文件要求：

1. policy.py
   保存当前最好的 heuristic，而且它应该已经经过“简化阶段”，是在不降低 best score 的前提下尽可能短、尽可能直接的版本。接口尽量简洁，比如：

```python
class Policy:
    def reset(self):
        ...
    def act(self, obs: np.ndarray, info: dict | None = None):
        ...
```

2. trials.jsonl
   每一次 trial 追加一行，至少包含：
   trial_index, timestamp, env_id, obs_mode, trial_name, episodes_finished, env_steps, score_mean, score_min, score_max, cumulative_env_steps, cumulative_episodes, policy_config, notes

3. summary.csv
   从 trials.jsonl 汇总出来。

4. sample_efficiency.png
   根据 summary.csv 画两张子图：
   - x = cumulative_env_steps，y = score_mean 和 running_best
   - x = cumulative_episodes，y = score_mean 和 running_best

5. README.md
   最后写清楚：当前最好分数、KNOWN_BEST_SCORE、REPEAT_INDEX、对应 trial、累计 env steps / episodes、FRAME_BUDGET、复现命令、最终保留下来的简化版 policy 逻辑、主要失败方向、以及为什么你认为已经满足停止规则。

sample 统计口径：

- 所有实际 step 过环境的 trial / probe / debug rollout 都要计入 cumulative_env_steps 和 cumulative_episodes，不能偷偷漏掉。
- score 以 episode return 为准；如果一个 trial 跑多个 episodes，就记 mean / min / max。

现在直接开始执行。先创建 RUN_DIR，探测 ENV_ID 的 action_space、observation_space、reset obs shape、step 返回结构，然后自己决定怎么迭代。中途不要问我，也不要汇报进展，直到满足停止规则后再输出最终总结。
````

</details>

下面这张 Atari57 图里的分数，是这种无人工介入批量运行自己跑出来的。我没有中途看视频挑方向，也没有把某个游戏的失败原因喂回去让它继续改。人工介入当然能把单点分数推得更高，Breakout 和 Ant 这种实验过程已经说明了上限可以继续往上走。但我不觉得人工介入是这条路线的必要条件，它更像现在模型能力还不够时的加速器。模型再聪明一些以后，很多“看失败视频、判断缺哪个机制、继续改代码”的步骤应该可以直接在无人工介入里完成。每个运行都像一个小的编程智能体研究员。它先探测动作、观测和奖励，再生成候选启发式策略或局部搜索脚本，然后不断评测、记录最高分、看失败模式、继续改。

![Atari57 样本效率对比 OpenRL Benchmark](atari57_openrl_sample_efficiency_context.png)

这张图的横轴从 `10^4` 个环境步开始，因为更早的部分基本看不出变化；纵轴是 Atari 人类归一化分数，也就是 HNS。Codex 曲线按 Atari 常见的中位数口径画：每个游戏先把 3 次独立运行取中位数，再在 57 个游戏上取中位数。这个口径不会被少数特别高分的游戏拉爆，所以它看的是覆盖率：到底有多少游戏已经被这套流程摸到要害。

在这种完全无人工介入的批量运行里，`native_obs` 到 `9.7M` 步附近是 `0.81`，`ram` 是 `0.59`。同一张图里，[OpenRL Benchmark](https://arxiv.org/abs/2402.03046) 保存的 PPO2 / CleanRL EnvPool PPO 中位数 HNS 曲线到 `10M` 步大概是 `0.88` / `0.92`。所以这个结果并不是 Codex 已经全面打穿传统强化学习；更准确的说法是，一个很粗糙的编程智能体批量流程，在完全不看中途结果的情况下，已经能把 Atari57 的中位数推进到接近这些基线的区间。

聚合曲线会把差异压到一个中位数里，所以我又把 57 个游戏逐个摊开画了一张。Atari 原始回报跨游戏没有可比性，这里仍然用每个游戏自己的 HNS；虚线 `1.0` 是人类分数。左图按 heuristic 分数排序，右图把 heuristic 和 OpenRL 里的 CleanRL EnvPool PPO 直接对到同一个坐标上。

![Atari57 每个游戏 HNS 对比](atari57_per_game_hns_comparison.png)

这张图能看出两件事。第一，重合是有的：Breakout、Krull、DoubleDunk、Boxing、DemonAttack 这些游戏里，heuristic 和强化学习基线都能拿到明显高于人类基线的分数。第二，差异也很大：heuristic 在 Asterix、Jamesbond、Centipede、Bowling、Skiing、Tennis 这类游戏上相对更突出；PPO 在 Atlantis、VideoPinball、UpNDown、Assault、RoadRunner、StarGunner 上明显强很多。这个分布比一个中位数更有信息：heuristic 不是均匀地学到“玩 Atari”，而是在某些游戏里很快写出了有效机制，在另一些游戏里还没找到对的状态表示或长期策略。

这里也要把账算清楚。Codex 曲线来自这次 342 条批量运行的原始 `summary.csv`，我把所有结果重新整理后，按 `cumulative_env_steps` 还原每条搜索轨迹。它仍然不能当严格排行榜：Codex 写代码、读日志和看视频的计算量没有按神经网络训练计算量计入；PPO 曲线则是它保存的中位数 HNS 曲线。这段比较主要看一个信号：在规整输入下，编程智能体维护的启发式策略已经可以用很少环境交互，把不少游戏推到有竞争力的区间，而且 RAM 和原生观测没有差得离谱。

我觉得 Atari57 最有意思的地方，是样本效率的来源变了。传统神经网络 Atari 学习要在每个环境里从高维输入重新学表示、信用分配和动作含义；这里 Codex 做的是把环境拆成可维护的小程序系统：射击游戏的瞄准/躲避，接球游戏的反弹，躲避游戏的位置规则，环境包装器细节，以及每个环境自己的失败实验记录。它没有训练出一个通用神经网络，它是在批量生成和维护一批局部启发式/条件反射系统。

## 为什么我觉得这件事值得想

和神经网络比，启发式策略最明显的两个优点是样本效率和可解释性。样本效率前面已经很直观：很多时候 Codex 跑几万到几十万步，就能把一个可工作的策略结构搭起来。可解释性也同样重要。Breakout 的每个提升都能落到一个具体机制上：球路卡住、低位高速球、后期偏移收敛、挡板动作延迟。Ant 的每个提升也能落到具体模块上：相位、偏航反馈、谐波、残差规划、热启动。失败的时候可以直接打开代码和视频，看是哪条条件、哪个状态估计、哪个动作延迟出了问题；这和只盯着一条 reward 曲线的体验很不一样。

神经网络当然有自己的优势，尤其是表达能力、复杂视觉和跨状态泛化。但这次实验让我觉得，很多局部控制任务不一定要全部交给端到端训练。一个可维护的启发式层可以先把“反射动作”和“局部几何”吃掉，再把更难的感知、长期规划、策略选择交给学习系统。这样系统未必更简单，但调试会更像软件工程：有日志、有回放、有消融、有失败来源。

这里的核心还是维护成本。过去启发式系统难维护，维护者要同时当程序员、实验员、调试员、数据记录员和半个控制工程师。一个 Breakout 策略要记住卡循环、后期逐步收偏移、挡板延迟、视觉检测器、RAM 映射；一个 Ant 策略要记住关节顺序、相位、平衡反馈、展开打分目标、规划预算、失败配置。人很难长期维护这种复杂度，所以最后大家自然会把希望放到端到端训练上。

编程智能体让这个取舍不太一样了。启发式系统可以变成一种“可执行研究笔记”：状态提取、策略结构、参数扫描、失败诊断、视频证据、样本效率曲线都在同一个代码系统里。策略也不再只是一次性手写的脆弱脚本，它可以变成一个智能体持续维护、局部重构、跨输入迁移的软件产物。

## 对机器人控制的启发

这对机器人仍然有想象空间，但边界要说清楚。Ant 是仿真里的移动控制，和真实机器人不是一回事。仿真里可以让 Codex 几万步、几百万步试错，摔了也只是重置；真实世界里每次试错都有时间、硬件磨损、安全、场地复位和传感器漂移成本。所以这套闭环不能直接搬到真机上，让它像在仿真里那样盲跑。

我更愿意把它看成机器人系统里某一层工具，而不是完整机器人方案。它比较适合的地方，是那些状态能被可靠观测、失败可以安全回滚、局部控制目标清楚的部分：保持身体稳定、某个关节回到安全角度、脚落地以后卸力、快摔倒时先做恢复动作、夹爪接近物体时限力限速。这些更像条件反射，确实可能被程序化维护和回归测试。

但操作任务会麻烦很多。叠衣服、整理线缆、打开软包装这种任务，难点不只是“机器人关节怎么动”，还有物体本身的状态：布料形变、遮挡、接触历史、摩擦、皱褶、目标形状。这里写启发式策略不是调几个关节相位就能解决的；如果没有好的感知表示、可恢复的动作原语和足够真实的仿真，编程智能体也会在错误的状态变量上写出很脆的规则。

所以更现实的形态可能是混合系统：仿真和离线数据先用来生成、筛掉候选启发式；真机上只做小步、安全、有护栏的验证；神经网络负责感知、物体状态估计和长程价值；启发式/条件反射系统负责低延迟、安全约束、局部恢复和测试判据；编程智能体负责维护接口、检测、失败处理和回归测试。这样它不是“用启发式手写叠衣服”，而是把一些能被写成程序的局部规律从端到端训练里拆出来。

## 限制

样本效率：Atari57 图里的环境步数是策略本地的真实 EnvPool 步数，但不包括 Codex 写代码、读日志、看视频的耗时，也不等同于标准神经网络训练预算。Ant 里的残差模型预测规划还使用本地 MuJoCo 模型内展开，这些计算量也不能和真实 EnvPool 环境步数混在一起。

RAM 和图像/原生观测也要分开讲。Atari57 聚合结果里 RAM 和原生观测的整体效果没有差得离谱，这是最有意思的现象；但单个 Breakout 里的纯图像 `14.5K` 是迁移预算，继承了 RAM 阶段已经发现的几何策略。它证明的是结构可以被智能体迁移和重构，不能写成“纯图像从零 14.5K 学到满分”。

还有些环境不适合这种反应式启发式策略。Montezuma's Revenge 是最典型的反例：这篇实验记录里的状态图搜索能把钥匙距离从 `72` 推到 `28`，但奖励仍然是 `0`。后面我又试了很久，才把它摸到过 `400` 分左右；那时策略确实已经学会了“爬楼梯、跳过怪物、捡钥匙、开门”这一串动作。问题是这种策略用普通代码分支表达起来很别扭：动作必须对齐到时机，失败后要能恢复，中间状态还要能重新进入计划。这里的限制更像是 `policy.py` 这种手写状态机的表达能力不够。Montezuma 让我觉得，策略的表现形式本身也很关键：有些环境需要可组合的宏动作、可恢复的搜索状态，甚至需要一个比普通 if-else 更适合长期规划的程序结构。

所以文章里最强的结论应该是“很有潜力”，还不到“已经替代强化学习基准”。但这个潜力确实很强：一旦输入规整，而且编程智能体能维护复杂启发式策略，样本效率可以进入过去神经网络学习曲线很少触及的量级。这就是我觉得值得写出来的地方。

## 结论

这次实验改变了我对启发式策略的看法。以前我会把它归到脆弱、临时拼凑、不可扩展那一类；现在我更愿意把它看成一种可以由编程智能体持续维护的软件系统。它早就不像几条固定规则，更像一组能被测试、可视化、调参、重构、迁移、记录来源的程序化条件反射。

Atari57 说明，在规整输入下，由编程智能体维护的启发式策略可以用很少环境步数把中位数 HNS 推到接近神经网络基线的区间，而且这是无人工介入批量运行的结果。它也暴露了现在的问题：搜索轨迹很重尾，有些游戏很快找到结构，有些游戏还找不到。Breakout 说明人工介入时单点上限还能更高：先发现几何控制，再维护卡住处理和后期逐步收偏移，最后把 RAM 状态读取迁移成 RGB 检测器。Ant 说明，即使在连续控制里，Codex 也能从简单节律步态出发，自己引入残差模型预测规划，把一个多关节条件反射系统维护到很高分。

如果继续做下去，我已经不太满足于“再调高某个游戏的分数”。更想看的事情是系统化：给编程智能体一个环境、一个严格实验记录、一个采样预算、一个渲染/诊断接口，让它自己发现、维护和回归测试启发式/条件反射层。过去这件事靠人类维护太累，所以几乎没人认真做；有了编程智能体，它突然变成了一个很合理的研究方向。而且我觉得随着模型智能继续增长，这种迭代时间和成本会降得很快，生成一套启发式策略会比今天还要便宜很多。

我尤其想看的机器人版本也会更收敛：先从安全、可复现的局部反射和仿真到真机验证工具开始，而不是直接宣称能手写复杂操作任务。真实世界迭代慢，操作任务还牵涉物体状态，所以这条路线更可能先变成训练和测试系统的一部分：底层反射负责安全和局部反馈，高层策略和感知模型负责目标、物体理解和组合。这样每个关节不一定都要从零训练，也不一定每个失败都要靠更大的网络去吸收。

## 致谢

感谢 [Costa Huang](https://costa.sh/) 和 [Tairan He](https://tairanhe.com/) 对这篇文章的反馈。

## 引用

如果需要在 LaTeX 里引用这篇文章，可以先用下面这个 BibTeX。正式发布以后，把 `url` 换成最终链接就行。

```bibtex
@misc{weng2026codex_heuristic_policy,
  title = {Make Heuristics Great Again：让 Codex 从零构建启发式系统},
  author = {Weng, Jiayi},
  year = {2026},
  month = apr,
  howpublished = {\url{https://example.com/codex-heuristic-policy}},
  note = {Blog post}
}
```

正文里可以写成：

```latex
\usepackage{url}

... as discussed in \cite{weng2026codex_heuristic_policy}.
```
