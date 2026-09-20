# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**简体中文** | [English](README.md)

**一个 0.6B 并行决策模型：输入状态与问题，直接得到完整概率分布，无需生成答案 token。**

[在线演示](https://nanojev-dev.tianyuchen99.chatgpt.site/side-by-side?autoplay=1#maze) · [模型](https://huggingface.co/C-Tianyu/NanoJev-dev) · [数据集](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data-dev)

## 更新内容

**2026 年 9 月 20 日：一个模型，四款游戏。**

- **统一 checkpoint：** 同一个模型支持 Maze、Snake、ViZDoom Basic 和 Predict Position。
- **更大的游戏实录：** 225 次行动完成 50×50 迷宫；完整存活 256 步的 Snake 对局，吃到 30 个食物。
- **移动目标射击：** Predict Position 测试成功数从 11/128 提升到 **27/128**，Basic 保持 **128/128**。
- **新版模型与数据：** step-400 checkpoint、覆盖五个分区的 18,760 条混合任务数据，以及可复现的评测记录。

## 三个模型，并排回放

**[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、NanoJev 和未微调 Qwen** 的真实网页回放。下方动图自动循环，点击即可进入交互播放器。四款演示使用同一个当前 NanoJev checkpoint。

### 找到出口 · 50×50 Maze

[![Jev、当前 NanoJev 与未微调 Qwen 在网页三栏播放器中探索同一张 50×50 迷宫](assets/maze_unified_autoplay.gif)](https://nanojev-dev.tianyuchen99.chatgpt.site/side-by-side?autoplay=1#maze)

NanoJev 用 **225 次行动**到达出口，[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 用 **2,738 次**，未微调 Qwen 用 **4,726 次**。三组都通过局部安全概率驱动相同探索代码，并记住已经走通的路径。

### 把握开火时机 · Predict Position

[![NanoJev 等待后命中移动目标，Jev 和未微调 Qwen 未命中，三组按同一游戏时钟播放](assets/predict_position_unified_autoplay.gif)](https://nanojev-dev.tianyuchen99.chatgpt.site/predict-position?autoplay=1)

一个移动目标，一枚火箭。NanoJev 在 **5.06 秒**发射、**5.94 秒**命中；[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 和未微调 Qwen 在 **1.40 秒**发射后落空。播放器保留两个精选 NanoJev 独胜案例，展示原始画面、动作概率与实际发射时刻。

[打开 Snake](https://nanojev-dev.tianyuchen99.chatgpt.site/side-by-side?autoplay=1#snake) · [打开 Basic](https://nanojev-dev.tianyuchen99.chatgpt.site/?autoplay=1)

## 核心能力

- **并行决策：** 将独立状态、问题和候选路径放入同一次 backbone 前向计算。
- **动态候选：** Choice 通过共享评分头，返回所提供的 2–255 个候选的完整分布。
- **布尔与等级问题：** 输出一个命题成立的概率，或 2–10 个有序等级的分布及期望。
- **直接输出概率：** 可用于排序、贪心选择或采样，无需生成答案 token。
- **轻量统一底座：** Qwen3-0.6B 加决策头，支持四款游戏及持久推理服务。

每个请求包含**状态、问题和候选集合**。模型编码候选路径，再由共享决策头输出概率。Choice 使用集合注意力与 softmax，Boolean 使用 sigmoid，Score 返回等级的概率加权期望。

## 完整测试结果

以下为 **274 个测试案例**的成功数。各系统使用相同观察接口、候选动作和固定随机种子的 epsilon-greedy 控制器：

| 模型 | Maze | Snake | Basic | Predict Position |
|---|---:|---:|---:|---:|
| **NanoJev** | **4/10** | **8/8** | **128/128** | **27/128** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 7/10 | 8/8 | 56/128 | 11/128 |
| 未微调 Qwen3-0.6B | 2/10 | 0/8 | 56/128 | 11/128 |

测试与 OOD 合计为**每个模型 548 个案例**，全部评测轨迹均通过独立模拟器重放。上方大规模导航演示使用页面标明的局部问题与代码规划设置。

[完整测试及 OOD 结果](docs/SONIC_PREDICT_POSITION_RESULTS.md) · [训练流程](docs/SONIC_PREDICT_POSITION.md)

## 模型与数据

当前版本为 **`hard_lr1e5`，step 400**，使用完整问题交叉熵训练同一个共享模型。每次更新按 **1/3、1/3、1/6、1/6** 的权重混合 Maze、Snake、Basic 和 Predict Position。

hard-target 与 soft-target 两种版本在 train、dev、calibration、test 和 OOD 五个分区各包含 **18,760 条数据**。选定的 hard-target 训练分区为 **10,898 条**，包含 **6,788 条 Predict Position 问题**；通过目标有效性检查、用于训练的问题为 **10,893 条**。原有 Maze、Snake 和 Basic 分区保持一致。数据包还包含匹配的 soft-target 版本、专家轨迹和评测记录。

开发版模型和数据位于上方链接的 Hugging Face 仓库，可登录获授权账户下载。

## 快速开始

```bash
git clone --branch feature/unified-game-policy-iteration https://github.com/TianyuCodings/NanoJev-dev.git
cd NanoJev-dev
python -m pip install -r requirements-toy.txt huggingface_hub
hf auth login
```

下载当前 checkpoint 与数据：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev-dev",
    local_dir="checkpoints/NanoJev-unified",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data-dev",
    repo_type="dataset",
    local_dir="data/NanoJev-unified",
)
```

在 CUDA 环境启动推理服务：

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev-unified \
  --web-root web --port 8765 --disable-native-triton
```

模型只加载一次。向 **`POST http://127.0.0.1:8765/api/evaluate`** 发送状态与问题批次即可调用。

本地体验真实游戏回放：

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

打开 **http://127.0.0.1:8080/dev/side-by-side.html?autoplay=1#maze** 或 **http://127.0.0.1:8080/dev/predict-position.html?autoplay=1**。

## 开发文档

[输入契约](docs/TYPESAFE_CONTRACT.md) · [统一环境](docs/UNIFIED_GAMES.md) · [原子判断与规划](docs/ATOMIC_PLANNING.md) · [Predict Position 回放](docs/PREDICT_POSITION_DEMO.md) · [射击回放](docs/SHOOTING_DEMO.md)

## 路线图

- [x] 一个统一 checkpoint 支持 Maze、Snake 和两款射击任务。
- [x] 50×50 迷宫、长局 Snake 与三模型同步网页回放。
- [x] 混合任务 SFT、可复现的数据划分与独立重放评测。
- [ ] 面向更多长程任务的 RLCD 后训练。
- [ ] 共享前缀推理与更大的候选批次。
- [ ] 更多射击场景与结构化输入支持。
