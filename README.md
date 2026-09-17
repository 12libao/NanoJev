# NanoJev

**简体中文** | [English](README.en.md)

**从 Qwen3-0.6B 训练的并行决策模型原型：输入状态、问题和动态候选集合，直接输出候选概率。** 批内多个状态、多个问题和全部候选在一次 backbone 前向中计算，无输出 token 自回归解码。

项目包含查询生成、Jev 概率标注、程序监督、训练、留出集评估、闭环游戏、模型服务和可视化。当前复现的是 Jev 有辨识度的接口与决策方式；官方 RLCD 的内部训练配方尚未公开取得，也没有把本项目的交叉熵训练称为 RLCD。

## 看实际表现

同一张地图上并排运行 **NanoJev（训练后）、Jev API、原始 Qwen（未做本任务微调）**。以下动画来自保存的真实模型轨迹，不是人工绘制的示意行为。每段完整视频包含事先固定的 4 个例子：前 2 张 4×4 test 地图和前 2 张 6×6 OOD 地图。

### 概率采样：按完整动作分布选择

[![NanoJev、Jev 和原始 Qwen 的概率采样对照](assets/comparison_sample.gif)](assets/comparison_sample.mp4)

[观看完整高清 MP4](assets/comparison_sample.mp4) · [静态预览](assets/comparison_sample.png)

### 贪心选择：每步取最高概率动作

[![NanoJev、Jev 和原始 Qwen 的贪心对照](assets/comparison_greedy.gif)](assets/comparison_greedy.mp4)

[观看完整高清 MP4](assets/comparison_greedy.mp4) · [静态预览](assets/comparison_greedy.png)

GIF 展示各视频的前 12 秒；MP4 保留全部 4 例，包括循环和失败。画面按**环境步数**同步，已完成的一列停在终点。这不是推理延迟竞赛；概率柱分别表示学生动作分布、Jev 的归一化舍入概率、Qwen 的选项条件概率。

在本地查看可暂停、切换案例的交互回放，无需模型、GPU 或 API：

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

打开 **http://127.0.0.1:8080/comparison.html**。根页面另保留 V2/V3 的全部历史对照、失败轨迹和多状态推理说明。静态回放与实时模型服务明确分开。

## 同一批 40 张地图的结果

每个系统运行 greedy 与 T=1 sampling，各含 20 张新 4×4 地图和 20 张 6×6 OOD 地图，共 **240 局**。无在线 oracle 纠错、访问惩罚或循环提前停止；每局最多 `2 × size²` 步。

| 系统 | 贪心：4×4 | 贪心：6×6 | 采样：4×4 | 采样：6×6 |
|---|---:|---:|---:|---:|
| **训练后的 NanoJev** | 18/20（90%） | 10/20（50%） | 19/20（95%） | 18/20（90%） |
| **Jev：本次真实 API** | 20/20（100%） | 16/20（80%） | 20/20（100%） | 19/20（95%） |
| **原始 Qwen3-0.6B** | 6/20（30%） | 1/20（5%） | 7/20（35%） | 3/20（15%） |

“未训练”指已预训练、没有本任务微调的原始 Qwen。它使用原生 LM head 对 A–D 答案标签取条件分布，不是随机权重或随机新头。NanoJev 使用新决策头，输入格式也不同，因此这张表比较三个完整系统，不能解释为只改变训练这一项的因果消融。

NanoJev 固定使用 `v3_teacher_coords_multi_seed17`，未按本次结果重新挑选 checkpoint。单个训练 seed、单个采样 seed 和每尺寸 20 图只能支持这个 toy benchmark 的结论。6×6 上 greedy 仍明显弱于 Jev；原有 V3 程序监督主组、消融和非导航回归结果均保留。

- [三方比较协议、固定案例与概率语义](research/nanojev_comparison_protocol_zh.md)
- [三方完整结果与解释](research/nanojev_comparison_zh.md) · [完整指标及最小公开轨迹](research/nanojev_comparison_public.json) · [240 局独立核验](research/nanojev_comparison_verification.json)
- [V3 全部训练对照与失败](research/navigation_v3_report_zh.md) · [非导航能力回归](research/navigation_v3_regression_zh.md)
- [视频来源、帧与文件哈希](assets/comparison_media_manifest.json)

## 模型怎样做决策

训练样本明确包含 `(state, question, candidate_set, target_distribution)`。问题不能省略；候选集合是每题的输入，不是固定类别词表。

1. **直接初始化 LLM。** 使用 `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，没有先训练 yes/no reranker。
2. **编码候选路径。** 每条路径含状态、问题与当前候选语义；批内路径一次送入 backbone。共享标量头与集合注意力生成每题 K 个 logits，softmax 得到 K 维分布，无需为不同 K 新增分类头。集合注意力只用于 Choice；Score 绕过集合注意力，Boolean 使用单路径 sigmoid。
3. **对完整分布训练。** `L = -Σ qᵢ log pᵢ`；目标可来自 Jev 完整候选概率或独立程序真值。完整问题共同计算归一化分母，不能拆成互不相关的二分类损失。
4. **并行返回结果。** Choice 返回动态候选分布，Boolean 返回概率，Score 返回完整等级分布和期望。没有输出 token 解码；下一步游戏状态依赖本步动作，因此环境仍按步推进。

实际服务已验证 **6 states / 18 questions / 44 candidate paths / 1 backbone forward**。当前接口支持 Choice K=2–255、Score 2–10 级。同一架构已检查 K=2/5/20/64/255 的合法分布；这不等于已证明大 K 的语义质量。当前路径重复编码前缀，尚未实现 Jev 级吞吐或共享前缀优化。[6 状态同批实测](research/parallel_example_v3.json) · [持久服务检查](research/live_service_check_v3.json) · [动态候选检查](research/dynamic_candidates_v2.json)

## 完整训练与 benchmark 流程

**[执行手册：从数据到可运行模型的完整命令](research/pipeline_runbook_zh.md)** · [每项算法决定的理由](research/implementation_plan_zh.md) · [真实 Fable 逐点讨论与复核](research/algorithm_fable_decisions_zh.md)

| 阶段 | 实现与检查 |
|---|---|
| 查询分布 | 自写规则、目录、概率事件、井字棋、网格生成器；先生成环境，再写 question 与合法候选 |
| 数据隔离 | train / dev / calibration / test / OOD；同一地图、规则源及改写继承分组，防止泄漏 |
| 监督 | 参考分布监督与独立程序监督分别记录；原始舍入输出与派生分布分开保存 |
| 训练 | Qwen 初始化、决策头预热、全参训练；按完整问题做 microbatch，dev 选 checkpoint |
| 概率评估 | 对 Jev 的 KL/TV 衡量教师一致性；对独立真值的 NLL/Brier/ECE 和解析分布 TV 衡量正确性与概率质量 |
| 闭环评估 | 同一批地图、同一控制器规则，报告完成率、路径效率、重复访问及全部失败 |
| 部署与演示 | 本地 checkpoint 服务一次加载，多次请求；保存真实轨迹供网页和视频回放 |

不能把“像 Jev”与“概率校准正确”混为一谈。V2 可解析概率题上，程序监督学生的 test TV 为 `.0454 ± .0081`，参考分布监督学生为 `.3060 ± .0479`（各 3 个训练 seed）。动作策略概率也不是终局获胜率。[V2 完整结果](research/pipeline_v2_report_zh.md)

手册提供完整的 **零 API 程序监督路径** 和可切换教师的路径。V2 为 2,312 个状态 / 6,936 题；V3 为新增 500 张地图 / 3,000 状态 / 9,000 题。V3 五组各在单张 A100 80GB 上训练 1,200 步，约 8.3–10.0 分钟；五组均从同一 V2 teacher checkpoint 继续训练，不能把 gold 更新分支称为从未接触教师的权重。

真实在线推理需自行准备兼容 checkpoint：

```bash
python scripts/serve_decisions.py   --checkpoint-dir runs/v3_teacher_coords_multi_seed17   --web-root web --port 8765
```

Python/CUDA 依赖见 [requirements-toy.txt](requirements-toy.txt)。`--disable-native-triton` 仅为原实验宿主的兼容回退。仓库发布源码、生成器、评测证据与回放；**训练权重和历史私有教师训练标签未随 Git 仓库上传**。可重建新的程序监督实验；复算历史训练则需相应冻结标签与 checkpoint，不能用公开摘要反推。[交付与复现边界](research/release_scope_zh.md)

## 复核三方演示

```bash
# 只验证公开结果与轨迹，不调用 API、不训练模型
python3 scripts/verify_nanojev_comparison.py --public-only \
  --output /tmp/nanojev-public-verification.json
```

验证器检查合法动作、真实状态转移、完整 horizon、概率选择、采样 RNG、地图 cohort 和 Jev 调用来源引用。历史私有日志与原始数据的加强核验在本次发布前执行；公开副本可复核的范围见验证器输出和协议。

原始 Qwen 的模型下载、数据构建与 GPU 基线复跑见 [完整基线命令](research/navigation_v3_native_qwen_zh.md)。新采集 Jev 响应需要 Node.js ≥22，安装锁定依赖并配置本机密钥：

```bash
npm ci
cp .env.example .env
# 在 .env 中设置 AI_GATEWAY_API_KEY；不要提交该文件

# 先按基线文档重建 data/native_rebuild_v3，再采集新的闭环 Jev 结果
python3 scripts/evaluate_live_jev_navigation.py \
  --data-dir data/native_rebuild_v3 \
  --journal-dir data/jev_rebuild_journal \
  --output-dir artifacts/jev_rebuild --budget-usd 2
```

三方新增 Jev 对照实际发出 145 次成功请求 / 435 题，花费 **$0.003966648**；循环状态复用本次运行的精确输入缓存，未使用旧训练标签冒充现场 API。全部网关实验累计 **$0.138601428**，余额 **$24.861398572**（2026-09-17）。Claude CLI 的标价用量和用户 GPU 资源另记。[预算明细](research/budget_summary.json)

## 研究记录与许可

[官网、案例与公开来源](research/jev_public_sources_zh.md) · [X、Discord 线索与 RLCD](research/rlcd_public_discussion_zh.md) · [RLCD 数学分析](research/rlcd_theory_zh.md) · [已有 OpenJev 仓库审计](research/openjev_repo_audit_zh.md) · [jevlike 审计](research/jevlike_repo_audit_zh.md) · [早期 toy 实验](research/toy_experiment_report_zh.md)

代码采用 [MIT License](LICENSE)，自写程序生成数据标记 CC0；第三方模型、服务材料及其权利分别记录。NanoJev 是独立研究项目，与 TypeSafe/Jev 无官方关联。历史文件中的 OpenJev 名称与 schema 保留以便追溯，项目现名 NanoJev。
