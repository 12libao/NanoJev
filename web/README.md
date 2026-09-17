# NanoJev 实验阅读页

无框架、无依赖的中文页面，读取同目录的 `demo_results.json`。展示的是实际记录的回放，不把回放标成实时推理；文件缺失时显示空状态，不生成假成功数据。

从仓库根目录启动只读静态服务：

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

打开 `http://127.0.0.1:8080`。不要直接双击 HTML：`file://` 下浏览器通常无法读取 JSON。静态服务不要指向整个仓库，避免把 `.env` 和私有研究文件暴露给网页访问。

页面提供模型/基线选择、对局选择、前后步进、速度调节、播放暂停、完整步骤 JSON、概率条、执行证据及来源链接。概率原值不重新归一化；缺失分布不会补 one-hot。动作概率不等于终局胜率。窄屏可用，使用本地系统字体，无 CDN 或外部脚本。

V3 使用 `summary.test/ood`，V2 使用 `summary[split/game]`；页面按各自结构展示，并标明版本、控制器和评测集合。V2/V3 不是同一批地图，不能把两组数字当配对比较。若文件含预登记的 `v3_gold_coords_multi_seed17` greedy，首次载入默认选择该组；不会读取测试成绩来决定默认项，刷新时保留用户选择。同 checkpoint 的 greedy/sample 是两个控制器，并非两个独立训练的模型。

## 数据契约

基本结构如下，只有字段示意，没有成功轨迹或概率示例：

```json
{
  "generated_at": "实验实际生成时间",
  "models": [
    {
      "name": "模型名称或明确的基线名称",
      "checkpoint_sha256": "实际权重哈希；基线可以不提供",
      "summary": {},
      "episodes": []
    }
  ],
  "parallel_batches": [],
  "sources": []
}
```

每个 episode：`id, game, split, initial_state, steps, outcome`；可附 `final_state`。`steps` 每项：

- `state`：动作前的环境状态。
- `action`：实际执行动作。井字棋 `cell_1`–`cell_9`；网格 `north/east/south/west`。
- `controller` 可为 `greedy` / `sample`；采样时始终用 `step.action` 表示实际动作，不用 `answers.action.choice` 的 argmax 覆盖。`distribution_argmax` 可另外记录，概率条标出实际抽中项。
- `probabilities`：动作 ID 到原生概率的映射。缺失时页面尝试从 `answers` 的 Choice 中读取。
- `answers`：可选的完整同状态多题答案；不把它称为多状态同批。
- `next_state`：动作后的环境状态。
- `model_forward`：布尔值或实际执行信息对象；缺失明确显示“未记录”。
- `forced`：该动作是否由唯一合法选项/环境强制；`actor` 可明确模型或对手。
- 可选 `latency_ms` / `model_latency_ms` / `execution`。

支持的环境状态与当前 `scripts/game_tasks.py` 一致：

```text
grid_navigation: { game, size, walls:[[row,col],...], position:[row,col], goal:[row,col] }
tic_tac_toe:       { game, board:"九个字符，. / X / O", player:"X 或 O" }
```

坐标为 0 起始行列。环境状态也可包装在 `environment_state` 或 `metadata.environment_state`；其它 state 会以 JSON 降级展示，不猜测棋盘。

多状态同批须提供独立的执行记录：

```text
parallel_batches: [{
  id, model,
  execution: { forward_passes, precision, total_paths, ... },
  states: [{ id, state?, answers: { question_id: { type, probabilities, choice?, p_true?, score?, level? } } }]
}]
```

可以放在顶层或某个 model 内。没有 `forward_passes` 时，页面不宣称一次前向；不会把不同时刻的多个步骤伪装成同一个批次。Boolean 只有 `p_true` 时，可视化展示 `{false:1-p_true,true:p_true}`，原始 JSON 保持服务返回。

`sources` 支持 `{title,url}` 或 URL 字符串。只接受 HTTP/HTTPS 链接。页面固定列出官网 Doom、X 原帖、第三方 Jevlike 并说明网格/井字棋只是独立启发；其它源可以追加。

## 可选实时接口

“实时接口”页把编辑器中的 JSON 原样 POST 到**当前同源** `/api/evaluate`，不读取 `.env`，不含任何 API 密钥，不直接调用 Jev 或其它付费模型。

静态 `http.server` 不实现这个接口，页面会明确提示启动模型推理服务。后端可在同一端口同时提供本目录静态文件和模型接口。请求 schema 由后端决定；默认输入编辑器采用 `{states:[{id,state,questions}]}`。推荐返回 `{states:[{id,answers}],execution}`，页面也保留所有原始响应 JSON。

模型应在后端启动时加载一次，各请求实际执行前向；页面只显示真实响应，不回退预设答案。120 秒浏览器超时不保证终止服务端计算。浏览器往返耗时与纯 GPU 时延分开解读。

## 检查

```bash
node --check web/app.js
```

结果生成后应在浏览器检查：模型/对局切换、最后一步 next_state、暂停与滑块、窄屏、强制动作、失败轨迹、非单位和概率、同批证据、JSON 展开，以及静态服务下实时接口的缺失提示。具体实际验证记录由本次交付报告说明，不把语法检查写成完整浏览器验收。

2026-09-17 已用临时 Playwright 与隔离无登录 Chrome 153 验收空数据/错误状态，以及首版真实 artifact 的 4 模型、320 局加载。逐模型抽查两类游戏，核对前后步进、终局状态与原始记录，检查 80 状态/1 次 forward 批次、自动播放和 390px 窄屏；页面 JavaScript 异常为 0。首个网格截图保留模型达到 32 步上限的失败，不挑成功片段。

证据：[空状态检查](../research/web_empty_browser_check.json)、[真实记录检查](../research/web_real_browser_check.json)、[截图目录](../research/web_screenshots/)。这次检查不等于对每一局每一帧穷尽验证。

随后已通过实际 GPU 服务的浏览器 POST：3 states、9 questions、21 candidate paths、1 次 forward、0 decode、0 网络模型调用，persistent load count=1、call index=3。页面展示与真实响应 JSON 一致；浏览器往返约 132 ms，服务计时约 66.7 ms，这是一条请求记录而不是性能分位数。见 [实时检查](../research/web_live_browser_check.json)、[实际请求和响应](../research/web_live_browser_response.json)。

V3 整合后已在真实浏览器验证 16 个展示项、800 局的加载和分组摘要，默认预定主组 greedy。实际 sample 步骤抽中 east（39.79%）而 argmax 为 north（60.21%），页面标记正确；主组 OOD 失败、6 states/18 questions/1 forward 及 390px 窄屏均检查通过，pageerror=0。见 [V3 检查](../research/web_v3_browser_check.json) 和 [完整验收范围](../research/web_reader_review_zh.md)。这些是回放验收，不把 800 局宣称为逐帧穷尽检查。

服务切换到预定主 V3 checkpoint 后，也已完成新的真实浏览器 POST：3 states/9 questions/21 paths/1 forward、0 decode、0 网络模型调用、HTTP 200。见 [V3 实时检查](../research/web_v3_live_browser_check.json) 与 [原始响应](../research/web_v3_live_browser_response.json)；浏览器约 130 ms、服务约 65.7 ms 仅是这条请求的观察。

## NanoJev 三方同例视频

`comparison.html` 读取同目录 `comparison_results.json`，三方名称固定为“训练后的 NanoJev / Jev API / 原始 Qwen”。NanoJev 使用固定的 `v3_teacher_coords_multi_seed17`，原始 Qwen 使用未经本项目微调的 Qwen3-0.6B。案例固定为原 V3 前两个 TEST、前两个 OOD，greedy/sample 分开；不根据结果选地图，也不以推理耗时控制动画速度。每格按同一环境步推进，先结束的轨迹保持真实终局。

制作实际媒体需要已有的三方轨迹、Chrome、Playwright 和 FFmpeg。渲染器不调用 API 或 GPU，不安装依赖，目标文件存在时拒绝覆盖：

```bash
python3 scripts/render_comparison_video.py \
  --trained research/navigation_v3_v3_teacher_coords_multi_seed17_greedy.json \
  --trained research/navigation_v3_v3_teacher_coords_multi_seed17_sample.json \
  --jev research/navigation_v3_jev_api_greedy.json \
  --jev research/navigation_v3_jev_api_sample.json \
  --base research/navigation_v3_native_qwen_greedy.json \
  --base research/navigation_v3_native_qwen_sample.json \
  --ffmpeg /path/to/ffmpeg --playwright-module /path/to/playwright/index.mjs
```

输出为实际 reader 数据与 `assets/comparison_greedy.mp4`、`comparison_sample.mp4`，各有 GIF 和 PNG 海报。GIF 是各完整视频固定开头片段；不是另挑的最佳表现。manifest 记录源轨迹、页面与媒体 SHA256，并逐帧核对实际动作、同步环境和终局。原始 Qwen 的概率是提供的 A–D 答案标签条件分布，不等于完整词表分布；动作概率也不是最终通关率。

本次已实际生成并验收：两条 H.264 MP4 均为 1600×1000、约 63.42 秒，分别约 688 KB / 757 KB；两个 GIF 均为 800×500、48 帧、固定前 12 秒，分别约 414 KB / 455 KB。逐帧捕获共 424 帧，全部检查环境状态、实际动作和每个案例最后一帧；浏览器切换控制器/地图、步进、播放暂停和终局保持均通过。导出 MP4 可在 Chrome 解码并精确跳转末帧，另已提取编码后末帧目视检查。证据在 `assets/comparison_media_manifest.json` 与 `comparison_playback_check.json`。

静态阅读页本身无依赖。重新制作媒体需要 Python 标准库、Node.js、Playwright（本次 1.63.0）、Chrome（本次 153）及带 libx264 的 FFmpeg（本次 7.1）；临时安装的 Pillow 只用于额外检查 GIF 帧数，不是渲染器依赖。现有媒体不会覆盖，重新生成请另加 `--data-output /tmp/nanojev-comparison.json --output-dir /tmp/nanojev-media`。本次制作没有新 API 或 GPU 推理调用。
