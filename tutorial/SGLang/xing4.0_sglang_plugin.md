# 使用 SGLang 插件（开源方式）部署 Xing4.0-29B-A4B 模型

　　本文介绍如何通过 SGLang 的**开源插件机制**，以官方/原生 SGLang 环境为基础，在不改动 SGLang 源码树的前提下部署 Xing4.0-29B-A4B 模型的 OpenAI 兼容 API 服务。主要覆盖以下内容：

- 环境准备（SGLang 环境）
- Xing4.0-29B-A4B 模型下载
- 获取插件并启动容器
- 启动 SGLang 服务
- 插件加载状态验证与开关说明
- OpenAI 兼容 API 调用示例
- 启动参数说明与推理参数建议
- 常见问题排查

　　Xing4.0 的 SGLang 适配代码以**开源插件**方式发布，托管于 [XingChen-AGI/xing4-sglang-plugin](https://github.com/XingChen-AGI/xing4-sglang-plugin)。插件以 Python 包形式分发，通过 Python entry point 注册到 `sglang.srt.plugins` 组，由 SGLang 在进程启动时自动加载，可直接配合原生 SGLang 环境使用，无需改动 SGLang 源码树，也便于独立升级与维护。插件**不包含自定义 CUDA 内核，也不需要单独编译**，其实现复用 SGLang 既有的运行期算子（MLA / MoE / DeepGEMM / FlashInfer）。

> [!NOTE]
> 插件注册的内容包括：`Xing4_0ForCausalLM` 模型实现、`xing4_0` 模型配置（复用 SGLang 的 MLA + MoE 配置基础设施）、`xing4_0` 思维链解析器（reasoning parser）、`xing4_0` 工具调用解析器（tool-call parser），以及 `xing4_0` 投机解码（EAGLE / NextN）与 MLA 识别支持。此外，插件通过 `sglang_xing4_0_plugin.compat` 下的 HookRegistry 钩子补齐上游 PR 尚未合入 SGLang 主干的若干内部改动，这些钩子均为惰性、幂等，可按需单独关闭（见「插件加载状态验证」）。

## 环境准备

　　请确认宿主机满足以下软件要求：

- Docker（建议 24.0 及以上，使用容器方式时）
- NVIDIA 驱动（建议 535 及以上）
- NVIDIA Container Toolkit（`nvidia-docker` 运行时）
- GPU 支持 bfloat16（mHC 模式要求 bfloat16）

　　本文以**官方 SGLang 镜像**`lmsysorg/sglang` 作为基础环境：**无需在宿主机上安装 SGLang，也无需自行编译 SGLang**，只需在容器内安装本插件即可启用 Xing4.0 支持。若你已在物理机或自建镜像中安装了 SGLang，也可跳过容器步骤，直接在**同一 Python 环境**中安装插件。

　　校验 GPU 是否可被容器访问：

```bash
docker run --rm --gpus all lmsysorg/sglang:latest nvidia-smi
```

> [!NOTE]
> 镜像 tag 可按需固定为具体的 SGLang 版本。插件由 SGLang 上游 PR [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793)（分支 `support_xing4_0`）移植而来，基线 commit 记录在 `src/sglang_xing4_0_plugin/__init__.py` 的 `TESTED_SGLANG` 常量中；生产环境建议固定版本，而非使用 `latest`。

> [!NOTE]
> 本文以 `--tp-size 2` 为例，需要至少 2 张 GPU；可根据实际卡数调整该参数。文中的硬件环境仅为**演示与验证所用**，并非强制要求。

## 下载 Xing4.0-29B-A4B 模型

　　Xing4.0-29B-A4B 模型可从以下平台下载，三端权重一致：

| 平台 | 地址 |
|------|------|
| Hugging Face | https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B |
| ModelScope | https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B |
| Modelers | https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B |

　　以 ModelScope 为例，下载到本地目录：

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.0-29B-A4B \
  --local_dir /yourpath/models/Xing4.0-29B-A4B
```

　　如需进一步降低显存占用，也可使用 FP8 版本（[Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)），下载与启动方式与下方一致，仅需将挂载路径指向 FP8 目录即可。

> [!NOTE]
> mHC 模式（`hc_mult > 1`）仅支持 `bfloat16`，且暂不支持流水线并行。插件会在不满足条件时给出明确报错。

## 获取插件并启动容器

　　第一步，获取插件源码，供挂载进容器使用：

```bash
git clone https://github.com/XingChen-AGI/xing4-sglang-plugin.git
```

　　第二步，启动容器（以 bash 作为入口点并保持运行），挂载模型目录与插件仓库，并暴露 API 端口：

```bash
docker run -d \
  --name xing4-sglang-plugin \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v /yourpath/models/Xing4.0-29B-A4B:/models/Xing4.0-29B-A4B \
  -v /yourpath/xing4-sglang-plugin:/workspace/xing4-sglang-plugin \
  lmsysorg/sglang:latest \
  -c "sleep infinity"
```

　　第三步，进入容器，在容器内安装插件（容器内已具备 SGLang 环境）：

```bash
docker exec -it xing4-sglang-plugin bash

pip install -e /workspace/xing4-sglang-plugin --no-deps
```

　　第四步，验证 entry point 对容器内的解释器可见：

```bash
python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='sglang.srt.plugins')])"
```

　　输出中应包含：

```
('xing4_0', 'sglang_xing4_0_plugin:register')
```

> [!NOTE]
> 仅把 `src/` 加入 `PYTHONPATH` **不能**完成插件发现，因为 entry point 来自已安装包的元数据，必须完成安装步骤。`--no-deps` 用于避免插件连带替换容器内的 SGLang 安装。

## 启动 SGLang 服务

　　在容器内执行以下命令启动服务，无需任何额外的插件参数：

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1

sglang serve --model-path /models/Xing4.0-29B-A4B \
   --trust-remote-code \
   --host 0.0.0.0 \
   --port 8000 \
   --served-model-name Xing4.0-29B-A4B \
   --tp-size 2 \
   --context-length 262144 \
   --mem-fraction-static 0.90 \
   --max-running-requests 4 \
   --reasoning-parser xing4_0 \
   --tool-call-parser xing4_0 \
   --speculative-algorithm EAGLE
```

　　启动后通过 `http://localhost:8000/v1` 即可访问 OpenAI 兼容 API。当服务日志中出现类似 `Uvicorn running on http://0.0.0.0:8000` 的字样时，表示服务已对外可用。

> [!NOTE]
> `export FLASHINFER_DISABLE_VERSION_CHECK=1` 用于跳过 FlashInfer 与 CUDA 版本的校验（例如 FlashInfer 0.7.x + CUDA 13.0 组合），不设置时可能因版本门禁而启动失败。

> [!NOTE]
> `--speculative-algorithm EAGLE` 可按需去掉。Xing4_0 的 checkpoint 自带 NextN（MTP）草稿头，插件已将 `--speculative-draft-model-path` 默认指向主模型路径，因此**无需单独指定草稿模型**。

> [!IMPORTANT]
> SGLang 0.5.x 对 EAGLE 系列参数有**三选一**约束：要么完全自动（不设置 `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens`），要么完全手动（三者全设），要么使用 `--speculative-adaptive`。**只设置其中两个会触发 `AssertionError`。**

### 插件加载状态验证

　　服务启动时，请在日志中确认插件的注册记录：

```
sglang-xing4_0-plugin: registered Xing4_0ForCausalLM (models=sglang_xing4_0_plugin.models, parser=xing4_0, vendored from sglang commit aa737beda5 (0.4.x) / 0.5.22)
```

　　若运行环境中已内置 Xing4_0 支持（例如预编译镜像 `quay.io/xingchen-agi/xingchen-inference-sglang:v0.5.20.rc1-xing4_0`），插件会打印让位日志并将环境内实现保持为活动实现：

```
sglang-xing4_0-plugin: Xing4_0ForCausalLM is already supported in tree; standing down. Set XING4_0_PLUGIN_OVERRIDE=1 to override with the vendored copy.
```

　　如需显式使用本插件（移植）实现，设置环境变量后启动：

```bash
XING4_0_PLUGIN_OVERRIDE=1 sglang serve --model-path /models/Xing4.0-29B-A4B ...
```

　　compat 层的开关（用于在上游 PR 逐步合入后逐项关闭对应钩子，全部幂等）：

```bash
# 关闭整个 compat 层
XING4_0_COMPAT_OFF=1 sglang serve --model-path /models/Xing4.0-29B-A4B ...

# 关闭单个钩子项
XING4_0_COMPAT_MHC_PRE=0 sglang serve --model-path /models/Xing4.0-29B-A4B ...
```

> [!NOTE]
> SGLang 目前未提供类似 `VLLM_PLUGINS` 的插件白名单环境变量。若需屏蔽本插件，可通过调整 `PYTHONPATH` 或卸载插件实现。

## 调用 API

　　服务启动后，可使用 OpenAI 兼容接口进行调用。以下示例中的 `model` 字段需与启动时的 `--served-model-name` 一致。

### 基础对话（curl）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Xing4.0-29B-A4B",
    "messages": [
      {"role": "user", "content": "用一句话介绍你自己。"}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "chat_template_kwargs":{"enable_thinking":true},"skip_special_tokens": false
  }'
```

　　关闭思维链输出时，在请求体中传入 `"chat_template_kwargs":{"enable_thinking":false}`。开启思维链时生成内容形如 `<think>推理过程</think>正文内容`；关闭时仅输出正文内容。

## 启动参数说明

| 参数 | 说明 |
|------|------|
| `--model-path` | 模型目录路径 |
| `--host 0.0.0.0` | 监听所有网卡，使容器外可访问 |
| `--port 8000` | API 服务端口 |
| `--served-model-name` | 对外暴露的模型名，调用 API 时需与之对应 |
| `--tp-size 2` | 张量并行卡数，需与可用 GPU 数一致 |
| `--trust-remote-code` | 信任模型仓库内的远程代码 |
| `--context-length 262144` | 最大上下文长度，Xing4.0 原生支持 256K |
| `--mem-fraction-static 0.90` | 预留显存比例上限，按显存余量调整 |
| `--max-running-requests 4` | 最大并发请求数，按业务并发与显存调整 |
| `--reasoning-parser xing4_0` | 解析 Xing4.0 的思维链输出（由插件注册） |
| `--tool-call-parser xing4_0` | 解析 Xing4.0 的工具调用格式（由插件注册） |
| `--speculative-algorithm EAGLE` | 启用 EAGLE 投机解码以加速生成（可选，参数约束见上文） |
| `--speculative-adaptive` | 使用自适应投机解码参数（与手动/自动模式三选一） |
| `--chat-template` | 可选：指定模型仓库外的 `chat_template.jinja` 路径 |

## 推理参数建议

　　参考 Xing4.0 官方建议，不同场景下的采样参数选择如下：

| 场景 | temperature | top_p | repetition_penalty |
|------|-------------|-------|---------------------|
| 复杂推理 / 通用任务 | 1.0 | 0.95 | 1.05 |
| 代码 / 智能体任务 | 0.8 | 0.95 | 1.05 |

## 常见问题

### 插件没有加载

- 确认插件安装在**与 SGLang 相同的 Python 环境**中，且完成的是包安装而非仅设置 `PYTHONPATH`
- 用「获取插件并启动容器」一节的命令检查 entry point 是否可见，输出应包含 `('xing4_0', 'sglang_xing4_0_plugin:register')`
- 日志中未出现 `sglang-xing4_0-plugin: registered ...` 时，先确认是否触发了「让位」逻辑（详见「插件加载状态验证」）

### 模型架构不被识别（Xing4_0ForCausalLM 不存在）

- 插件未成功注册时，SGLang 会因架构缺失而拒绝加载模型。按上一条排查注册与日志
- 若使用已内置支持的环境（如预编译镜像），可设置 `XING4_0_PLUGIN_OVERRIDE=1` 强制改用插件实现

### EAGLE 启动报 AssertionError

- SGLang 0.5.x 要求 `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens` 三者同时设置（手动模式），或全部不设置（自动模式），或改用 `--speculative-adaptive`；只设置其中两个会触发断言失败
- 对于 MTP 类自带草稿头，自动模式（不设置三者）通常已足够，`topk` 取大于 1 的值反而可能降低接受率

### mHC 相关报错（bfloat16 / 流水线并行）

- mHC 模式要求模型与推理均为 `bfloat16`；FP16 或量化路径会报错
- mHC 暂不支持流水线并行

### 提示 FlashInfer 版本不匹配

- 设置 `FLASHINFER_DISABLE_VERSION_CHECK=1` 后重启服务

### 显存不足（OOM）

- 降低 `--mem-fraction-static`（如 0.85）或 `--context-length`（如 131072）
- 使用 FP8 版本模型以减少显存占用
- 减少 `--max-running-requests` 以降低并发显存峰值

### 张量并行卡数不匹配

- `--tp-size` 必须能整除可见 GPU 数；如使用 4 卡可设为 `4` 或 `2`

### 容器内无法访问 GPU

- 确认已安装 NVIDIA Container Toolkit，并在启动时使用 `--gpus all`
- 若只希望使用部分卡，可改用 `--gpus '"device=0,1"'`

### 服务无法从外部访问

- 确认 `--host` 设为 `0.0.0.0`，并使用 `-p 8000:8000` 映射端口

## 升级与维护

　　插件依赖 SGLang 的内部运行期 API，`TESTED_SGLANG` 常量记录了移植所基于的 SGLang 版本。升级 SGLang（即更换镜像 tag 或重新安装）后建议：

- 在容器内的插件仓库目录 `/workspace/xing4-sglang-plugin` 下重新运行插件自带的测试（无需 GPU 与模型权重），确认注册入口与解析器功能正常：

```bash
python -m pytest tests/ -q
```

- 观察启动日志中插件打印的「移植基线版本」与「运行中的 SGLang 版本」，若两者偏差较大，需关注内部 API 是否发生迁移，必要时重新移植对应模块
- 随着上游 PR [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793) 逐步合入主干，可通过 `XING4_0_COMPAT_*` 逐项关闭对应的 compat 钩子；当全部内容均已进入主干后，`register()` 会自动让位
