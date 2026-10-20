# 使用 vLLM 插件（开源方式）部署 Xing4.0-29B-A4B 模型

　　本文介绍如何通过 vLLM 的**开源插件（general plugin）**机制，以 vLLM 官方镜像为基础环境，在不改动 vLLM 源码树的前提下部署 Xing4.0-29B-A4B 模型的 OpenAI 兼容 API 服务。主要覆盖以下内容：

- 环境准备（vLLM 官方镜像）
- Xing4.0-29B-A4B 模型下载
- 获取插件并启动容器
- 启动 vLLM 服务
- 插件加载状态验证与开关说明
- OpenAI 兼容 API 调用示例
- 启动参数说明与推理参数建议
- 常见问题排查

　　Xing4.0 的 vLLM 适配代码以**开源插件**方式发布，托管于 [XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin)。插件以 Python 包形式分发，通过 Python entry point 注册到 `vllm.general_plugins` 组，由 vLLM 在进程启动时自动加载，可直接配合 vLLM 官方镜像使用，无需改动 vLLM 源码树，也便于独立升级与维护。插件**不包含自定义 CUDA 内核，也不需要单独编译**，其实现复用 vLLM 既有的运行期算子（包括 `vllm.model_executor.layers.mhc`）。

> [!NOTE]
> 插件注册的内容包括：`Xing4_0ForCausalLM` 模型实现、`xing4_0` 模型配置（复用 vLLM 的 MLA + MoE 配置基础设施）、`xing4_0` 思维链解析器（reasoning parser）、`xing4_0` 工具调用解析器（tool-call parser），以及 `xing4_0` 投机解码（MTP）与 MLA 识别支持。

## 环境准备

　　请确认宿主机满足以下软件要求：

- Docker（建议 24.0 及以上）
- NVIDIA 驱动（建议 535 及以上）
- NVIDIA Container Toolkit（`nvidia-docker` 运行时）
- GPU 支持 bfloat16（mHC 模式要求 bfloat16）

　　本文以 vLLM 官方镜像 `vllm/vllm-openai` 作为基础环境：**无需在宿主机上安装 vLLM，也无需自行编译 vLLM**，只需在容器内安装本插件即可启用 Xing4.0 支持。

　　校验 GPU 是否可被容器访问：

```bash
docker run --rm --gpus all vllm/vllm-openai:latest nvidia-smi
```

　　容器内所需的其他依赖（如 tilelang，CUDA 上 mHC 路径所需）通常已随镜像中的 vLLM 一并安装；若缺失，可在容器内执行 `pip install tilelang`。

> [!NOTE]
> 镜像 tag 可按需固定为具体的 vLLM 版本。插件由 vLLM 上游 commit `2e09247c2d7b6b97d13af6e71a85bf8d1271deb6` 移植而来，该 commit 记录在 `src/vllm_xing4_0_plugin/__init__.py` 的 `TESTED_VLLM` 常量中；生产环境建议固定版本，而非使用 `latest`。

> [!NOTE]
> 本文以 `--tensor-parallel-size 2` 为例，需要至少 2 张 GPU；可根据实际卡数调整该参数。文中的 NVIDIA H100（单卡 80GB 显存，2 卡共 160GB）仅为**演示与验证所用的硬件环境**，并非强制要求。

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
> mHC 模式（`hc_mult > 1`）仅支持 `bfloat16`，且暂不支持流水线并行（pipeline parallelism）。插件会在不满足条件时给出明确报错。

## 获取插件并启动容器

　　第一步，获取插件源码，供挂载进容器使用：

```bash
git clone https://github.com/XingChen-AGI/xing4-vllm-plugin.git
```

　　第二步，启动容器（以 bash 作为入口点并保持运行），挂载模型目录与插件仓库，并暴露 API 端口：

```bash
docker run -d \
  --name xing4-vllm-plugin \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v /yourpath/models/Xing4.0-29B-A4B:/models/Xing4.0-29B-A4B \
  -v /yourpath/xing4-vllm-plugin:/workspace/xing4-vllm-plugin \
  vllm/vllm-openai:latest \
  -c "sleep infinity"
```

　　第三步，进入容器，在容器内安装插件（容器内已具备 vLLM 环境）：

```bash
docker exec -it xing4-vllm-plugin bash

pip install -e /workspace/xing4-vllm-plugin --no-deps
```

　　第四步，验证 entry point 对容器内的解释器可见：

```bash
python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

　　输出中应包含：

```
('xing4_0', 'vllm_xing4_0_plugin:register')
```

> [!NOTE]
> 仅把 `src/` 加入 `PYTHONPATH` **不能**完成插件发现，因为 entry point 来自已安装包的元数据，必须完成安装步骤。`--no-deps` 用于避免插件连带替换容器内的 vLLM 安装。

## 启动 vLLM 服务

　　在容器内执行以下命令启动服务，无需任何额外的插件参数：

```bash
vllm serve /models/Xing4.0-29B-A4B \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.0-29B-A4B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 4 \
    --reasoning-parser xing4_0 \
    --tool-call-parser xing4_0 \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"mtp", "num_speculative_tokens": 1}'
```

　　启动后通过 `http://localhost:8000/v1` 即可访问 OpenAI 兼容 API。当前终端会持续输出服务日志，当日志中出现类似 `Uvicorn running on http://0.0.0.0:8000` 的字样时，表示服务已对外可用。

> [!NOTE]
> 上述命令启用了 MTP 投机解码（模型仓库自带 MTP 层时可用），如不需要可去掉 `--speculative-config`。`mtp` 是 vLLM 对 MTP 类草稿模型的投机解码方法名；不指定 `model` 时，vLLM 会自动复用目标模型权重中的 MTP 层，插件已在注册时把 `xing4_0` 的 MTP 草稿配置映射到该路径。

### 插件加载状态验证

　　服务启动时，请在日志中确认插件的注册记录：

```
vllm-xing4_0-plugin: registered Xing4_0ForCausalLM -> vllm_xing4_0_plugin.xing4_0:Xing4_0ForCausalLM (tested against vLLM commit 2e09247c2d7b6b97d13af6e71a85bf8d1271deb6, running <vLLM 版本>)
```

　　若运行环境中已内置 Xing4_0 支持（例如预编译镜像 `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0`），插件会打印让位日志并将环境内实现保持为活动实现：

```
vllm-xing4_0-plugin: Xing4_0ForCausalLM is already supported in tree; standing down. Set XING4_0_PLUGIN_OVERRIDE=1 to override.
```

　　如需显式使用本插件（移植）实现，设置环境变量后启动：

```bash
XING4_0_PLUGIN_OVERRIDE=1 vllm serve /models/Xing4.0-29B-A4B ...
```

　　如果环境中使用了 `VLLM_PLUGINS` 白名单，需将本插件的 entry point 名称 `xing4_0` 加入白名单，否则插件不会被加载：

```bash
VLLM_PLUGINS=xing4_0 vllm serve /models/Xing4.0-29B-A4B ...
```

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
| `--host 0.0.0.0` | 监听所有网卡，使容器外可访问 |
| `--port 8000` | API 服务端口 |
| `--served-model-name` | 对外暴露的模型名，调用 API 时需与之对应 |
| `--tensor-parallel-size 2` | 张量并行卡数，需与可用 GPU 数一致 |
| `--trust-remote-code` | 信任模型仓库内的远程代码 |
| `--max-model-len 262144` | 最大上下文长度，Xing4.0 原生支持 256K |
| `--gpu-memory-utilization 0.90` | 预留显存比例上限，按显存余量调整 |
| `--max-num-seqs 4` | 最大并发请求数，按业务并发与显存调整 |
| `--reasoning-parser xing4_0` | 解析 Xing4.0 的思维链输出（由插件注册） |
| `--tool-call-parser xing4_0` | 解析 Xing4.0 的工具调用格式（由插件注册） |
| `--enable-auto-tool-choice` | 启用自动工具调用（function calling），请求带 `tools` 参数时必须开启 |
| `--speculative-config` | 启用 MTP 投机解码以加速生成，`method` 使用 `mtp` |
| `--chat-template` | 可选：指定模型仓库外的 `chat_template.jinja` 路径 |

## 推理参数建议

　　参考 Xing4.0 官方建议，不同场景下的采样参数选择如下：

| 场景 | temperature | top_p | repetition_penalty |
|------|-------------|-------|---------------------|
| 复杂推理 / 通用任务 | 1.0 | 0.95 | 1.05 |
| 代码 / 智能体任务 | 0.8 | 0.95 | 1.05 |

## 常见问题

### 插件没有加载

- 确认插件安装在**容器内与 vLLM 相同的 Python 环境**中，且完成的是包安装而非仅设置 `PYTHONPATH`
- 用「获取插件并启动容器」一节的命令检查 entry point 是否可见，输出应包含 `('xing4_0', 'vllm_xing4_0_plugin:register')`
- 若设置了 `VLLM_PLUGINS` 白名单，需包含 `xing4_0`
- 日志中未出现 `vllm-xing4_0-plugin: registered ...` 时，先确认是否触发了「让位」逻辑（详见「插件加载状态验证」）

### 模型架构不被识别（Xing4_0ForCausalLM 不存在）

- 插件未成功注册时，vLLM 会因架构缺失而拒绝加载模型。按上一条排查注册与日志
- 若使用预编译镜像等已内置支持的环境，可设置 `XING4_0_PLUGIN_OVERRIDE=1` 强制改用插件实现

### 提示缺少 tilelang

- CUDA 上 mHC 路径需要 tilelang，在容器内执行 `pip install tilelang` 后重启服务

### mHC 相关报错（bfloat16 / 流水线并行）

- mHC 模式要求模型与推理均为 `bfloat16`；FP16 或量化路径会报错
- mHC 暂不支持流水线并行，请勿设置 `--pipeline-parallel-size > 1`

### 显存不足（OOM）

- 降低 `--gpu-memory-utilization`（如 0.85）或 `--max-model-len`（如 131072）
- 使用 FP8 版本模型以减少显存占用
- 减少 `--max-num-seqs` 以降低并发显存峰值

### 张量并行卡数不匹配

- `--tensor-parallel-size` 必须能整除可见 GPU 数；如使用 4 卡可设为 `4` 或 `2`

### 容器内无法访问 GPU

- 确认已安装 NVIDIA Container Toolkit，并在启动时使用 `--gpus all`
- 若只希望使用部分卡，可改用 `--gpus '"device=0,1"'`

### 服务无法从外部访问

- 确认 `--host` 设为 `0.0.0.0`，并使用 `-p 8000:8000` 映射端口

## 升级与维护

　　插件依赖 vLLM 的内部运行期 API，`TESTED_VLLM` 常量记录了移植所基于的 vLLM 版本。升级 vLLM（即更换镜像 tag）后建议：

- 在容器内的插件仓库目录 `/workspace/xing4-vllm-plugin` 下重新运行插件自带的冒烟测试（无需 GPU 与模型权重），确认注册入口、MTP/MLA 补丁与解析器功能正常：

```bash
python -m pytest tests/test_smoke.py -q
```

- 观察启动日志中插件打印的「移植基线版本」与「运行中的 vLLM 版本」，若两者偏差较大，需关注内部 API 是否发生迁移，必要时重新移植对应模块
