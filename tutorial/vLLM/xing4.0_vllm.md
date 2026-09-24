# 使用 vLLM 部署 Xing4.0-29B-A4B 模型

　　本文介绍如何通过 vLLM 启动 Xing4.0-29B-A4B 模型的 OpenAI 兼容 API 服务。主要覆盖以下内容：

- 预编译镜像拉取
- Xing4.0-29B-A4B 模型下载
- 通过 Docker 部署 vLLM 服务
- OpenAI 兼容 API 调用示例
- 启动参数说明与推理参数建议

　　Xing4.0 的 vLLM 适配代码（[vllm-project/vllm#57135](https://github.com/vllm-project/vllm/pull/57135)）目前尚未合入 vLLM 主分支。在合入前，本文使用已集成该 PR 的预编译镜像 `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0`，无需自行从源码编译即可快速部署。

## 环境准备

　　请确认宿主机满足以下软件要求：

- Docker（建议 24.0 及以上）
- NVIDIA 驱动（建议 535 及以上）
- NVIDIA Container Toolkit（`nvidia-docker` 运行时）

　　校验 GPU 是否可被容器访问：

```bash
docker run --rm --gpus all quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0 nvidia-smi
```

　　本文启动命令以 `--tensor-parallel-size 2` 为例，需要至少 2 张 GPU；可根据实际卡数调整该参数。

> [!NOTE]
> 本文中的 NVIDIA H100（单卡 80GB 显存，2 卡共 160GB）仅为**演示与验证所用的硬件环境**，并非强制要求。用户可根据自身硬件选择其他支持 CUDA 的 NVIDIA GPU，只需根据实际显存与卡数调整 `--tensor-parallel-size`、`--max-model-len`、`--gpu-memory-utilization` 等参数即可；显存有限时也可参考下文使用 FP8 版本模型。

## 拉取镜像

```bash
docker pull quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0
```

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

　　如需进一步降低显存占用，也可使用 FP8 版本（[Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)），下载与启动方式与下方一致，仅需将 `MODEL_PATH` 指向 FP8 目录即可。

## 启动 vLLM 服务

　　通过 Docker 部署分为两步：先启动容器，再在容器内执行 vLLM 启动脚本。

　　第一步，启动容器（以 bash 作为入口点并保持运行），挂载本地模型目录并暴露 API 端口：

```bash
docker run -d \
  --name xing4-vllm \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v /yourpath/models/Xing4.0-29B-A4B:/models/Xing4.0-29B-A4B \
  quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0 \
  -c "sleep infinity"
```

　　第二步，进入容器后执行以下命令启动 vLLM 服务，日志将直接输出到当前终端：

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

### 启动参数说明

| 参数 | 说明 |
|------|------|
| `--host 0.0.0.0` | 监听所有网卡，使容器外可访问 |
| `--port 8000` | API 服务端口 |
| `--served-model-name` | 对外暴露的模型名，调用 API 时需与之对应 |
| `--tensor-parallel-size 2` | 张量并行卡数，需与可用 GPU 数一致 |
| `--trust-remote-code` | 信任模型仓库内的远程代码 |
| `--max-model-len 262144` | 最大上下文长度，Xing4.0 原生支持 256K |
| `--gpu-memory-utilization 0.90` | 预留显存比例上限，按显存余量调整 |
| `--max-num-seqs 4` | 最大并发请求数 |
| `--reasoning-parser xing4_0` | 解析 Xing4.0 的思维链输出（`</think>` 之前为推理过程） |
| `--tool-call-parser xing4_0` | 解析 Xing4.0 的工具调用格式 |
| `--enable-auto-tool-choice` | 启用自动工具调用（function calling），请求带 `tools` 参数时必须开启 |
| `--speculative-config` | 启用 MTP 投机解码以加速生成 |

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

## 推理参数建议

　　参考 Xing4.0 官方建议，不同场景下的采样参数选择如下：

| 场景 | temperature | top_p | repetition_penalty |
|------|-------------|-------|---------------------|
| 复杂推理 / 通用任务 | 1.0 | 0.95 | 1.05 |
| 代码 / 智能体任务 | 0.8 | 0.95 | 1.05 |

## 常见问题

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
