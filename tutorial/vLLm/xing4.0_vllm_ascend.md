# xing4_0 模型推理服务使用说明

> 适用镜像: `harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0`
> 适用范围: 基于 vLLM-Ascend 的 OpenAI 兼容推理服务部署、启动与测试

---

## 1. 概述

本文档说明如何在昇腾(Ascend)NPU 环境下,基于 vLLM-Ascend 推理框架部署并调用 **Xing4.0**大语言模型。

整体流程:

1. 加载镜像并启动容器;
2. 准备主模型权重(草稿模型权重仅 dspark 投机采样模式需要);
3. 启动 OpenAI 兼容的推理服务(`/v1/completions` 等接口);
4. 通过 `curl` 或 OpenAI SDK 调用验证。

---

## 2. 镜像与模型简介

| 项目 | 内容 |
| --- | --- |
| 开源镜像 | `harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0` |
| 镜像离线文件 | `/data01/workspace/lzx/llm/images/xing4_0-29b-v26.tar.gz` |
| 推理框架 | vLLM-Ascend `v0.26.0rc1` |
| 模型规模 | 29B(主模型) |
| 对外服务名 | `xingchen4`(由 `--served-model-name` 指定) |

该镜像已内置针对 Xing4.0 优化的 vLLM-Ascend 环境,并开启多项加速能力(详见第 7 节启动参数详解)。

---

## 3. 环境与资源要求

- **硬件**: 昇腾 **910B2** 系列 NPU,至少 **2 张**卡(启动脚本中 `--tensor-parallel-size 2` 使用 2 卡张量并行);参考容器脚本透传 8 张卡(`/dev/davinci0` ~ `/dev/davinci7`),服务按需取用其中 2 张;
- **软件**: 宿主机需安装与镜像配套的昇腾驱动/CANN 运行环境,并正确挂载 `davinci` 设备节点;
- **存储**:
  - 主模型权重目录(脚本中 `$weight_path`),两种启动方式均需要;
  - 草稿模型权重目录(仅 dspark 投机采样需要):`/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best`。mtp 模式无需草稿模型,详见第 6 节。

---

## 4. 镜像加载与容器启动

### 4.1 加载离线镜像

```bash
docker load -i /data01/workspace/lzx/llm/images/xing4_0-29b-v26.tar.gz
```

加载完成后确认镜像存在:

```bash
docker images | grep vllm-ascend
```

### 4.2 启动容器(参考脚本)

昇腾容器需透传 NPU 设备与驱动目录。参考脚本如下(`$1` 为容器名,`$2` 为镜像名):

```bash
docker run -itd -u 0 \
    --name $1 \
    --net=host \
    --privileged=true \
    --shm-size=512g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt/sfs_turbo/.cache:/root/.cache \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -e VLLM_USE_V1=1 \
    -it $2 /bin/bash
```

使用示例(`$1` = 容器名,`$2` = 镜像名):

```bash
docker run -itd -u 0 --name xing4_0-vllm --net=host --privileged=true --shm-size=512g \
  [--device 与 -v 参数同上方脚本] \
  -e VLLM_USE_V1=1 -it harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0 /bin/bash
```

> 说明:
> - 本脚本适用于 **昇腾 910B2** 环境,透传 `/dev/davinci0` ~ `/dev/davinci7` 共 **8 张** NPU,服务通过 `--tensor-parallel-size 2` 使用其中 2 张;
> - `--net=host` 使用宿主机网络,服务端口(`--port`,dspark=8000 / mtp=8009)直接绑定宿主机,无需 `-p` 端口映射;
> - `-u 0` 以 root 运行;`--privileged=true` 与 `--shm-size=512g` 为昇腾推理容器常用配置;
> - 权重等业务目录按实际环境按需添加 `-v` 挂载;
> - `-e VLLM_USE_V1=1` 启用 vLLM V1 执行引擎。

---

## 5. 模型权重准备

1. 将主模型权重放在宿主机目录,并在启动脚本中通过 `$weight_path` 传入,例如:

   ```bash
   export weight_path=/weights/xing4_0
   ```

2. **草稿模型权重(仅 dspark 模式需要,可选)**:若使用 dspark 投机采样,需确认草稿模型目录存在且可读:

   ```bash
   ls /hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best
   ```

   > mtp 模式使用主模型自带的 MTP(Multi-Token Prediction)预测头,无需单独的草稿模型权重。

---

## 6. 启动推理服务

服务支持两种投机采样方式,**二者选一**;也可都不配置(去掉 `--speculative-config` 即不启用投机采样,服务正常推理,仅吞吐略低):

| 方式 | 投机采样方法 | 是否需要草稿模型 | 监听端口 | 日志文件 |
| --- | --- | --- | --- | --- |
| 方式一 | dspark(可选) | 需要 | `8000` | `dspark-baseline-think.txt` |
| 方式二 | mtp | 不需要(使用主模型 MTP 预测头) | `8009` | `mtp-9.txt` |

### 6.1 方式一:dspark 投机采样(可选)

> **dspark 为可选项**:它依赖额外的草稿模型,不启用时服务同样可正常推理,仅吞吐略低。若需启用,请确保草稿模型权重就绪(见第 5 节)。

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 7, "method": "dspark", "model":"/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best"}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --additional-config '{"enable_cpu_binding": true,"enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    > dspark-baseline-think.txt 2>&1 &
```

> ⚠️ 原始脚本中 `--model $weight_path` 行末多了一个 `i`,为笔误,请删除,否则参数解析会出错。

### 6.2 方式二:mtp 投机采样

> mtp(Multi-Token Prediction)使用主模型自身的 MTP 预测头,一次前向可预测多个 token,**无需额外草稿模型**,`--speculative-config` 中不需要配置 `model` 字段。

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8009 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}' \
    --additional-config '{"enable_cpu_binding": true,"enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    > mtp-9.txt 2>&1 &
```

### 6.3 两种方式参数差异对比

| 参数 | 方式一(dspark) | 方式二(mtp) |
| --- | --- | --- |
| `--port` | `8000` | `8009` |
| `--tensor-parallel-size` | `2` | `2` |
| `--max-num-batched-tokens` | `8192` | `4096` |
| `--gpu-memory-utilization` | `0.85` | `0.6` |
| `--speculative-config` | `method=dspark`、`num_speculative_tokens=7`,含草稿模型 `model` 路径 | `method=mtp`、`num_speculative_tokens=2`,无草稿模型 |
| 草稿模型 | 需要 | 不需要 |
| 日志文件 | `dspark-baseline-think.txt` | `mtp-9.txt` |

其余参数(模型、服务名、上下文长度、prefix caching、图捕获与附加配置等)两方式一致。

### 6.4 查看启动日志

```bash
# dspark 方式
tail -f dspark-baseline-think.txt
# mtp 方式
tail -f mtp-9.txt
```

当日志出现类似 `Application startup complete` 或监听端口的信息时,服务即就绪。

---

## 7. 启动参数详解

### 7.1 基本参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--model` | `$weight_path` | 主模型权重目录路径 |
| `--port` | `8000`(dspark)/ `8009`(mtp) | 服务监听端口,需与所选启动方式对应 |
| `--served-model-name` | `xingchen4` | 对外模型名称,API 请求体中的 `model` 字段需与此一致 |
| `--trust-remote-code` | - | 允许加载模型目录中的自定义代码 |
| `--max-num-seqs` | `64` | 最大并发序列(请求)数 |
| `--max-model-len` | `131072` | 最大上下文长度,即 128K tokens |
| `--max-num-batched-tokens` | `8192`(dspark)/ `4096`(mtp) | 单次 batch 的最大 token 数,控制显存/内存占用与吞吐 |

### 7.2 并行与资源参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tensor-parallel-size` | `2` | 使用 2 张 NPU 做张量并行(正式参数名;`--tensor-parallel` 为其兼容写法,效果相同) |
| `--data-parallel-size` | `1` | 数据并行度,单副本部署为 1 |
| `--gpu-memory-utilization` | `0.85`(dspark)/ `0.6`(mtp) | NPU 显存利用率上限,越低越保守,防止 OOM |

### 7.3 加速特性参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--enable-prefix-caching` | - | 开启前缀缓存,相同前缀的请求可复用 KV 缓存,提升吞吐 |
| `--speculative-config` | 见脚本(可选) | 投机采样配置,**二选一**:① dspark:`method=dspark`、`num_speculative_tokens=7`,需 `model` 指向草稿模型路径;② mtp:`method=mtp`、`num_speculative_tokens=2`,无需草稿模型。**不配置该参数则不启用投机采样** |
| `--compilation-config` | 见脚本 | CUDA Graph 捕获配置,`FULL_AND_PIECEWISE` 模式配合多组捕获尺寸,减少算子调度开销 |
| `--additional-config` | 见脚本 | 昇腾/模型附加配置:`enable_cpu_binding`(CPU 绑核)、关闭 DSA CP、开启共享专家多流重叠与门控多流重叠 |

---

## 8. 服务测试

### 8.1 健康检查

```bash
curl http://localhost:8000/health    # dspark 方式(端口 8000)
curl http://localhost:8009/health    # mtp 方式(端口 8009)
```

预期返回 `OK`。

### 8.2 接口调用(Completions)

```bash
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "xingchen4",
        "prompt": "Hello, please introduce you",
        "max_tokens": 1024,
        "temperature": 0,
        "seed": 1234
    }'
```

> ⚠️ **端口一致性**:测试端口必须与服务实际监听端口一致——dspark 方式为 `8000`,mtp 方式为 `8009`。下方示例以 `8000` 为例,使用 mtp 方式时请替换为 `8009`。

请求体字段说明:

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `model` | `xingchen4` | 必须与 `--served-model-name` 一致 |
| `prompt` | 文本 | 输入提示词 |
| `max_tokens` | `1024` | 最大生成 token 数 |
| `temperature` | `0` | 采样温度,0 表示贪婪解码 |
| `seed` | `1234` | 随机种子,便于结果复现 |

### 8.3 Chat 接口(可选)

该服务同样兼容 OpenAI Chat 格式:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "xingchen4",
        "messages": [{"role": "user", "content": "Hello, please introduce you"}],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

---

## 9. mtp 模式测试基线

以下为 **mtp 模式**(端口 `8009`,见 6.2 节)在昇腾 910B4 上的压测基线数据,表中时延指标单位均为 **ms**。`run` 为轮次,`avg` 为平均行:

| run | TTFT_avg | TTFT_p90 | TTFT_max | TPOT_avg | TPOT_p90 | TPOT_max | ITL_p90 | ITL_p99 | Spec% | gen/s | E2E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 132.15 | 142.43 | 142.43 | 8.64 | 10.14 | 10.14 | 29.55 | 36.30 | 69.42 | 92.02 | 710.27 |
| 2 | 130.84 | 142.76 | 142.76 | 8.73 | 10.12 | 10.12 | 29.30 | 55.53 | 69.42 | 91.49 | 709.48 |
| 3 | 145.70 | 169.62 | 169.62 | 8.77 | 10.13 | 10.13 | 29.30 | 56.26 | 69.42 | 89.00 | 736.90 |
| avg | 136.23 | 151.60 | 151.60 | 8.71 | 10.13 | 10.13 | 29.38 | 49.36 | 69.42 | 90.84 | 718.88 |

字段说明:

| 字段 | 含义 |
| --- | --- |
| `run` | 轮次编号(1/2/3),`avg` 为平均行 |
| `TTFT_avg / TTFT_p90 / TTFT_max` | 首 token 时延(ms)的平均值 / P90 / 最大值 |
| `TPOT_avg / TPOT_p90 / TPOT_max` | 每个输出 token 时延(ms)的平均值 / P90 / 最大值 |
| `ITL_p90 / ITL_p99` | 输出 token 间隔时延(ms)的 P90 / P99 |
| `Spec%` | 投机采样接受率(本组稳定在 69.42%) |
| `gen/s` | 生成速率(token/s) |
| `E2E` | 端到端耗时(ms) |

---

## 10. 附录

### 附录 A:容器启动脚本 `run_container.sh`

```bash
#!/bin/bash
# 用法:bash run_container.sh <容器名> <镜像名>
CONTAINER_NAME=$1
IMAGE=$2

docker run -itd -u 0 \
    --name $CONTAINER_NAME \
    --net=host \
    --privileged=true \
    --shm-size=512g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt/sfs_turbo/.cache:/root/.cache \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -e VLLM_USE_V1=1 \
    -it $IMAGE /bin/bash
```

使用示例:

```bash
bash run_container.sh xing4_0-vllm harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0
```

### 附录 B:dspark 启动脚本 `start_dspark.sh`(可选)

```bash
#!/bin/bash
export weight_path=/weights/xing4_0   # 按实际权重路径修改

nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 7, "method": "dspark", "model":"/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best"}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --additional-config '{"enable_cpu_binding": true,"enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    > dspark-baseline-think.txt 2>&1 &
```

### 附录 C:mtp 启动脚本 `start_mtp.sh`

```bash
#!/bin/bash
export weight_path=/weights/xing4_0   # 按实际权重路径修改

nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8009 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}' \
    --additional-config '{"enable_cpu_binding": true,"enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    > mtp-9.txt 2>&1 &
```

### 附录 D:测试脚本 `test_api.sh`

```bash
#!/bin/bash
# 端口需与启动参数 --port 保持一致:dspark=8000,mtp=8009
PORT=${PORT:-8000}
curl -X POST http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "xingchen4",
        "prompt": "Hello, please introduce you",
        "max_tokens": 1024,
        "temperature": 0,
        "seed": 1234
    }'
```

