# Deploying Xing4.0-29B-A4B with the SGLang Plugin (Open-Source Path)

　　This document describes how to deploy an OpenAI-compatible API service for the Xing4.0-29B-A4B model through SGLang's **open-source plugin mechanism**, using an official / stock SGLang environment as the base, without modifying the SGLang source tree. It covers:

- Prerequisites (SGLang environment)
- Downloading the Xing4.0-29B-A4B model
- Getting the plugin and starting the container
- Launching the SGLang service
- Verifying plugin loading and its switches
- OpenAI-compatible API call examples
- Startup parameter descriptions and inference parameter recommendations
- Troubleshooting

　　The SGLang adaptation code for Xing4.0 is released as an **open-source plugin**, hosted at [XingChen-AGI/xing4-sglang-plugin](https://github.com/XingChen-AGI/xing4-sglang-plugin). The plugin is distributed as a Python package and registers itself through a Python entry point in the `sglang.srt.plugins` group, so SGLang loads it automatically at process startup. It works with a stock SGLang environment, requires no changes to the SGLang source tree, and can be upgraded and maintained independently. The plugin **contains no custom CUDA kernels and needs no separate build**; its implementation reuses runtime operators already present in SGLang (MLA / MoE / DeepGEMM / FlashInfer).

> [!NOTE]
> The plugin registers: the `Xing4_0ForCausalLM` model implementation, the `xing4_0` model configuration (reusing SGLang's MLA + MoE configuration infrastructure), the `xing4_0` reasoning parser, the `xing4_0` tool-call parser, and `xing4_0` speculative decoding (EAGLE / NextN) plus MLA detection support. In addition, the plugin supplies the upstream-PR internals that have not yet landed in SGLang mainline through lazy, idempotent `HookRegistry` hooks under `sglang_xing4_0_plugin.compat`; each hook can be disabled individually (see "Verifying Plugin Loading").

## Prerequisites

　　Make sure the host machine meets the following software requirements:

- Docker (24.0 or later recommended, when using the container path)
- NVIDIA driver (535 or later recommended)
- NVIDIA Container Toolkit (the `nvidia-docker` runtime)
- A GPU that supports bfloat16 (mHC mode requires bfloat16)

　　This document uses the **official SGLang image** `lmsysorg/sglang` as the base environment: **there is no need to install SGLang on the host, nor to build SGLang yourself** — simply install this plugin inside the container to enable Xing4.0 support. If you already have SGLang installed on a bare-metal host or a custom image, you may skip the container steps and install the plugin into the **same Python environment** instead.

　　To check that the GPU is accessible from a container:

```bash
docker run --rm --gpus all lmsysorg/sglang:latest nvidia-smi
```

> [!NOTE]
> The image tag can be pinned to a specific SGLang version as needed. The plugin is ported from the upstream SGLang PR [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793) (branch `support_xing4_0`), whose baseline commit is recorded in the `TESTED_SGLANG` constant in `src/sglang_xing4_0_plugin/__init__.py`; for production, pin the version rather than using `latest`.

> [!NOTE]
> The commands in this document use `--tp-size 2` as an example, which requires at least 2 GPUs; adjust this parameter according to the actual number of GPUs available. The hardware environment used in this document is only the **environment used for demonstration and validation**; it is not a mandatory requirement.

## Download the Xing4.0-29B-A4B Model

　　The Xing4.0-29B-A4B model can be downloaded from the following platforms; the weights are identical across all three:

| Platform | Address |
|----------|---------|
| Hugging Face | https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B |
| ModelScope | https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B |
| Modelers | https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B |

　　Taking ModelScope as an example, download to a local directory:

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.0-29B-A4B \
  --local_dir /yourpath/models/Xing4.0-29B-A4B
```

　　To further reduce memory usage, you can also use the FP8 version ([Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)); the download and startup steps are the same as below, you only need to point the mounted path to the FP8 directory.

> [!NOTE]
> mHC mode (`hc_mult > 1`) supports `bfloat16` only and does not support pipeline parallelism. The plugin reports a clear error when these conditions are not met.

## Getting the Plugin and Starting the Container

　　Step 1: get the plugin source code, so that it can be mounted into the container:

```bash
git clone https://github.com/XingChen-AGI/xing4-sglang-plugin.git
```

　　Step 2: start a container (with bash as the entry point, kept running), mounting the model directory and the plugin repository and exposing the API port:

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

　　Step 3: enter the container and install the plugin inside it (the SGLang environment is already present there):

```bash
docker exec -it xing4-sglang-plugin bash

pip install -e /workspace/xing4-sglang-plugin --no-deps
```

　　Step 4: verify that the entry point is visible to the interpreter inside the container:

```bash
python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='sglang.srt.plugins')])"
```

　　The output should include:

```
('xing4_0', 'sglang_xing4_0_plugin:register')
```

> [!NOTE]
> Adding `src/` to `PYTHONPATH` alone does **not** make the plugin discoverable, because the entry point comes from the metadata of an installed package. The installation step above is required. `--no-deps` prevents the plugin from replacing the SGLang installation inside the container.

## Launch the SGLang Service

　　Run the following command inside the container to start the service — no additional plugin-specific arguments are needed:

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

　　Once launched, the OpenAI-compatible API is available at `http://localhost:8000/v1`. Once a line such as `Uvicorn running on http://0.0.0.0:8000` appears in the service logs, the service is available externally.

> [!NOTE]
> `export FLASHINFER_DISABLE_VERSION_CHECK=1` skips the FlashInfer/CUDA version gate (for example the FlashInfer 0.7.x + CUDA 13.0 combination); without it the service may fail to start due to the version check.

> [!NOTE]
> `--speculative-algorithm EAGLE` can be dropped if not needed. The Xing4_0 checkpoint ships its own NextN (MTP) draft head, and the plugin already defaults `--speculative-draft-model-path` to the main model path, so **no separate draft model is required**.

> [!IMPORTANT]
> SGLang 0.5.x imposes an **exactly-one-of-three** rule on the EAGLE family: either fully automatic (set none of `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens`), or fully manual (set all three), or use `--speculative-adaptive`. **Setting only two of the three triggers an `AssertionError`.**

### Verifying Plugin Loading

　　At service startup, confirm the plugin's registration record in the logs:

```
sglang-xing4_0-plugin: registered Xing4_0ForCausalLM (models=sglang_xing4_0_plugin.models, parser=xing4_0, vendored from sglang commit aa737beda5 (0.4.x) / 0.5.22)
```

　　If the runtime environment already ships Xing4_0 support (such as the prebuilt image `quay.io/xingchen-agi/xingchen-inference-sglang:v0.5.20.rc1-xing4_0`), the plugin logs a stand-down message and keeps the in-environment implementation active:

```
sglang-xing4_0-plugin: Xing4_0ForCausalLM is already supported in tree; standing down. Set XING4_0_PLUGIN_OVERRIDE=1 to override with the vendored copy.
```

　　To explicitly use this plugin (ported) implementation, set the environment variable before starting:

```bash
XING4_0_PLUGIN_OVERRIDE=1 sglang serve --model-path /models/Xing4.0-29B-A4B ...
```

　　Switches for the compat layer (used to turn individual hooks off as the upstream PR lands piece by piece; all are idempotent):

```bash
# disable the whole compat layer
XING4_0_COMPAT_OFF=1 sglang serve --model-path /models/Xing4.0-29B-A4B ...

# disable a single hook item
XING4_0_COMPAT_MHC_PRE=0 sglang serve --model-path /models/Xing4.0-29B-A4B ...
```

> [!NOTE]
> SGLang does not currently provide a plugin allowlist environment variable equivalent to `VLLM_PLUGINS`. To suppress this plugin, adjust `PYTHONPATH` or uninstall it.

## Calling the API

　　Once the service is running, you can call it through the OpenAI-compatible interface. The `model` field in the examples below must match the `--served-model-name` used at startup.

### Basic Chat (curl)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Xing4.0-29B-A4B",
    "messages": [
      {"role": "user", "content": "Introduce yourself in one sentence."}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "chat_template_kwargs":{"enable_thinking":true},"skip_special_tokens": false
  }'
```

　　To disable chain-of-thought output, pass `"chat_template_kwargs":{"enable_thinking":false}` in the request body. With thinking enabled, the generated content looks like `<think>reasoning process</think>final content`; with thinking disabled, only the final content is produced.

## Startup Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `--model-path` | Path to the model directory |
| `--host 0.0.0.0` | Listen on all network interfaces so the service is reachable from outside the container |
| `--port 8000` | API service port |
| `--served-model-name` | The model name exposed externally; API calls must use the same name |
| `--tp-size 2` | Number of GPUs for tensor parallelism; must match the number of available GPUs |
| `--trust-remote-code` | Trust remote code shipped in the model repository |
| `--context-length 262144` | Maximum context length; Xing4.0 natively supports 256K |
| `--mem-fraction-static 0.90` | Upper limit of the fraction of GPU memory to reserve; adjust according to available headroom |
| `--max-running-requests 4` | Maximum number of concurrent requests; adjust according to traffic and memory |
| `--reasoning-parser xing4_0` | Parses Xing4.0 chain-of-thought output (registered by the plugin) |
| `--tool-call-parser xing4_0` | Parses the Xing4.0 tool-call format (registered by the plugin) |
| `--speculative-algorithm EAGLE` | Enables EAGLE speculative decoding to accelerate generation (optional; see the parameter constraint above) |
| `--speculative-adaptive` | Uses adaptive speculative decoding parameters (one of the three mutually exclusive modes) |
| `--chat-template` | Optional: path to a `chat_template.jinja` outside the model repository |

## Recommended Inference Parameters

　　Following Xing4.0's official recommendations, the sampling parameters for different scenarios are:

| Scenario | temperature | top_p | repetition_penalty |
|----------|-------------|-------|---------------------|
| Complex reasoning / general tasks | 1.0 | 0.95 | 1.05 |
| Coding / agent tasks | 0.8 | 0.95 | 1.05 |

## Troubleshooting

### The plugin was not loaded

- Make sure the plugin is installed in the **same Python environment as SGLang**, and that you completed a package installation rather than only setting `PYTHONPATH`
- Use the command in "Getting the Plugin and Starting the Container" to check that the entry point is visible; the output should include `('xing4_0', 'sglang_xing4_0_plugin:register')`
- If `sglang-xing4_0-plugin: registered ...` does not appear in the logs, first check whether the stand-down logic was triggered (see "Verifying Plugin Loading")

### The model architecture is not recognized (Xing4_0ForCausalLM missing)

- If the plugin did not register successfully, SGLang refuses to load the model because the architecture is missing. Follow the previous item to check registration and logs
- In environments that already ship support (such as the prebuilt image), you can set `XING4_0_PLUGIN_OVERRIDE=1` to force the plugin implementation instead

### EAGLE fails to start with an AssertionError

- SGLang 0.5.x requires `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens` to be set together (manual mode), or all left unset (automatic mode), or `--speculative-adaptive` to be used instead; setting only two of the three fails the assertion
- For a bundled MTP-style draft head, the automatic mode (setting none of the three) is usually sufficient, and a `topk` greater than 1 may in fact lower the acceptance rate

### mHC-related errors (bfloat16 / pipeline parallelism)

- mHC mode requires both the model and inference to run in `bfloat16`; FP16 or quantized paths raise errors
- mHC does not currently support pipeline parallelism

### FlashInfer reports a version mismatch

- Set `FLASHINFER_DISABLE_VERSION_CHECK=1` and restart the service

### Out of memory (OOM)

- Lower `--mem-fraction-static` (e.g. 0.85) or `--context-length` (e.g. 131072)
- Use the FP8 version of the model to reduce memory usage
- Reduce `--max-running-requests` to lower the peak memory usage under concurrency

### Tensor parallelism does not match the GPU count

- `--tp-size` must divide the number of visible GPUs; for example, with 4 GPUs it can be set to `4` or `2`

### The container cannot access the GPU

- Make sure the NVIDIA Container Toolkit is installed and start the container with `--gpus all`
- To use only a subset of the GPUs, use `--gpus '"device=0,1"'` instead

### The service is not reachable externally

- Make sure `--host` is set to `0.0.0.0` and the port is published with `-p 8000:8000`

## Upgrade and Maintenance

　　The plugin depends on SGLang's internal runtime APIs, and the `TESTED_SGLANG` constant records the SGLang version the port is based on. After upgrading SGLang (i.e. switching the image tag or reinstalling):

- Re-run the plugin's own tests (no GPU or model weights required) from the plugin repository inside the container, at `/workspace/xing4-sglang-plugin`, to confirm that the registration entry point and the parsers work as expected:

```bash
python -m pytest tests/ -q
```

- Check the "ported baseline version" and the "running SGLang version" printed by the plugin in the startup logs; if the two diverge significantly, watch for internal API migrations and re-port the relevant modules when necessary
- As upstream PR [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793) lands in mainline piece by piece, the corresponding compat hooks can be turned off one at a time via `XING4_0_COMPAT_*`; once everything is in-tree, `register()` stands down automatically
