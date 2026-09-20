<div align="center">
<h1>
  Xing4.0
</h1>
</div>

<p align="center">
   🦉 <a href="https://github.com/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">GitHub</a> • 🤗 <a href="https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">ModelScope</a> • 🪐 <a href="https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">Modelers</a> • 📖 <a href="./README_zh.md">中文版</a>
</p>

# Table of Contents

- [Introduction](#introduction)
- [News](#news)
- [Model](#model)
- [Evaluation](#evaluation)
- [Quickstart](#quickstart)
  - [Local Inference](#local-inference)
  - [Serving](#serving)
    - [vLLM](#vllm)
    - [SGLang](#sglang)
    - [KTransformers](#ktransformers)
  - [Recommended Parameters](#recommended-parameters)
- [Fine-tuning](#fine-tuning)
  - [LLaMA-Factory](#llama-factory)
  - [MindFormers](#mindformers)
  - [FlagOS](#flagos)
- [Acknowledgments](#acknowledgments)
- [Disclaimer, License & Citation](#disclaimer-license--citation)


# Introduction

### Xing4.0

**Xing4.0** is developed by China Telecom Artificial Intelligence Technology Co., Ltd. It is the latest generation of [TeleChat](https://github.com/Tele-AI/TeleChat3), trained entirely on the Ascend NPU platform.

**Xing4.0-29B-A4B** has 29B total parameters and only 4B activated per token. It natively supports a 256K context length, extensible to 512K. It is the first model of this scale trained entirely on the Ascend NPU platform with the MindSpore framework, and deeply optimized for complex engineering tasks. Key features include:

- **Agent-Oriented Architecture**: Built on the mHC + MLA + MTP architecture, supporting multi-step planning, tool calling, and complex reasoning chain execution, ensuring task coherence and execution stability under long contexts.
- **Deep Co-optimization with Ascend NPU**: Adapted for Ascend 910C clusters using MindSpore/MindFormers, including feature adaptation for mHC, cross-framework numerical alignment, and fused operator development, enabling stable and efficient training on the Ascend platform.
- **Significant Training Efficiency Gains**: Through multi-level co-optimization — including fine-grained MoE communication optimization, selective recomputation, DVM automatic graph-operator fusion, and Ascend C mHC fused operators — overall training throughput was improved by approximately **96%** over out-of-the-box performance.
- **Full Open-Source Ecosystem Compatibility**: Supports LLaMA-Factory and MindFormers for fine-tuning; SGLang, vLLM, and KTransformers for inference and deployment; with targeted adaptation and format alignment for agent frameworks such as OpenCode, Claude Code, OpenClaw, and Hermes, enabling seamless integration into existing workflows. On the hardware side, multi-chip platform adaptation has been completed via BAAI FlagOS, supporting cross-architecture one-click deployment.
- **Easy Adaptation for Domain-Specific Scenarios**: Well-suited for downstream task fine-tuning, allowing lightweight customization on proprietary data for vertical domains such as intent classification, table understanding, contract auditing, and knowledge-based QA, enabling rapid domain capability development and deployment at low cost.

# News
- 2026-09-17 Open-sourced **Xing4.0-29B-A4B** [[HuggingFace Hub](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B)]
- 2026-09-17 Open-sourced **Xing4.0-29B-A4B-FP8** [[HuggingFace Hub](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)] and **Xing4.0-29B-A4B-GGUF** [[HuggingFace Hub](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF)]

# Model
The model architecture of **Xing4.0-29B-A4B** is as follows:

|            | Layers | Hidden Size | Dense FFN Intermediate Size | Expert Intermediate Size | Attention Type | Number of Routed Experts | Active Experts per Token | Number of Shared Experts |
|------------|--------|-------------|----------------------------|--------------------------|----------------|--------------------------|--------------------------|--------------------------|
| Xing4.0-29B-A4B  | 40     | 3584        | 9216                   | 1024                | MLA       | 64             | 4                 | 1              |


# Evaluation

We systematically evaluated Xing4.0-29B-A4B across a range of representative benchmarks. The results are as follows:

- **Coding Agent**: Xing4.0-29B-A4B achieves strong results on SWE-bench Verified (75.0), SWE-bench Multilingual (66.0), and Terminal-Bench 2.1 (57.5). On SWE-bench Verified (75.0), it approaches Qwen3.6-35B-A3B (76.0) and significantly outperforms Gemma4-26B-A4B (53.0). On Terminal-Bench 2.1 (57.5), it surpasses both Qwen3.6-35B-A3B (51.5) and Gemma4-26B-A4B (30.0).

- **General Agent**: Xing4.0-29B-A4B demonstrates strong performance on Claw-Eval (76.55), DeepresearchBII (60.80), and Tau3-Bench (64.63). On Claw-Eval (76.55), it outperforms both Qwen3.6-35B-A3B (74.54) and Gemma4-26B-A4B (71.49). On DeepresearchBII (60.80), it surpasses Qwen3.6-35B-A3B (59.70) and Gemma4-26B-A4B (39.30).


| Benchmark              | Xing4.0-29B-A4B | Gemma4-26B-A4B | Qwen3.6-35B-A3B |
|------------------------|-------------------|----------------|-----------------|
| IFBench                | 69.67             | 72.67          | 65.50           |
| AIME2026               | 90.00             | 88.30          | 92.70           |
| AA.LCR                 | 61.00             | 66.00          | 62.00           |
| Tau3-Bench             | 64.63             | 58.90          | 67.20           |
| Claw-Eval              | 76.55             | 71.49          | 74.54           |
| SWE-bench Verified     | 75.00             | 53.00          | 76.00           |
| Terminal-Bench 2.1     | 57.50             | 30.00          | 51.50           |
| SWE-bench Multilingual | 66.00             | 51.00          | 67.20           |
| DeepresearchBII        | 60.80             | 39.30          | 59.70           |


# Quickstart

### Local Inference

**Inference Example**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "XingChen-AGI/Xing4.0-29B-A4B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16)

prompt = "Briefly explain the basic principles of quantum computing."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    top_p=0.95,
    temperature=1.0,
    repetition_penalty=1.05,
    max_new_tokens=32768
)

response = tokenizer.decode(generated_ids[0], skip_special_tokens=False, spaces_between_special_tokens=False)
# The model output contains a Chain-of-Thought; the final answer follows </think>
answer = response.split("</think>")[-1].strip()
print(answer)
```


### Serving
> [!NOTE]
> Xing4.0 model support has been submitted to the following inference frameworks via pull requests, which are currently under review and **not yet merged into their main branches**. Before the PRs are merged, please install from the corresponding PR branch to enable Xing4.0 support.
>
> | Framework | Pull Request | Status | Commit Hash |
> |-----------|-------------|--------|----------------|
> | SGLang | [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793) | Pending | f22026fa0cb20c661614ded7f3b0e81a844b1906 |
> | vLLM | [vllm-project/vllm#57135](https://github.com/vllm-project/vllm/pull/57135) | Pending | 25aa52a29131753c10e56528d3fa0c4464cead63 |
> | TensorRT-LLM | [NVIDIA/TensorRT-LLM#19283](https://github.com/NVIDIA/TensorRT-LLM/pull/19283) | Pending | 1b03b640fa93dc956f482b9d74f7124f14b28d71 |
> | llama.cpp | [ggml-org/llama.cpp#29012](https://github.com/ggml-org/llama.cpp/pull/29012) | Pending | 63c16fb9797d00f13d70b5a618b4deb08953aef3 |
> | KTransformers | [kvcache-ai/ktransformers#2168](https://github.com/kvcache-ai/ktransformers/pull/2168) | Pending | f2f0e3a8c65cbb47249d429c6ca488a8652963c8 |
#### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput, low-latency LLM inference and serving engine that provides an OpenAI-compatible API out of the box.

```shell
vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.0-29B-A4B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --reasoning-parser xing4 \
    --tool-call-parser xing4 \
    --speculative-config '{"method":"mtp", "num_speculative_tokens": 1}'
```

Once launched, the OpenAI-compatible API is available at `http://localhost:8000/v1`.

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) is a high-performance serving framework for large language models and vision-language models.

```shell
sglang serve --model-path ${MODEL_PATH} \
   --trust-remote-code \
   --host 0.0.0.0 \
   --port 8000 \
   --served-model-name Xing4.0-29B-A4B \
   --tp-size 2 \
   --context-length 262144 \
   --mem-fraction-static 0.90 \
   --max-running-requests 32 \
   --reasoning-parser xing4 \
   --tool-call-parser xing4 \
   --speculative-algorithm EAGLE
```

Once launched, the OpenAI-compatible API is available at `http://localhost:8000/v1`.

#### KTransformers

[KTransformers](https://github.com/kvcache-ai/ktransformers) is a flexible inference optimization framework that enables efficient execution of large-scale MoE models on limited GPU resources through heterogeneous computing (GPU/CPU co-processing).

```
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path ${MODEL_PATH} \
  --served-model-name Xing4.0-29B-A4B \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --attention-backend triton \
  --kt-weight-path ${MODEL_PATH} \
  --kt-method BF16 \
  --kt-cpuinfer 50 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 24 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 512 \
  --max-total-tokens 65536 \
  --tool-call-parser xing4_0 \
  --reasoning-parser xing4_0
```

### Recommended Parameters
#### Xing4.0-29B-A4B Inference Parameters
- Complex reasoning / general tasks: `temperature=1.0, top_p=0.95, repetition_penalty=1.05`

- Coding / agent tasks: `temperature=0.8, top_p=0.95, repetition_penalty=1.05`




# Fine-tuning

### LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is an open-source framework for training and fine-tuning large language models, supporting pre-training, supervised fine-tuning (SFT), reward modeling (RM), PPO, DPO, KTO, ORPO, SimPO, and more.

Xing4.0 supports fine-tuning, weight merging, inference, and deployment via LLaMA-Factory. For details, see [Xing4.0 LLaMA-Factory Fine-tuning Guide](./tutorial/llama_factory/xing4.0_llama_factory.md).

### MindFormers

[MindFormers](https://atomgit.com/mindspore/mindformers) is a large model development toolkit built on MindSpore, covering the full pipeline from pre-training, fine-tuning, inference, evaluation, to deployment. It includes built-in distributed strategies such as tensor parallelism, pipeline parallelism, and sequence parallelism, providing out-of-the-box capabilities for large model training on the Ascend platform.

Xing4.0 has been adapted for Ascend Atlas 800T A3 clusters, supporting distributed training and evaluation with MindSpore + MindFormers. For details, see [Xing4.0 MindFormers Training Guide](./tutorial/mindformers/xing4.0_MindFormers.md).

### FlagOS

[FlagOS](https://docs.flagos.io/en/latest/) is an open-source AI system software stack for large model training, inference, and deployment, with adaptation support for mainstream frameworks and hardware platforms.

Xing4.0 has integrated mHC Triton-Ascend fused operators into MindFormers via FlagOS, with single-node 16-card training validation completed. For details, see [Xing4.0 FlagOS Triton Operator Integration Guide](./tutorial/flagos/xing4.0_FlagOS_mHC_MindFormers.md).



# Acknowledgments

We extend our deepest gratitude to the open-source community — it is by standing on the shoulders of these giants that we are able to see further.

Special thanks to the DeepSeek team. Drawing on the design wisdom of their model architecture brought significant stability and efficiency to our training process, making the path of exploration smoother and clearer.


# Disclaimer, License & Citation

### Disclaimer

Do not use Xing4.0 or its derivative models for any activities that endanger national or social security, or violate applicable laws and regulations. Do not use it for internet services that have not undergone security review and filing.

We have made every effort to ensure the compliance of the training data, but due to the complexity of models and data, unforeseen issues may still arise. We assume no liability for any issues arising from the use of this open-source model — including but not limited to data security risks, public opinion risks, and risks arising from the model being misled, misused, disseminated, or improperly utilized.


### Citation
If you find our work useful, please cite:

```
@misc{liu2025trainingreporttelechat3moe,
      title={Training Report of TeleChat3-MoE}, 
      author={Xinzhang Liu and Chao Wang and Zhihao Yang and Zhuo Jiang and Xuncheng Zhao and Haoran Wang and Lei Li and Dongdong He and Luobin Liu and Kaizhe Yuan and Han Gao and Zihan Wang and Yitong Yao and Sishi Xiong and Wenmin Deng and Haowei He and Kaidong Yu and Yu Zhao and Ruiyu Fang and Yuhao Jiang and Yingyan Li and Xiaohui Hu and Xi Yu and Jingqi Li and Yanwei Liu and Qingli Li and Xinyu Shi and Junhao Niu and Chengnuo Huang and Yao Xiao and Ruiwen Wang and Fengkai Li and Luwen Pu and Kaipeng Jia and Fubei Yao and Yuyao Huang and Xuewei He and Zhuoru Jiang and Ruiting Song and Rui Xue and Qiyi Xie and Jie Zhang and Zilu Huang and Zhaoxi Zhang and Zhilong Lu and Yanhan Zhang and Yin Zhang and Yanlei Xue and Zhu Yuan and Teng Su and Xin Jiang and Shuangyong Song and Yongxiang Li and Xuelong Li},
      year={2025},
      eprint={2512.24157},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.24157}, 
}

@misc{wang2025technicalreporttelechat2telechat25,
      title={Technical Report of TeleChat2, TeleChat2.5 and T1}, 
      author={Zihan Wang and Xinzhang Liu and Yitong Yao and Chao Wang and Yu Zhao and Zhihao Yang and Wenmin Deng and Kaipeng Jia and Jiaxin Peng and Yuyao Huang and Sishi Xiong and Zhuo Jiang and Kaidong Yu and Xiaohui Hu and Fubei Yao and Ruiyu Fang and Zhuoru Jiang and Ruiting Song and Qiyi Xie and Rui Xue and Xuewei He and Yanlei Xue and Zhu Yuan and Zhaoxi Zhang and Zilu Huang and Shiquan Wang and Xin Wang and Hanming Wu and Mingyuan Wang and Xufeng Zhan and Yuhan Sun and Zhaohu Xing and Yuhao Jiang and Bingkai Yang and Shuangyong Song and Yongxiang Li and Zhongjiang He and Xuelong Li},
      year={2025},
      eprint={2507.18013},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18013}, 
}
```
