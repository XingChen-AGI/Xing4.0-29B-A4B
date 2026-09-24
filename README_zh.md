<div align="center">
<h1>
  星辰语义大模型-Xing4.0
</h1>
</div>

<p align="center">
   🦉 <a href="https://github.com/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">GitHub</a> • 🤗 <a href="https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">ModelScope</a> • 🪐 <a href="https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B" target="_blank">Modelers</a> • 📖 <a href="./README.md">English</a>
</p>

# 目录

- [介绍](#介绍)
- [最新动态](#最新动态)
- [模型](#模型)
- [能力评估](#能力评估)
- [快速开始](#快速开始)
  - [本地推理](#本地推理)
  - [服务化推理](#服务化推理)
    - [vLLM](#vllm)
    - [SGLang](#sglang)
    - [KTransformers](#ktransformers)
  - [超参选择](#超参选择)
- [微调](#微调)
  - [LLaMA-Factory](#llama-factory)
  - [MindFormers](#mindformers)
  - [FlagOS](#flagos)
- [致谢](#致谢)
- [声明、协议、引用](#声明协议引用)


# 介绍

### 星辰语义大模型-Xing4.0

**Xing4.0** 由中电信人工智能科技有限公司研发，是星辰语义大模型系列（原 [TeleChat](https://github.com/Tele-AI/TeleChat3)）的新一代模型，**完全基于国产算力**训练。

**Xing4.0-29B-A4B** 模型总参数量 29B，激活参数仅 4B，原生支持 256K 上下文，可扩展至 512K，是国内首个基于国产算力与国产框架完成训练、面向复杂工程任务深度优化的百亿参数大模型。主要特性如下：

- **面向 Agent 的架构设计**：采用 mHC + MLA + MTP 架构，支持多步骤规划、工具调用与复杂推理链路执行，保障长上下文下任务连贯与执行稳定。
- **国产算力深度协同**：面向昇腾 910C 集群，基于 MindSpore/MindFormers 完成 mHC 等新特性适配、跨框架精度对齐及融合算子开发，实现国产算力平台上的稳定高效训练。
- **训练效率大幅提升**：通过细粒度 MoE 通信优化、选择性重计算、DVM 图算自动融合、Ascend C mHC 融合算子等多层次协同优化，整体训练吞吐较开箱性能**提升约 96%**。
- **开源生态全面兼容**：训练微调支持 LLaMA-Factory、MindFormers；推理部署支持 SGLang、vLLM、KTransformers；针对 OpenCode、Claude Code、OpenClaw、Hermes 等 Agent 框架进行定向适配与格式对齐，可直接接入现有工作流。算力方面，基于智源 FlagOS 完成多款 AI 芯片平台适配，支持跨架构一键部署。
- **低资源场景易于适配**：模型具备良好的下游任务微调能力，可面向私域数据进行轻量化定制，适用于意图识别、表格理解、合同审计、知识问答等垂直场景，以较低成本实现领域能力的快速构建与落地。

# 最新动态
- 2026-09-17 开源 **Xing4.0-29B-A4B** [[HuggingFace Hub](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B)]
- 2026-09-17 开源 **Xing4.0-29B-A4B-FP8** [[HuggingFace Hub](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)] and **Xing4.0-29B-A4B-GGUF** [[HuggingFace Hub](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF)]
# 模型
**Xing4.0-29B-A4B** 的模型结构配置如下表所示：

|            | Layers | Hidden Size | Dense FFN Intermediate Size | Expert Intermediate Size | Attention Type | Number of Routed Experts | Active Experts per Token | Number of Shared Experts |
|------------|--------|-------------|----------------------------|--------------------------|----------------|--------------------------|--------------------------|--------------------------|
| Xing4.0-29B-A4B  | 40     | 3584        | 9216                   | 1024                | MLA       | 64             | 4                 | 1              |


# 能力评估

为了系统的评估Xing4.0-29B-A4B的表现，我们选取了多个代表性benchmark进行评测。具体表现如下：

- 在Coding Agent方面，Xing4.0-29B-A4B 在 SWE-bench Verified（75.0）、SWE-bench Multilingual（66.0）和 Terminal-Bench 2.1（57.5） 上表现突出。其中，SWE-bench Verified （75.0）接近 Qwen3.6-35B-A3B（76.0），并超过 Gemma4-26B-A4B（53.0）；Terminal-Bench 2.1（57.5） 分别超过 Qwen3.6-35B-A3B（51.5）和 Gemma4-26B-A4B（30.0）。

- 在通用Agent方面，Xing4.0-29B-A4B 在 Claw-Eval（76.55）、DeepresearchBII（60.80）和 tau3-Bench（64.63） 上展现出较强表现。其中，Claw-Eval（76.55） 超过 Qwen3.6-35B-A3B（74.54）和 Gemma4-26B-A4B（71.49），DeepresearchBII（60.80） 超过 Qwen3.6-35B-A3B（59.70）和 Gemma4-26B-A4B（39.30）。


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


# 快速开始

### 本地推理

**模型推理方法示范**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "XingChen-AGI/Xing4.0-29B-A4B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16)

prompt = "假设有一巨大的游泳池，现有四根水管可以注满游泳池。当泳池为空时，一根水管可以在两天内注满游泳池；一根水管可以在三天内注满游泳池；一根水管可以在四天内注满游泳池；一根水管可以在六小时内注满游泳池。还有水泵负责抽水，泳池已满时，一个水泵可以在十二小时内抽空泳池。初始泳池为空，同时打开所有水管和一个水泵，游泳池需要多长时间才能注满?"
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
# 模型输出包含思维链，</think> 之后为最终回答
answer = response.split("</think>")[-1].strip()
print(answer)
```

输出结果如下（思维链部分已省略）：

````text
这是一个经典的工程问题（流水问题），我们可以通过计算**注水速率**和**排水速率**来解决。

为了方便计算，我们先把所有时间单位统一为**小时**。

### 第一步：计算各水管的注水速率
假设游泳池的总容量为 1 个单位。

1. 第一根水管：2天注满 = 48小时注满，速率 = $1/48$（池/小时）
2. 第二根水管：3天注满 = 72小时注满，速率 = $1/72$（池/小时）
3. 第三根水管：4天注满 = 96小时注满，速率 = $1/96$（池/小时）
4. 第四根水管：6小时注满，速率 = $1/6$（池/小时）

### 第二步：计算水泵的排水速率
1个水泵在12小时内抽空泳池，所以排水速率为：
$1/12$（池/小时）

### 第三步：计算总的净注水速率
同时打开所有水管和抽水泵，净注水速率 = 所有水管注水速率之和 - 水泵排水速率。

$$净速率 = \frac{1}{48} + \frac{1}{72} + \frac{1}{96} + \frac{1}{6} - \frac{1}{12}$$

为了相加，我们需要找到这些分数的公分母。48、72、96、6、12 的最小公倍数是 **288**。

将每个分数通分：
* $\frac{1}{48} = \frac{6}{288}$
* $\frac{1}{72} = \frac{4}{288}$
* $\frac{1}{96} = \frac{3}{288}$
* $\frac{1}{6} = \frac{48}{288}$
* $\frac{1}{12} = \frac{24}{288}$

现在计算总和：
$$净速率 = \frac{6 + 4 + 3 + 48 - 24}{288} = \frac{37}{288} \text{（池/小时）}$$

### 第四步：计算注满所需的时间
总容量 ÷ 净速率 = 注满时间

$$时间 = 1 \div \frac{37}{288} = \frac{288}{37} \text{ 小时}$$

将 $\frac{288}{37}$ 化为带分数：$288 \div 37 = 7$ 余 $29$，即 **7又 $\frac{29}{37}$ 小时**。

换算成更直观的小时和分钟：
* $\frac{29}{37}$ 小时 $\approx 0.7838$ 小时
* $0.7838 \times 60 \approx 47.03$ 分钟

**结论：**
初始泳池为空，同时打开所有水管和一个水泵，游泳池大约需要 **7小时47分钟**（精确值为 $\frac{288}{37}$ 小时）才能注满。
````


### 服务化推理
> [!NOTE]
> Xing4.0 模型的适配代码已向以下推理框架提交 Pull Request，目前**尚未合入主分支**。在 PR 合入之前，请从对应 PR 分支安装以启用 Xing4.0 支持。
>
> | 框架 | Pull Request | 状态 | commit hash |
> |------|-------------|------|----------------|
> | SGLang | [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793) | 待合入 | f22026fa0cb20c661614ded7f3b0e81a844b1906 |
> | vLLM | [vllm-project/vllm#57135](https://github.com/vllm-project/vllm/pull/57135) | 待合入 | 25aa52a29131753c10e56528d3fa0c4464cead63 |
> | TensorRT-LLM | [NVIDIA/TensorRT-LLM#19283](https://github.com/NVIDIA/TensorRT-LLM/pull/19283) | 待合入 | 1b03b640fa93dc956f482b9d74f7124f14b28d71 |
> | llama.cpp | [ggml-org/llama.cpp#29012](https://github.com/ggml-org/llama.cpp/pull/29012) | 待合入 | 63c16fb9797d00f13d70b5a618b4deb08953aef3 |
> | KTransformers | [kvcache-ai/ktransformers#2168](https://github.com/kvcache-ai/ktransformers/pull/2168) | 待合入 | f2f0e3a8c65cbb47249d429c6ca488a8652963c8 |
>
> 为方便快速部署，我们同时提供了已集成对应 PR 的预编译 Docker 镜像，无需自行从源码编译：
>
> | 框架 | 预编译镜像 |
> |------|-----------|
> | vLLM | `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0` |
> | SGLang | `quay.io/xingchen-agi/xingchen-inference-sglang:v0.5.20.rc1-xing4_0` |
>
> 推荐通过以下文档使用镜像完成部署：[vLLM 部署文档](./tutorial/vLLm/xing4.0_vllm.md) | [SGLang 部署文档](./tutorial/SGLang/xing4.0_sglang.md)


#### vLLM

[vLLM](https://github.com/vllm-project/vllm) 是高吞吐、低延迟的 LLM 推理与服务引擎，可一键启动 OpenAI 兼容的 API 服务。

> [!TIP]
> 可直接拉取预编译镜像 `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0`，通过 Docker 部署服务。具体步骤、参数说明与调用示例请参考 [Xing4.0 vLLM 部署文档](./tutorial/vLLm/xing4.0_vllm.md)。


```shell
vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.0-29B-A4B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 4 \
    --enable-auto-tool-choice \
    --reasoning-parser xing4 \
    --tool-call-parser xing4 \
    --speculative-config '{"method":"mtp", "num_speculative_tokens": 1}'
```

启动后即可通过 `http://localhost:8000/v1` 访问 OpenAI 兼容 API。

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) 是面向大语言模型与视觉语言模型的高性能服务框架。

> [!TIP]
> 可直接拉取预编译镜像 `quay.io/xingchen-agi/xingchen-inference-sglang:v0.5.20.rc1-xing4_0`，通过 Docker 部署服务。具体步骤、参数说明与调用示例请参考 [Xing4.0 SGLang 部署文档](./tutorial/SGLang/xing4.0_sglang.md)。

```shell
sglang serve --model-path ${MODEL_PATH} \
   --trust-remote-code \
   --host 0.0.0.0 \
   --port 8000 \
   --served-model-name Xing4.0-29B-A4B \
   --tp-size 2 \
   --context-length 262144 \
   --mem-fraction-static 0.90 \
   --max-running-requests 4 \
   --reasoning-parser xing4 \
   --tool-call-parser xing4 \
   --speculative-algorithm EAGLE
```

启动后即可通过 `http://localhost:8000/v1` 访问 OpenAI 兼容 API。

#### KTransformers

[KTransformers](https://github.com/kvcache-ai/ktransformers) 是一个灵活的推理优化框架，支持在有限 GPU 资源下通过异构计算（GPU/CPU 协同）高效运行大规模 MoE 模型。
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

### 超参选择
#### Xing4.0-29B-A4B 模型推理参数选择
- 复杂推理、通用任务，建议使用`temperature=1.0, top_p=0.95, repetition_penalty=1.05`

- 代码、智能体任务，建议使用`temperature=0.8, top_p=0.95, repetition_penalty=1.05`




# 微调

### LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一个开源的大语言模型训练与微调框架，支持预训练、监督微调（SFT）、奖励模型训练（RM）、PPO、DPO、KTO、ORPO、SimPO 等多种训练方式。

Xing4.0 已支持使用 LLaMA-Factory 进行微调、权重合并、推理与部署，具体使用方式参考 [Xing4.0-LLaMA-Factory 微调文档](./tutorial/llama_factory/xing4.0_llama_factory.md)。

### MindFormers

[MindFormers](https://atomgit.com/mindspore/mindformers) 是基于昇思 MindSpore 构建的大模型开发套件，覆盖预训练、微调、推理、评测与部署全流程，内置张量并行、流水线并行、序列并行等分布式策略，为昇腾平台的大模型训练提供开箱即用的能力。

Xing4.0 已完成对昇腾 Atlas 800T A3 集群的适配，支持基于 MindSpore + MindFormers 进行分布式训练与评测，具体使用方式参考 [Xing4.0-MindFormers 训练文档](./tutorial/mindformers/xing4.0_MindFormers.md)。

### FlagOS

[FlagOS](https://docs.flagos.io/en/latest/) 是面向大模型训练、推理与部署的开源 AI 系统软件栈，提供多种主流大模型框架和硬件平台的适配能力。

Xing4.0 基于 FlagOS 完成了 mHC Triton-Ascend 融合算子在 MindFormers 上的接入与单机 16 卡训练验证，具体使用方式参考 [Xing4.0-FlagOS Triton 算子接入文档](./tutorial/flagos/xing4.0_FlagOS_mHC_MindFormers.md)。



# 致谢

在此，我们要向开源社区的伟大贡献致以最深切的谢意——正是站在这些巨人的肩膀上，我们才得以眺望更远的风景。

特别的向DeepSeek团队表达我们诚挚的感激。借鉴其模型架构的设计智慧，为我们模型的训练过程赋予了显著的稳定性与效率，使探索之路更为平稳而清晰。


# 声明、协议、引用

### 声明

请勿将 Xing4.0 及其衍生模型用于任何危害国家社会安全或违反法律法规的活动，也请勿将其用于未经安全审查与备案的互联网服务。

我们已尽力确保训练数据的合规性，但受限于模型与数据的复杂性，仍可能存在无法预见的问题。对于因使用本开源模型所引发的任何问题——包括但不限于数据安全、舆论风险，以及模型被误导、滥用、传播或不当利用所带来的风险——我们不承担相关责任。


### 引用
如需引用我们的工作，请使用如下参考:

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
