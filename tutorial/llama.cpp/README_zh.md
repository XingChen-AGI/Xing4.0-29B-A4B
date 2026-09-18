# Xing4.0-29B-A4B-llama-pc-deploy

https://github.com/user-attachments/assets/6fad2612-27a9-497c-b03e-340e98c8e86f


两种方案，从零到对话只需 10 分钟

## 一、模型简介
Xing4.0-29B-A4B 是 MoE 架构大模型（MLA + MoE + HC 定制），总参数 29B，int4权重采用 IQ4_NL 混合精度量化，量化后约 19GB，可在单张消费级 GPU 上运行。  
### 硬件环境
本次测试环境如下：  
| **项目** | **配置** |
|---|---|
| 操作系统 | Windows 11 |
| GPU | RTX 3090 24GB |
| CUDA Driver Version | 13.1 |
| 推理框架 | llama.cpp |
### 部署方案
本文提供两种部署方案，按需选择：
| 对比项 | 方案一：免编译分发 | 方案二：一键编译 |
| :--- | :--- | :--- |
| **适合人群** | 普通用户、快速上手 | 开发者、需要灵活定制 |
| **操作步骤** | 解压、放模型、双击 bat 运行 | PowerShell 脚本，自动编译启动 |
| **耗时** | 约 3 分钟（解压 + 启动） | 首次约 15-30 分钟（编译） |
| **文件大小** | 预编译包约 390MB | 脚本约 16KB |
| **环境要求** | 仅显卡驱动 | Git + CMake + VS2022 + CUDA Toolkit |
| **灵活性** | 仅可调启动参数 | 可修改源码、切换分支、调整编译选项 |
| **可移植性** | 3090（sm_86）及以上架构 GPU | 任意 Windows + CUDA 机器 |
| **更新方式** | 重新分发部署包 | `git pull` + 重新编译 |

## 二、方案一：免编译分发
### 2.1 适用场景
●目标机器 GPU 架构与编译机相同或更新（如 3090/3050 同为 sm_86，4090 sm_89 向下兼容）   
●已安装 NVIDIA 显卡驱动
●不想安装开发工具链（Git / CMake / VS2022 / CUDA Toolkit）
### 2.2 部署包内容
部署包是一个[tar压缩包](https://github.com/shuxiaoqiong/xing4_0-llama-pc-deploy/releases/download/deploy-without-compile/Xing4.0-29B-A4B-deploy.rar)，包含以下文件：
| 文件 | 说明 | 大小 |
|---|---|---|
| `llama-server.exe` | 静态编译的推理引擎（含嵌入式 Web UI） | ~43 MB |
| `cudart64_13.dll` | CUDA 运行时库 | ~0.5 MB |
| `cublas64_13.dll` | CUDA 矩阵运算库 | ~51 MB |
| `cublasLt64_13.dll` | CUDA 矩阵运算库（轻量版） | ~453 MB |
| `run-server.bat` | 一键启动脚本 | ~2 KB |

模型已经开源，欢迎关注TeleAI:  
huggingface仓库：https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF  
modelscope仓库：https://www.modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF

### 2.3 部署步骤
Step 1：解压部署包
将整个文件夹拷贝到目标机器任意目录（如 D:\xing4_0-llama-pc-deploy\）。
Step 2：放入模型文件
如果模型文件不在部署包中，将 GGUF 权重放入部署目录，与 run-server.bat 同级，最终结构如下：
```
D:\xing4_0-llama-pc-deploy\
  ├── llama-server.exe
  ├── cudart64_13.dll
  ├── cublas64_13.dll
  ├── cublasLt64_13.dll
  ├── run-server.bat
  ├── xing4_0-29b-IQ4_NL-00001-of-00003.gguf ← 放这里
  ├── xing4_0-29b-IQ4_NL-00002-of-00003.gguf ← 放这里
  ├── xing4_0-29b-IQ4_NL-00003-of-00003.gguf ← 放这里
```
Step 3：双击启动
双击 run-server.bat，会弹出命令行窗口显示启动日志，随后浏览器自动打开对话页面。
看到以下输出说明启动成功：
```
model loaded
listening on http://0.0.0.0:8086
```
浏览器地址栏会自动跳转到对应服务端，即可开始对话。
### 2.4 自定义参数
用记事本打开 run-server.bat，修改文件顶部的参数值即可，无需碰下方的启动逻辑：
```
set MODEL=xing4_0-29b-IQ4_NL-00001-of-00003.gguf   rem 模型文件名
set NGL=999                                    rem GPU层数（0=纯CPU）
set CTX=65536                                  rem 上下文长度
set NTOKENS=8192                               rem 最大生成token数
set FA=on                                      rem Flash Attention
set CACHEK=q8_0                                rem KV缓存量化
set CACHEV=q8_0
set PORT=8086                                  rem 端口
set HOST=0.0.0.0                               rem 监听地址
set STATICPATH=                                rem Web UI目录（空=用内置）
```
常见参数：
| 需求 | 修改 |
|---|---|
| 减少显存占用 | `set CTX=32768`（减小上下文） |
| 纯 CPU 运行 | `set NGL=0` |
| 更换端口 | `set PORT=8086` |
| 仅本机访问 | `set HOST=127.0.0.1` |

## 三、方案二：一键编译部署
### 3.1 环境准备
以下软件需提前安装，安装后重启终端使环境变量生效：
1. **Git**  下载：https://git-scm.com
2. **CMake**  下载：https://cmake.org/download
3. **Visual Studio 2022**（勾选"使用 C++ 的桌面开发"）  下载：https://visualstudio.microsoft.com/downloads
4. **CUDA Toolkit**（需与 GPU 驱动版本匹配）  下载：https://developer.nvidia.com/cuda-toolkit-archive
验证环境：
```
git --version
cmake --version
nvcc --version
```
三条命令都有输出，说明环境就绪。
### 3.2 获取脚本
将部署脚本 [Deploy-Xing4.0-29B-A4B.ps1](https://github.com/shuxiaoqiong/xing4_0-llama-pc-deploy/blob/main/Deploy-Xing4.0-29B-A4B.ps1) 放到当前工作目录，脚本会在此目录下自动克隆 llama.cpp 仓库。
### 3.3 运行脚本
在 PowerShell 中执行：
```
.\Deploy-Xing4.0-29B-A4B.ps1
```
脚本会自动完成以下 7 个步骤：
```
步骤	说明
Step 1	检查依赖（Git、CMake、VS2022、nvcc）
Step 2	自动检测 CUDA Toolkit 路径
Step 3	克隆 / 更新 llama.cpp 仓库
Step 4	切换到 xing4_0-port 分支
Step 5	下载 Web UI 静态资源（从 HF 镜像）
Step 6	编译 llama.cpp（GPU 或 CPU 后端）
Step 7	启动 llama-server 并自动打开浏览器
```
### 3.4 可选参数
脚本支持以下参数，按需指定：
```
# CPU 模式（不使用 GPU）
.\Deploy-Xing4.0-29B-A4B.ps1 -Backend cpu
# 自定义端口
.\Deploy-Xing4.0-29B-A4B.ps1 -Port 8086
# 自定义模型路径
.\Deploy-Xing4.0-29B-A4B.ps1 -ModelPath "D:\models\xing4_0-29b-mtp-IQ4_NL.gguf"
# 自定义上下文长度
.\Deploy-Xing4.0-29B-A4B.ps1 -ContextSize 65536
```
完整参数列表：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `-ModelPath` | 无 | 模型文件路径 |
| `-ContextSize` | 262144 | 上下文窗口大小 |
| `-Backend` | gpu | 后端选择：gpu 或 cpu |
| `-GpuLayers` | 999 | GPU 层数（0 = 纯 CPU） |
| `-Port` | 8086 | 服务端口 |
| `-HostAddr` | 0.0.0.0 | 监听地址 |
| `-StaticPath` | 自动检测 | Web UI 静态文件目录 |

### 3.5 编译完成后
编译成功后，脚本会自动启动服务并打开浏览器。看到类似输出说明成功：
```
[OK] Build complete
[OK] Starting API server: http://0.0.0.0:8086
```
浏览器会自动弹出对话页面，即可开始使用。

## 四、常见问题
Q: 启动后浏览器显示 “Server unavailable”    
A:检查命令行窗口是否有报错。常见原因：模型文件路径不对、端口被占用、GPU 显存不足。
 
Q: 浏览器打开 0.0.0.0:8086，页面无法访问  
A:0.0.0.0 是服务端监听地址，客户端需用 http://127.0.0.1:8086 访问。run-server.bat 已自动用 127.0.0.1 打开浏览器。

Q: CUDA DLL 缺失报错  
A:方案二中，三个 CUDA DLL 必须和 llama-server.exe 在同一目录。如果目标机器 GPU 型号不同，需替换为对应版本的 CUDA DLL。

Q: 显存不够（oom）    
A:减小上下文大小（-c 32768 或更小）、使用 KV 缓存量化（--cache-type-k q4_0 --cache-type-v q4_0）、减少 GPU 层数（-ngl 值）。

Q: CPU 模式怎么用  
A:方案一：.\Deploy-Xing4.0-29B-A4B.ps1 -Backend cpu
  方案二：编辑 run-server.bat，将 set NGL=0。
