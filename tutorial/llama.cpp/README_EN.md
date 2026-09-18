# Xing4.0-29B-A4B-llama-pc-deploy

https://github.com/user-attachments/assets/6fad2612-27a9-497c-b03e-340e98c8e86f


Two deployment options — from zero to chat in just 10 minutes.

## 1. Model Overview
Xing4.0-29B-A4B is a MoE-architecture LLM (MLA + MoE + HC customization) with 29B total parameters. The int4 weights use IQ4_NL mixed-precision quantization, producing a quantized size of about 19GB — runnable on a single consumer-grade GPU.
### Hardware Environment
The test environment used is as follows:
| **Item** | **Configuration** |
|---|---|
| OS | Windows 11 |
| GPU | RTX 3090 24GB |
| CUDA Driver Version | 13.1 |
| Inference Framework | llama.cpp |
### Deployment Options
This document provides two deployment options — choose as needed:
| Item | Option 1: Pre-built Release | Option 2: One-click Compile |
| :--- | :--- | :--- |
| **Target users** | General users, quick start | Developers, flexible customization |
| **Steps** | Unzip, drop in model, double-click bat | PowerShell script, auto-compile and launch |
| **Time** | ~3 minutes (unzip + start) | ~15-30 minutes first time (compile) |
| **File size** | Pre-built package ~390MB | Script ~16KB |
| **Requirements** | GPU driver only | Git + CMake + VS2022 + CUDA Toolkit |
| **Flexibility** | Only startup params adjustable | Modify source, switch branches, adjust build options |
| **Portability** | 3090 (sm_86) and newer GPU architectures | Any Windows + CUDA machine |
| **Updates** | Re-download release package | `git pull` + recompile |

## 2. Option 1: Pre-built Release
### 2.1 Applicable Scenarios
- Target machine GPU architecture is identical to or newer than the build machine (e.g., 3090/3050 are both sm_86; 4090 sm_89 is backward compatible)
- NVIDIA GPU driver already installed
- No desire to install the development toolchain (Git / CMake / VS2022 / CUDA Toolkit)
### 2.2 Package Contents
The deployment package is a [tar archive](https://github.com/shuxiaoqiong/xing4_0-llama-pc-deploy/releases/download/deploy-without-compile/Xing4.0-29B-A4B-deploy.rar) containing the following files:
| File | Description | Size |
|---|---|---|
| `llama-server.exe` | Statically compiled inference engine (with embedded Web UI) | ~43 MB |
| `cudart64_13.dll` | CUDA runtime library | ~0.5 MB |
| `cublas64_13.dll` | CUDA matrix operation library | ~51 MB |
| `cublasLt64_13.dll` | CUDA matrix operation library (lite) | ~453 MB |
| `run-server.bat` | One-click launch script | ~2 KB |

The model is open-source — welcome to follow TeleAI:
HuggingFace repository: https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF
ModelScope repository: https://www.modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF

### 2.3 Deployment Steps
Step 1: Unzip the package
Copy the entire folder to any directory on the target machine (e.g., D:\xing4_0-llama-pc-deploy\).
Step 2: Place the model files
If the model files are not in the package, drop the GGUF weights into the deploy directory, at the same level as run-server.bat. The final structure should be:
```
D:\xing4_0-llama-pc-deploy\
  ├── llama-server.exe
  ├── cudart64_13.dll
  ├── cublas64_13.dll
  ├── cublasLt64_13.dll
  ├── run-server.bat
  ├── xing4_0-29b-IQ4_NL-00001-of-00003.gguf ← place here
  ├── xing4_0-29b-IQ4_NL-00002-of-00003.gguf ← place here
  ├── xing4_0-29b-IQ4_NL-00003-of-00003.gguf ← place here
```
Step 3: Double-click to launch
Double-click run-server.bat. A command-line window will appear showing startup logs, then the browser will automatically open the chat page.
The following output indicates a successful start:
```
model loaded
listening on http://0.0.0.0:8086
```
The browser address bar will automatically jump to the corresponding server, and you can start chatting.
### 2.4 Custom Parameters
Open run-server.bat with Notepad and modify the parameter values at the top of the file — no need to touch the launch logic below:
```
set MODEL=xing4_0-29b-IQ4_NL-00001-of-00003.gguf   rem model filename
set NGL=999                                    rem GPU layers (0 = CPU only)
set CTX=65536                                  rem context length
set NTOKENS=8192                               rem max generated tokens
set FA=on                                      rem Flash Attention
set CACHEK=q8_0                                rem KV cache quantization
set CACHEV=q8_0
set PORT=8086                                  rem port
set HOST=0.0.0.0                               rem listen address
set STATICPATH=                                rem Web UI directory (empty = built-in)
```
Common parameters:
| Goal | Change |
|---|---|
| Reduce VRAM usage | `set CTX=32768` (smaller context) |
| CPU-only mode | `set NGL=0` |
| Change port | `set PORT=8086` |
| Local access only | `set HOST=127.0.0.1` |

## 3. Option 2: One-click Compile Deployment
### 3.1 Environment Preparation
The following software must be installed in advance. Restart the terminal after installation so environment variables take effect:
1. **Git**  Download: https://git-scm.com
2. **CMake**  Download: https://cmake.org/download
3. **Visual Studio 2022** (check "Desktop development with C++")  Download: https://visualstudio.microsoft.com/downloads
4. **CUDA Toolkit** (must match the GPU driver version)  Download: https://developer.nvidia.com/cuda-toolkit-archive
Verify the environment:
```
git --version
cmake --version
nvcc --version
```
If all three commands produce output, the environment is ready.
### 3.2 Get the Script
Place the deployment script [Deploy-Xing4.0-29B-A4B.ps1](https://github.com/shuxiaoqiong/xing4_0-llama-pc-deploy/blob/main/Deploy-Xing4.0-29B-A4B.ps1) in the current working directory. The script will automatically clone the llama.cpp repository in this directory.
### 3.3 Run the Script
In PowerShell, run:
```
.\Deploy-Xing4.0-29B-A4B.ps1
```
The script automatically performs the following 7 steps:
```
Step    Description
Step 1  Check dependencies (Git, CMake, VS2022, nvcc)
Step 2  Auto-detect CUDA Toolkit path
Step 3  Clone / update llama.cpp repository
Step 4  Switch to the xing4_0-port branch
Step 5  Download Web UI static assets (from HF mirror)
Step 6  Compile llama.cpp (GPU or CPU backend)
Step 7  Launch llama-server and auto-open browser
```
### 3.4 Optional Parameters
The script supports the following parameters, specify as needed:
```
# CPU mode (no GPU)
.\Deploy-Xing4.0-29B-A4B.ps1 -Backend cpu
# Custom port
.\Deploy-Xing4.0-29B-A4B.ps1 -Port 8086
# Custom model path
.\Deploy-Xing4.0-29B-A4B.ps1 -ModelPath "D:\models\xing4_0-29b-mtp-IQ4_NL.gguf"
# Custom context length
.\Deploy-Xing4.0-29B-A4B.ps1 -ContextSize 65536
```
Full parameter list:
| Parameter | Default | Description |
|---|---|---|
| `-ModelPath` | none | Model file path |
| `-ContextSize` | 262144 | Context window size |
| `-Backend` | gpu | Backend choice: gpu or cpu |
| `-GpuLayers` | 999 | GPU layers (0 = CPU only) |
| `-Port` | 8086 | Service port |
| `-HostAddr` | 0.0.0.0 | Listen address |
| `-StaticPath` | auto-detect | Web UI static file directory |

### 3.5 After Compilation
After a successful build, the script will automatically start the service and open the browser. Output like the following indicates success:
```
[OK] Build complete
[OK] Starting API server: http://0.0.0.0:8086
```
The browser will automatically pop up the chat page, and you can start using it.

## 4. FAQ
Q: After launch, the browser shows "Server unavailable"
A: Check the command-line window for errors. Common causes: wrong model file path, port occupied, insufficient GPU VRAM.

Q: Browser opens 0.0.0.0:8086 but the page is inaccessible
A: 0.0.0.0 is the server listen address; clients must access via http://127.0.0.1:8086. run-server.bat already opens the browser with 127.0.0.1.

Q: CUDA DLL missing error
A: In Option 2, the three CUDA DLLs must be in the same directory as llama-server.exe. If the target machine has a different GPU model, replace them with the matching version of CUDA DLLs.

Q: Insufficient VRAM (oom)
A: Reduce the context size (-c 32768 or smaller), use KV cache quantization (--cache-type-k q4_0 --cache-type-v q4_0), or reduce GPU layers (the -ngl value).

Q: How to use CPU mode
A: Option 1: .\Deploy-Xing4.0-29B-A4B.ps1 -Backend cpu
   Option 2: Edit run-server.bat and set NGL=0.
