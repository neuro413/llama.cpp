# llama.cpp

> [!NOTE]
> **本分支 `prism-ptq1_0-avx2` 是 [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) 的 AVX2 优化分支**，
> 基于上游 `prism` 分支的 [`d7b2896`](https://github.com/PrismML-Eng/llama.cpp/commit/d7b2896)，
> 为 PTQ1_0 三值格式（如 Ternary-Bonsai-2-27B）新增了一个 CPU 重排内核，
> 面向没有 AVX-512 VNNI 的 AVX2 处理器（Haswell 及之后的同类 CPU）。
> 改动只涉及 `ggml/src/ggml-cpu/repack.cpp` 一个文件。

## PTQ1_0 AVX2 重排内核

### 效果

测试模型 Ternary-Bonsai-2-27B-PTQ1_0（27B，权重约 6GB），4 核 AVX2（无 VNNI）CPU + 双通道 DDR3，纯 CPU 推理，4 线程：

| 项目 | 上游 CPU 路径 | 本分支 |
|---|---|---|
| 生成速度 (tg) | 1.05 t/s | **2.87–2.91 t/s** |
| 提示处理 (pp64) | 1.03 t/s | **6.73 t/s** |

生成阶段已接近内存带宽上限。贪婪解码输出与上游路径逐字一致。

### 原理

加载时把 PTQ1_0 权重重排成适合 AVX2 的布局（不超过原始行大小）：

```
每行： [G 组 × 128 字节：纯 5-trit 打包，每字节已异或 0x80]
      [r 个尾块 × 32 字节，每字节 4 个 trit]
      [nb 个 fp16 缩放]
      其中 nb = 每行 128 元素块数，G = nb / 5，r = nb % 5
```

- **解 trit**：同一个 32 字节向量连续乘 3（`x + x + x`），每次用两次有符号比较
  `t = [x > -43] - [x > 42]` 取出最高位 trit（得到 w+1 ∈ {0,1,2}），不需要查表、洗牌或移位。
  字节预先异或 0x80，乘 3 后偏置不变（384 ≡ 128 mod 256）。
- **点积**：`vpmaddubsw(t, q)` 计算 `(w+1)·q`，多出的 `Σq` 偏移项在激活量化时预先算好（`ds = d·Σq`），每行最后统一向量化扣除。
- **按组收尾**：每组 5 个块用一次 `vcvtph2ps` 转换 5 个缩放。
- **多列共享解码**：提示处理时一次解码的权重同时与 4 列激活相乘；行再切成约 64KB 的小块，让权重在遍历所有列期间留在 L2。
- **软件预取**：提前 2KB 预取，弥补硬件预取在 4KB 页边界处中断的问题。
- **编译器屏障**：LLVM 会把连续的 `x*3` 合并成乘 9/27/81，而 AVX2 没有字节乘法，会展开成大量只能走 port 0 的指令，因此每次乘 3 后加 `__asm__("" : "+x"(x))`。

启用条件：编译目标有 AVX2 且没有 AVX-512 VNNI，张量为 2D 的 PTQ1_0，且行内尾块能放进原行大小。
使用 `GGML_CPU_ALL_VARIANTS` 构建时，`haswell` 等对应变体会自动启用。

### 推荐运行参数

```bash
# OpenMP 构建时，-C/--cpu-strict 对 OpenMP 线程池无效，要用环境变量把线程分散到各物理核
export OMP_PLACES=cores OMP_PROC_BIND=spread
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf --mmproj Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
  -t <物理核数> -lm dio -fa on -ctk q8_0 -ctv q8_0 -np 1 -cram 512
```

- **绑核**：Windows 等系统可能把两个计算线程放到同一物理核的两个超线程上，它们共用同一组执行端口，
  矩阵乘法会被最慢的核拖住。绑核后生成速度约提升 25%。线程数取物理核数，开超线程反而更慢。
- **`-lm dio`**：DirectIO 加载，系统不保留文件缓存，避免文件缓存和重排后的内存各占一份。
- **`-cram 512`**：llama-server 默认用 8GB 内存做提示缓存，内存较小的机器建议调低。

---

> [!IMPORTANT]
> **This is the PrismML fork of llama.cpp**, the main line behind the [Bonsai](https://huggingface.co/collections/prism-ml/bonsai) models (branch `prism`, developed as `prism-v7`). It tracks current mainline llama.cpp and adds the fork's low-bit formats and runtime features on top.
>
> **New here? Start with the [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) repo.** It downloads the right models and the correct prebuilt binaries for your hardware/backend automatically.
>
> **Which ternary model file to use:**
>
> - `*-PQ2_0.gguf` (fork group-128, ggml id 142): preferred on Metal, CUDA, HIP and CPU. About 6% smaller than group-64.
> - `*-Q2_0_g64.gguf` / 27B `*-Q2_g64.gguf` (official group-64, ggml id 42): runs on every backend here AND on mainline llama.cpp. If unsure, use this. Newer model releases name this file plain `*-Q2_0.gguf`.
> - `*-Q2_0.gguf` on OLDER model repos is the **deprecated legacy format** (group 128 stored as id 42). It does not load on these builds; the error tells you which file to get instead. If you must run it, use the frozen [`prism-v5`](https://github.com/PrismML-Eng/llama.cpp/tree/prism-v5) line and its final release [`prism-b9601`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9601-68faa14).
>
> **Speculative decoding (dspark)** is supported via mainline's draft-dspark plus fork patches. Drafters published for older model releases need a one-time conversion with `gguf-dspark-to-dflash` (see [SPECULATIVE.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) in Bonsai-demo); newer releases ship ready-to-use drafters.
>
> Do NOT build from `prism-v6` (stale mid-migration snapshot) and do NOT mix this fork's `ggml-*` libraries with a stock llama.cpp build.

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
