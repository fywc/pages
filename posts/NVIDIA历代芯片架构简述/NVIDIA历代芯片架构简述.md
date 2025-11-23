# NVIDIA历代芯片架构简述

1999 年，英伟达发明了 GPU（graphics processing unit），英伟达 GPU 从 Fermi 到 Blackwell 共 9 代架构，时间跨度从 2010 年至 2024 年，具体包括费米（Feimi）、开普勒（Kepler）、麦克斯韦（Maxwell）、帕斯卡（Pashcal）、伏特（Volt）、图灵（Turing）、安培（Ampere）和赫柏（Hopper）和布莱克韦尔（Blackwell）架构。经过 15 年的发展，CUDA 已成为英伟达的技术“护城河”，Tensor Core5.0，NVLink5.0，NVswitch4.0，Transformer Engine2.0 等技术迭代更新，正如英伟达公司官方宣传语所言：人工智能计算领域的领导者，推动了 AI、HPC、游戏、创意设计、自动驾驶汽车和机器人开发领域的进步。 ​

| 架构         | 发布时间  | 核心参数                                                                    | 特点&优势                                                   |
| ---------- | ----- | ----------------------------------------------------------------------- | ------------------------------------------------------- |
| Fermi      | 2010  | 16 个 SM，每个 SM 包含 32 个 CUDA Cores，一共 512 CUDA Cores                      | 首个完整 GPU 计算架构，支持与共享存储结合的 Cache 层次 GPU 架构，支持 ECC GPU 架构  |
| Kepler     | 2012  | 16 个 SM，每个 SM 包含 32 个 CUDA Cores，一共 512 CUDA Cores                      | 游戏性能大幅提升，首次支持 GPU Direct 技术                             |
| Maxwell    | 2014  | 16 个 SM，每个 SM 包括 4 个处理块，每个处理块包括 32 个 CUDA Cores+8 个 LD/ST Unit + 8 SFU  | 每组 SM 单元从 192 个减少到每组 128 个，每个 SMM 单元拥有更多逻辑控制电路          |
| Pascal     | 2016  | GP100 有 60 个 SM，每个 SM 包括 64 个 CUDA Cores，32 个 DP Cores                  | NVLink 第一代，双向互联带宽 160GB/s，P100 拥有 56 个 SM HBM           |
| Volta      | 2017  | 80 个 SM，每个 SM 包括 32 个 FP64+64 Int32+64 FP32+8 个 Tensor Cores            | NVLink2.0，Tensor Cores 第一代，支持 AI 运算，NVSwitch1.0         |
| Turing     | 2018  | 102 核心 92 个 SM，SM 重新设计，每个 SM 包含 64 个 Int32+64 个 FP32+8 个 Tensor Cores   | Tensor Core2.0，RT Core 第一代                              |
| Ampere     | 2020  | 108 个 SM，每个 SM 包含 64 个 FP32+64 个 INT32+32 个 FP64+4 个 Tensor Cores       | Tensor Core3.0，RT Core2.0，NVLink3.0，结构稀疏性矩阵 MIG1.0      |
| Hopper     | 2022  | 132 个 SM，每个 SM 包含 128 个 FP32+64 个 INT32+64 个 FP64+4 个 Tensor Cores      | Tensor Core4.0，NVLink4.0，结构稀疏性矩阵 MIG2.0                 |
| Blackwell  | 2024  | 160个SM                                                                  | Tensor Core5.0，NVLink5.0, 第二代 Transformer 引擎，支持 RAS     |

Hopper 架构是第一个真正的异构加速平台，适用于高性能计算（HPC）和 AI 工作负载。英伟达 Grace CPU 和英伟达 Hopper GPU 实现英伟达 NVLink-C2C 互连，高达 900 GB/s 的总带宽的同时支持 CPU 内存寻址为 GPU 内存。NVLink4.0 连接多达 256 个英伟达 Grace Hopper 超级芯片，最高可达 150 TB 的 GPU 可寻址内存。

H100 一共有 8 组 GPC、66 组 TPC、132 组 SM，总计有 16896 个 CUDA 核心、528 个 Tensor 核心、50MB 二级缓存。显存为新一代 HBM3，容量 80 GB，位宽 5120-bit，带宽高达 3 TB/s。

2024 年 3 月，英伟达发布 Blackwell 架构，专门用于处理数据中心规模的生成式 AI 工作流，能效是 Hopper 的 25 倍，新一代架构在以下方面做了创新：

- **新型 AI 超级芯片**：Blackwell 架构 GPU 具有 2080 亿个晶体管，采用专门定制的台积电 4NP 工艺制造。所有 Blackwell 产品均采用双倍光刻极限尺寸的裸片，通过 10 TB/s 的片间互联技术连接成一块统一的 GPU。
- **第二代 Transformer 引擎**：将定制的 Blackwell Tensor Core 技术与英伟达 TensorRT-LLM 和 NeMo 框架创新相结合，加速大语言模型 (LLM) 和专家混合模型 (MoE) 的推理和训练。
- **第五代 NVLink**：为了加速万亿参数和混合专家模型的性能，新一代 NVLink 为每个 GPU 提供 1.8TB/s 双向带宽，支持多达 576 个 GPU 间的无缝高速通信，适用于复杂大语言模型。
- **RAS 引擎**：Blackwell 通过专用的可靠性、可用性和可服务性 (RAS) 引擎增加了智能恢复能力，以识别早期可能发生的潜在故障，从而更大限度地减少停机时间。
- **安全 AI**：内置英伟达机密计算技术，可通过基于硬件的强大安全性保护敏感数据和 AI 模型，使其免遭未经授权的访问。
- **解压缩引擎**：拥有解压缩引擎以及通过 900GB/s 双向带宽的高速链路访问英伟达 Grace CPU 中大量内存的能力，可加速整个数据库查询工作流，从而在数据分析和数据科学方面实现更高性能。

英伟达 GB200 Grace Blackwell 超级芯片通过 900GB/s 超低功耗的片间互联，将两个英伟达 B200 Tensor Core GPU 与英伟达 Grace CPU 相连。在 90 天内训练一个 1.8 万亿参数的 MoE 架构 GPT 模型，需要 8000 个 Hopper 架构 GPU，15 兆瓦功率，Blackwell 架构只需要 2000 个 GPU，以及 ¼ 的能源消耗。
