# Prefill-Decode Disaggregated Architectures

Prefill-Decode (P/D) Disaggregated Architectures constitute a leading paradigm for optimizing large language model (LLM) inference at scale. By decoupling the compute-intensive prefill (prompt/context processing) stage from the memory-intensive decode (tokenwise autoregressive generation) stage, these architectures allow for separate, phase-aware hardware allocation, minimizing resource contention and enabling throughput and latency optimizations subject to diverse service-level objectives (SLOs). The following sections survey the defining principles, architectural variants, resource management strategies, scheduling solutions, and empirical findings from key research threads, with a focus on the most impactful methods, metrics, and trade-offs.

## 1. Architectural Principles and Variants

P/D disaggregated architectures physically or logically separate the prefill and decode stages of LLM inference, typically assigning each phase to distinct GPU (or accelerator) pools, hardware configurations, or temporal resource partitions. The fundamental rationale is the orthogonality of resource bottlenecks: prefill is compute-FLOPs bound (requiring peak GPU arithmetic throughput), whereas decode is usually memory bandwidth and working set bound, due to the need for holding long KV caches and streaming tokens sequentially.

Major architectural variants include:

| Variant                   | Disaggregation Dimension            | Characteristic Features         |
|---------------------------|-------------------------------------|-------------------------------|
| Classical P/D             | Resource (node or device-level)     | Prefill and decode on separate devices; KVCache transfer via RDMA or NVLink | 
| Temporal (PaDG, TD-Pipe)  | Time (intra-instance time-sharing)  | Single device alternates prefill and decode in rolling slots or pipeline phases |
| Intra-GPU (Nexus, semi-PD)| Hardware partitioning (SM-level)    | Dynamic allocation of GPU SMs or logical cores per phase, decoupling prefill/decode computation within a device |
| Multi-stage (EPD, HydraInfer)| Functional (encode, prefill, decode) | Explicit splitting for multimodal models: image/audio encode, textual prefill, token decode, possibly run on separate heterogeneous clusters |

Innovations such as KVCache-centric buffering (Mooncake), block-contiguous D2D KVCache transfer (P/D-Serve, FlowKV), homomorphic KVCache quantization (HACK), and hierarchical scheduling/control (TD-Pipe, Arrow) are tailored to reduce phase interference, resource underutilization, and communication bottlenecks [2407.00079][2408.08147][2502.03589][2504.03775][2506.10470].

## 2. Phase Decoupling and Resource Management

Successful P/D disaggregation requires tightly coordinated resource management:

- **KVCache Handling:** Effective reuse, remote retrieval, or cross-node migration of key-value caches is central. Designs such as Mooncake feature disaggregated KVCache that resides in CPU/DRAM/SSD, accessible via GPUDirect-RDMA, supporting cache prefix sharing across requests.
- **Fine-grained Partitioning:** Partitioning computational or memory resources at sub-device (SM/block/core) granularity enables adaptive response to phase-specific workload (semi-PD, Nexus). For instance, semi-PD partitions streaming multiprocessors between phases using CUDA MPS, supporting asynchronous workers with unified memory pointers and atomic allocation to prevent write-after-read hazards [2504.19867][2507.06608].
- **Topology-aware Deployment:** In large heterogeneous clusters, intelligent partitioning and placement (HexGen-2, HeteroScale) optimize the mapping of prefill and decode groups onto GPUs and network subgroups to match bandwidth and compute requirements and minimize cross-switch contention [2502.07903][2508.19559].

## 3. Scheduling, Load Balancing, and Autoscaling

Dynamic and predictive scheduling is essential to meet latency objectives and maximize throughput in the face of varying workloads and phase imbalances.

- **KVCache-aware Scheduling:** Schedulers (e.g., Mooncake’s Conductor) analyze cache prefix overlaps, queue delays ($T_{queue}$), prefill compute times ($T_{prefill}$), and transfer latencies ($T_{transfer}$), selecting instance pairs that minimize time-to-first-token (TTFT) and inter-token decode time (TBT), subject to kvcache_balancing_thresholds:

  $$
  \text{TTFT} = T_\text{queue} + T_\text{prefill} \quad \text{or} \quad T_\text{transfer} + T_\text{queue} + T_\text{prefill}
  $$

- **Early Rejection and Load Prediction:** Imposing SLOs, systems like Mooncake and P/D-Serve employ early rejection by predicting whether a request would cause downstream SLO violation, often using rolling predictions of average decoding time per token (\(t_d\)) and comparing predicted ratios to SLO thresholds:

  $$
  \text{Predicted TBT Ratio} = \frac{\text{Average decoding time per token}}{l_\text{tbt}}
  $$
  
  If the ratio exceeds threshold, requests are dropped preemptively.
  
- **Decoupled and Elastic Scheduling:** Arrow, DynaServe, and HyperFlexis dynamically flip instance roles (prefill/decode), size task/batch pools, and enable real-time resource reallocation under SLOs. Arrow, for instance, treats prefill/decode as per-request attributes and maintains multiple elastic pools, using minimal-load greedy assignment to predict and bound TTFT/TPOT per request [2505.11916][2504.09285][2508.15919].

- **Autoscaling Policies:** HeteroScale demonstrates that decode phase tokens-per-second (TPS) is the robust metric for coordinated scaling, leading to significant GPU utilization gains over naive SM or tail-latency signals. The autoscaling logic maintains the P/D pool ratio within empirically determined ranges and uses proportional controllers and topology-aware schedulers to maximize token throughput even in hardware-diverse and network-constrained environments [2508.19559].

## 4. KVCache Transfer and Compression Strategies

Transmission of KVCache between prefill and decode stages can be a principal bottleneck in P/D architectures, especially under long-context scenarios:

- **Contiguous and Block-free KVCache Transfer:** Systems such as FlowKV and P/D-Serve restructure KVCache memory layouts to enable merging of layer-wise blocks into large segments, reducing per-request NVLink/NCCL transfer calls by up to $L \times 2$ and cutting transmission latency by nearly 96% [2504.03775][2408.08147].
- **Pull-based Tensor-Centric RPC:** KVDirect eliminates heavy synchronization by letting decode workers “pull” required KV cache blocks in coalesced RDMA reads, thus decoupling memory reservation/provision and reducing per-request latency by up to 55% over push-mode baselines [2501.14743].
- **Lossy and Lossless Compression:** HACK applies asymmetric 2-bit quantization to KV data and directly performs homomorphic matrix multiplications—approximate attention—on the quantized data, circumventing costly dequantization. The key approximation is:

  $$
  \sum_{z} a_{iz} b_{zj} \approx s_{a_i} s_{b_j} \sum_z a'_{iz} b'_{zj}
  + m_{b_j} s_{a_i} \sum_z a'_{iz}
  + m_{a_i} s_{b_j} \sum_z b'_{zj}
  + Z m_{a_i} m_{b_j}
  $$

  This achieves JCT reductions of up to 70.9% over uncompressed baselines [2502.03589].

## 5. Specializations: Multimodal and Heterogeneous Deployments

Adapting P/D architectures for heterogeneous hardware and multimodal input increases system complexity but yields notable performance improvements.

- **Multimodal EPD Disaggregation:** For large multimodal models, the EPD architecture (HydraInfer, EPD Disaggregation) inserts an explicit “encode” stage (for images/audio/video), allocating it to dedicated GPUs and separating prefill/decode to maximize batch size and minimize memory contention. Caching multimedia tokens, parallelization of intra-request image patch encodings, and black-box resource optimization improve memory efficiency by up to 15× and batch size by up to 22× [2505.12658][2501.05460].
- **Heterogeneous GPU Scheduling:** HexGen-2 formalizes scheduling as a constraint optimization problem, partitioning the GPU communication topology graph via spectral partitioning and max-flow algorithms to assign heterogeneous GPUs to prefill or decode, considering distinct parallelism strategies (tensor, pipeline) per group [2502.07903]. HeteroScale’s topology-aware scheduler ensures that prefill and decode groups are assigned to network-affine subgroups (RDMA affinity) for minimal transfer bottlenecks.

## 6. Comparative Metrics and Empirical Outcomes

Experimental results across the literature consistently report:

| System      | Throughput Gain          | TTFT / TPOT Improvements            | Additional Notes                                      |
|-------------|-------------------------|-------------------------------------|-------------------------------------------------------|
| Mooncake    | up to 525% (simulated)  | 75% more requests at SLO            | Long-context, early rejection, KVCache reuse [2407.00079] |
| P/D-Serve   | 6.7× over aggregation   | 42% TTFT reduction, 46% transfer    | 8+ month prod deployment, fine-group RoCE [2408.08147] |
| FlowKV      | 15.2–48.9% over baseline| 96% lower KVCache transfer latency  | Heterogeneous GPU support, segment management [2504.03775] |
| TD-Pipe     | 1.91–2.73× over pipeline| Pipeline bubble minimization        | Hierarchical controller, prefill predicting [2506.10470]   |
| Arrow       | 5.62–7.78× request rate | TTFT predictively bounded           | Stateless instances, elastic pools [2505.11916]         |
| Nexus       | 2.2× throughput vs vLLM | 20× TTFT, 2.5× TBT reductions       | Intra-GPU SM partitioning, cost model [2507.06608]     |
| TaiChi      | Up to 77% more goodput  | 13.2× TTFT, 1.69× TPOT reductions   | Hybrid PD (aggreg/disagg), per-request scheduling [2508.01989] |

These results demonstrate dramatic improvements in both throughput and SLO adherence, especially under diverse prompt lengths, tidal (burst) request patterns, and heterogeneous infrastructures.

## 7. Practical Considerations and Trade-offs

Common operational trade-offs in P/D architectures include:

- **Communication vs. Interference:** While disaggregation eliminates prefill/decode interference, it introduces potentially costly KVCache transfers. Hybrid and partially-disaggregated strategies (EcoServe, semi-PD, TaiChi) attempt to balance these competing demands by using temporal disaggregation or intra-instance resource splits [2504.18154][2504.19867][2508.01989].
- **Granularity of Partitioning:** Finer-grained (SM-level, chunked pipelining, micro-request splitting) approaches allow more dynamic adaptation but increase scheduler complexity and may reduce hardware efficiency at very high concurrency.
- **Adaptive and Predictive Scheduling:** Early rejection, SLO-aware token budgeting, and dynamic pooling (Arrow, HyperFlexis) are most effective when accurate performance models are available for queueing, memory, and transfer overheads [2505.11916][2508.15919]. Metric-driven autoscaling policies based on empirically validated signals (decode TPS) are necessary for efficient cloud deployment at scale [2508.19559].
- **Storage and Weight Duplication:** Disaggregated computation with unified storage (semi-PD) avoids weight replication and storage imbalance, achieving both low overhead and high request capacity [2504.19867].

***

P/D Disaggregated Architectures, through phase-aware hardware allocation, fine-grained resource control, predictive and dynamic scheduling, and advanced KVCache transfer and compression techniques, set a robust foundation for scalable, efficient, and SLO-compliant large model inference in modern datacenter deployments. Their continued evolution is characterized by the fusion of algorithmic, systems, and hardware-affinity optimizations, as substantiated by rigorous empirical studies and production-scale deployments.

Source: https://www.emergentmind.com/articles/prefill-decode-p-d-disaggregated-architectures