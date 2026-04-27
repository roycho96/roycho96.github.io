---
title: NCCL and Communication Collectives
date: 2026-04-21 10:00:00 +0900
categories: [Distributed, NCCL]
tags: [nccl, mpi, collective-communication]
lang: en
ref: nccl-collectives
permalink: /posts/nccl-collectives/
math: true
mermaid: true
---

## 1. Why Collective?

When many processes are involved, building group-wide actions like broadcast or reduce out of 1:1 communication alone makes communication time scale linearly with node count.

For example, suppose the root rank wants to broadcast the same data to the other $p-1$ ranks. With P2P (Send/Recv) alone, the root has to call 1:1 Send $p-1$ times in sequence, and every call funnels data through the root's single outgoing link, so total time scales linearly with the node count. Call the same broadcast as an NCCL collective, however, and NCCL automatically picks one algorithm (ring, tree, NVLS, CollNet, PAT) based on topology, message size, and the collective itself. For a large-message broadcast or allreduce the ring family is the right intuition: every link is active simultaneously, so total time becomes nearly independent of node count (concrete comparison in §5.1).

So parallel computing exposes group-level communication patterns (collectives) as first-class APIs. The abstraction has been settled since the MPI era, and NCCL is the same idea ported onto GPUs and NVLink / InfiniBand (IB) / RDMA (Remote Direct Memory Access).

This post is NCCL-centric, but the vocabulary is MPI-compatible. Names like AllReduce, AllGather are identical, and the algorithm-selection logic uses a similar cost model.

## 2. MPI vs NCCL

| Aspect | MPI | NCCL |
|---|---|---|
| Primary target | CPU cluster | GPU cluster |
| Where comm runs | Host-side library | GPU kernel (single-kernel comm + reduction) |
| Data path | Host memory ↔ network | GPU memory ↔ GPU memory (GPUDirect P2P / RDMA)¹ |
| Collective contract | MPI standard | MPI-compatible + extras (NVLS, PAT) |
| Algorithm selection | implementation cost model (Open MPI tuned / MPICH) | NCCL auto + `NCCL_ALGO` |
| P2P | `MPI_Send/Recv` | `ncclSend/Recv` (NCCL 2.7+) |
| One-sided | `MPI_Put/Get/Win` | `ncclPutSignal` / `ncclSignal` / `ncclWaitSignal` + `ncclWindow_t` |

> ¹ With no direct GPU ↔ NIC path (GPUDirect RDMA unavailable), traffic detours through host RAM as an intermediate staging buffer; intra-node, when two GPUs can't talk via direct P2P, host RAM plays the same role (SHM transport). See §4.5.

API behavior is compatible, implementation is GPU-specialized. Calling NCCL "MPI redesigned for GPU-native execution" isn't a stretch.

## 3. The Four Base Patterns

There are four fundamentals. Everything else is a composition.

| Pattern | Direction | Data shape | Typical use |
|---|---|---|---|
| Broadcast | root → all | replicate same value | initial weight/buffer sync |
| Scatter | root → all | distinct chunks | batch / partition distribution |
| Gather | all → root | concat | result collection |
| Reduce | all → root | element-wise op (sum/max/…) | loss/gradient aggregation |

The NCCL official user guide has one diagram per pattern.

![Broadcast: root → all ranks with same value](/assets/img/posts/nccl-collectives/broadcast.png){: width="420"}
_Figure 1. Broadcast. The root copies the same value to all ranks._

![Reduce: all → root, element-wise op](/assets/img/posts/nccl-collectives/reduce.png){: width="420"}
_Figure 2. Reduce. All ranks' values are combined; only the root receives the result._

![Scatter: root → all ranks with distinct chunks](/assets/img/posts/nccl-collectives/scatter.png){: width="420"}
_Figure 3. Scatter. The root's large buffer is split into per-rank pieces._

![Gather: all → root, concatenated](/assets/img/posts/nccl-collectives/gather.png){: width="420"}
_Figure 4. Gather. All ranks' chunks are concatenated at the root in rank order._

Compose these four and you get the rest. AllGather is Gather + Broadcast (every rank holds every chunk). AllReduce is Reduce + Broadcast (the reduced result reaches every rank). ReduceScatter is Reduce + Scatter (combine, then redistribute by chunk). AlltoAll is Scatter × N transposed: every rank sends a different chunk to every rank.

AllReduce can be implemented as Reduce + Broadcast or as ReduceScatter + AllGather. The latter is what MPICH/NCCL use under the names Rabenseifner / Ring. Same semantics, but the choice of decomposition determines performance, and we revisit this with NCCL code in §5.

## 4. NCCL Primitive Catalog

NCCL's public API splits into three groups: collectives, P2P, and one-sided RMA.

### 4.1 Eight Collectives

> Note. The list below tracks NCCL's official user guide. Sources analyzing earlier NCCL series (e.g., NCCL 2.19 in Hu et al., *Demystifying NCCL*) describe the official collectives as five (AllReduce, Broadcast, Reduce, AllGather, ReduceScatter). Gather, Scatter, AlltoAll appear as collective APIs in current docs but are essentially grouped P2P internally (see §4.4).

| Name | Meaning | Input | Output | ML use |
|---|---|---|---|---|
| `ncclAllReduce` | element-wise reduce across all ranks; all ranks receive | `[count]` per rank | `[count]` per rank | DDP gradient sync |
| `ncclBroadcast` | replicate root's value to all ranks | `[count]` on root | `[count]` per rank | init param sync |
| `ncclReduce` | reduce across all ranks; only root receives | `[count]` per rank | `[count]` on root | norm aggregation |
| `ncclAllGather` | each rank's chunk concatenated by all | `[count]` per rank | `[count × nranks]` per rank | ZeRO-3 / FSDP param |
| `ncclReduceScatter` | reduce, then split per-rank | `[count × nranks]` per rank | `[count]` per rank | FSDP gradient |
| `ncclGather` | all ranks' chunks concatenated at root | `[count]` per rank | `[count × nranks]` on root | result collection |
| `ncclScatter` | distribute root's per-rank chunks | `[count × nranks]` on root | `[count]` per rank | batch distribution |
| `ncclAlltoAll` | each rank sends/receives chunks to/from every rank | `[count × nranks]` per rank | `[count × nranks]` per rank | MoE token dispatch |

NCCL official user guide diagrams:

![AllReduce](/assets/img/posts/nccl-collectives/allreduce.png){: width="480"}
_Figure 5. AllReduce. The reduce result reaches all ranks._

![AllGather](/assets/img/posts/nccl-collectives/allgather.png){: width="480"}
_Figure 6. AllGather. All ranks receive the concatenation of every chunk in rank order._

![ReduceScatter](/assets/img/posts/nccl-collectives/reducescatter.png){: width="480"}
_Figure 7. ReduceScatter. Combine, then split into per-rank chunks._

![AlltoAll](/assets/img/posts/nccl-collectives/alltoall.png){: width="480"}
_Figure 8. AlltoAll. Every rank sends a distinct chunk to every other rank. This is the core operation behind MoE expert dispatch._

### 4.2 Point-to-Point

| Name | Meaning |
|---|---|
| `ncclSend` | send to a specific peer |
| `ncclRecv` | receive from a specific peer |

Official since NCCL 2.7. Wrap multiple Send/Recv calls in `ncclGroupStart/End` and you can build scatter / gather / all-to-all patterns out of P2P alone.

### 4.3 One-sided RMA + Signal

**RMA** (Remote Memory Access) is a model where *only one side calls in*, unlike two-sided where a sender's `Send` has to be matched by a receiver's `Recv`. The receiver pre-registers a portion of its memory as a *window*, and the sender reads/writes that window directly whenever it wants. The receiver never calls `Recv` at all. The rendezvous coupling between the two sides is what disappears, and in that sense two-sided is a sync communication model while one-sided RMA is async (we revisit this in §7 Layer 2). The MPI-2 idioms `MPI_Put` / `MPI_Get` / `MPI_Win` are the prototype, and NCCL has shipped the host-side one-sided RMA API (`ncclPutSignal` / `ncclSignal` / `ncclWaitSignal`) since NCCL 2.29.2 (CUDA 12.5+ required).

A **window** here is the handle for "a memory region exposed for RMA," registered with a communicator. In NCCL you call `ncclCommWindowRegister(comm, buff, size, *win, flags)` to turn a region of GPU vidmem into a window. Once registered, only other ranks in the same communicator can RMA into that region, and only through this window. Explicit registration rather than blanket memory exposure, for safety.

**Signal** is the lightweight notification primitive paired with RMA, the doorbell for "I'm done writing, you can read now," decoupled from data movement. The producer calls `ncclPutSignal` to write + notify; the consumer calls `ncclWaitSignal` to wait until ready. This fits producer/consumer patterns where the consumer doesn't need to pre-post a `Recv` (e.g., GPU-resident KV cache, prefill/decode separation).

#### Mapping to familiar OS concepts

Two-sided Send/Recv resembles message-passing IPC (pipes or message queues). Both sides have to explicitly match write/read calls, and one side stalling blocks the other. RMA windows, in contrast, correspond to shared-memory IPC, or `mmap`. One side registers a memory region as shared, and the other accesses it as if it were in its own address space.

`ncclPutSignal`'s data transfer follows the same picture as DMA (Direct Memory Access): no CPU or OS kernel in the path; the NIC or GPU writes directly into the peer's memory (window). That is zero-copy. Signal / WaitSignal, abstractly, is closer to a condition variable or hardware doorbell. Data movement and the ready notification are decoupled, and the consumer no longer has to pre-post a matching `ncclRecv`. Whichever progress model the implementation uses (polling / interrupt / sequence counter) depends on the transport and hardware; the device-side spin-wait in §5.3 is one form of NCCL synchronization among others.

#### NCCL API

The API surface from `nccl.h.in`:

| Name | Meaning |
|---|---|
| `ncclPutSignal(localbuff, count, dtype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)` | push data + signal to peer's window in one call |
| `ncclSignal(peer, sigIdx, ctx, flags, comm, stream)` | signal-only, no data |
| `ncclWaitSignal(nDesc, signalDescs, comm, stream)` | wait on multiple signals via a descriptor array |
| `ncclCommWindowRegister(comm, buff, size, *win, winFlags)` | register a memory window for RMA |
| `ncclCommWindowDeregister(comm, win)` | deregister |
| `ncclWinGetUserPtr(comm, win, **outUserPtr)` | get the symmetric memory pointer |

> `ncclWaitSignal`'s `signalDescs` is an array of `{opCnt, peer, sigIdx, ctx}` structs. `sigIdx` / `ctx` are pinned to 0 for now (must be 0). `ncclPutSignal` / `ncclSignal`'s `flags` is also reserved (0). All reserved fields for future multi-context / multi-signal extensions.

The `ncclWindow_t peerWin` taken by `ncclPutSignal` is an opaque handle to a GPU-vidmem-backed window. This fits distributed reader/writer patterns or a GPU-resident KV cache, anywhere "one side just needs to write into the other side's memory."

#### A concrete case: disaggregated prefill/decode

Splitting LLM inference into prefill (read the user's prompt and produce the KV cache) and decode (use the KV cache to generate tokens one at a time) nodes, a *disaggregated serving* architecture, is becoming standard (vLLM, SGLang, Mooncake, DistServe, Splitwise). The heavy KV cache produced by the prefill node has to be transferred into the decode node's GPU memory, and if you implement that with two-sided `ncclSend` / `ncclRecv` the decode side has to pre-post `ncclRecv` and match timing on every handoff, paying a coupling cost per KV cache transfer.

Implemented with RMA + Signal instead, the decode node registers its KV cache region as a window with `ncclCommWindowRegister`, and the moment prefill finishes it just DMA-writes into that window via `ncclPutSignal` and fires a signal. Decode wakes from `ncclWaitSignal`, finds the data already sitting in its memory, and starts decoding immediately. No `ncclRecv` call, no rendezvous. NVIDIA NIXL (open-sourced at GTC 2025, adopted by vLLM, SGLang, Dynamo) and the Mooncake Transfer Engine implement RDMA transfers exactly along these lines. Workloads like disaggregation, where producer and consumer timing has to be decoupled, are the primary use case for RMA.

> Note: counting NCCL's internal IDs there are even more functions. The `ncclFunc_t` enum (with entries like `AllGatherV`) brings the dispatch function count to 15. From a user's perspective the eight + two + six above are enough.

### 4.4 `ncclScatter` / `ncclGather` / `ncclAlltoAll`

The eight-collective table includes `ncclScatter`, `ncclGather`, and `ncclAlltoAll`, but their internals are not ring/tree algorithms; they're bundles of Send/Recv pairs. The dispatch in `enqueue.cc` makes this clear.

```c
// from src/enqueue.cc
if (info->coll == ncclFuncAlltoAll) {
  for (int r=0; r<comm->nRanks; r++) {
    p2pTaskAppend(comm, info, ncclFuncSend, ...);
    p2pTaskAppend(comm, info, ncclFuncRecv, ...);
  }
} else if (info->coll == ncclFuncGather){
  p2pTaskAppend(comm, info, ncclFuncSend, ..., info->root, allowUB);
  if (comm->rank == info->root)
    for (int r=0; r<comm->nRanks; r++)
      p2pTaskAppend(comm, info, ncclFuncRecv, ...);
} else if (info->coll == ncclFuncScatter) {
  if (comm->rank == info->root)
    for (int r = 0; r < comm->nRanks; r++)
      p2pTaskAppend(comm, info, ncclFuncSend, ...);
  p2pTaskAppend(comm, info, ncclFuncRecv, ..., info->root, allowUB);
}
```

A single `ncclAlltoAll` call expands into rank-count Sends + the same number of Recvs (the `comm->nRanks` loop). The user calls one collective; internally it becomes a single batch of P2P operations dispatched in one kernel launch.

### 4.5 Intra-node vs Inter-node Data Paths

The communicator API is the same, but what happens on the wire differs entirely depending on whether the GPUs sit inside one node or across nodes.

**Intra-node** (between GPUs in the same node):

| Priority | Path | Condition |
|---|---|---|
| 1 | P2P over NVLink | direct NVLink |
| 2 | P2P over PCIe | no NVLink |
| 3 | SHM (via host memory) | P2P unavailable, or inter-socket P2P is inefficient |
| 4 | NIC loopback | multi-socket + per-GPU local NIC + GPUDirect RDMA |

Within a single process, ranks get P2P_DIRECT and bypass the FIFO entirely (the `direct*` primitives in §5.2).

The *SHM transport* (Shared Memory) in row 3 is the fallback when two same-node GPUs can't talk via direct P2P (different PCIe root complexes / NUMA sockets, etc.). GPU A writes into host pinned memory and GPU B reads from the same region. Network isn't involved, so it's faster than inter-node but slower than direct GPU-to-GPU P2P.

**Inter-node** (across nodes):

```
GPU kernel → GPU vidmem → CPU proxy thread → NIC → wire → NIC → ...
                                  │
                                  └→ RDMA write (IB/RoCE) or socket send
```

- Once a GPU kernel fills a buffer, the **CPU proxy thread** (`ncclProxyService`, `src/proxy.cc`) posts the NIC's RDMA write or socket send. The CPU never touches the data itself, but orchestrating NIC operations is host-thread work.
- **GPUDirect RDMA available** (NIC and GPU share a PCIe switch or sit within the same complex of bridges; gated by `ncclTopoCheckGdr` in `src/graph/paths.cc`) means the intermediate buffer lives in GPU vidmem and the NIC reads/writes GPU memory directly. `NCCL_NET_GDR_LEVEL` tunes the threshold.
- **Unavailable** routes through host pinned memory: GPU → host copy → NIC RDMA → peer host → GPU copy. Two extra PCIe traversals.
- A **rendezvous** where the two sides agree on buffer readiness precedes every data transfer.

DMA devices like GPUs and NICs bypass the OS and read/write physical addresses directly, so if a page got swapped mid-transfer the DMA would land on the wrong memory. Staging buffers therefore have to be pinned. The familiar analogy: PyTorch DataLoader's `pin_memory=True` produces exactly this memory type. There it speeds up dataset → GPU H2D copies (and lets `non_blocking=True` actually run async). NCCL uses host pinned memory as a staging buffer for the same reason whenever GPUDirect RDMA isn't available. The underlying CUDA API for both is `cudaHostAlloc` / `cudaHostRegister`.

#### Topology graph

A *topology graph* models a node's hardware (GPUs, NICs, PCIe bridges, NVSwitches, NUMA domains) and the links between them (NVLink, PCIe, C2C) as nodes and edges. NCCL builds it at init. The intra-node priority, the GPUDirect RDMA gating, and the algorithm picked in §5 are all decisions on this graph. `ncclTopoGetSystem` (`src/graph/topo.cc`) walks sysfs (`/sys/class/pci_bus`, `/sys/devices/system/node`), queries NVML for NVLink and C2C, and pulls NIC properties from the network plugin to detect GPU, NIC, NVSwitch, PCIe-bridge, and CPU-NUMA nodes. Intra-node ranks then bootstrap-allgather their local views and fuse them into one system graph, and `ncclTopoComputePaths` (`src/graph/paths.cc`) BFS-labels every (source, target) pair with a *path type* (`PATH_LOC` < `NVL` < `NVB` < `C2C` < `PIX` < `PXB` < `P2C` < `PXN` < `PHB` < `SYS` < `NET`) and an aggregate bandwidth.

Transport selection is simple. `selectTransport` walks `{p2p, shm, net, collNet}` and keeps the first one whose `canConnect` accepts the path. `NCCL_P2P_LEVEL` (default `PATH_PXB`) caps how far P2P reaches, and `NCCL_NET_GDR_LEVEL` (default `PATH_P2C` on C2C systems, else `PATH_PXB`) caps GDR. Algorithm search in `src/graph/search.cc` walks the same graph to score Ring, Tree, NVLS, and CollNet patterns. To see what NCCL detected on a given system, run with `NCCL_TOPO_DUMP_FILE=topo.xml`; rank 0 dumps the full system XML. To override detection on a non-standard chassis, point `NCCL_TOPO_FILE` at a hand-edited XML.

This proxy thread is what makes "host async" in §7 Layer 2 more precise. `ncclSend` returning immediately is just the stream enqueue; the GPU kernel fills the buffer next, and only after that does the proxy thread post the NIC operation. Wire traffic happens long after the host call returned. The true end of a collective is the proxy thread's last RDMA completion, not the host call's return.

## 5. Collective Decomposition and the NCCL Kernel

Several relations let you express the same semantics through different primitive compositions.

$$
\text{AllReduce} \equiv \text{ReduceScatter} + \text{AllGather}
$$

$$
\text{AllReduce} \equiv \text{Reduce} + \text{Broadcast}
$$

$$
\text{AllGather} \equiv \text{Gather} + \text{Broadcast}
$$

ZeRO-3 / FSDP communication design uses the first decomposition (AR = RS + AG) directly: split AllReduce into RS + AG and keep gradients only on the partition. The body of NCCL's Ring AllReduce kernel is also exactly two loops: one for the RS phase, one for the AG phase.

### 5.0 Communication Channels: one collective = N parallel rings

So far we've been drawing "ring" as a single path, but NCCL actually splits one collective call across multiple channels. The reason is simple: if one SM handled all the data, that SM would bottleneck large messages, and the NVLink links / per-node NICs would sit underused. Channels are the *parallel pipeline* abstraction that arrived for exactly this.

- kernel grid: `dim3 grid = {(unsigned)nChannels, 1, 1};` (`src/enqueue.cc`). One channel = one CUDA block.
- input buffer: partitioned into per-channel disjoint contiguous regions.
- each channel runs its own ring (or tree) instance *independently*.
- if per-channel chunks get too small, NIC FIFOs sit underfilled and network throughput tanks. For small messages NCCL heuristically reduces `nChannels` (`enqueue.cc::scheduleP2pTasksToPlan`).

So the `runRing` we'll meet in §5.2 is the ring run *for one channel*, and within the same kernel launch nChannels blocks run the same code over different data segments simultaneously. This doesn't contradict the single-kernel model of §7 Layer 2; it sharpens it. One kernel launch, with a grid of nChannels inside.

### 5.1 Naive vs Ring

A picture before the code. Suppose we broadcast across 4 GPUs. The simplest approach (Naive) is master/slave: the root sends data directly to every other GPU. Ring views all GPUs as a neighbor chain (`G0 → G1 → G2 → G3 → G0`), splits the data into small chunks, and only neighbors hand chunks off.

![Master/slave (Naive) topology](/assets/img/posts/nccl-collectives/gibiansky-master-slave.png){: width="500"}
_Figure 9. The Naive master/slave topology. The root sends to every other GPU directly. The root's outgoing link is the bottleneck every round, while inter-GPU links sit idle._

![Ring topology](/assets/img/posts/nccl-collectives/gibiansky-ring.png){: width="500"}
_Figure 10. The Ring topology. Each GPU only talks to its immediate predecessor/successor. Splitting data into chunks lets every link carry a different chunk simultaneously, so bandwidth becomes nearly independent of node count._

Take the same 4 GPUs broadcasting the same data. Looking at per-round link activity makes the difference plain. Notation:

- $p$ = number of GPUs (ranks)
- $n$ = total bytes to send
- $B$ = per-link bandwidth (bytes/sec)
- $c$ = chunk size in the ring split

**Naive (full $n$ each round, only the root's link active)**

| round | `G0→G1` | `G0→G2` | `G0→G3` | total |
|---|---|---|---|---|
| 1 | $n$ |  |  | $n$ |
| 2 |  | $n$ |  | $n$ |
| 3 |  |  | $n$ | $n$ |

Total time ≈ $(p-1) \cdot n / B$. While one link works, the others sit idle; the entire dataset traverses again every round.

**Ring (data split into 3 chunks $a, b, c$, all links active)**

| round | `G0→G1` | `G1→G2` | `G2→G3` |
|---|---|---|---|
| 1 | $a$ |  |  |
| 2 | $b$ | $a$ |  |
| 3 | $c$ | $b$ | $a$ |
| 4 |  | $c$ | $b$ |
| 5 |  |  | $c$ |

Setting $m = \lceil n/c \rceil$ for the chunk count, ring/chain pipeline broadcast in general:

$$
T \approx (m + p - 2)\frac{c}{B} = \frac{n}{B} + O\!\left(\frac{pc}{B}\right)
$$

The first term $n/B$ is the time the whole dataset spends crossing one link; the second term $O(pc/B)$ is pipeline fill / drain cost. As $m$ grows with smaller chunks, the second term becomes negligible and $T \to n/B$, almost independent of GPU count $p$ (the table above shows $m=3$).

Ring AllReduce extends the same principle. It's not a broadcast: instead, an RS phase ($p-1$ steps accumulating reduces) and an AG phase ($p-1$ steps propagating to all) run consecutively on the same ring.

The bandwidth-optimal $2(p-1)K/(pB)$ of §5.1 falls out as cleanly as it does because NCCL forces one design choice: putting every primitive (send / recv / reduce / forward) inside *one* CUDA kernel. A naïve design that launched separate kernels for the network primitives and the reduce would pay CUDA launch latency on every step instead of once, leave the SM idle between launches, and force the reduce kernel to re-read the just-arrived bytes from HBM. NCCL folds the whole ring traversal plus the reductions into a single kernel. Launch overhead collapses to one. `recv → reduce → send` happens inside the same thread's register set, so the bytes never round-trip through HBM (§5.3, §7 Layer 2). The channel model in §5.0 and the code dive ahead both fall out of that choice.

### 5.2 The NCCL Ring AllReduce Kernel

Ring AllReduce splits data into $p$ chunks and starts GPU $i$ from chunk $i$. At iteration $k$:

- ReduceScatter phase: send chunk $(i + k) \bmod p$ to the next GPU; receive chunk $(i + k - 1) \bmod p$ from the previous GPU and accumulate it with the local value.
- AllGather phase: send chunk $(i + 1 + k) \bmod p$; receive $(i + k) \bmod p$ and overwrite.

![Scatter-Reduce iteration](/assets/img/posts/nccl-collectives/gibiansky-scatter-reduce-step.png){: width="560"}
_Figure 11. One Scatter-Reduce iteration. Every GPU simultaneously sends one chunk to the next GPU and receives one chunk from the previous GPU, accumulating it with the local value._

![AllGather iteration](/assets/img/posts/nccl-collectives/gibiansky-allgather-step.png){: width="560"}
_Figure 12. One AllGather iteration. After RS finishes, the same ring is traversed once more, this time overwriting instead of reducing._

After $p-1$ iterations of each phase, RS leaves GPU $i$ holding the reduced result for chunk $(i+1) \bmod p$, and AG ends with all GPUs holding all chunks. The `offset` variable in the code carries this indexing.

```c
// src/device/all_reduce.h::runRing
for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  // ReduceScatter phase
  prims.directSend(offset, offset, nelem);                                    // step 0
  for (int j = 2; j < nranks; ++j)
    prims.directRecvReduceDirectSend(offset, offset, nelem);                  // recv + reduce + send
  prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
  // RS done in nranks-1 steps. Each rank now holds one reduced chunk.

  // AllGather phase
  for (int j = 1; j < nranks - 1; ++j)
    prims.directRecvCopyDirectSend(offset, offset, nelem);                    // recv + copy + send
  prims.directRecv(offset, nelem);
  // AG done in nranks-1 steps. All ranks now hold all chunks.
}
```

The first loop is the ReduceScatter phase ($p-1$ steps), the second is the AllGather phase ($p-1$ steps). Both phases run consecutively on the same ring within a single kernel, and `directRecvReduce*` finishes the reduce in the same kernel as soon as data arrives (we revisit this fused structure in §7 Layer 2). Ring AllReduce's $2(p-1)$-step cost is precisely the step count of these two loops.

> Counting algorithm rounds: RS $(p-1)$ + AG $(p-1)$ = $2(p-1)$. Counting NCCL device-primitive calls: RS issues `directSend` + `directRecvReduceDirectSend × (p-2)` + `directRecvReduceCopyDirectSend` for $p$ calls; AG issues `directRecvCopyDirectSend × (p-2)` + `directRecv` for $p-1$, totaling $2p-1$ primitive invocations. Same algorithm in different units: $2(p-1)$ rounds, $2p-1$ primitive calls. AG's first chunk forward is fused into the last RS `directRecvReduceCopyDirectSend`, so there is no separate `directSend` kicking off the AG phase.

The `direct*` prefix on these primitives (`directSend`, `directRecv`, `directRecvReduceDirectSend`, etc.) marks the P2P_DIRECT shortcut. Within the same host *and* the same process (the typical DDP/FSDP single-process-multi-GPU scenario), data goes directly from source GPU vidmem into destination GPU vidmem, skipping the intermediate FIFO that ordinary P2P uses. The gate is `P2P_SAME_PID = (hostHash == peerHostHash) && (pidHash == peerPidHash)` (`src/transport/p2p.cc`); cross-process intra-node falls back to P2P_IPC / P2P_CUMEM. Non-direct variants (`recvReduceSend`, `recvCopySend`, etc.) go through the FIFO.

This difference shows up directly in latency. With P2P_DIRECT active, each send/recv saves one memory copy, which is why most fused primitives in §5.3 come in `direct*` flavors.

Looking at per-GPU wire traffic instead of step count, this algorithm is also *bandwidth-optimal*. Ring AllReduce's per-rank **send** traffic is $(p-1)K/p$ in RS plus $(p-1)K/p$ in AG, totaling $2(p-1)K/p$. **Receive** traffic is the same (full-duplex links assumed). Time models usually express the bandwidth term in terms of send direction.

$$
\text{per-GPU send traffic} = \frac{2(p-1)K}{p} \xrightarrow{p \to \infty} 2K
$$

($K$ = total data in bytes). The byte-cost lower bound for AllReduce is $2K$ (each GPU sends its own data once, receives the result once), so the ring achieves the information-theoretic minimum exactly.

> **Protocol dimension**. The code below (and `waitPeer` in §6.2) is NCCL's **Simple** protocol. NCCL has a separate protocol dimension: Simple, LL, LL128. The same ring algorithm uses different sync mechanisms and transfer granularities depending on the protocol.
>
> - **Simple**: large chunks + memory fence for sync. The `waitPeer` + step-counter pattern below is this. Strong on large messages, but fence overhead makes small-message latency high.
> - **LL** (Low Latency): flag-based sync via 8 B atomics (4 B data + 4 B flag). No memory fence. Strong on small messages, but bandwidth tops out at 25-50% of peak.
> - **LL128**: 128 B atomics (120 B data + 8 B flag). Reaches ~95% of peak on intra-node NVLink.

### 5.3 What `directRecvReduceDirectSend` Actually Does

How a single `directRecvReduceDirectSend` call accomplishes "recv + reduce + send" becomes clear from the primitive definitions in `src/device/prims_simple.h`.

```c
// from src/device/prims_simple.h
__device__ __forceinline__
void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx,
                                ssize_t eltN, bool postOp=false) {
  genericOp</*DirectRecv*/1, /*DirectSend*/1, /*Recv*/1, /*Send*/1,
            Input, /*DstBuf*/-1>(inpIx, outIx, eltN, postOp);
}

__device__ __forceinline__
void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
  genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
}
```

The variants (`directSend`, `directRecv`, `directRecvCopyDirectSend`, `recvReduceSend`, …) are all instances of the same `genericOp<DirectRecv, DirectSend, Recv, Send, SrcBuf, DstBuf>` template. The 25 combinations differing only in template parameters basically *are* NCCL's kernel-level vocabulary.

What one `genericOp` call does internally:
1. `waitPeer()`. Spin until the peer's step counter advances.
2. `subBarrier()`. Synchronize worker threads in the block.
3. `reduceCopy<...>(srcs, dsts, workSize)`. Take the received data, element-wise reduce with the local Input, and store into the next fifo.
4. `barrier()`. Block-wide barrier after the reduce-copy.
5. `postPeer()`. Increment our own step counter to notify the next peer.

(Network-device transports also slot in an `ncclNetDeviceUnpack` plus an extra `subBarrier` between steps 2 and 3.)

So one `directRecvReduceDirectSend` is a single cycle of spin-wait → fused reduce-copy → notify, with thread-block barriers between stages. This cycle runs once per ring step, with the host CPU never involved. We come back to what this single-kernel design implies in §7 Layer 2.

## 6. P2P vs Collective

### 6.1 Roles

> A collective must be called by every rank in the communicator in the same order, with the same `count` and `datatype`. Mismatch is undefined behavior or a hang. (NCCL docs, *Collective Operations*)

| Aspect | Collective | P2P (two-sided) |
|---|---|---|
| Participants | every rank in the communicator | only sender + receiver |
| Call form | every rank with the same op / count / datatype | one side Send, the other Recv |
| Sync strength | strong (group-wide barrier feel) | weak (only the peer pair has to match) |
| Expressiveness | fixed patterns only | arbitrary peer subsets, irregular routing |

### 6.2 P2P sync / async

NCCL `Send/Recv` is GPU-blocking but host-async. Start with the receiver-side device function:

```c
// src/device/sendrecv.h::runRecv
__device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
  Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
    prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr, ...);
  size_t cursor = 0;
  do {
    int n = min(size_t(chunkSize), bytes - cursor);
    prims.directRecv(cursor, n);   // GPU spins here until the peer's step arrives
    cursor += n;
  } while (cursor < bytes);
}
```

`prims.directRecv` ultimately calls `waitPeer`. That's the actual spin-wait body.

```c
// src/device/prims_simple.h::waitPeer
void waitPeer(...) {
  int spins = 0;
  while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
    connStepCache = loadStepValue(connStepPtr);  // volatile load of the peer's step counter
    if (checkAbort(flags, Aborted, spins)) break;
  }
}
```

The GPU thread spins until the peer's step counter reaches the value it's waiting for. `connStepPtr` is a counter mapped into the peer GPU's vidmem; volatile loads re-read it every iteration. Until the receiver consumes, the sender can't move on. Meanwhile, on the host side, `ncclSend` / `ncclRecv` just enqueue onto the CUDA stream and return immediately (§7 Layer 2).

`NCCL_STEPS` (default 8, in `src/include/device.h`) determines how many slots a per-channel buffer is divided into. On the same channel's ring, while one chunk is being reduced in slot 0, the next chunk arrives at slot 1 and the one after that gets enqueued at slot 2: a multi-stage pipeline.

Two dimensions, which is the point.

- ring step: one round with the peer (the $i$-th reduce in RS, the $i$-th forward in AG).
- slot step: one cell of the FIFO. Within a single ring step, different slots can be at different stages.

The `waitPeer` step counter is in slot units. `step + StepPerSlice` in the code above is "the slot I want to write next," while the peer's `connStepCache` is "the slot the peer has already advanced to". The wait releases only once the peer is sufficiently ahead. This slot pipelining is orthogonal to the channel parallelism in §5.0: another layer of pipelining inside the same channel.

Running multiple P2Ps concurrently requires `ncclGroupStart/End`. The NCCL header is explicit:

> "This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations need to progress concurrently to complete, they must be fused within a ncclGroupStart/ncclGroupEnd section." (`nccl.h.in`)

The `ncclGroupDepth` counter in `src/group.cc` is thread-local. While depth > 0, collective calls don't launch immediately; they queue, and `ncclGroupEnd` flushes them as a single kernel launch. The send/recv pairs need to live in one kernel for GPUs to avoid waiting on each other forever, which is why the group call is the central piece for deadlock prevention.

## 7. Sync vs Async

The "is an NCCL collective sync or async?" question gets confusing because two perspectives are mixed.

### Layer 1. Training-level perspective

Large-scale LLM training is typically synchronous (BSP). You can't move on to the next step's weight update until the required collective / P2P finishes. PyTorch DDP docs call constructor / forward / backward "distributed synchronization points" for this reason. Even with overlap / prefetch options on, that's Layer-2 concurrent execution, not async by definition.

### Layer 2. NCCL API / CUDA stream

Both collectives and P2P calls return immediately after enqueueing onto the CUDA stream. Host-async. In code:

```c
// src/enqueue.cc::ncclLaunchKernel
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  cudaStream_t launchStream = planner->streams->stream;
  // ...
  CUCHECKGOTO(cuLaunchKernel(fn, grid.x, ..., launchStream, nullptr, extra),
              ret, do_return);
  // returns here. The kernel runs asynchronously on the GPU.
}
```

One `cuLaunchKernel` invocation is one collective call. That's why a `dist.all_reduce(...)` looks like it finishes in milliseconds from Python; the actual wire traffic happens later, on the GPU.

NCCL implements communication and computation as a single kernel. Looking at `reduceCopyPacks`'s inner loop (the heart of the `genericOp` we saw in §5.3) makes the fused structure obvious:

```c
// from src/device/common_kernel.h::reduceCopyPacks
while (...) {
  BytePack<BytePerPack> acc[Unroll];
  // 1) load received data from the peer fifo (volatile = read fresh each time)
  acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
  // 2) element-wise reduce against the local input or another source
  acc[u] = applyReduce(redFn, acc[u], tmp[u]);
  if (postOp) acc[u] = applyPostOp(redFn, acc[u]);
  // 3) store to the next peer's fifo (or the output buffer)
  st_global<BytePerPack>(minDsts[d], acc[u]);
}
```

A single thread runs `ld_volatile_global → applyReduce → st_global` in the same register set. No CPU, no other kernel. When the kernel actually runs on the GPU, chunks flow around the ring while reductions happen inside the same kernel (cf. §5). From the host's point of view it's async; from a distributed-systems point of view it's a rendezvous. Both views are simultaneously correct.

A bit more on what "rendezvous from a distributed-systems point of view" means: NCCL's `ncclSend` / `ncclRecv` returns immediately on the host call, so it looks async on the surface, but as long as it's two-sided, the data is never delivered if the receiver doesn't call `ncclRecv`. MPI's non-blocking `MPI_Isend` / `MPI_Irecv` is the same story: the call returns immediately, but both sides still have to logically post a matching call, and that coupling (rendezvous) survives. Whether the API is blocking or not, two-sided is a sync communication model.

The coupling disappears only with the one-sided RMA in §4.3, namely `ncclPutSignal` / `ncclWaitSignal`. Once the receiver registers a window via `ncclCommWindowRegister`, it makes no further calls; the sender writes directly into that memory. So "two-sided is sync, one-sided is async" describes this architectural coupling, not whether the API call is blocking. That's why patterns where producer and consumer timing has to be decoupled (the prefill/decode disaggregation in §4.3) fit RMA naturally.

To recap, whether NCCL is sync or async depends on the perspective.

| Perspective | NCCL behavior |
|---|---|
| Host API | returns right after CUDA stream enqueue. host-async |
| GPU kernel | spins on peer step / FIFO state. device-blocking |
| Two-sided P2P | sender / receiver matching required. rendezvous coupling remains |
| Training step | BSP, so the optimizer step blocks until collectives finish |
| One-sided RMA | window-registered, initiator-driven put/signal. no matching Recv |

---

## Reference

- Hu, Z. et al. "Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms." arXiv:2507.04786, 2026.
- <https://docs.nvidia.com/deeplearning/nccl/>
- <https://github.com/NVIDIA/nccl>
- <https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/>
