---
title: NCCL 과 Communication Collectives
date: 2026-04-21 10:00:00 +0900
categories: [Distributed, NCCL]
tags: [nccl, mpi, collective-communication]
lang: ko
ref: nccl-collectives
permalink: /posts/nccl-collectives/
math: true
mermaid: true
---

## 1. 왜 Collective 인가

여러 프로세스가 있을 때 1:1 통신만으로 broadcast 나 reduce 같은 집단 동작을 짜면 통신 시간이 노드 수에 선형으로 늘어난다.

예를 들어 root rank 가 다른 $p-1$ 개 rank 에 같은 데이터를 broadcast 한다고 하자. P2P (Send/Recv) 만 쓰면 root 가 1:1 Send 를 $p-1$ 번 순차로 부르고, 매번 root 의 송신 링크 하나로만 데이터가 나가니 시간이 노드 수에 비례해 늘어난다. 같은 broadcast 를 NCCL collective 로 부르면 NCCL 이 topology / message size / collective 종류에 따라 ring / tree / NVLS / CollNet / PAT 같은 알고리즘 중 하나를 자동 선택한다. 큰 메시지의 broadcast / allreduce 라면 ring 계열을 떠올리면 직관적이고, 내부에서 모든 링크가 동시에 활성화되어 시간이 노드 수에 거의 무관해진다 (구체 비교는 §5.1).

Parallel computing 은 그래서 집단 단위 통신 패턴 (collective) 을 공식 API 로 제공한다. MPI 시대부터 굳어진 추상이고, NCCL 은 같은 개념을 GPU 와 NVLink / InfiniBand (IB) / RDMA (Remote Direct Memory Access) 위에 옮겨 놓은 것.

이 글은 NCCL 기준이지만 어휘 자체는 MPI 와 호환된다. AllReduce, AllGather 같은 이름이 똑같고, 알고리즘 선택도 비슷한 cost model 을 쓴다.

## 2. MPI 와 NCCL

| 기준 | MPI | NCCL |
|---|---|---|
| 주 타겟 | CPU cluster | GPU cluster |
| 통신 실행 | Host side 라이브러리 | GPU kernel (single-kernel comm + reduction) |
| 데이터 이동 | Host memory ↔ network | GPU memory ↔ GPU memory (GPUDirect P2P / RDMA)¹ |
| Collective 의미 | MPI 표준 | MPI 호환 + extra (NVLS, PAT) |
| 알고리즘 선택 | 구현체 cost model (Open MPI tuned / MPICH) | NCCL auto + `NCCL_ALGO` |
| P2P | `MPI_Send/Recv` | `ncclSend/Recv` (NCCL 2.7+) |
| One-sided | `MPI_Put/Get/Win` | `ncclPutSignal` / `ncclSignal` / `ncclWaitSignal` + `ncclWindow_t` |

> ¹ GPU ↔ NIC 직접 경로 (GPUDirect RDMA) 가 없으면 host RAM 을 중간 staging buffer 로 거쳐 보내고, intra-node 에서 두 GPU 가 직접 P2P 못 할 때도 host RAM 으로 우회 (SHM transport). 자세한 건 §4.5.

API 동작은 호환, 구현은 GPU 특화. NCCL 을 "MPI 의 GPU-native 재설계" 로 봐도 무리는 없다.

## 3. 기본 4 패턴

기본은 네 개. 나머지는 다 조합이다.

| 패턴 | 방향 | 데이터 형태 | 대표 용도 |
|---|---|---|---|
| Broadcast | root → all | 같은 값 복제 | 초기 weight/buffer 동기화 |
| Scatter | root → all | 서로 다른 chunk | 배치/파티션 분배 |
| Gather | all → root | concat | 결과 수집 |
| Reduce | all → root | element-wise op (sum/max/…) | loss/gradient 집계 |

NCCL 공식 user guide 의 도식이 이 네 패턴을 한 장씩 보여준다.

![Broadcast: root → all ranks 에 같은 값](/assets/img/posts/nccl-collectives/broadcast.png){: width="420"}
_Figure 1. Broadcast. root 가 같은 값을 all ranks 에 복사한다._

![Reduce: all → root, element-wise op](/assets/img/posts/nccl-collectives/reduce.png){: width="420"}
_Figure 2. Reduce. all ranks 의 값을 합쳐 root 한 명만 결과를 받는다._

![Scatter: root → all ranks 에 서로 다른 chunk](/assets/img/posts/nccl-collectives/scatter.png){: width="420"}
_Figure 3. Scatter. root 가 가진 큰 버퍼를 rank 마다 다른 조각으로 나눠준다._

![Gather: all → root 에 concat](/assets/img/posts/nccl-collectives/gather.png){: width="420"}
_Figure 4. Gather. all ranks 의 chunk 를 root 에 rank 순서대로 concat._

이 넷을 조합하면 자주 쓰는 collective 가 다 나온다. AllGather 는 Gather + Broadcast (all ranks 의 chunk 를 all ranks 가 갖는다), AllReduce 는 Reduce + Broadcast (reduce 결과를 all ranks 에), ReduceScatter 는 Reduce + Scatter (합친 뒤 rank 별 chunk 로 배포), AlltoAll 은 각 rank 가 각각 다른 대상에게 각각 다른 데이터를 보내는 Scatter × N 의 전치 버전.

AllReduce 는 Reduce + Broadcast 로 짜도 되고 ReduceScatter + AllGather 로 짜도 된다. 후자가 MPICH/NCCL 의 Rabenseifner / Ring 방식이다. 같은 의미라도 어느 쪽으로 짜느냐에 따라 성능이 갈리고, §5 에서 NCCL 코드로 다시 본다.

## 4. NCCL Primitive 카탈로그

NCCL 의 공개 API 는 collective, P2P, 그리고 1-sided RMA 세 부류로 나뉜다.

### 4.1 Collective 8 종

> 참고. 아래 API 목록은 NCCL 공식 user guide 기준이다. NCCL 2.19 계열을 분석한 일부 자료 (예: Hu et al., *Demystifying NCCL*) 에서는 공식 collective 를 AllReduce / Broadcast / Reduce / AllGather / ReduceScatter 의 5 개로 설명한다. Gather / Scatter / AlltoAll 은 최신 docs 에 collective API 로 들어와 있지만, 구현상 grouped P2P 로 펼쳐지는 성격이 강하다 (§4.4 참고).

| 이름 | 의미 | 입력 | 출력 | ML 용도 |
|---|---|---|---|---|
| `ncclAllReduce` | all ranks 의 값을 element-wise reduce, all ranks 가 결과 | `[count]` per rank | `[count]` per rank | DDP gradient sync |
| `ncclBroadcast` | root 값을 all ranks 에 복제 | `[count]` on root | `[count]` per rank | init param sync |
| `ncclReduce` | all ranks 의 값을 reduce, root 만 결과 | `[count]` per rank | `[count]` on root | norm 집계 |
| `ncclAllGather` | 각자의 chunk 를 all ranks 가 concat | `[count]` per rank | `[count × nranks]` per rank | ZeRO-3 / FSDP param |
| `ncclReduceScatter` | reduce 후 rank 별 chunk 로 분배 | `[count × nranks]` per rank | `[count]` per rank | FSDP gradient |
| `ncclGather` | all ranks 의 chunk 를 root 에 concat | `[count]` per rank | `[count × nranks]` on root | 결과 취합 |
| `ncclScatter` | root 의 chunk 를 rank 별로 배포 | `[count × nranks]` on root | `[count]` per rank | 배치 분배 |
| `ncclAlltoAll` | 각 rank 가 각 rank 에게 chunk 교환 | `[count × nranks]` per rank | `[count × nranks]` per rank | MoE token dispatch |

NCCL 공식 user guide 의 도식:

![AllReduce](/assets/img/posts/nccl-collectives/allreduce.png){: width="480"}
_Figure 5. AllReduce. reduce 결과를 all ranks 가 받는다._

![AllGather](/assets/img/posts/nccl-collectives/allgather.png){: width="480"}
_Figure 6. AllGather. 각자의 chunk 를 all ranks 가 rank 순서로 concat 해서 받는다._

![ReduceScatter](/assets/img/posts/nccl-collectives/reducescatter.png){: width="480"}
_Figure 7. ReduceScatter. 합친 뒤 rank 별 chunk 로 분배._

![AlltoAll](/assets/img/posts/nccl-collectives/alltoall.png){: width="480"}
_Figure 8. AlltoAll. 각 rank 가 각 rank 에게 다른 chunk. MoE expert dispatch 의 핵심 연산._

### 4.2 Point-to-Point

| 이름 | 의미 |
|---|---|
| `ncclSend` | 특정 peer 에게 데이터 전송 |
| `ncclRecv` | 특정 peer 로부터 데이터 수신 |

NCCL 2.7 부터 공식. 여러 Send/Recv 를 `ncclGroupStart/End` 로 묶으면 scatter / gather / all-to-all 같은 패턴도 P2P 만으로 짤 수 있다.

### 4.3 One-sided RMA + Signal

**RMA** (Remote Memory Access) 는 sender 의 `Send` + receiver 의 `Recv` 매칭이 필요한 two-sided 와 달리 *한 쪽만 호출*하면 되는 모델. Receiver 는 자기 메모리의 일부 영역을 *window* 로 미리 등록해두기만 하면, sender 가 원하는 시점에 그 window 에 직접 read/write 한다. Receiver 쪽에서 `Recv` 를 부를 일 자체가 없다. 양측 사이 결합 (rendezvous) 이 풀리는 것이 핵심이고, 그 의미에서 two-sided 는 sync, one-sided RMA 는 async 한 통신 모델이다 (§7 Layer 2 에서 다시). MPI-2 의 `MPI_Put` / `MPI_Get` / `MPI_Win` 이 원형이고, NCCL 은 host-side one-sided RMA API (`ncclPutSignal` / `ncclSignal` / `ncclWaitSignal`) 를 NCCL 2.29.2 부터 제공한다 (CUDA 12.5+ 필요).

여기서 **window** 는 "RMA 대상으로 노출할 메모리 영역" 을 communicator 에 등록한 handle. NCCL 에서는 `ncclCommWindowRegister(comm, buff, size, *win, flags)` 로 GPU vidmem 의 한 영역을 window 로 만든다. 등록되면 같은 communicator 의 다른 rank 들이 이 window 를 통해서만 그 영역에 RMA 를 걸 수 있다 (전체 메모리 노출이 아니라 명시 등록만, 보안/안정성 측면).

**Signal** 은 RMA 와 짝을 이루는 가벼운 동기화 noti. "데이터 다 썼으니 읽어도 돼" 를 데이터 이동과 별개로 던지는 doorbell. Producer 는 `ncclPutSignal` 로 write + notify, consumer 는 `ncclWaitSignal` 로 ready 시점만 기다리면 된다. 미리 Recv 안 띄워도 되는 producer/consumer 패턴 (GPU-resident KV cache, prefill/decode 분리 등) 에 적합.

#### OS 의 익숙한 개념으로 정리

Two-sided Send/Recv 는 OS 의 message passing IPC (pipe 나 message queue) 에 가깝다. 양쪽이 명시적으로 write/read 호출을 맞춰야 하고, 한 쪽이 늦으면 다른 쪽이 block 된다. 반면 RMA window 는 shared memory IPC 또는 `mmap` 에 대응한다. 한 쪽이 특정 메모리 영역을 공유 가능한 영역으로 등록해두면 다른 쪽은 그 영역을 자기 주소공간처럼 접근한다.

`ncclPutSignal` 의 데이터 전송은 DMA (Direct Memory Access) 와 같은 그림이다. CPU / OS kernel 을 거치지 않고 NIC / GPU 가 직접 상대방 메모리 (window) 에 write 한다. 이것이 zero-copy. Signal / WaitSignal 은 추상 수준에서 condition variable 또는 hardware doorbell 에 가까운 동기화이다. 핵심은 데이터 이동과 ready 알림이 분리되고, consumer 가 매번 matching `ncclRecv` 를 미리 걸 필요가 없다는 점. 내부에서 어떤 progress 모델로 구현되는지 (polling / interrupt / sequence counter) 는 transport 와 하드웨어에 따라 달라질 수 있다 (§5.3 의 device-side spin-wait 도 NCCL 동기화의 한 형태).

#### NCCL API

`nccl.h.in` 의 head 에서 본 API:

| 이름 | 의미 |
|---|---|
| `ncclPutSignal(localbuff, count, dtype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)` | 데이터 + signal 을 peer 의 window 에 한 번에 push |
| `ncclSignal(peer, sigIdx, ctx, flags, comm, stream)` | 데이터 없이 signal 만 |
| `ncclWaitSignal(nDesc, signalDescs, comm, stream)` | descriptor array 에 따라 다수의 signal 대기 |
| `ncclCommWindowRegister(comm, buff, size, *win, winFlags)` | RMA 용 메모리 window 등록 |
| `ncclCommWindowDeregister(comm, win)` | 해제 |
| `ncclWinGetUserPtr(comm, win, **outUserPtr)` | symmetric 메모리 포인터 가져오기 |

> `ncclWaitSignal` 의 `signalDescs` 는 `{opCnt, peer, sigIdx, ctx}` 구조체 배열. `sigIdx` / `ctx` 는 현재 0 으로 고정 (must be 0 for now). `ncclPutSignal` / `ncclSignal` 의 `flags` 도 reserved 라 0. multi-context / multi-signal 확장을 위한 reserved field 들.

`ncclPutSignal` 이 받는 `ncclWindow_t peerWin` 은 GPU vidmem 기반 window 의 opaque handle. 분산 reader/writer pattern 이나 GPU-resident KV cache 처럼 "한 쪽이 다른 쪽 메모리에 쓰기만 하면 되는" 경우에 어울린다.

#### 구체 사례: disaggregated prefill/decode

LLM 추론을 prefill (사용자 prompt 를 읽어 KV cache 생성) 과 decode (KV cache 로 토큰을 한 개씩 생성) 노드로 분리하는 disaggregated serving 아키텍처가 최근 표준이 되어가고 있다 (vLLM, SGLang, Mooncake, DistServe, Splitwise). Prefill 노드가 만든 무거운 KV cache 를 decode 노드 GPU 메모리로 넘겨야 하는데, two-sided `ncclSend` / `ncclRecv` 로 짜면 매번 decode 쪽에서 `ncclRecv` 를 미리 띄워두고 timing 을 맞춰야 해서 KV cache 핸드오프마다 결합도 비용이 든다.

RMA + Signal 로 짜면 decode 가 자기 KV cache 영역을 `ncclCommWindowRegister` 로 window 등록해두고, prefill 이 끝나는 즉시 그 window 에 직접 `ncclPutSignal` 로 DMA write + signal 만 던진다. Decode 는 `ncclWaitSignal` 깨어난 뒤 자기 메모리에 이미 올라온 데이터를 바로 읽어 디코딩 시작. `ncclRecv` 호출도 rendezvous 도 없다. NVIDIA NIXL (GTC 2025 오픈소스, vLLM·SGLang·Dynamo 가 채택) 과 Mooncake Transfer Engine 이 정확히 이 모델로 RDMA 전송을 구현한다. Disaggregation 처럼 producer 와 consumer 의 timing 결합을 끊어야 하는 워크로드가 RMA 의 1 차 사용처다.

> 참고: NCCL 내부 ID 까지 포함하면 함수는 더 많다. `ncclFunc_t` 가 정의하는 enum (`AllGatherV` 등) 까지 합치면 디스패치 함수는 15 개. 사용자 입장에선 위의 8 + 2 + 6 만 알면 된다.

### 4.4 `ncclScatter` / `ncclGather` / `ncclAlltoAll`

위 표에 `ncclScatter` / `ncclGather` / `ncclAlltoAll` 이 다 있지만, 내부 구현은 ring/tree 알고리즘이 아니라 Send/Recv 쌍의 묶음이다. `enqueue.cc` 의 dispatch 코드를 보면 명확하다.

```c
// src/enqueue.cc 발췌
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

`ncclAlltoAll` 한 호출이 곧 rank 수만큼의 Send + 같은 수의 Recv 로 펼쳐진다 (코드의 `comm->nRanks` 루프). 사용자는 group 으로 묶을 필요 없이 collective 한 줄을 호출하고, 내부에선 한 batch 의 P2P 가 한 kernel launch 로 처리된다.

### 4.5 Intra-node vs Inter-node 데이터 경로

Communicator API 는 같지만 wire 위에서 일어나는 일은 같은 노드 안인지 노드 사이인지에 따라 완전히 다르다.

**Intra-node** (같은 노드 안의 GPU 사이):

| 우선순위 | 경로 | 조건 |
|---|---|---|
| 1 | P2P over NVLink | NVLink 직결 |
| 2 | P2P over PCIe | NVLink 없을 때 |
| 3 | SHM (host memory 경유) | P2P 불가능하거나 inter-socket P2P 가 비효율적일 때 |
| 4 | NIC loopback | multi-socket + GPU 마다 local NIC, GDRDMA 가능 |

같은 process 안의 rank 끼리는 P2P_DIRECT 가 활성화되어 staging FIFO 를 우회한다 (§5.2 의 `direct*` primitive).

표 3 행의 *SHM transport* (Shared Memory) 는 같은 노드 안에서도 두 GPU 가 직접 P2P 못 할 때 (서로 다른 PCIe root complex / NUMA socket 등) host RAM 을 공유 버퍼로 두고 만나는 fallback. GPU A 가 host pinned memory 에 write → GPU B 가 같은 영역에서 read 하는 방식이라, 네트워크는 안 타니까 inter-node 보다 빠르지만 GPU-direct P2P 보다는 느리다.

**Inter-node** (노드 사이):

```
GPU kernel ─→ GPU vidmem ─→ (CPU proxy thread) ─→ NIC ─→ wire ─→ NIC ─→ ...
                                  │
                                  └─→ RDMA write (IB/RoCE) 또는 socket send
```

- GPU kernel 이 buffer 를 채우면, **CPU proxy thread** (`ncclProxyService`, `src/proxy.cc`) 가 NIC 의 RDMA write 또는 socket send 를 post 한다. CPU 가 데이터 자체를 만지지는 않지만 NIC 작업 orchestration 은 host thread 의 일.
- **GDRDMA 가능** (NIC 와 GPU 가 같은 PCIe switch 또는 그 안의 multiple bridges, default `PATH_PXB`) 하면 intermediate buffer 가 GPU vidmem 에 올라가고 NIC 가 GPU memory 를 직접 read/write. 게이트는 `ncclTopoCheckGdr` (`src/graph/paths.cc`) 가 결정하고, `NCCL_NET_GDR_LEVEL` 환경변수로 override 가능.
- **불가능**하면 host pinned memory 에 staging: GPU → host copy → NIC RDMA → 반대편 host → GPU copy. PCIe 를 두 번 더 건너는 셈.
- 양쪽이 buffer readiness 를 합의하는 **rendezvous** 가 데이터 전송 앞에 깔린다.

 DMA 장치 (GPU, NIC) 는 OS 를 우회해 직접 물리 주소에 access 하니까 page 가 swap 되면 엉뚱한 메모리에 쓰는 사고가 남. 그래서 staging buffer 는 반드시 pinned 여야 한다. 익숙한 비유로, PyTorch DataLoader 의 `pin_memory=True` 가 만드는 게 정확히 같은 메모리 type. 거기서는 dataset → GPU H2D copy 를 빠르게 (그리고 `non_blocking=True` 가 진짜 async 로 동작하게) 하는 용도로 쓰고, NCCL 도 같은 이유로 GDRDMA 가 안 되는 환경에서 host pinned memory 를 staging buffer 로 둔다. CUDA API 로는 둘 다 `cudaHostAlloc` / `cudaHostRegister`.

proxy thread 가 끼면 §7 Layer 2 의 "host async" 가 더 정밀해진다. `ncclSend` 호출이 즉시 리턴하는 건 stream enqueue 직후이고, 그 다음 GPU kernel 이 buffer 를 채우고, 그 다음에야 proxy thread 가 NIC 작업을 post 한다. wire 위의 traffic 시점은 host 호출 완료보다 한참 뒤다. 한 collective 의 진짜 끝은 proxy thread 의 마지막 RDMA completion 이지 host call 의 return 이 아니다.

## 5. Collective Decomposition 과 NCCL 커널 구조

같은 의미를 다른 primitive 조합으로 달성하는 관계가 몇 개 있다.

$$
\text{AllReduce} \equiv \text{ReduceScatter} + \text{AllGather}
$$

$$
\text{AllReduce} \equiv \text{Reduce} + \text{Broadcast}
$$

$$
\text{AllGather} \equiv \text{Gather} + \text{Broadcast}
$$

ZeRO-3 / FSDP 의 통신 설계가 첫 번째 decomposition (AR = RS + AG) 을 그대로 쓴다. AllReduce 를 RS + AG 로 쪼개고 gradient 를 partition 에만 두는 구조. NCCL Ring AllReduce 커널의 본체도 정확히 RS phase 와 AG phase 두 loop 이다.

### 5.0 Communication Channels: 한 collective = N 개 ring 의 병렬 실행

지금까지 "ring" 을 단일 경로처럼 그렸지만, 실제 NCCL 은 collective 한 호출을 여러 channel 로 쪼갠다. 이유는 단순. 단일 SM 이 모든 데이터를 처리하면 큰 메시지에서 SM 하나가 병목이 되고, NVLink 의 여러 링크나 노드의 여러 NIC 가 다 활용되지 못한다. channel 은 그래서 *parallel pipeline* 으로 들어온 추상.

- kernel grid: `dim3 grid = {(unsigned)nChannels, 1, 1};` (`src/enqueue.cc`). channel 1 개 = CUDA block 1 개
- 입력 버퍼: channel 별 disjoint contiguous region 으로 partition
- 각 channel 은 자기 ring (또는 tree) 인스턴스를 *독립적으로* 돌림
- channel 별 chunk 가 너무 작아지면 NIC FIFO 가 덜 차서 network throughput 저하. 작은 메시지에서는 NCCL 이 휴리스틱으로 `nChannels` 를 줄임 (`enqueue.cc::scheduleP2pTasksToPlan`)

§5.2 의 `runRing` 도 *한 channel 의* ring 실행이고, 같은 kernel launch 안에서 nChannels 개의 block 이 같은 코드를 다른 데이터 segment 에 대해 동시 실행한다. 이 구조는 §7 Layer 2 의 single-kernel 모델과 모순이 아니라 보강이다. kernel launch 는 1 회, 그 안의 grid 가 nChannels 만큼.

### 5.1 Naive vs Ring

본격 코드 보기 전에 그림 한 장. GPU 4 장 으로 broadcast 를 짠다고 하자. 가장 단순한 방식 (Naive) 은 root 한 장이 다른 모두에게 직접 데이터를 쏘는 master/slave 구조. Ring 은 모든 GPU 를 이웃 chain 으로 보고 (`G0 → G1 → G2 → G3 → G0`), 데이터를 작은 조각으로 쪼개 이웃끼리만 chunk 를 넘기는 구조.

![Master/slave (Naive) topology](/assets/img/posts/nccl-collectives/gibiansky-master-slave.png){: width="500"}
_Figure 9. Naive 의 master/slave 구조. root GPU 가 다른 모든 GPU 에게 직접 데이터를 쏜다. root 의 송신 링크 하나가 매번 bottleneck 이고 다른 GPU 끼리의 링크는 놀고 있다._

![Ring topology](/assets/img/posts/nccl-collectives/gibiansky-ring.png){: width="500"}
_Figure 10. Ring 구조. 각 GPU 는 자기 직전/직후 이웃하고만 통신. 데이터를 chunk 로 쪼개면 모든 링크가 동시에 다른 chunk 를 운반할 수 있어 bandwidth 가 노드 수에 거의 무관해진다._

같은 4 GPU, 같은 양의 데이터를 broadcast 한다고 했을 때 round 별 링크 활성도를 보면 차이가 명확하다. 비교용 기호:

- $p$ = GPU (rank) 개수
- $n$ = 보낼 데이터 전체 크기 (bytes)
- $B$ = 링크 한 줄의 bandwidth (bytes/sec)
- $c$ = ring 에서 데이터를 쪼갠 chunk 하나 크기

**Naive (round 마다 전체 $n$, root 의 한 링크만 활성)**

| round | `G0→G1` | `G0→G2` | `G0→G3` | total |
|---|---|---|---|---|
| 1 | $n$ |  |  | $n$ |
| 2 |  | $n$ |  | $n$ |
| 3 |  |  | $n$ | $n$ |

총 시간 ≈ $(p-1) \cdot n / B$. 한 링크가 일하는 동안 나머지는 놀고, 데이터가 매 round 통째로 다시 흐른다.

**Ring (chunk $a, b, c$ 로 3 분할, 모든 링크 동시 활성)**

| round | `G0→G1` | `G1→G2` | `G2→G3` |
|---|---|---|---|
| 1 | $a$ |  |  |
| 2 | $b$ | $a$ |  |
| 3 | $c$ | $b$ | $a$ |
| 4 |  | $c$ | $b$ |
| 5 |  |  | $c$ |

chunk 개수를 $m = \lceil n/c \rceil$ 라고 두면, ring/chain pipeline broadcast 의 총 시간은 일반식으로:

$$
T \approx (m + p - 2)\frac{c}{B} = \frac{n}{B} + O\!\left(\frac{pc}{B}\right)
$$

첫 항 $n/B$ 는 데이터 전체가 한 link 를 통과하는 시간, 둘째 항 $O(pc/B)$ 는 pipeline fill / drain 비용. chunk 가 충분히 작아 $m$ 이 커지면 둘째 항이 무시되고 $T \to n/B$ 로 수렴, GPU 개수 $p$ 에 거의 무관해진다 (위 표는 $m=3$ 인 경우).

Ring AllReduce 도 같은 원리의 연장이다. 단지 broadcast 가 아니라 RS phase ($p-1$ step 동안 reduce 누적) + AG phase ($p-1$ step 동안 모두에게 전파) 의 두 단계가 같은 ring 위에서 연속 진행될 뿐.

### 5.2 NCCL Ring AllReduce 커널

Ring AllReduce 는 데이터를 $p$ 개 chunk 로 자르고 GPU $i$ 가 chunk $i$ 를 시작점으로 잡는다. iteration $k$ 에서:

- ReduceScatter phase: chunk $(i + k) \bmod p$ 를 다음 GPU 로 보내고, chunk $(i + k - 1) \bmod p$ 를 이전 GPU 에서 받아 local 값과 누적
- AllGather phase: chunk $(i + 1 + k) \bmod p$ 를 보내고, $(i + k) \bmod p$ 를 받아 덮어쓰기

![Scatter-Reduce iteration](/assets/img/posts/nccl-collectives/gibiansky-scatter-reduce-step.png){: width="560"}
_Figure 11. Scatter-Reduce 한 iteration. 모든 GPU 가 동시에 다음 GPU 로 한 chunk 를 보내고 이전 GPU 에서 한 chunk 를 받아 local 값과 합산한다._

![AllGather iteration](/assets/img/posts/nccl-collectives/gibiansky-allgather-step.png){: width="560"}
_Figure 12. AllGather 한 iteration. RS 가 끝난 뒤엔 reduce 가 아니라 덮어쓰기로 같은 ring 을 한 바퀴 더 돈다._

각 phase 가 $p-1$ iteration 끝나면, RS 후엔 GPU $i$ 가 chunk $(i+1) \bmod p$ 의 reduced 결과를 보유, AG 후엔 모두가 모든 chunk 를 보유. 코드의 `offset` 변수가 이 인덱싱을 들고 다닌다.

```c
// src/device/all_reduce.h::runRing
for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  // ReduceScatter phase
  prims.directSend(offset, offset, nelem);                                    // step 0
  for (int j = 2; j < nranks; ++j)
    prims.directRecvReduceDirectSend(offset, offset, nelem);                  // recv + reduce + send
  prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
  // 위까지가 RS 의 nranks-1 step. 각 rank 가 reduced chunk 하나를 보유.

  // AllGather phase
  for (int j = 1; j < nranks - 1; ++j)
    prims.directRecvCopyDirectSend(offset, offset, nelem);                    // recv + copy + send
  prims.directRecv(offset, nelem);
  // 위까지가 AG 의 nranks-1 step. 모두가 모든 chunk 보유.
}
```

앞 loop 이 ReduceScatter phase ($p-1$ step), 뒤 loop 이 AllGather phase ($p-1$ step). 두 phase 가 한 kernel 안에서 같은 ring 위로 연속 실행되고, `directRecvReduce*` 는 받는 즉시 같은 kernel 에서 reduce 를 끝낸다 (이 fused 구조는 §7 Layer 2 에서 다시). Ring AllReduce 의 $2(p-1)$ step cost 가 그대로 위 두 loop 의 step 수다.

> 알고리즘 round 로 세면 RS $(p-1)$ + AG $(p-1)$ = $2(p-1)$ round. NCCL device primitive 호출 수 기준으로는 RS 가 `directSend` + `directRecvReduceDirectSend × (p-2)` + `directRecvReduceCopyDirectSend` 의 $p$ 개, AG 가 `directRecvCopyDirectSend × (p-2)` + `directRecv` 의 $p-1$ 개, 합 $2p-1$ 개 primitive. 같은 알고리즘을 다른 단위로 보는 것. round 로 세면 $2(p-1)$, primitive call 로 세면 $2p-1$. AG 의 첫 chunk forward 가 RS 의 마지막 `directRecvReduceCopyDirectSend` 안에 fuse 되어 있어 별도 `directSend` 로 시작하지 않는 것이 NCCL 구현 디테일.

코드의 `direct*` prefix 가 붙은 primitive (`directSend`, `directRecv`, `directRecvReduceDirectSend` 등) 는 P2P_DIRECT 모드의 단축 경로를 가리킨다. 같은 host *그리고* 같은 process 안의 rank 끼리 통신할 때 (typical DDP/FSDP single-process-multi-GPU) 활성화되어, 일반 P2P 가 거치는 intermediate FIFO buffer 를 건너뛰고 source GPU vidmem 에서 destination GPU vidmem 으로 *직접* 쓴다. 게이트는 `P2P_SAME_PID = (hostHash == peerHostHash) && (pidHash == peerPidHash)` (`src/transport/p2p.cc`); cross-process intra-node 는 P2P_IPC / P2P_CUMEM 으로 fallback. non-direct 변종 (`recvReduceSend`, `recvCopySend` 등) 은 FIFO 경유.

이 차이가 latency 에 직접 영향을 준다. P2P_DIRECT 가 활성화되면 send/recv 마다 메모리 복사 한 번이 절약되고, 그래서 §5.3 의 fused primitive 가 대부분 `direct*` 변종으로 등장하는 것.

Step 수가 아니라 각 GPU 의 wire traffic 으로 봐도 같은 알고리즘이 *bandwidth-optimal* 이라는 게 보인다. Ring AllReduce 의 per-rank **send** traffic 은 RS 에서 $(p-1)K/p$, AG 에서 $(p-1)K/p$, 총 $2(p-1)K/p$. **receive** traffic 도 같은 양이다 (full-duplex 링크 가정). 시간 모델에서 bandwidth term 은 보통 send 방향 기준으로 쓴다.

$$
\text{per-GPU send traffic} = \frac{2(p-1)K}{p} \xrightarrow{p \to \infty} 2K
$$

($K$ = 전체 데이터 byte). AllReduce 의 byte cost 하한이 $2K$ (자기 데이터 한 번 내보내고 결과 한 번 받기) 이므로 ring 이 정보 이론적 최소치를 그대로 달성한다.

> **참고: 프로토콜 차원**. 아래 코드 (그리고 §6.2 의 `waitPeer`) 는 NCCL 의 **Simple** 프로토콜 동작이다. NCCL 은 프로토콜이라는 별도 차원을 가진다. Simple, LL, LL128 의 셋이고, 같은 ring 알고리즘이라도 프로토콜에 따라 sync 메커니즘과 transfer granularity 가 달라진다.
>
> - **Simple**: 큰 chunk + memory fence 로 sync. 아래 `waitPeer` + step counter 패턴이 이쪽. large message 에 강하지만 fence overhead 때문에 small message 는 latency 큼.
> - **LL** (Low Latency): 4B data + 4B flag 의 8B atomic 으로 flag-based sync. memory fence 없음. small message 에 강하지만 대역폭은 peak 의 25-50% 수준.
> - **LL128**: 120B data + 8B flag 의 128B atomic. NVLink intra-node 에서 peak 의 ~95%.

### 5.3 `directRecvReduceDirectSend` 가 하는 일

위 loop 의 `directRecvReduceDirectSend` 가 어떻게 "recv + reduce + send" 를 한 호출로 끝내는지는 `src/device/prims_simple.h` 의 primitive 정의를 보면 명확하다.

```c
// src/device/prims_simple.h 발췌
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

여러 변종 (`directSend`, `directRecv`, `directRecvCopyDirectSend`, `recvReduceSend`, ...) 이 모두 같은 `genericOp<DirectRecv, DirectSend, Recv, Send, SrcBuf, DstBuf>` 템플릿의 인스턴스다. template parameter 만 다른 25 가지 조합이 사실상 NCCL 커널의 동작 어휘 전부.

`genericOp` 의 한 호출이 내부에서 수행하는 것:
1. `waitPeer()`. peer 의 step 카운터가 올라올 때까지 spin
2. `subBarrier()`. block 안 worker thread 동기화
3. `reduceCopy<...>(srcs, dsts, workSize)`. 받은 데이터를 local Input 과 element-wise reduce 해서 다음 fifo 에 store
4. `barrier()`. reduce-copy 끝난 뒤 block 전체 barrier
5. `postPeer()`. 자기 step 카운터 증가시켜 다음 peer 에 알림

(network-device transport 의 경우 step 2 와 3 사이에 `ncclNetDeviceUnpack` + 추가 `subBarrier` 가 더 끼어든다.)

`directRecvReduceDirectSend` 한 호출이 곧 spin-wait → fused reduce-copy → notify 한 사이클이고, 각 stage 사이에 thread-block barrier 가 들어간다. 이 사이클이 ring step 마다 한 번씩 돌고, host CPU 는 전혀 끼지 않는다. §7 Layer 2 에서 이 single-kernel 설계의 의미를 다시 짚는다.

## 6. P2P vs Collective

### 6.1 역할 차이

> Collective 는 communicator 의 모든 rank 가 같은 순서, 같은 `count`, 같은 `datatype` 으로 호출해야 한다. 불일치 시 undefined behavior 또는 hang. (NCCL docs, *Collective Operations*)

| 구분 | Collective | P2P (two-sided) |
|---|---|---|
| 참여자 | communicator 의 모든 rank | sender + receiver 둘만 |
| 호출 | 전원이 같은 op / count / datatype | 한 쪽 Send, 다른 쪽 Recv |
| sync 강도 | 강 (group-wide barrier 느낌) | 약 (peer 쌍만 맞으면 됨) |
| expressiveness | 고정 패턴만 | 임의 peer subset, irregular routing |

### 6.2 P2P sync / async

NCCL `Send/Recv` 는 GPU 에서 blocking, host 에서는 async 다. 먼저 receiver 쪽 device 함수부터:

```c
// src/device/sendrecv.h::runRecv
__device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
  Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
    prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr, ...);
  size_t cursor = 0;
  do {
    int n = min(size_t(chunkSize), bytes - cursor);
    prims.directRecv(cursor, n);   // 여기서 GPU 가 peer 의 step 이 올 때까지 spin
    cursor += n;
  } while (cursor < bytes);
}
```

`prims.directRecv` 안에서 결국 호출되는 게 `waitPeer`. 이게 진짜 spin-wait 의 본체다.

```c
// src/device/prims_simple.h::waitPeer
void waitPeer(...) {
  int spins = 0;
  while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
    connStepCache = loadStepValue(connStepPtr);  // peer 의 step 카운터 volatile load
    if (checkAbort(flags, Aborted, spins)) break;
  }
}
```

GPU 쪽 thread 는 peer 의 step 카운터가 자기가 기다리는 값까지 올라갈 때까지 빙빙 돈다. `connStepPtr` 은 peer GPU 의 vidmem 에 매핑된 카운터고, volatile load 로 매번 새로 읽어 check. Receiver 가 받기 전엔 sender 의 다음 op 이 못 진행한다. 한편 host 쪽 `ncclSend` / `ncclRecv` 함수는 CUDA stream 에 launch 만 하고 즉시 리턴한다 (§7 Layer 2).

`NCCL_STEPS` (default 8, `src/include/device.h`) 는 channel 버퍼 하나를 몇 개 slot 으로 쪼갤지 결정한다. 같은 channel 의 ring 위에서, 어떤 chunk 가 slot 0 에서 reduce 되고 있는 동안 다음 chunk 가 slot 1 에 도착하고 그 다음 chunk 가 slot 2 로 enqueue 되는 multi-stage pipeline 이 가능해진다.

차원이 두 개라는 점이 핵심.

- ring step: peer 와의 한 round (RS 단계의 $i$ 번째 reduce, AG 단계의 $i$ 번째 forward)
- slot step: FIFO 의 한 칸. 같은 ring step 안에서도 slot 별로 stage 가 다를 수 있음

`waitPeer` 의 step counter 는 slot 단위. 위 코드의 `step + StepPerSlice` 가 "내가 다음에 쓰고 싶은 slot 위치" 이고, peer 의 `connStepCache` 가 "peer 가 이미 진행한 slot 위치" 라서, peer 가 충분히 앞서가야만 wait 이 풀린다. 이 slot pipelining 이 §5.0 의 channel 병렬성과 직교 차원. 같은 channel 안에서도 slot 으로 한 번 더 pipeline 이 걸린다.

여러 P2P 를 동시에 굴리려면 `ncclGroupStart/End` 로 묶어야 한다. NCCL 헤더가 명시:

> "This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations need to progress concurrently to complete, they must be fused within a ncclGroupStart/ncclGroupEnd section." (`nccl.h.in`)

`src/group.cc` 의 `ncclGroupDepth` 카운터는 thread-local. depth > 0 동안 collective 호출은 즉시 launch 하지 않고 큐에 쌓아뒀다가 `ncclGroupEnd` 에서 single kernel launch 로 한꺼번에 보낸다. send/recv 매칭이 한 kernel 안에 들어가야 GPU 끼리 서로 기다리다 멈추는 일이 없으니, group 호출이 deadlock 방지의 핵심이다.

## 7. Sync vs Async

"NCCL collective 는 sync 인가 async 인가" 같은 질문이 헷갈리는 건 두 관점을 혼동해서 그렇다.

### Layer 1. 학습 관점

대규모 LLM 학습은 보통 synchronous training (BSP) 이다. 필요한 collective / P2P 가 끝나기 전엔 같은 step 의 weight update 로 못 넘어간다. PyTorch DDP 문서가 constructor / forward / backward 를 "distributed synchronization point" 라고 부르는 것도 그래서. overlap / prefetch 같은 옵션이 켜져 있어도 그건 Layer 2 의 동시 실행이지 정의상 async 가 아니다.

### Layer 2. NCCL API / CUDA stream

Collective / P2P 모두 호출 자체는 CUDA stream enqueue 후 즉시 리턴. host 관점에서는 async. 코드로:

```c
// src/enqueue.cc::ncclLaunchKernel
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  cudaStream_t launchStream = planner->streams->stream;
  // ...
  CUCHECKGOTO(cuLaunchKernel(fn, grid.x, ..., launchStream, nullptr, extra),
              ret, do_return);
  // 여기서 즉시 리턴. kernel 은 GPU 에서 비동기 실행
}
```

`cuLaunchKernel` 한 번이 collective 1 호출에 해당한다. Python 단에서 `dist.all_reduce(...)` 가 ms 안에 끝난 듯 보이는 것도 그래서고, 실제 wire traffic 은 그 뒤 GPU 에서 일어난다.

NCCL 은 communication 과 computation 을 single kernel 로 구현한다. §5.3 에서 본 `genericOp` 의 핵심에 해당하는 `reduceCopyPacks` 의 inner loop 을 보면 fused 구조가 명확하다.

```c
// src/device/common_kernel.h::reduceCopyPacks 발췌
while (...) {
  BytePack<BytePerPack> acc[Unroll];
  // 1) 받은 데이터를 peer fifo 에서 로드 (volatile = 매번 새로 읽기)
  acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
  // 2) 로컬 input 또는 다른 source 와 element-wise reduce
  acc[u] = applyReduce(redFn, acc[u], tmp[u]);
  if (postOp) acc[u] = applyPostOp(redFn, acc[u]);
  // 3) 다음 peer 의 fifo (또는 출력 buffer) 에 store
  st_global<BytePerPack>(minDsts[d], acc[u]);
}
```

한 thread 가 `ld_volatile_global → applyReduce → st_global` 을 같은 register set 에서 연속 수행. CPU 도 안 거치고 다른 kernel 도 안 거친다. Stream 에 올라간 kernel 이 GPU 에서 실제로 돌 때 chunk 가 ring 을 따라 흐르고, 동시에 reduction 이 같은 kernel 안에서 일어난다 (§5 참고). 호스트 관점에선 async, 분산 관점에선 rendezvous. 두 관점이 동시에 맞다.

여기서 "분산 관점에선 rendezvous" 라고 한 의미를 좀 더 풀어보면: NCCL 의 `ncclSend` / `ncclRecv` 는 host 호출이 즉시 리턴해서 표면상 async 처럼 보이지만, two-sided 인 이상 receiver 가 `ncclRecv` 를 호출하지 않으면 데이터는 영원히 전달되지 않는다. MPI 의 non-blocking 버전 `MPI_Isend` / `MPI_Irecv` 도 마찬가지로 호출은 즉시 반환되지만 양측이 논리적으로 matching 호출을 맞춰야 한다는 결합 (rendezvous) 은 그대로 남는다. API 가 blocking 이냐 와 무관하게 two-sided 는 본질적으로 sync 한 통신 모델이다.

결합이 사라지는 건 §4.3 의 one-sided RMA, 즉 `ncclPutSignal` / `ncclWaitSignal` 뿐. Receiver 는 `ncclCommWindowRegister` 로 window 만 등록해두면 자기 쪽 호출은 더 이상 없고, sender 가 알아서 그 메모리에 쓴다. "two-sided 는 sync, one-sided 는 async" 라는 대비는 API 가 blocking 이냐의 문제가 아니라 이 아키텍처 결합도의 차이고, 그래서 §4.3 에서 본 prefill/decode disaggregation 같이 producer 와 consumer 의 timing 을 떼어내야 하는 패턴에 RMA 가 자연스럽게 맞는 것.

결국 어느 쪽이 맞느냐는 관점 문제다.

| 관점 | NCCL 동작 |
|---|---|
| Host API | CUDA stream enqueue 직후 리턴. host 관점 async |
| GPU kernel | peer step / FIFO 상태 spin-wait. device 관점 blocking |
| Two-sided P2P | sender / receiver matching 필요. rendezvous 결합 존재 |
| Training step | BSP 라 collective 완료 전 optimizer step 불가 |
| One-sided RMA | window 등록 후 initiator 중심 put/signal. matching Recv 제거 |

---

## Reference

- Hu, Z. et al. "Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms." arXiv:2507.04786, 2026.
- <https://docs.nvidia.com/deeplearning/nccl/>
- <https://github.com/NVIDIA/nccl>
- <https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/>
