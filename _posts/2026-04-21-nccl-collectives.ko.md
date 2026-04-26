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

예를 들어 root rank 가 다른 $p-1$ 개 rank 에 같은 데이터를 broadcast 한다고 하자. P2P (Send/Recv) 만 쓰면 root 가 1:1 Send 를 $p-1$ 번 순차로 부르고, 매번 root 의 송신 링크 하나로만 데이터가 나가니 시간이 노드 수에 비례해 늘어난다. 같은 broadcast 를 NCCL collective 로 부르면 내부에서 ring 으로 짜져서 모든 링크가 동시에 활성화되어 시간이 노드 수에 거의 무관해진다 (구체 비교는 §5.1).

Parallel computing 은 그래서 집단 단위 통신 패턴 (collective) 을 공식 API 로 제공한다. MPI 시대부터 굳어진 추상이고, NCCL 은 같은 개념을 GPU 와 NVLink / InfiniBand (IB) / RDMA (Remote Direct Memory Access) 위에 옮겨 놓은 것.

이 글은 NCCL 기준이지만 어휘 자체는 MPI 와 호환된다. AllReduce, AllGather 같은 이름이 똑같고, 알고리즘 선택도 비슷한 cost model 을 쓴다.

## 2. MPI 와 NCCL

| 기준 | MPI | NCCL |
|---|---|---|
| 주 타겟 | CPU cluster | GPU cluster |
| 통신 실행 | Host side 라이브러리 | GPU kernel (single-kernel comm + reduction) |
| 데이터 이동 | Host memory ↔ network | GPU memory ↔ GPU memory (GPUDirect P2P / RDMA) |
| Collective 의미 | MPI 표준 | MPI 호환 + extra (NVLS, PAT) |
| 알고리즘 선택 | 구현체 cost model (Open MPI tuned / MPICH) | NCCL auto + `NCCL_ALGO` |
| P2P | `MPI_Send/Recv` | `ncclSend/Recv` (NCCL 2.7+) |
| One-sided | `MPI_Put/Get/Win` | `ncclPutSignal` / `ncclSignal` / `ncclWaitSignal` + `ncclWindow_t` |

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

**RMA** (Remote Memory Access) 는 sender 의 `Send` + receiver 의 `Recv` 매칭이 필요한 two-sided 와 달리 *한 쪽만 호출*하면 되는 모델. Target 이 자기 메모리의 일부 영역을 *window* 로 미리 등록해두면 origin 이 그 window 에 직접 read/write. MPI-2 의 `MPI_Put` / `MPI_Get` / `MPI_Win` 이 원형이고, NCCL 에도 같은 모델이 들어와 있다.

여기서 **window** 는 "RMA 대상으로 노출할 메모리 영역" 을 communicator 에 등록한 handle. NCCL 에서는 `ncclCommWindowRegister(comm, buff, size, *win, flags)` 로 GPU vidmem 의 한 영역을 window 로 만든다. 등록되면 같은 communicator 의 다른 rank 들이 이 window 를 통해서만 그 영역에 RMA 를 걸 수 있다 (전체 메모리 노출이 아니라 명시 등록만, 보안/안정성 측면).

**Signal** 은 RMA 와 짝을 이루는 가벼운 동기화 noti. "데이터 다 썼으니 읽어도 돼" 를 데이터 이동과 별개로 던지는 doorbell. Producer 는 PutSignal 로 write + notify, consumer 는 WaitSignal 로 ready 시점만 기다리면 된다. 미리 Recv 안 띄워도 되는 producer/consumer 패턴 (GPU-resident KV cache, prefill/decode 분리 등) 에 적합.

`nccl.h.in` 의 head 에서 본 API:

| 이름 | 의미 |
|---|---|
| `ncclPutSignal(sendbuff, peerWin, ...)` | 데이터 + signal 을 peer 의 window 에 한 번에 push |
| `ncclSignal(peer, sigIdx, ctx, flags, ...)` | 데이터 없이 signal 만 |
| `ncclWaitSignal(peer, sigIdx, ctx, flags, ...)` | 특정 signal 도착 대기 |
| `ncclCommWindowRegister(comm, buff, size, *win, flags)` | RMA 용 메모리 window 등록 |
| `ncclCommWindowDeregister(comm, win)` | 해제 |
| `ncclWinGetUserPtr(comm, win, **outUserPtr)` | symmetric 메모리 포인터 가져오기 |

`ncclPutSignal` 이 받는 `ncclWindow_t peerWin` 은 GPU vidmem 기반 window 의 opaque handle. 분산 reader/writer pattern 이나 GPU-resident KV cache 처럼 "한 쪽이 다른 쪽 메모리에 쓰기만 하면 되는" 경우에 어울린다.

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

총 시간 ≈ $n/B + (p-1) \cdot c/B$. chunk 가 충분히 작으면 두 번째 항이 무시되어 $n/B$ 로 수렴, GPU 개수 $p$ 에 거의 무관해진다.

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

Step 수가 아니라 각 GPU 의 wire traffic 으로 봐도 같은 알고리즘이 *bandwidth-optimal* 이라는 게 보인다. RS phase 에서 GPU 한 장이 보내고 받는 byte 합은 $\frac{(p-1)K}{p}$, AG phase 도 같은 양이라 총합은:

$$
\text{per-GPU 전송량} = \frac{2(p-1)K}{p} \xrightarrow{p \to \infty} 2K
$$

($K$ = 전체 데이터 byte). GPU 가 몇 장이든 결국 한 장 분량을 두 번 전송하면 끝난다. AllReduce 의 byte cost 하한이 $2K$ (자기 데이터 한 번 내보내고 결과 한 번 받기) 라서 이게 정보 이론적 최소치고, ring 이 그걸 그대로 달성한다.

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

여러 변종 (`directSend`, `directRecv`, `directRecvCopyDirectSend`, `recvReduceSend`, ...) 이 모두 같은 `genericOp<DirectRecv, DirectSend, Recv, Send, SrcBuf, DstBuf>` 템플릿의 인스턴스다. template parameter 만 다른 21 가지 조합이 사실상 NCCL 커널의 동작 어휘 전부.

`genericOp` 의 한 호출이 내부에서 수행하는 것:
1. `waitPeer()`. peer 의 step 카운터가 올라올 때까지 spin
2. `subBarrier()`. block 안 worker thread 동기화
3. `reduceCopy<...>(srcs, dsts, workSize)`. 받은 데이터를 local Input 과 element-wise reduce 해서 다음 fifo 에 store
4. `postPeer()`. 자기 step 카운터 증가시켜 다음 peer 에 알림

즉 `directRecvReduceDirectSend` 한 번이 곧 spin-wait → barrier → fused reduce-copy → notify 한 사이클. 이 사이클이 ring step 마다 한 번씩 돌고, host CPU 는 전혀 끼지 않는다. §7 Layer 2 에서 이 single-kernel 설계의 의미를 다시 짚는다.

## 6. P2P vs Collective

### 6.1 역할 차이

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

---

## Reference

- <https://docs.nvidia.com/deeplearning/nccl/>
- <https://github.com/NVIDIA/nccl>
- <https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/>
