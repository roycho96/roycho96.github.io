---
title: NCCL 알고리즘
date: 2026-04-28 11:00:00 +0900
categories: [Distributed, NCCL]
tags: [nccl, algorithms]
lang: ko
ref: nccl-algorithms
permalink: /posts/nccl-algorithms/
math: true
mermaid: true
---

이 글은 [NCCL 과 Communication Collectives](/posts/nccl-collectives/) 의 후속이다.

> 코드 인용과 함수 이름은 NCCL master (2026-04 시점, v2.30) 기준.

## 1. 같은 Collective, 다른 Schedule

같은 `AllReduce` 한 줄이라도 NCCL 이 매번 같은 방식으로 작동하지는 않는다. 메시지가 1 MB 인지 64 MB 인지, 단일 노드인지 16 노드인지, NVSwitch 가 있는 머신인지 아닌지에 따라 ring 이 골라지기도 하고 tree 가 골라지기도 하고 NVSwitch multicast 한 번에 끝나기도 한다.

`AllReduce` 의 **의미** (semantics) 는 한 줄로 정해진다. "모든 rank 의 값을 합쳐 모든 rank 가 결과를 받는다." 그걸 **어떻게** 수행하느냐는 별개의 층위다. 어느 schedule 이 빠른지는 메시지 크기, rank 수, topology, hardware 가 다 결정한다.

그래서 NCCL 은 collective 호출이 들어올 때마다 후보 알고리즘 7 개와 protocol 3 개로 7 × 3 = 21 칸짜리 표를 만든다. 셀마다 예상 실행 시간을 cost model 로 계산하고 가장 짧은 셀을 고른다 (argmin, 최솟값을 주는 위치).

표의 모든 칸을 다 쓰지는 않는다. eligibility 조건이 셀 대부분을 미리 빼버리고 시작한다. AllReduce 한정으로 보면 21 셀 중 실제 평가되는 건 10 셀 (PAT 가 AllReduce 자체에서 제외 3 셀, NVLS / NVLSTree / CollNet 이 Simple protocol 만 받음 8 셀). 그래도 골격은 같다. 표 + argmin.

이 글의 목표는 그 표가 어떻게 만들어지고, 호스트가 사용자의 tensor 크기를 보고 어떤 (algorithm, protocol) 을 고르는지를 NCCL master 코드로 따라가는 것.

## 2. αβγ Cost Model

Cost model 은 통신 한 번이 걸리는 시간을 모델링한 것. 알고리즘 A 와 B 중 어느 게 빠를지 비교하거나 NCCL 같은 라이브러리가 자동 선택을 할 때 쓰는 도구다. NCCL 의 cost model 도 결국 한 줄짜리 식이고 (`time = lat + bytes / bw`, §6.4 에서 본다), 이 글 §4 의 algorithm 별 분석도 다 그 cost model 기반이다.

가장 표준적인 형태는 한 메시지의 시간을 세 항으로 쪼개는 것이다.

- $\alpha$: 메시지 한 번 보내는 startup latency. RTT 비슷. 마이크로초 단위.
- $\beta$: byte 당 wire 전송 비용. 단위 1/bandwidth.
- $\gamma$: byte 당 reduction 연산 비용. 덧셈 같은 거. 보통 $\beta$ 보다 작아 무시되기도 한다.

$n$ byte 메시지 한 번 = $\alpha + n\beta$. Reduction 포함이면 $\alpha + n\beta + n\gamma$. 이 글의 모든 식이 이 변수로 적힌다 (αβ 는 Hockney 1994, γ 까지 합쳐 αβγ 모델로 굳어진 건 Thakur, Rabenseifner, Gropp 의 MPICH 분석 2005 가 표준 reference).

알고리즘 분석에서 보통 두 한계가 갈린다.

- **Latency-bound**. 메시지가 작아 $n\beta$ 가 무시될 만하면 비용이 step 수 × $\alpha$ 로 결정. 이 영역에서는 step 수 적은 알고리즘 ($\log p$) 이 이긴다.
- **Bandwidth-bound**. 메시지가 커서 $n\beta$ 가 지배하면 step 수보다 *총 송신 byte* 가 중요. 모든 link 를 동시에 활용하는 알고리즘이 이긴다.

알고리즘 선택의 본질은 이 둘 사이 어디에 있느냐다.

## 3. NCCL 의 알고리즘

§4 deep-dive 로 들어가기 전에, NCCL 이 코드에서 직접 들고 있는 algorithm 7 종 + topology pattern 6 종 + protocol 3 종을 먼저 본다.

### 3.1 Algorithm 7 종

```c
// src/include/device.h 발췌
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {
  "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT"
};
```

| Algorithm | 핵심 구조 | 적용 collective | 비고 |
|---|---|---|---|
| `Ring` | nearest-neighbor pipeline | 거의 모든 collective | per-rank traffic $\sim 2n$ |
| `Tree` | Double Binary Tree (§4.2) | AllReduce 만 | tree latency + ring BW |
| `CollNetDirect` / `CollNetChain` | NVIDIA SHARP (in-network reduce) | AllReduce, RS, AG | IB SHARP NIC 필요 |
| `NVLS` | NVSwitch multicast + reduce | AllReduce 등 | Hopper+, NVSwitch 필요 |
| `NVLSTree` | NVLS + multi-node tree | AllReduce | 2 노드 이상 |
| `PAT` | Bruck 변형 (§4.3) | AllGather, ReduceScatter | 2.23+, 1-GPU/노드 |

Eligibility 가 어떤 collective 에 어떤 algorithm 이 후보인지를 일찍 자른다 (`src/graph/tuning.cc::ncclTopoTuneModel`):

```c
// src/graph/tuning.cc 발췌
if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
    && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING
    && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
if (coll == ncclFuncAllReduce && a == NCCL_ALGO_PAT) continue;
```

요지: Broadcast / Reduce 는 Ring 만, AllGather / ReduceScatter 는 {Ring, PAT, NVLS, CollNet_Direct}, AllReduce 는 PAT 빼고 다.

### 3.2 Topology Pattern 6 종

알고리즘과 별개로 그래프 search 단계 (`src/graph/search.cc::ncclTopoCompute`) 에서 보는 topology pattern 이 따로 있다. 같은 Tree algorithm 이라도 그 안에서 NIC 트래픽을 어떻게 분산할지 결정한다.

```c
// src/include/graph.h 발췌
#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // tree parent + 자식 1 = GPU A, 자식 2 = GPU B
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // tree parent = GPU A, 자식들 = GPU B
#define NCCL_TOPO_PATTERN_TREE 3            // 모든 NIC 트래픽이 같은 GPU
#define NCCL_TOPO_PATTERN_RING 4
#define NCCL_TOPO_PATTERN_NVLS 5
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6
```

BALANCED / SPLIT 은 NIC traffic 을 두 GPU 에 나눠 PCIe / NVLink 병목을 푸는 변종. Tree 가 골라지면 그 안에서 graph search 가 이 셋 중 어느 pattern 이 가장 균형 있는지 따로 본다.

### 3.3 Protocol 3 종

같은 algorithm 도 wire format 이 셋. data:flag 비율이 다르다.

| Protocol | Cache line | data 효율 | 적합 |
|---|---|---|---|
| `LL` | 8B (4B data + 4B flag) | 50% | 짧은 메시지, latency |
| `LL128` | 128B (120B data + 8B flag) | 93.75% | NVLink intra-node 중간 메시지 |
| `Simple` | full data + 별도 fence | ~100% | 큰 메시지, throughput |

LL / LL128 의 핵심: data 옆에 flag 를 같이 보내서 receiver 가 단일 word load 로 ready 폴링 가능. PCIe doorbell 따로 안 받음. 그 대가가 효율 손실. LL128 은 NVLink cache line 단위 (128B) 를 그대로 활용해서 line 당 8B 만 flag 에 양보. 그래서 NVLink intra-node 에서 sweet spot. Enable 의 정확한 조건은 §5.

## 4. Algorithm Deep-dive

§3 의 algorithm 7 종 중 실제로 자주 골라지는 핵심 다섯 (Ring, Tree = Double Binary Tree, PAT, NVLS / NVLS_TREE, CollNet) 을 cost 분석 + NCCL 코드로 따라간다.

먼저 감각을 잡자. 같은 8-rank AllReduce 를 NCCL 의 핵심 셋으로 굴리면 라운드 수와 특성이 이렇게 다르다.

| Algorithm | 라운드 | 주 특성 |
|---|---|---|
| Ring | $2(p-1) = 14$ | 모든 link 동시 활성, bandwidth-optimal |
| Double Binary Tree | $\sim 2 \log p = 6$ | log latency + ring 급 bandwidth |
| NVLS | 1 | NVSwitch 가 multicast + reduce 를 hardware 로 처리 |

Ring 은 step 수가 많아도 한 step 의 메시지가 $K/p$ 로 작아 link bandwidth 를 끝까지 짠다. Tree 는 step 수가 적지만 두 phase (reduce-up + broadcast-down) 라 latency 가 $2 \log p$. NVLS 는 GPU 가 한 번 보내고 switch 가 나머지를 한다 (§4.4).

작은 메시지에서는 라운드 수 적은 쪽이 유리, 큰 메시지에서는 per-step 송신량 작은 쪽이 유리. 이 trade-off 위에 NCCL 이 §6 의 selection 머신을 얹는다.

### 4.1 Ring

각 rank 가 두 이웃과만 chunk 를 파이프라인. Step 수는 $O(p)$ 로 많지만 step 마다의 메시지가 작고 모든 link 가 동시에 활성화돼 wire bandwidth 를 끝까지 쓴다.

![Ring AllReduce step-by-step on 4 GPUs](/assets/img/posts/nccl-algorithms/demystify-ring-allreduce.png){: width="1080"}
_Figure 1. 4 GPU ring 위 AllReduce 의 한 iteration. 각 GPU 가 chunk 를 인접 rank 로 보내고, 받은 chunk 를 자기 partial 과 reduce 한 뒤 다음으로 넘긴다. RS phase 가 끝나면 각 rank 가 한 chunk 의 final reduced 값을 갖고, AG phase 에서 그 chunk 가 ring 을 한 바퀴 돌아 모두에게 전달된다._

Ring AllReduce 는 RS + AG 두 phase 로 구성된다 (①편 §3, §5.1). Phase 1 ReduceScatter ($p-1$ step), phase 2 AllGather ($p-1$ step). Per-step chunk size $K/p$. 총 step $2(p-1)$, per-rank 송신 byte $\approx 2K(p-1)/p$.

$$T_{\text{ring}}(K) = 2(p-1)\alpha + \frac{2(p-1)}{p} K\beta + \frac{p-1}{p} K\gamma$$

큰 $K$ 에서 $\beta$ 항이 $2K\beta$ 로 수렴. AllReduce 의 정보 이론적 하한 ($2K$, 자기 데이터 한 번 내보내고 결과 한 번 받기) 을 그대로 달성. 이게 ring 이 bandwidth-optimal 인 의미.

NCCL 은 ring 을 multi-channel 로 굴린다. ①편 §5.0 의 channel 모델 그대로. `ncclBuildRings` (`src/graph/rings.cc`) 가 channel 마다 독립 ring 을 만들고, kernel grid 가 channel 개수만큼 block 을 띄운다.

### 4.2 Tree (Double Binary Tree, NCCL 2.4+)

부모-자식 구조로 partial result 를 위로 모으거나 (reduce) 아래로 뿌리는 (broadcast) 방식. 깊이 $O(\log p)$ 라 step 수는 적다. 순진한 binary tree 의 cost 가:

$$T_{\text{naive tree}} \approx 2 \log p \cdot (\alpha + n\beta + n\gamma)$$

step 수는 짧지만 internal node 부하 비대칭이 약점이다. Power-of-two binary tree 에서 root 와 internal 들은 자식 수만큼 송수신, leaf 는 한 번. 그래서 internal 에 부하가 쏠려 link bandwidth 를 못 채운다. Tree 가 latency 는 좋아도 bandwidth 는 ring 한테 지는 이유가 여기 있다.

![Power-of-two binary tree](/assets/img/posts/nccl-algorithms/Btree.png){: width="640"}
_Figure 2. 8 rank power-of-two binary tree. Internal 인 0, 4, 2, 6 은 자식이 있어 송수신 부하가 크고, leaf 인 1, 3, 5, 7 은 한 번만. 이 비대칭이 §4.2 의 핵심 문제._

NCCL 2.4 가 이걸 complementary 한 binary tree 두 개로 푼다 (Sanders, Speck, Träff 2007).

핵심 trick.

- Tree A 에서 rank $r$ 이 internal 이면, tree B 에서는 leaf 가 되도록 두 tree 를 짠다.
- Payload 를 반씩 두 tree 에 태우면 모든 rank 의 송수신 byte 가 균일.
- 결과: tree latency $\log p$ + ring 급 bandwidth.

![Two complementary binary trees of NCCL Double Binary Tree](/assets/img/posts/nccl-algorithms/DBtree.png){: width="720"}
_Figure 3. NCCL 의 Double Binary Tree. 왼쪽 Tree A 와 오른쪽 Tree B 가 complementary 하다. Tree A 에서 internal 인 노드들이 (Figure 2 의 0, 4, 2, 6) Tree B 에서는 leaf 가 된다. Payload 를 반반 태우면 모든 rank 의 송수신 부하가 같아져서 link bandwidth 를 다 쓴다._

코드는 짝수면 mirror, 홀수면 shift (`src/graph/trees.cc::ncclGetDtree`).

```c
// src/graph/trees.cc 발췌 (요지)
ncclResult_t ncclGetDtree(int nranks, int rank,
    int* s0, int* d0_0, int* d0_1, int* parentChildType0,
    int* s1, int* d1_0, int* d1_1, int* parentChildType1) {
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);   // Tree A
  if (nranks % 2 == 1) {
    int shiftrank = (rank-1+nranks) % nranks;                     // shift by 1
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
    *s1   = u  == -1 ? -1 : (u +1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0+1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1+1) % nranks;
  } else {                                                         // mirror: r → nranks-1-r
    int u, d0, d1;
    ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1, parentChildType1);
    *s1   = u  == -1 ? -1 : nranks-1-u;
    *d1_0 = d0 == -1 ? -1 : nranks-1-d0;
    *d1_1 = d1 == -1 ? -1 : nranks-1-d1;
  }
  return ncclSuccess;
}
```

Cost model 에서도 "두 tree 가 bandwidth 를 반씩" 이 그대로 박혀있다.

```c
// src/graph/tuning.cc 발췌 — Tree AllReduce latency
if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) {
  comm->latencies[coll][a][p] +=
    2 * ((nRanks/nNodes - 1) * intraLat + log2i(nNodes) * interLat);
}
```

`2 ×` 의 의미. Ring AllReduce 가 RS + AG 두 phase 인 것처럼, tree AllReduce 도 reduce-up + broadcast-down 두 phase. 한 phase 의 latency 가 (intra-node leg) + (inter-node 트리 $\log_2 N$) 이라 그게 두 배.

### 4.3 PAT — Parallel Aggregated Trees (NCCL 2.23+)

Ring AllGather 의 latency 가 $(p-1)\alpha$ 로 큰 $p$ 에서 선형으로 늘어난다는 게 PAT 가 푸는 문제. Bruck (1997) 의 algorithm 변형이 base. Round 마다 partner 가 1, 2, 4, 8 식으로 두 배씩 늘어나는 dataflow 인데, 이 패턴이 FFT (Cooley-Tukey 1965) 의 butterfly diagram 과 닮아서 분산 컴퓨팅 문헌에서는 butterfly pattern 이라고도 부른다. NCCL 자체 코드 / docs 에는 이 용어가 등장하지 않고 PAT 라는 이름으로만 들고 있다.

![Bruck algorithm dataflow](/assets/img/posts/nccl-algorithms/bruck-algorithm.png){: width="720"}
_Figure 4. Bruck algorithm 의 데이터 흐름. Round 마다 partner 가 두 배씩 늘어나면서 (1, 2, 4, ...) 모든 rank 가 모든 데이터를 받는다. 라운드 수가 $\log p$._

Bruck 의 장점이 그대로 PAT 로 넘어온다. 라운드 수 $\log p$, power-of-two rank 요구 없음. 거기에 NCCL 만의 producer / worker kernel 구조가 얹혀 launch overhead 도 0 으로.

![PAT algorithm with 8 ranks](/assets/img/posts/nccl-algorithms/pat-8ranks.png){: width="720"}
_Figure 5. NCCL 의 PAT algorithm 을 8 rank 에서 본 모습. Bruck 의 binomial tree 패턴이 각 rank 에 대해 shift 된 형태로 동시에 도는 게 핵심._

#### 4.3.1 Enable 조건이 좁다

PAT 가 잘 안 보이는 이유. `ncclPatEnable` (`src/graph/tuning.cc:209`) 가 세 조건을 다 요구한다.

```c
// src/graph/tuning.cc 발췌
NCCL_PARAM(PatEnable, "PAT_ENABLE", 2);
static int ncclPatEnable(struct ncclComm* comm) {
  int patEnable = ncclParamPatEnable();
  if (comm->minCompCap < 60) return 0;             // SM60+ 필요 (CUDA atomics)
  if (patEnable != 2) return patEnable;
  if (comm->nNodes != comm->nRanks) return 0;      // 1 GPU per node 만
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;
  return 1;
}
```

특히 `nNodes == nRanks` 가 결정적. PAT 는 1 GPU per node 클러스터 전용. 노드 안에 GPU 가 여럿이면 (8-GPU H100 머신처럼) PAT 는 enable 안 된다. 이게 PAT 를 본 적 없다는 사람이 많은 이유.

언제 의미 있나. Scale-out 1-GPU/노드 클러스터, irregular topology (NVSwitch 없는 환경), AllGather / ReduceScatter 를 자주 부르는 워크로드. NCCL 2.23 release note 가 large-scale GPU 클러스터 (수천 노드) 에서의 의미를 강조한다.

#### 4.3.2 Cost

```c
// src/graph/tuning.cc 발췌
if (a == NCCL_ALGO_PAT
    && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter)) {
  comm->latencies[coll][a][p] +=
    log2i(nNodes) * (interLat / 3.5)
    + nRanks * 2.8;  // Still a linear part; hopefully we'll manage to remove it at some point.
}
```

식은 $\log p \cdot \frac{\alpha_{\text{inter}}}{3.5} + p \cdot 2.8$. 첫 항이 Bruck 의 log-step 부분, 둘째 항이 아직 남은 linear part. 코드 주석 ("hopefully we'll manage to remove it") 이 NCCL 도 이 linear term 을 미완성으로 본다는 걸 알려준다.

#### 4.3.3 Kernel — producer 1 + worker n

여기가 PAT 의 흥미로운 부분. 같은 CUDA block 안에서 thread 한 개가 알고리즘을 진행하고, 나머지가 데이터를 옮긴다.

```c
// src/device/all_gather.h 발췌 (NCCL_ALGO_PAT, 요지)
struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);

if (tid == nworkers) {
  // 알고리즘 thread 1 개. 다음 step 의 source/dst/size 를 shmem 에 push
  PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, ...);
  int step = 0;
  while (1) {
    struct ncclPatStep* ps = shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
    cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
    while (poll.load(cuda::memory_order_acquire) != 0) pollCount++;
    patAlgo.getNextOp(ps);
    if (ps->last == 2) break;
    step++;
  }
} else if (tid < nworkers) {
  // worker thread n 개. shmem 의 step descriptor 보고 실제 copy
  Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims(..., primsModePatAg);
  int step = group;
  while (1) {
    struct ncclPatStep* ps = shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
    cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
    while (poll.load(cuda::memory_order_acquire) == 0) pollCount++;
    prims.patCopy(ps, shmem);
    if (tidInGroup == 0) poll.store(0, cuda::memory_order_release);
    if (last) break;
    step += nGroups;
  }
}
```

읽어내야 할 점.

- 알고리즘 진행 (다음 step 어디로 가나) 과 데이터 이동을 같은 kernel 안에서 다른 thread 가 동시에 한다.
- Algorithm thread 가 한 step 앞서 plan 을 적재해두면, worker 가 이전 step 의 copy 를 진행하는 동안 다음 step 이 준비된다 (slot pipelining).
- ReduceScatter 용 PAT 은 `src/device/reduce_scatter.h` 에 같은 구조. `prims.patCopy` 만 `prims.patReduce` 로 바뀐다.

이 producer / worker 분리가 PAT 의 latency 를 더 줄이는 trick. log-step 알고리즘이라도 매 step 마다 host 가 launch 하면 launch overhead 가 누적되는데, NCCL 은 single-kernel 안에서 step 진행을 통제해 그 overhead 를 0 으로.

### 4.4 NVLS / NVLS_TREE

Hopper SXM (NVSwitch 4) 에서 가능한 in-switch reduction. NVSwitch 가 multicast + reduce 를 hardware 로 처리하므로 GPU 가 한 번 보내면 switch 가 나머지를 한다.

```c
// src/graph/tuning.cc 발췌
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f,   // Volta    — NVLS 미지원
  0.0f,   // Ampere   — NVLS 미지원
  0.85f,  // Hopper   — 85%
  0.74f,  // Blackwell
};
```

조건.

- Hopper / Blackwell GPU (compcap 9.0 / 10.0).
- NVSwitch 노드 (`system->nodes[NVS].count > 0`).
- 채널 ≥ 2 개.
- 단일 노드면 `NVLS`, 2 노드 이상이면 `NVLS_TREE` (multi-node tree 가 NVLS 위에 얹힘).

Cost 식이 단순한 것도 NVLS 의 특징.

```c
// src/graph/tuning.cc 발췌 — NVLS latency
if (a == NCCL_ALGO_NVLS) {
  comm->latencies[coll][a][p] = intraLat;
  if (nNodes > 1) comm->latencies[coll][a][p] += interLat;
}
```

`α × 1` (intra) + 옵션으로 `α × 1` (inter). $p$ 가 식에서 사라진다. 이게 NVSwitch multicast 의 본질. 8-GPU H100 / H200 NVSwitch 머신에서 큰 AllReduce 는 거의 자동으로 NVLS 가 골라진다.

### 4.5 CollNet Direct / Chain

InfiniBand SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 활용. NIC 와 IB switch 가 reduce 를 hardware 로 처리.

```c
// src/graph/tuning.cc 발췌
} else if (a == NCCL_ALGO_COLLNET_DIRECT) {
  comm->latencies[coll][a][p] +=
    2 * (std::min(1, (nRanks/nNodes-1)) * intraLat + (nRanks/nNodes-1) * 0.4) + interLat;
} else if (a == NCCL_ALGO_COLLNET_CHAIN) {
  comm->latencies[coll][a][p] += 2 * (nRanks/nNodes-1) * intraLat + interLat;
}
```

Inter-node 비용이 `interLat × 1` 한 번. 노드 사이 reduction 이 switch 안에서 끝나기 때문. 조건은 hardware 가 SHARP 를 지원하고 NCCL 이 인식한 IB SHARP NIC 가 있어야 한다는 것. HPC 클러스터의 InfiniBand 환경에서 의미 있고, RoCE 나 GPU 직결 NVLink 환경에서는 NVLS 가 더 자주 골라진다.

## 5. Protocol Simple / LL / LL128

§3.3 의 표를 코드까지 늘리면.

```c
// src/device/primitives.h 발췌
struct ProtoSimple {  // NCCL_PROTO_SIMPLE = 2
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
};
struct ProtoLL {      // NCCL_PROTO_LL = 0
  // 16B line = 8B data + 8B flag → 50%
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2;
  }
};
struct ProtoLL128 {   // NCCL_PROTO_LL128 = 1
  // 128B NVLink line 중 120B data → 93.75%
  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)
           * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
  }
};
```

LL128 은 atomicity 가 보장되는 transport 에서만 enable 된다. 조건이 꽤 까다롭다 (`src/graph/tuning.cc:486`).

```c
// src/graph/tuning.cc 발췌 — LL128 enable gating
if (pEnable == 2 && p == NCCL_PROTO_LL128) {
  pEnable = 1;
  if (ncclParamLl128C2c() && minCompCap >= 90) {
    pEnable &= (graphs[a]->typeInter <= PATH_PXN);  // Hopper+ + LL128_C2C=1 면 PXN 까지
  } else {
    pEnable &= (graphs[a]->typeInter <= PATH_PXB);  // 기본은 PXB 까지
  }
  pEnable &= (graphs[a]->typeIntra <= PATH_NVB);    // intra 는 NVLink 만
  pEnable &= (minCompCap == maxCompCap || minCompCap >= 90);  // compcap uniform 또는 ≥ Hopper
  pEnable &= !(minCompCap < 70 || ...);
}
```

읽으면.

- Intra-node 는 NVLink-Bridge 이하 (즉 NVLink 직결만). PCIe 거치면 LL128 안 씀.
- Inter-node 는 PXB 이하 (NIC 가 GPU 와 같은 PCIe switch). Hopper+ + `NCCL_LL128_C2C=1` 이면 PXN 까지 허용.
- Compute capability 가 uniform 이거나 모두 ≥ Hopper.

PATH 타입의 정확한 의미는 ①편 §4.5 참고. 가까운 순서: NVL > NVB > C2C > PIX > PXB > P2C > PXN > PHB > SYS > NET.

## 6. Selection 머신

호스트가 사용자의 collective 호출 (예: `ncclAllReduce(buf, ..., count, dtype, op, comm, stream)`) 을 받아 어떤 (algo, proto, nChannels, chunkSize) 로 launch 할지 결정하는 흐름. 두 단계로 나뉜다.

### 6.1 Init: α/β 테이블 빌드

`ncclTopoTuneModel` (`src/graph/tuning.cc:238`) 이 communicator init 시 모든 (collective, algorithm, protocol) 셀에 대해 두 표를 채운다.

- `comm->bandwidths[coll][algo][proto]` (GB/s)
- `comm->latencies[coll][algo][proto]` (µs)

bandwidth 는 topology 에서 BFS-측정한 link bandwidth + nvlsEfficiency / collnetEfficiency 같은 보정 인자로 시작해서, 알고리즘별 step 수와 $(p-1)/p$ 계수로 깎아 내려간다. Latency 는 baseLat + hwLat 의 합 + 알고리즘별 추가 항 (§4 의 수식들).

### 6.2 baseLat / hwLat verbatim

이 표가 모든 cost 계산의 입력이다.

```c
// src/graph/tuning.cc — baseLatencies (µs, [algo][proto] = [LL, LL128, Simple])
{
  {  6.8, 14.0,  8.4 }, {  6.6, 14.0,  8.4 },  // Tree, Ring
  {    0,    0,    0 }, {    0,    0,    0 },  // CollNetDirect, CollNetChain
  {    0,    0,    0 }, {    0,    0,    0 },  // NVLS, NVLSTree
  {  8.0,  8.0,  8.0 }                         // PAT
};

// hwLatencies[hw][algo][proto]  (µs, hw = NVLINK / PCI / NET)
{
/* NVLINK */
{ { 0.6,  1.25, 4.0 }, { 0.6,  1.9,  3.4 },   // Tree, Ring
  {   0,     0, 3.7 }, {   0,    0,  2.8 },   // CollNetDirect, Chain
  {   0,     0,  25 }, {   0,    0,   25 },   // NVLS, NVLSTree
  {   0,     0, 4.0 }                       },// PAT
/* PCI */
{ { 1.0,  1.9,  4.0 }, { 1.0,  2.5,  5.7 },
  {   0,     0, 3.7 }, {   0,    0,  2.8 },
  {   0,     0,   0 }, {   0,    0,    0 },
  {   0,     0, 4.0 }                       },
/* NET */
{ { 5.0,  8.5,  14  }, { 2.7,  4.0, 14.0 },
  {   0,     0,  31 }, {   0,    0,   30 },
  {   0,     0,  18 }, {   0,    0, 20.9 },
  {   0,     0,  14 }                       },
};
```

읽는 법. `hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 14.0` µs. NIC 한 hop 의 ring Simple latency 가 14 µs. 같은 ring 도 NVLink 면 3.4 µs 로 4 배 빠르다.

이 값들이 §7 numerical example 의 입력.

### 6.3 Eligibility filter

§3.1 / §3.3 / §5 의 조건들을 코드로 한 번에 보면.

```c
// src/graph/tuning.cc::ncclTopoTuneModel 발췌
for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
  // collective × algorithm 호환성 (§3.1)
  if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce)
      && a != NCCL_ALGO_RING) continue;
  if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
      && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING
      && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
  if (coll == ncclFuncAllReduce && a == NCCL_ALGO_PAT) continue;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // NVLS / NVLS_TREE → Simple only
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE)
      continue;
    // PAT → Simple only + 자체 enable
    if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
        && a == NCCL_ALGO_PAT
        && (p != NCCL_PROTO_SIMPLE || ncclPatEnable(comm) == 0))
      continue;
    // LL128 enable (§5)
    // ...
    // 살아남은 셀에 bandwidth / latency 계산
  }
}
```

비활성된 셀은 `comm->bandwidths[c][a][p] = 0` 으로 마크 (`tuning.cc:504`). 나중에 argmin 에서 자동 탈락.

### 6.4 Per-call: argmin

사용자가 `ncclAllReduce(...)` 를 부르면 collective task 가 만들어지고, `topoGetAlgoInfo` (`src/enqueue.cc:1940`) 가 message size + group size 를 보고 §1 의 7 × 3 표에서 argmin 을 찾는다.

```c
// src/enqueue.cc 발췌 (요지)
// 1. 모든 (algo, proto) 셀에 대해 ncclTopoGetAlgoTime 계산
for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    float time;
    ncclTopoGetAlgoTime(comm, coll, a, p, nBytes, numPipeOps, &time);
    table[a][p] = (bw == 0) ? -1.0 : time;
  }
}

// 2. argmin
float minTime = FLT_MAX;
int algorithm = NCCL_ALGO_UNDEF, protocol = NCCL_PROTO_UNDEF;
for (int a=0; a<NCCL_NUM_ALGORITHMS; a++)
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
    if (table[a][p] >= 0.0 && table[a][p] < minTime) {
      algorithm = a; protocol = p; minTime = table[a][p];
    }
  }
```

`ncclTopoGetAlgoTime` 자체는 한 줄.

```c
// src/graph/tuning.cc:609 발췌
*time = lat * latCount + nBytes / (1000 * bw);
```

$T = (\text{lat} \times \text{latCount}) + \frac{\text{nBytes}}{1000 \times \text{bw}}$. 단순한 $\alpha + n\beta$ 의 NCCL 구현. 단 `latCount` 는 algorithm 별로 다르고 (Ring 은 `numPipeOps`, 나머지는 `DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS)`), Tree AllReduce 는 message size 별 보정 (`treeCorrectionFactor[protocol][logSize]`) 이 들어간다.

### 6.5 Channel / chunk 사이즈

(algo, proto) 가 정해지면 같은 cost model 위에서 nChannels 와 chunkSize 도 결정. CollNet 은 자체 search, NVLS 는 `comm->nvlsChannels` 로 클램프, Ring / Tree 는 `nBytes < nc × nt × threadThreshold` 가 될 때까지 nc 를 줄이고 그 다음 nt 를 줄인다.

### 6.6 정리 그림

```mermaid
flowchart TD
    A[ncclCommInit] --> B[ncclTopoGetSystem<br/>topology graph 빌드]
    B --> C[ncclTopoTuneModel<br/>α/β 테이블 빌드<br/>+ eligibility filter]
    C --> D[(comm-&gt;bandwidths<br/>comm-&gt;latencies)]

    E[ncclAllReduce 호출] --> F[message size + group size]
    F --> G[topoGetAlgoInfo<br/>모든 algo×proto 셀에 대해<br/>α + n/β 계산]
    D --> G
    G --> H[argmin algo, proto]
    H --> I[nChannels, chunkSize 결정]
    I --> J[ncclLaunchKernel]
```

호스트가 tensor 크기를 보고 어떤 algorithm 으로 launch 할지 결정하는 흐름이 이 그림이다. Init 시 한 번 빌드, per-call 마다 size 보고 argmin.

## 7. Numerical Example

위 수식과 §6.2 의 표를 구체 환경에 박아 ring vs tree vs NVLS 가 어디서 갈리는지 본다.

### 7.1 시나리오 A: 8-GPU H100 단일 노드 (NVLink + NVSwitch)

가정.

- $p = 8$, single node, NVSwitch.
- NVLink per-direction $\approx 450$ GB/s, aggregate intra-node $B \approx 900$ GB/s.
- $\alpha_{\text{intra}} \approx 1$ µs (NVLink 한 hop, hwLat 표의 PAT 항).
- baseLat (Tree, Simple) $= 8.4$ µs, baseLat (Ring, Simple) $= 8.4$ µs.

각 algorithm 의 $T(K)$ 근사 (Simple protocol 기준):

$$T_{\text{ring}}(K) = 2(p-1)\alpha_{\text{intra}} + \frac{2(p-1)}{p} \cdot \frac{K}{B} \approx 14\,\mu s + \frac{1.75 K}{B}$$

$$T_{\text{tree}}(K) \approx 2 \log_2 p \cdot \alpha_{\text{intra}} + \frac{2K}{B} \approx 6\,\mu s + \frac{2K}{B}$$

$$T_{\text{NVLS}}(K) \approx \alpha_{\text{intra}} + \frac{K}{0.85 B} \approx 1\,\mu s + \frac{1.18 K}{B}$$

| $K$ | Ring | Tree | NVLS |
|---|---|---|---|
| 1 MB | $\sim 16$ µs | $\sim 8$ µs | $\sim 2$ µs |
| 16 MB | $\sim 45$ µs | $\sim 42$ µs | $\sim 22$ µs |
| 256 MB | $\sim 512$ µs | $\sim 575$ µs | $\sim 335$ µs |

이 머신에서는 NVLS 가 모든 size 에서 우세. 예상대로 H100 NVSwitch 환경의 큰 AllReduce 는 거의 NVLS 로 간다.

### 7.2 시나리오 B: 8 노드 × 8 GPU IB (NVLS 없음)

가정.

- $P = 64$, $p_{\text{node}} = 8$, 8 노드.
- IB per-direction $\approx 25$ GB/s, $\alpha_{\text{ib}} \approx 5$ µs (hwLat 의 NET ring Simple = 14 µs 의 일부).
- $\alpha_{\text{intra}} \approx 1$ µs.

Cross-node 비용이 지배적이라 ring 의 $2(P-1) = 126$ step 이 부담이다. Double Binary Tree 가 이 부담을 $2 \log_2 8 = 6$ inter-node step 으로 줄인다 (§4.2 의 Tree latency 식).

| $K$ | Ring | Tree (Double Binary) |
|---|---|---|
| 1 MB | $\sim 700$ µs | $\sim 100$ µs |
| 16 MB | $\sim 2.5$ ms | $\sim 1.4$ ms |
| 256 MB | $\sim 35$ ms | $\sim 22$ ms |

작은 $K$ 에서 tree 가 7 배 가까이 빠르고, 큰 $K$ 에서도 Sanders trick 덕에 tree 가 ring 보다 우세. NCCL 이 multi-node AllReduce 에서 Tree 를 자주 고르는 이유.

### 7.3 정리

시나리오별 자동 선택 패턴.

- **Single-node H100 NVSwitch**: 큰 AllReduce 는 NVLS, 작은 건 Tree 또는 NVLS.
- **Multi-node IB**: 큰 AllReduce 는 Tree (Double Binary), 작은 건 Tree, AllGather/RS 는 Ring 또는 PAT (1-GPU/노드면).
- **InfiniBand SHARP NIC**: CollNetDirect / Chain 이 Tree 를 밀어내는 경우.

(plot placeholder) 위 표를 Python + matplotlib 로 그려 같은 axes 에 ring / tree / NVLS 곡선을 올리면 두 crossover 가 시각적으로 보인다. 시나리오 A 에서는 NVLS 가 항상 아래, 시나리오 B 에서는 small $K$ 의 tree 와 large $K$ 의 ring 사이 crossover.

수치 분석은 그렇고, 실측은 어땠나. NCCL 2.4 의 Double Binary Tree 가 Summit 에서 24,576 GPU 까지 어떻게 scale 했는지 보자.

![NCCL bus bandwidth on Summit, up to 24,576 GPUs](/assets/img/posts/nccl-algorithms/Summit-BW.png){: width="640"}
_Figure 6. Summit 의 24K GPU 까지 거의 flat 한 bus bandwidth. Double Binary Tree + multi-channel ring 이 large-scale 에서도 bandwidth 를 유지한다._

![NCCL latency on Summit](/assets/img/posts/nccl-algorithms/Summit-Latency.png){: width="640"}
_Figure 7. 같은 측정의 latency. 24K GPU 까지 $O(\log p)$ 곡선. Pure ring 이었으면 $O(p)$ 직선이라 폭증했을 부분을 log-step 알고리즘 (Tree, PAT) 이 해결한다._

작은 메시지 + huge GPU count 조합에서 ring 만 쓰면 latency 가 폭증한다는 걸 보여주는 데이터. AllReduce 의 Double Binary Tree 와 AllGather / RS 의 PAT 가 푸는 문제가 이거다.

## 8. 환경변수, Determinism, 디버그

### 8.1 NCCL_ALGO / NCCL_PROTO override

자동 선택을 강제하는 방법.

```bash
# 모든 collective 에 Ring + CollNetDirect, AllReduce 만 Tree + CollNetDirect
NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect"

# LL128 빼고 다 허용
NCCL_PROTO="^LL128"
```

코드 흐름은 `parseList` 가 bitmask 로 만들고, 비활성 셀의 bandwidth 를 0 으로 (`tuning.cc:504`). 위의 argmin 이 자동으로 그 셀을 거른다.

### 8.2 Determinism

Reduction tree 가 달라지면 부동소수점 덧셈 순서가 달라져 낮은 비트에서 차이.

```bash
NCCL_ALGO=Tree
NCCL_PROTO=Simple
```

이 둘로 algorithm 과 protocol 을 고정하면 reduction 순서가 결정적이 된다. bf16 환경에서는 무시하기 어려운 차이를 줄 수 있어 reproducibility 검증 시 자주 쓴다.

### 8.3 NCCL_DEBUG_SUBSYS=TUNING

가장 직접적인 디버그 도구. NCCL 이 collective dispatch 시점에 로그 한 줄을 찍는다.

{% raw %}
```c
// src/enqueue.cc 발췌
INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d}",
  ncclFuncToString(task->func),
  task->count * ncclTypeSize(task->datatype),
  ncclAlgoToString(task->algorithm),
  ncclProtoToString(task->protocol),
  devWork->channelLo, devWork->channelHi);
```
{% endraw %}

활성화.

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

출력 예.

```
NCCL INFO AllReduce: 4194304 Bytes -> Algo Tree proto Simple channel{Lo..Hi}={0..7}
NCCL INFO AllGather: 1048576 Bytes -> Algo Ring proto LL128 channel{Lo..Hi}={0..3}
```

8-GPU H100 머신에서 AllReduce 가 NVLS 로 안 간다? `NCCL_DEBUG_SUBSYS=GRAPH` 로 채널 수와 NVS detection 먼저 확인. PAT 가 안 보인다? `nNodes != nRanks` 가 거의 항상 원인.

PyTorch profiler trace 의 kernel 이름도 같은 정보를 담는다. `ncclKernel_AllReduce_RING_LL_Sum_bfloat16` 같은 식.

---

## NCCL Source

| 주제 | 파일 | 함수 / 영역 |
|---|---|---|
| Algorithm enum | `src/include/device.h` | `ncclAlgoStr[]` |
| Pattern enum | `src/include/graph.h` | `NCCL_TOPO_PATTERN_*` |
| Ring 구성 | `src/graph/rings.cc` | `ncclBuildRings` |
| Double Binary Tree 구성 | `src/graph/trees.cc` | `ncclGetDtree`, `ncclGetBtree` |
| Topology pattern search | `src/graph/search.cc` | `ncclTopoCompute` |
| α/β 테이블 빌드 | `src/graph/tuning.cc` | `ncclTopoTuneModel` |
| baseLat / hwLat | `src/graph/tuning.cc` | `baseLatencies`, `hwLatencies` |
| NVLS efficiency | `src/graph/tuning.cc` | `nvlsEfficiency` |
| LL128 enable | `src/graph/tuning.cc` | `pEnable` 분기 |
| PAT enable | `src/graph/tuning.cc` | `ncclPatEnable` |
| Cost model 계산 | `src/graph/tuning.cc` | `ncclTopoGetAlgoTime` |
| Argmin selection | `src/enqueue.cc` | `topoGetAlgoInfo` |
| Tuning 로그 | `src/enqueue.cc` | `INFO(NCCL_TUNING, ...)` |
| PAT AllGather kernel | `src/device/all_gather.h` | `RunWorkColl<NCCL_ALGO_PAT>` |
| PAT ReduceScatter kernel | `src/device/reduce_scatter.h` | 동일 구조 |
| Ring AllReduce kernel | `src/device/all_reduce.h` | `runRing` |
| Protocol 정의 | `src/device/primitives.h` | `ProtoSimple`, `ProtoLL`, `ProtoLL128` |

## Reference

- Hu, Z. et al. "Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms." arXiv:2507.04786, 2026.
- Sanders, P., Speck, J., Träff, J.L. "Full Bandwidth Broadcast, Reduction and Scan with Only Two Trees." PVM/MPI, 2007.
- Thakur, R., Rabenseifner, R., Gropp, W. "Optimization of Collective Communication Operations in MPICH." IJHPCA, 2005.
- Bruck, J. et al. "Efficient Algorithms for All-to-All Communications in Multiport Message-Passing Systems." IEEE TPDS, 1997.
- Jeaugey, S. "PAT: a new algorithm for all-gather and reduce-scatter operations at scale." arXiv:2506.20252, 2025.
- <https://docs.nvidia.com/deeplearning/nccl/>
- <https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/>
- <https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/>
- <https://github.com/NVIDIA/nccl>
