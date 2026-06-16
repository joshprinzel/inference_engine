# Engine-Agnostic Scheduler

## Why This Exists

The original `ContinuousScheduler` proved the first version of ORCA-style serving semantics:

- drain request queue
- admit requests into active slots
- decode one token per active request per step
- free slots when requests finish
- track queue length, active slots, finished requests, and KV pressure

That was the right first scheduler.

But it had one architectural problem: it was tied directly to `ModelRunner`.

That meant the scheduler knew too much about the execution backend. It called Hugging Face-backed methods directly, which made it hard to plug in the CUDA paged attention path cleanly.

The new `EngineScheduler` fixes that by depending on a `DecodeEngine` interface instead of depending on `ModelRunner`.

---

## Core Idea

The scheduler should not know whether the engine is:

```text
Hugging Face
Synthetic CUDA
Future custom model runtime
```

The scheduler should only know how to:

```text
1. drain incoming requests
2. admit requests into slots
3. reserve/free KV blocks
4. call the decode engine for one decode step
5. stream/record emitted text
6. mark requests finished or failed
7. expose metrics and snapshots
```

Execution details belong to the engine.

---

## New Runtime Shape

```text
RequestQueue
    ↓
EngineScheduler
    ↓
DecodeEngine interface
    ├── HFDecodeEngine
    │       └── ModelRunner / Hugging Face generation
    │
    └── SyntheticCudaDecodeEngine
            ├── KVCachePool
            ├── DecodeBatch
            ├── AttentionBackend
            └── CUDA paged attention kernel
```

This is the major architectural line we crossed.

Before this refactor, the Hugging Face path and CUDA path were separate.

After this refactor, both paths can run under the same scheduler contract.

---

## What `EngineScheduler` Owns

`EngineScheduler` owns scheduling and request lifecycle:

```text
RequestQueue draining
waiting list
active slots
finished list
admission policy
KV block reservation
KV block freeing
decode loop timing
request finish/failure handling
metrics snapshots
```

The scheduler is responsible for deciding *which requests run*.

It is not responsible for knowing *how model execution happens*.

---

## What `DecodeEngine` Owns

`DecodeEngine` owns execution.

The interface is:

```python
class DecodeEngine(Protocol):
    @property
    def device(self) -> str:
        ...

    def count_prompt_tokens(self, prompt: str) -> int:
        ...

    def init_request_state(self, request_state: RequestState) -> None:
        ...

    def decode_step(
        self,
        request_states: list[RequestState],
        kv_block_manager: KVBlockManager,
    ) -> DecodeStepOutput:
        ...
```

The engine is responsible for:

```text
prompt/token initialization
one decode step of execution
backend-specific KV behavior
updating request_state.next_token
updating request_state.generated_tokens
returning emitted text
returning backend timing
optionally returning DecodeBatch snapshots
```

The scheduler treats all engines the same.

---

## `HFDecodeEngine`

`HFDecodeEngine` wraps the existing `ModelRunner`.

It gives us a real text generation path:

```text
EngineScheduler
    -> HFDecodeEngine
    -> ModelRunner
    -> Hugging Face model.generate / forward path
```

This path uses Hugging Face internals for:

```text
attention
past_key_values
logits
token generation
```

The KV block manager is still runtime metadata in this mode. Hugging Face owns the real KV tensors internally.

### Why It Matters

This gives the project a working real-model baseline.

It also keeps the frontend/API useful while the custom CUDA engine is still synthetic at the model-math level.

---

## `SyntheticCudaDecodeEngine`

`SyntheticCudaDecodeEngine` exercises the custom CUDA paged attention runtime path.

It uses synthetic:

```text
query tensors
generated token IDs
generated-token K/V writes
```

But it uses real project infrastructure for:

```text
RequestState lifecycle
KVBlockManager block tables
DecodeBatch lowering
KVCachePool physical KV layout
AttentionBackend abstraction
CUDA paged attention kernel
```

The path is:

```text
EngineScheduler
    -> SyntheticCudaDecodeEngine
    -> DecodeBatch
    -> KVCachePool
    -> AttentionBackend
    -> CUDA paged attention kernel
```

### Why It Matters

This proves the CUDA backend is no longer just an isolated kernel benchmark.

It can run under the same continuous scheduler as the Hugging Face engine.

That is the engine-agnostic runtime milestone.

---

## What This Proves

This refactor proves:

```text
same RequestQueue
same RequestState
same EngineScheduler
same KVBlockManager
same slot lifecycle
same metrics path
different DecodeEngine implementations
```

Specifically:

```text
HFDecodeEngine:
    real text generation
    Hugging Face owns model internals

SyntheticCudaDecodeEngine:
    synthetic model math
    real paged KV/cache/backend plumbing
    custom CUDA paged attention execution
```

This is the core systems story.

---

## What This Does Not Prove Yet

This does not yet prove:

```text
real Q/K/V projections through the custom CUDA path
real logits through the custom CUDA path
real sampling through the custom CUDA path
full Hugging Face replacement
full custom transformer execution
production-grade scheduler policy
```

That is expected.

The current goal is engine-agnostic runtime architecture, not full model replacement.

---

## Why Not Hook CUDA Directly Into Hugging Face?

The current `ModelRunner` uses Hugging Face as a black box.

That path does not expose a clean boundary for:

```text
paged KV block tables
custom KVCachePool layout
DecodeBatch metadata
custom CUDA attention call
per-layer Q/K/V tensors
```

Trying to force the CUDA kernel into Hugging Face internals now would create a fragile monkey-patch project.

Instead, this project now has a cleaner ladder:

```text
1. HFDecodeEngine
   Real text generation baseline.

2. SyntheticCudaDecodeEngine
   Real scheduler/KV/DecodeBatch/CUDA backend path with synthetic model math.

3. Future CustomModelDecodeEngine
   Real model execution path using custom CUDA paged attention.
```

This keeps the current work honest and gives a direct path to the eventual goal.

---

## Long-Term Target

The eventual target is a real custom model decode engine:

```text
CustomModelDecodeEngine
    -> token embedding
    -> Q/K/V projections
    -> KVCachePool writeback
    -> CUDA paged attention
    -> MLP
    -> logits
    -> sampling
    -> streamed token output
```

That engine should plug into the same `EngineScheduler`.

The scheduler should not need to change.

That is the reason this refactor matters.

---

## Current Engine Modes

### Hugging Face Engine

Purpose:

```text
real text generation baseline
```

Path:

```text
EngineScheduler -> HFDecodeEngine -> ModelRunner -> Hugging Face
```

Strength:

```text
real model output
```

Limitation:

```text
custom CUDA attention is not used
```

---

### Synthetic CUDA Engine

Purpose:

```text
validate custom CUDA paged attention under the real scheduler runtime
```

Path:

```text
EngineScheduler -> SyntheticCudaDecodeEngine -> DecodeBatch -> CUDA attention
```

Strength:

```text
real scheduler/KV/backend integration
```

Limitation:

```text
synthetic Q/K/V and fake token outputs
```

---

## Interview Explanation

The important architecture decision was making the scheduler engine-agnostic.

Originally, the scheduler directly called the Hugging Face model runner. That made it hard to integrate a custom CUDA attention path because scheduling and execution were mixed together.

I introduced a `DecodeEngine` interface so the scheduler only manages request lifecycle, slots, KV block policy, and metrics. The execution backend is now pluggable.

I implemented two engines:

```text
HFDecodeEngine
    real Hugging Face text generation

SyntheticCudaDecodeEngine
    synthetic model math, but real DecodeBatch, paged KV cache, and CUDA attention backend
```

This gives me a working real-model baseline and a working CUDA-runtime path under the same scheduler. The next step is a future `CustomModelDecodeEngine` that replaces synthetic Q/K/V with real model layer execution.

---

## Why This Is Resume-Relevant

This turns the project from:

```text
a Hugging Face app plus a CUDA kernel experiment
```

into:

```text
an engine-agnostic inference runtime with pluggable decode engines
```

The key technical claim is:

> The same continuous scheduler can run either a Hugging Face decode engine for real text generation or a synthetic CUDA decode engine that exercises the custom paged KV cache, DecodeBatch lowering, and CUDA paged attention backend.

That is the runtime architecture milestone.