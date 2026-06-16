## Current Runtime Status

This project is an experimental LLM inference serving runtime.

The current runtime supports two decode engines under the same engine-agnostic scheduler:

```text
HFDecodeEngine
    Real text generation through Hugging Face.

SyntheticCudaDecodeEngine
    Synthetic model math, but real DecodeBatch lowering, paged KV cache layout,
    KV block management, and custom CUDA paged attention execution.
```

The main runtime path is:

```text
server.py -> EngineScheduler -> DecodeEngine
```

Legacy schedulers are retained for comparison only.

## Quickstart

Build CUDA backend:

```bash
make build-cuda
```

Run HF smoke tests:

```bash
make test-hf
```

Run synthetic CUDA smoke tests:

```bash
make test-synthetic
```

Run server with Hugging Face backend:

```bash
make run-hf
```

Run server with synthetic CUDA backend:

```bash
make run-synthetic
```