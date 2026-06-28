# Current Runtime Path

The current runtime path is:

```text
server.py
    -> EngineScheduler
    -> DecodeEngine
        -> HFDecodeEngine
        -> SyntheticCudaDecodeEngine
```

Legacy Schedulers are retained only for Comparison:
```text
scheduler.py
continuous_scheduler.py
synthetic_decode_engine.py
```

New runtime work should target:

```text
EngineScheduler
DecodeEngine
HFDecodeEngine
SyntheticCudaDecodeEngine
KVBlockManager
KVCachePool
DecodeBatch
AttentionBackend
```

Do not add new features to the legacy scheduler Path.



## `Makefile`

```makefile
.PHONY: build-cuda test-hf test-synthetic test-runtime run-hf run-synthetic clean

build-cuda:
	cd cuda_backend && python setup.py build_ext --inplace

test-hf:
	python experiments/tests/test_hf_decode_engine_smoke.py
	python experiments/tests/test_engine_scheduler_hf_smoke.py

test-synthetic: build-cuda
	python experiments/tests/test_engine_scheduler_synthetic_cuda_smoke.py

test-runtime: test-hf test-synthetic

run-hf:
	ENGINE_BACKEND=hf uvicorn server:app --reload

run-synthetic: build-cuda
	ENGINE_BACKEND=synthetic-cuda uvicorn server:app --reload

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf cuda_backend/build
	rm -rf cuda_backend/build-native
	rm -f cuda_backend/*.so
	rm -f cuda_backend/*.log
```