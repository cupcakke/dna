# Review and validation

## Corrections to the first delivery

- The first reply did not disclose the boundary between executable CPU tests, Zig semantic checks, and unverified Futhark/CUDA runtime behavior. It should not be interpreted as a guarantee of error-free production execution.
- The original GEMM regression used matrices below the optimized-path threshold. The revised test includes 137-by-131 outputs and reduction sizes 129, 257, and 259, covering packed kernels, partial tiles, and odd reduction tails. The native suite passed on explicit AVX2 (`x86_64_v3`) and AVX512 (`x86_64_v4`) targets.
- HTTP framing is now a separately tested production module. It rejects malformed or duplicate Content-Length headers and unsupported Transfer-Encoding, bounds individual frames rather than whole pipelines, and preserves the next request after consuming a frame. Body and header slices are obtained only after all buffer growth has finished.
- Temporary inference allocations now have request lifetime, not connection lifetime. Persistent connections no longer accumulate per-request arena allocations until disconnect.
- Rate-limiter cleanup is periodic rather than a full client-table scan on every request. Client keys remain independent of the source port, and storage remains bounded.
- Inference locking now also protects model availability checks and tokenization against concurrent model replacement. Both inference handlers use deferred unlocks. Embedding response buffers have one owner on both error and success paths.
- Exhausted training ranks now contribute zero gradients even when gradient averaging is disabled. A one-token padding row exists only to satisfy accelerator input-shape requirements; its gradient contribution is zero.
- Added an independent VPU check against the canonical coupling-plus-OFTB order, including forward/inverse round trips with diffusion enabled and disabled.
- Added executable regression checks for the actual benchmark parsing loop, Modal decorator placement, and watchdog marker. These checks do not import Modal or claim to execute a Modal deployment.

## Reconstruction semantics

The invalid direct embedding update from the numerical round-trip residual was removed. The reconstruction metric remains a diagnostic of finite-precision inversion. The derivative of an exact inverse composition is not a standalone `2 * residual` update to the input. This patch does not claim to introduce a separate trainable reconstruction objective; the existing reconstruction-alpha argument is retained for interface compatibility.

## Validation performed

Using Zig 0.14.1:

- `python -m ziglang build check test-rsf test-rsf-native test-tokenizer -Dgpu=false --summary all`: passed; 31 tests.
- `python -m ziglang build check -Dgpu=true --summary all`: passed semantic analysis; this does not link or execute CUDA kernels.
- `python -m ziglang build test-rsf-native -Dgpu=false -Dcpu=x86_64_v3 --summary all`: passed; 13 tests, including the revised packed-GEMM test.
- `python -m ziglang build test-rsf-native -Dgpu=false -Dcpu=x86_64_v4 --summary all`: passed; the same 13 tests on the AVX512 target.
- `python -m ziglang test src/api/http_framing.zig`: passed; 5 tests.
- `python -m ziglang test src/api/http_framing.zig -O ReleaseSafe`: passed; the same 5 tests.
- `python scripts/test_modal_status_bench.py`: passed; 3 tests.
- Python compilation and `git diff --check`: passed.

There are 39 distinct passing tests across the CPU suites, HTTP-framing tests, and benchmark regression checks. Repeated target/configuration runs are not counted as additional distinct tests.

## Not verified here

Futhark compilation, generated-C linking, live CUDA execution, multi-GPU/NCCL execution, Modal deployment, and accelerator-backed checkpoint save/load round trips have not been verified. The Futhark release download was blocked at GitHub's release-asset host by the environment's outbound-network restrictions. GPU semantic analysis is not a substitute for those checks. The HTTP tests exercise production framing and buffer consumption, not a live inference-server socket integration test.
