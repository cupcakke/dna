import csv
import hashlib
import json
import math
import os
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "jaide-status-bench"
LOCAL_PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_MOUNT_PATH = Path("/workspace/jaide")
DATA_MOUNT_PATH = Path("/data")
CHECKPOINT_MOUNT_PATH = Path("/checkpoints")
REPORT_MOUNT_PATH = Path("/reports")
BUILD_MOUNT_PATH = Path("/build_artifacts")
TOKENIZER_SOURCE_PATH = Path(
    os.environ.get("JAIDE_BENCH_TOKENIZER_SOURCE_PATH", "/workspace/jaide/tokenizer.vocab")
)

IGNORE_PATTERNS = [
    ".git",
    ".zig-cache",
    ".venv",
    ".venv-modal",
    "__pycache__",
    "*.o",
    "*.a",
    "*.bin",
    ".local",
    ".cache",
    ".upm",
    ".pythonlibs",
    ".config",
]

GPU_SPEC = os.environ.get("JAIDE_BENCH_GPU", "B200:1")


def _gpu_count_from_spec(spec: str) -> int:
    parts = spec.rsplit(":", 1)
    if len(parts) == 1:
        return 1
    try:
        count = int(parts[1])
    except ValueError as exc:
        raise ValueError("A JAIDE_BENCH_GPU értékének pozitív GPU számmal kell végződnie") from exc
    if count <= 0:
        raise ValueError("A JAIDE_BENCH_GPU értékének pozitív GPU számmal kell végződnie")
    return count

ALLOCATED_GPU_COUNT = _gpu_count_from_spec(GPU_SPEC)
TIMEOUT_SEC = int(os.environ.get("JAIDE_BENCH_TIMEOUT", "86400"))
CPU_TIMEOUT_SEC = int(os.environ.get("JAIDE_CPU_TIMEOUT", "86400"))
CPU_REQUEST = float(os.environ.get("JAIDE_BENCH_CPU_REQUEST", "32.0"))
CPU_LIMIT = float(os.environ.get("JAIDE_BENCH_CPU_LIMIT", "32.0"))
MEMORY_REQUEST_MB = int(os.environ.get("JAIDE_BENCH_MEMORY_REQUEST", "131072"))
MEMORY_LIMIT_MB = int(os.environ.get("JAIDE_BENCH_MEMORY_LIMIT", "131072"))
MODEL_DIM = int(os.environ.get("JAIDE_BENCH_MODEL_DIM", "8192"))
NUM_LAYERS = int(os.environ.get("JAIDE_BENCH_LAYERS", "24"))
BATCH_SIZE = int(os.environ.get("JAIDE_BENCH_BATCH", "32"))
EPOCHS = int(os.environ.get("JAIDE_BENCH_EPOCHS", "1"))
SAMPLE_CAP = int(os.environ.get("JAIDE_BENCH_SAMPLE_CAP", "50000"))
MAX_SEQ_LEN = int(os.environ.get("JAIDE_BENCH_MAX_SEQ_LEN", "256"))
LEARNING_RATE = os.environ.get("JAIDE_BENCH_LR", "0.0003")
REASONING_CYCLES = int(os.environ.get("JAIDE_BENCH_REASONING_CYCLES", "1"))
RELATIONAL_PASS_INTERVAL = int(os.environ.get("JAIDE_BENCH_RELATIONAL_PASS_INTERVAL", "1000"))
JAIDE_RELATIONAL_FAST = os.environ.get("JAIDE_RELATIONAL_FAST", "1")
NUM_GPUS = int(os.environ.get("JAIDE_BENCH_NUM_GPUS", str(ALLOCATED_GPU_COUNT)))
RECONSTRUCTION_ALPHA = os.environ.get("JAIDE_BENCH_RECONSTRUCTION_ALPHA", "0.3")
PHASE_A_STEPS = int(os.environ.get("JAIDE_BENCH_PHASE_A_STEPS", "500"))
PHASE_B_STEPS = int(os.environ.get("JAIDE_BENCH_PHASE_B_STEPS", "2000"))
SHUFFLE_TARGET_CONTROL = os.environ.get("JAIDE_BENCH_SHUFFLE_TARGET_CONTROL", "0")
TARGET_SOURCE_FROZEN = os.environ.get("JAIDE_BENCH_TARGET_SOURCE_FROZEN", "1")
SPECTRAL_DEPTH_COMPENSATION = os.environ.get("JAIDE_BENCH_SPECTRAL_DEPTH_COMPENSATION", "1")
INFERENCE_STARTUP_TIMEOUT_SEC = int(os.environ.get("JAIDE_INFERENCE_STARTUP_TIMEOUT", "600"))
MASTER_ADDR = os.environ.get("JAIDE_BENCH_MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("JAIDE_BENCH_MASTER_PORT", "29500")
NCCL_DEBUG = os.environ.get("JAIDE_BENCH_NCCL_DEBUG", "WARN")
MASTER_IS_LOOPBACK = MASTER_ADDR in {"127.0.0.1", "localhost", "::1"}
NCCL_IB_DISABLE = os.environ.get("JAIDE_BENCH_NCCL_IB_DISABLE", "1" if MASTER_IS_LOOPBACK else "0")
NCCL_SOCKET_IFNAME = os.environ.get("JAIDE_BENCH_NCCL_SOCKET_IFNAME", "lo" if MASTER_IS_LOOPBACK else "^lo,docker")
CUDA_DEVICE_ORDER = os.environ.get("JAIDE_BENCH_CUDA_DEVICE_ORDER", "PCI_BUS_ID")
DATASET_PATH = "/data/dataset/finephrase_bench.jsonl"
TOKEN_DATASET_PATH = "/data/dataset/finephrase_bench.bin"
CHECKPOINT_PATH = os.environ.get("JAIDE_BENCH_CHECKPOINT_PATH", "/checkpoints/tokenizer.vocab")
VOCAB_SIZE = int(os.environ.get("JAIDE_BENCH_VOCAB_SIZE", "32000"))
SPECTRAL_NORM_TARGET = os.environ.get("JAIDE_BENCH_SPECTRAL_NORM_TARGET", "0.9")
SPECTRAL_POWER_ITERATIONS = int(os.environ.get("JAIDE_BENCH_SPECTRAL_POWER_ITERATIONS", "10"))
SEED_OFFSET = int(os.environ.get("JAIDE_BENCH_SEED_OFFSET", "0"))
GRAD_MEAN = os.environ.get("JAIDE_BENCH_GRAD_MEAN", "true")
NORMALIZED_GRADIENT_FLOW = os.environ.get("JAIDE_BENCH_NORMALIZED_GRADIENT_FLOW", "true")
GRADIENT_CLIP_NORM = os.environ.get("JAIDE_BENCH_GRADIENT_CLIP_NORM", "1.0")
SFD_TRUST_RATIO = os.environ.get("JAIDE_BENCH_SFD_TRUST_RATIO", "0.1")
SFD_WEIGHT_FLOOR = os.environ.get("JAIDE_BENCH_SFD_WEIGHT_FLOOR", "0.001")
SPECTRAL_INTERVAL = int(os.environ.get("JAIDE_BENCH_SPECTRAL_INTERVAL", "100"))
LOGDET_WEIGHT = os.environ.get("JAIDE_BENCH_LOGDET_WEIGHT", "-0.001")
CLIP_MIN = os.environ.get("JAIDE_BENCH_CLIP_MIN", "-5.0")
CLIP_MAX = os.environ.get("JAIDE_BENCH_CLIP_MAX", "5.0")
CHECKPOINT_VERSION = int(os.environ.get("JAIDE_BENCH_CHECKPOINT_VERSION", "7"))
CHECKPOINT_INTERVAL_EPOCHS = int(os.environ.get("JAIDE_BENCH_CHECKPOINT_INTERVAL_EPOCHS", "5"))
RESUME_CHECKPOINT = os.environ.get("JAIDE_BENCH_RESUME_CHECKPOINT", "")
NCU_ENABLE = os.environ.get("JAIDE_BENCH_NCU", "0") == "1"
FORCE_REBUILD = os.environ.get("JAIDE_BENCH_FORCE_REBUILD", "1") == "1"
SKIP_PREP = os.environ.get("JAIDE_BENCH_SKIP_PREP", "0") == "1"
RUN_MODE = os.environ.get("JAIDE_BENCH_RUN_MODE", "all").strip().lower()
INIT_DEADLINE_SEC = int(os.environ.get("JAIDE_BENCH_INIT_DEADLINE", "1800"))
IDLE_WATCHDOG_SEC = int(os.environ.get("JAIDE_BENCH_IDLE_WATCHDOG", "900"))
SKIP_KNOWLEDGE_GRAPH = os.environ.get("JAIDE_BENCH_SKIP_KNOWLEDGE_GRAPH", "1")
GRAPH_CHUNK_SIZE = os.environ.get("JAIDE_BENCH_GRAPH_CHUNK_SIZE", "4096")
INIT_SPECTRAL_ITERS = os.environ.get("JAIDE_BENCH_INIT_SPECTRAL_ITERS", "0")

SOURCE_FINGERPRINT_PATHS = (
    "src/hw/accel/main.fut",
    "src/hw/accel/futhark_kernels.fut",
    "src/hw/accel/accel_interface.zig",
    "src/hw/accel/futhark_bindings.zig",
    "src/hw/accel/cuda_bindings.zig",
    "src/hw/accel/gpu_memory.zig",
    "src/hw/accel/compact_batch.zig",
    "src/hw/accel/futhark_abi_check.c",
    "src/hw/accel/tensor_core_matmul.zig",
    "src/distributed/distributed_trainer_futhark.zig",
    "src/tokenizer/mgt.zig",
    "src/distributed/checkpoint_envelope.zig",
    "src/distributed/nccl_bindings.zig",
    "src/main_distributed_futhark.zig",
    "src/api/inference_server.zig",
    "build.zig",
)

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("jaide-bench-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("jaide-bench-checkpoints", create_if_missing=True)
report_volume = modal.Volume.from_name("jaide-bench-reports", create_if_missing=True)
build_volume = modal.Volume.from_name("jaide-bench-build", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages "
        "git curl xz-utils build-essential wget ca-certificates pkg-config "
        "libnccl2 libnccl-dev opencl-headers ocl-icd-opencl-dev jq",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("pyarrow", "requests", "zstandard")
    .run_commands(
        "mkdir -p /opt",
        "curl -fsSL https://ziglang.org/download/0.14.1/zig-x86_64-linux-0.14.1.tar.xz "
        "| tar -xJ -C /opt",
        "ln -sf /opt/zig-x86_64-linux-0.14.1/zig /usr/local/bin/zig",
        "zig version",
    )
    .run_commands(
        "curl -fsSL https://github.com/diku-dk/futhark/releases/download/v0.26.4/"
        "futhark-0.26.4-linux-x86_64.tar.xz -o /tmp/futhark.tar.xz",
        "mkdir -p /opt/futhark",
        "tar -xJf /tmp/futhark.tar.xz -C /opt/futhark --strip-components=1",
        "ln -sf /opt/futhark/bin/futhark /usr/local/bin/futhark",
        "rm /tmp/futhark.tar.xz",
        "futhark --version | grep -F '0.26.4' || "
        "{ echo 'HIBA: futhark verzió eltérés a telepítés után'; exit 1; }",
    )
    .env(
        {
            "PATH": "/opt/zig-x86_64-linux-0.14.1:/opt/futhark/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs",
            "HF_HOME": "/data/hf_home",
            "HF_DATASETS_CACHE": "/data/hf_datasets_cache",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=str(PROJECT_MOUNT_PATH),
        ignore=IGNORE_PATTERNS,
    )
)


def _log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except FileNotFoundError:
        return


def _source_fingerprint(root: Path = LOCAL_PROJECT_DIR) -> str:
    digest = hashlib.sha256()
    digest.update(b"zig-0.14.1\n")
    digest.update(b"futhark-0.26.4-cuda-sm100\n")
    digest.update(b"checkpoint-v7\n")
    digest.update(b"rsf-coupling-2-stack-only\n")
    for relative in SOURCE_FINGERPRINT_PATHS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".fingerprint")


def _write_fingerprint(path: Path, fingerprint: str) -> None:
    sidecar = _fingerprint_sidecar(path)
    sidecar.write_text(fingerprint + "\n", encoding="utf-8")


def _fingerprint_matches(path: Path, fingerprint: str) -> bool:
    sidecar = _fingerprint_sidecar(path)
    if not sidecar.is_file():
        return False
    try:
        stored = sidecar.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return stored == fingerprint


def _find_latest_binary(name: str, fingerprint: Optional[str] = None) -> Optional[Path]:
    if not BUILD_MOUNT_PATH.exists():
        return None
    candidates: List[Path] = []
    for path in BUILD_MOUNT_PATH.rglob(name):
        try:
            if path.is_file() and path.stat().st_size > 0:
                if fingerprint is not None and not _fingerprint_matches(path, fingerprint):
                    continue
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    return candidates[0]


def _classify_training_failure(output: str, timed_out: bool, init_seen: bool) -> str:
    lowered = output.lower()
    if "vocabularyfull" in lowered:
        return "vocabulary_full"
    if "memory_admission_rejected" in lowered:
        return "admission_rejected"
    if "device_memory_query_failed" in lowered:
        return "device_query_failed"
    if "out of memory" in lowered or "cudaerror 2" in lowered or "memoryallocation" in lowered:
        return "oom"
    if "using managed memory" in lowered and not init_seen:
        return "unified_memory_stall"
    if timed_out and not init_seen:
        return "init_stall"
    if timed_out:
        return "idle_stall"
    return "unknown"


def _clear_rank_coordination_files(nccl_id_path: str) -> None:
    base = Path(nccl_id_path)
    _safe_unlink(base)
    _safe_unlink(Path(str(base) + ".ready"))
    try:
        for marker in base.parent.glob(base.name + ".*"):
            if marker.is_file() or marker.is_symlink():
                _safe_unlink(marker)
    except FileNotFoundError:
        pass


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return


def _run(
    cmd: List[str],
    cwd: str,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    timeout: int = 900,
    input_bytes: Optional[bytes] = None,
) -> Tuple[int, str, float]:
    if timeout <= 0:
        raise ValueError("Az időkorlátnak pozitívnak kell lennie")
    if not cmd:
        raise ValueError("A parancs nem lehet üres")
    _log(f">>> {' '.join(cmd)}  (munkakönyvtár={cwd})")
    t0 = time.monotonic()
    deadline = t0 + timeout
    stdin_mode = subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=stdin_mode,
        bufsize=0,
        start_new_session=True,
    )
    if proc.stdout is None:
        _terminate_process_group(proc)
        raise RuntimeError("A folyamat szabványos kimeneti csöve nem jött létre")
    if input_bytes is not None:
        if proc.stdin is None:
            _terminate_process_group(proc)
            raise RuntimeError("A folyamat szabványos bemeneti csöve nem jött létre")
        try:
            proc.stdin.write(input_bytes)
            proc.stdin.flush()
        finally:
            proc.stdin.close()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    output_chunks: List[bytes] = []
    timed_out = False
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                _terminate_process_group(proc)
                break
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining.decode("utf-8", errors="replace"), end="", flush=True)
                    output_chunks.append(remaining)
                break
            events = selector.select(timeout=max(0.0, min(1.0, deadline - now)))
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                output_chunks.append(chunk)
        if proc.poll() is None:
            proc.wait()
    finally:
        selector.close()
        proc.stdout.close()
    dt = time.monotonic() - t0
    out = b"".join(output_chunks).decode("utf-8", errors="replace")
    _log(f"<<< kilépési kód={proc.returncode}  időtartam={dt:.2f}s")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output=out.encode("utf-8"))
    if check and proc.returncode != 0:
        raise SystemExit(f"A parancs hibával fejeződött be (kód={proc.returncode}): {' '.join(cmd)}")
    return int(proc.returncode or 0), out, dt


def _run_multirank(
    cmd: List[str],
    cwd: str,
    base_env: Dict[str, str],
    num_gpus: int,
    nccl_id_path: str,
    timeout: int,
    init_deadline_sec: Optional[int] = None,
    idle_watchdog_sec: Optional[int] = None,
    init_marker: str = "[Step ",
) -> Tuple[int, str, float]:
    if num_gpus <= 0:
        raise ValueError("A GPU-k számának legalább 1-nek kell lennie")
    if timeout <= 0:
        raise ValueError("Az időkorlátnak pozitívnak kell lennie")
    if not cmd:
        raise ValueError("A parancs nem lehet üres")

    _log(f">>> több-GPU futtatás {' '.join(cmd)} GPU-szám={num_gpus} (munkakönyvtár={cwd})")
    t0 = time.monotonic()
    deadline = t0 + timeout

    _clear_rank_coordination_files(nccl_id_path)

    rank_envs: List[Dict[str, str]] = []
    for rank_index in range(num_gpus):
        rank_env = base_env.copy()
        rank_env["WORLD_SIZE"] = str(num_gpus)
        rank_env["RANK"] = str(rank_index)
        rank_env["LOCAL_RANK"] = str(rank_index)
        rank_env["JAIDE_NCCL_ID_PATH"] = nccl_id_path
        rank_envs.append(rank_env)

    procs: List[subprocess.Popen[Any]] = []
    fd_to_rank: Dict[int, int] = {}
    output_chunks: List[bytes] = []
    selector = selectors.DefaultSelector()
    timed_out = False
    last_output = time.monotonic()
    init_seen = False
    stall_reason = None

    try:
        for rank_index in range(num_gpus):
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=rank_envs[rank_index],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
            if proc.stdout is None:
                procs.append(proc)
                for started in procs:
                    _terminate_process_group(started)
                raise RuntimeError(f"A(z) {rank_index}. rangú folyamat kimeneti csöve nem jött létre")
            procs.append(proc)
            selector.register(proc.stdout, selectors.EVENT_READ)
            fd_to_rank[proc.stdout.fileno()] = rank_index

        open_streams = num_gpus
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                stall_reason = "global_timeout"
                for proc in procs:
                    _terminate_process_group(proc)
                break
            if not init_seen and init_deadline_sec is not None and (now - t0) >= init_deadline_sec:
                timed_out = True
                stall_reason = "init_deadline"
                for proc in procs:
                    _terminate_process_group(proc)
                break
            if idle_watchdog_sec is not None and (now - last_output) >= idle_watchdog_sec:
                timed_out = True
                stall_reason = "idle_watchdog"
                for proc in procs:
                    _terminate_process_group(proc)
                break

            if open_streams == 0 and all(proc.poll() is not None for proc in procs):
                break

            events = selector.select(timeout=max(0.05, min(1.0, deadline - now)))
            if not events:
                if open_streams == 0 and all(proc.poll() is not None for proc in procs):
                    break
                continue

            for key, _ in events:
                rank_index = fd_to_rank.get(key.fd, -1)
                try:
                    chunk = os.read(key.fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    open_streams -= 1
                    continue
                last_output = now
                decoded = chunk.decode("utf-8", errors="replace")
                if init_marker in decoded:
                    init_seen = True
                prefix = f"[GPU-{rank_index}] ".encode("utf-8")
                for line in chunk.splitlines(keepends=True):
                    output_chunks.append(prefix + line)
                for line in chunk.splitlines(keepends=True):
                    sys.stdout.buffer.write(prefix + line)
                sys.stdout.buffer.flush()

        for proc in procs:
            if proc.poll() is None:
                proc.wait()
    finally:
        selector.close()
        for proc in procs:
            if proc.stdout:
                try:
                    proc.stdout.close()
                except OSError:
                    pass

    dt = time.monotonic() - t0
    combined_out = b"".join(output_chunks).decode("utf-8", errors="replace")
    returncodes = [proc.returncode for proc in procs]
    failures = [int(code) for code in returncodes if code not in (None, 0)]
    combined_rc = 0 if not failures else max(failures)
    classification = _classify_training_failure(combined_out, timed_out, init_seen)
    _log(f"<<< több-GPU futtatás GPU-szám={num_gpus} kódok={returncodes} időtartam={dt:.2f}s class={classification} init_seen={init_seen} stall={stall_reason}")
    if timed_out:
        raise RuntimeError(
            f"training_watchdog class={classification} stall={stall_reason} init_seen={init_seen}\n{combined_out}"
        )
    return combined_rc, combined_out, dt


def _write_report(report_dir: Path, name: str, content: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fp = report_dir / name
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)
    _log(f"Jelentés elmentve: {fp}")


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _read_token_dataset_meta(path: Path) -> Optional[Dict[str, int]]:
    try:
        with path.open("rb") as source:
            header = source.read(64)
        if len(header) != 64 or header[:8] != b"JAIDETOK":
            return None
        version, token_type = struct.unpack_from("<II", header, 8)
        header_size, total_tokens, vocabulary_size, maximum_sequence_length = struct.unpack_from(
            "<QQQQ", header, 16
        )
        if version != 1 or token_type != 0 or header_size < 64 or header_size % 64 != 0:
            return None
        if total_tokens <= 0 or vocabulary_size <= 0 or maximum_sequence_length <= 0:
            return None
        expected_size = header_size + total_tokens * 4
        if path.stat().st_size != expected_size:
            return None
        return {
            "header_size": header_size,
            "total_tokens": total_tokens,
            "vocabulary_size": vocabulary_size,
            "maximum_sequence_length": maximum_sequence_length,
        }
    except (OSError, struct.error, ValueError):
        return None


def _parallel_pretokenize(
    pretokenize_bin: Path,
    input_path: Path,
    output_tmp: Path,
    vocab_path: Path,
    max_seq_len: int,
    sample_cap: int,
    project_dir: str,
    env: Dict[str, str],
    timeout: int,
) -> Tuple[int, int, float]:
    worker_setting = int(os.environ.get("JAIDE_BENCH_PRETOKENIZE_WORKERS", "16"))
    worker_count = max(1, min(worker_setting, os.cpu_count() or 1, sample_cap))
    samples_per_worker = (sample_cap + worker_count - 1) // worker_count
    work_dir = Path("/tmp") / f"jaide-pretokenize-{os.getpid()}"
    work_dir.mkdir(parents=True, exist_ok=True)
    part_counts = [0] * worker_count
    part_inputs = [work_dir / f"input-{index:03d}.jsonl" for index in range(worker_count)]
    part_outputs = [work_dir / f"output-{index:03d}.bin" for index in range(worker_count)]
    processes: List[subprocess.Popen[Any]] = []
    log_files: List[Any] = []
    t0 = time.time()

    try:
        handles = [path.open("wb") for path in part_inputs]
        try:
            written = 0
            with input_path.open("rb") as source:
                for raw_line in source:
                    if not raw_line.strip():
                        continue
                    if written >= sample_cap:
                        break
                    part_index = min(written // samples_per_worker, worker_count - 1)
                    handles[part_index].write(raw_line)
                    part_counts[part_index] += 1
                    written += 1
        finally:
            for handle in handles:
                handle.close()

        if written != sample_cap:
            raise RuntimeError(
                f"A bemeneti dataset csak {written} nem üres mintát tartalmaz, "
                f"de {sample_cap} szükséges"
            )

        for index, count in enumerate(part_counts):
            if count == 0:
                continue
            log_path = work_dir / f"worker-{index:03d}.log"
            log_file = log_path.open("w", encoding="utf-8")
            log_files.append(log_file)
            process = subprocess.Popen(
                [
                    str(pretokenize_bin),
                    str(part_inputs[index]),
                    str(part_outputs[index]),
                    str(vocab_path),
                    str(max_seq_len),
                    str(count),
                ],
                cwd=project_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            processes.append(process)

        deadline = time.monotonic() + timeout
        while True:
            unfinished = [process for process in processes if process.poll() is None]
            if not unfinished:
                break
            if time.monotonic() >= deadline:
                for process in unfinished:
                    _terminate_process_group(process)
                raise RuntimeError("A párhuzamos előtokenizálás túllépte az időkorlátot")
            time.sleep(0.1)

        failures = [process.returncode for process in processes if process.returncode != 0]
        if failures:
            worker_logs = []
            for index in range(worker_count):
                log_path = work_dir / f"worker-{index:03d}.log"
                if log_path.is_file():
                    worker_logs.append(log_path.read_text(encoding="utf-8", errors="replace"))
            raise RuntimeError(
                f"{len(failures)} előtokenizáló worker hibával állt le:\n"
                + "\n".join(worker_logs)
            )

        part_meta: List[Dict[str, int]] = []
        total_tokens = 0
        vocabulary_size = 0
        maximum_sequence_length = 0
        for index, count in enumerate(part_counts):
            if count == 0:
                continue
            meta = _read_token_dataset_meta(part_outputs[index])
            if meta is None:
                raise RuntimeError(f"Hibás worker output: {part_outputs[index]}")
            expected_tokens = count * (max_seq_len + 1)
            if (
                meta["total_tokens"] != expected_tokens
                or meta["maximum_sequence_length"] != max_seq_len
            ):
                raise RuntimeError(
                    f"A worker output mérete hibás: {part_outputs[index]} "
                    f"(várt tokenek: {expected_tokens}, kapott: {meta['total_tokens']})"
                )
            if vocabulary_size == 0:
                vocabulary_size = meta["vocabulary_size"]
                maximum_sequence_length = meta["maximum_sequence_length"]
            elif (
                vocabulary_size != meta["vocabulary_size"]
                or maximum_sequence_length != meta["maximum_sequence_length"]
            ):
                raise RuntimeError("A worker outputok fejlécei nem egyeznek")
            total_tokens += meta["total_tokens"]
            part_meta.append(meta)

        with output_tmp.open("wb") as target:
            target.write(
                struct.pack(
                    "<8sIIQQQQQQ",
                    b"JAIDETOK",
                    1,
                    0,
                    64,
                    total_tokens,
                    vocabulary_size,
                    maximum_sequence_length,
                    0,
                    0,
                )
            )
            for index, count in enumerate(part_counts):
                if count == 0:
                    continue
                with part_outputs[index].open("rb") as source:
                    source.seek(part_meta.pop(0)["header_size"])
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())

        return sample_cap, output_tmp.stat().st_size, time.time() - t0
    finally:
        for process in processes:
            if process.poll() is None:
                _terminate_process_group(process)
        for log_file in log_files:
            log_file.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def _is_current_vocab_format(path: Path) -> bool:
    try:
        with path.open("rb") as source:
            raw = source.read(4)
            if len(raw) != 4:
                return False
            token_count = int.from_bytes(raw, "little")
            if token_count == 0 or token_count > VOCAB_SIZE:
                return False
            for _ in range(token_count):
                length_raw = source.read(4)
                if len(length_raw) != 4:
                    return False
                token_length = int.from_bytes(length_raw, "little")
                if token_length > 16 * 1024 * 1024:
                    return False
                if len(source.read(token_length + 4)) != token_length + 4:
                    return False
            bpe_count_raw = source.read(4)
            if len(bpe_count_raw) != 4:
                return False
            bpe_count = int.from_bytes(bpe_count_raw, "little")
            if bpe_count == 0:
                return True
            first_record = source.read(8)
            if len(first_record) != 8:
                return False
            legacy_key_length = int.from_bytes(first_record[:4], "little")
            if legacy_key_length <= 4096 and legacy_key_length > 0:
                legacy_key = first_record[4:] + source.read(max(0, legacy_key_length - 4))
                if len(legacy_key) == legacy_key_length and all(32 <= byte < 127 for byte in legacy_key):
                    return False
            return True
    except (OSError, ValueError):
        return False


def _reformat_legacy_vocab(path: Path) -> bool:
    data = path.read_bytes()
    offset = 0

    def read_u32() -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise ValueError("A legacy tokenizer csonka")
        value = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        return value

    token_count = read_u32()
    if token_count == 0 or token_count > VOCAB_SIZE:
        return False
    tokens: Dict[bytes, int] = {}
    output = bytearray(struct.pack("<I", token_count))
    for _ in range(token_count):
        record_start = offset
        token_length = read_u32()
        if offset + token_length + 4 > len(data):
            return False
        token = data[offset : offset + token_length]
        offset += token_length
        token_id = read_u32()
        if token in tokens or token_id >= VOCAB_SIZE:
            return False
        tokens[token] = token_id
        output.extend(data[record_start:offset])

    bpe_count = read_u32()
    if bpe_count == 0:
        return False
    legacy_records: List[Tuple[bytes, int, int]] = []
    for _ in range(bpe_count):
        key_length = read_u32()
        if offset + key_length + 8 > len(data):
            return False
        key = data[offset : offset + key_length]
        offset += key_length
        token_id = read_u32()
        priority = read_u32()
        legacy_records.append((key, token_id, priority))

    output.extend(struct.pack("<I", bpe_count))
    for key, token_id, priority in legacy_records:
        pair_key: Optional[int] = None
        for split in range(1, len(key)):
            left_id = tokens.get(key[:split])
            right_id = tokens.get(key[split:])
            if left_id is not None and right_id is not None:
                pair_key = (left_id << 32) | right_id
                break
        if pair_key is None:
            return False
        output.extend(struct.pack("<QII", pair_key, token_id, priority))
    output.extend(data[offset:])

    temporary_path = path.with_name(path.name + ".reformatting")
    _safe_unlink(temporary_path)
    try:
        with temporary_path.open("wb") as target:
            target.write(output)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        _safe_unlink(temporary_path)
        raise
    return True


def _ensure_tokenizer_in_checkpoint() -> Dict[str, Any]:
    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        if _is_current_vocab_format(checkpoint_path):
            return {
                "source_path": str(checkpoint_path),
                "destination_path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "reused": True,
                "format": "numeric_pair_keys",
            }
        if _reformat_legacy_vocab(checkpoint_path) and _is_current_vocab_format(checkpoint_path):
            return {
                "source_path": str(checkpoint_path),
                "destination_path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "reused": True,
                "reformatted": True,
                "format": "numeric_pair_keys",
            }
        _log(f"Régi tokenizer-formátum elutasítva és törölve: {checkpoint_path}")
        checkpoint_path.unlink()

    if not TOKENIZER_SOURCE_PATH.is_file():
        return {
            "source_path": None,
            "destination_path": str(checkpoint_path),
            "bytes": 0,
            "reused": False,
            "format": "rebuild_required",
        }
    source_size = TOKENIZER_SOURCE_PATH.stat().st_size
    if source_size <= 0:
        raise RuntimeError(f"A projektben lévő tokenizer üres: {TOKENIZER_SOURCE_PATH}")
    if not _is_current_vocab_format(TOKENIZER_SOURCE_PATH):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TOKENIZER_SOURCE_PATH, checkpoint_path)
        if _reformat_legacy_vocab(checkpoint_path) and _is_current_vocab_format(checkpoint_path):
            return {
                "source_path": str(TOKENIZER_SOURCE_PATH),
                "destination_path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "reused": False,
                "reformatted": True,
                "format": "numeric_pair_keys",
            }
        checkpoint_path.unlink()
        raise RuntimeError(f"A legacy tokenizer átalakítása sikertelen: {TOKENIZER_SOURCE_PATH}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = checkpoint_path.with_name(checkpoint_path.name + ".uploading")
    _safe_unlink(temporary_destination)
    try:
        with TOKENIZER_SOURCE_PATH.open("rb") as source, temporary_destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if temporary_destination.stat().st_size != source_size:
            raise IOError(
                "A tokenizer feltöltése hiányos: "
                f"forrás={source_size} cél={temporary_destination.stat().st_size}"
            )
        os.replace(temporary_destination, checkpoint_path)
    except BaseException:
        _safe_unlink(temporary_destination)
        raise

    return {
        "source_path": str(TOKENIZER_SOURCE_PATH),
        "destination_path": str(checkpoint_path),
        "bytes": source_size,
        "reused": False,
        "format": "numeric_pair_keys",
    }


def _run_futhark_kernels(project_dir: str, env: Dict[str, str]) -> None:
    accel_dir = os.path.join(project_dir, "src", "hw", "accel")
    _log("Futhark csomagok szinkronizálása")
    _run(
        ["futhark", "pkg", "sync"],
        cwd=accel_dir,
        env=env,
        check=False,
        timeout=180,
    )
    _log("Futhark CPU könyvtár fordítása")
    _run(
        [
            "futhark",
            "c",
            "--library",
            os.path.join(accel_dir, "futhark_kernels.fut"),
            "-o",
            os.path.join(accel_dir, "futhark_kernels"),
        ],
        cwd=project_dir,
        env=env,
    )
    _log("Futhark CUDA könyvtár fordítása")
    _run(
        [
            "futhark",
            "cuda",
            "--library",
            os.path.join(accel_dir, "main.fut"),
            "-o",
            os.path.join(accel_dir, "main_gpu"),
        ],
        cwd=project_dir,
        env=env,
    )


def _parse_gpu_monitor(text: str) -> Dict[str, Any]:
    samples: Dict[int, Dict[str, List[float]]] = {}
    for row in csv.reader(text.splitlines()):
        if len(row) != 6:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        target = samples.setdefault(index, {
            "utilization_gpu_pct": [],
            "memory_used_mib": [],
            "memory_total_mib": [],
            "power_draw_w": [],
            "clock_sm_mhz": [],
        })
        for key, raw_value in zip(target, row[1:]):
            try:
                target[key].append(float(raw_value.strip()))
            except ValueError:
                pass
    result: Dict[str, Any] = {}
    for index, metrics in samples.items():
        result[str(index)] = {
            key: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for key, values in metrics.items()
            if values
        }
    return result


@app.function(
    image=image,
    cpu=(1.0, 2.0),
    memory=(4096, 8192),
    timeout=1800,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
        str(BUILD_MOUNT_PATH): build_volume,
    },
)
def inspect_prepared_state(run_id: int) -> Dict[str, Any]:
    data_volume.reload()
    checkpoint_volume.reload()
    build_volume.reload()

    source_fp = _source_fingerprint(PROJECT_MOUNT_PATH)
    token_dataset_path = Path(TOKEN_DATASET_PATH)
    dataset_meta = _read_token_dataset_meta(token_dataset_path)

    dataset_phase: Dict[str, Any] = {
        "dataset_path": str(token_dataset_path),
        "source_dataset_path": str(DATASET_PATH),
        "source": "modal_volume:jaide-bench-data",
        "separate_modal_run": True,
        "reused": True,
    }

    if dataset_meta is None:
        dataset_phase["sample_count"] = 0
        dataset_phase["dataset_bytes"] = 0
        dataset_phase["pretokenized"] = False
        _log(f"A token dataset hiányzik vagy érvénytelen: {token_dataset_path}")
    else:
        window_length = dataset_meta["maximum_sequence_length"] + 1
        if dataset_meta["total_tokens"] % window_length != 0:
            raise RuntimeError(
                f"A kész token dataset token-száma nem osztható a mintablokk méretével: {token_dataset_path}"
            )
        if dataset_meta["maximum_sequence_length"] != MAX_SEQ_LEN:
            raise RuntimeError(
                "A meglévő token dataset szekvenciahossza eltér a kért JAIDE_BENCH_MAX_SEQ_LEN értéktől: "
                f"{dataset_meta['maximum_sequence_length']} != {MAX_SEQ_LEN}"
            )
        if dataset_meta["vocabulary_size"] != VOCAB_SIZE:
            raise RuntimeError(
                "A meglévő token dataset szótármérete eltér a kért JAIDE_BENCH_VOCAB_SIZE értéktől: "
                f"{dataset_meta['vocabulary_size']} != {VOCAB_SIZE}"
            )
        dataset_phase["sample_count"] = dataset_meta["total_tokens"] // window_length
        dataset_phase["dataset_bytes"] = token_dataset_path.stat().st_size
        dataset_phase["pretokenized"] = True
        dataset_phase["maximum_sequence_length"] = dataset_meta["maximum_sequence_length"]
        dataset_phase["vocabulary_size"] = dataset_meta["vocabulary_size"]
        dataset_phase["total_tokens"] = dataset_meta["total_tokens"]
        _log(
            f"A token dataset ellenőrizve: {dataset_phase['sample_count']} minta, "
            f"{dataset_phase['dataset_bytes']} bájt"
        )

    distributed_binary = _find_latest_binary("jaide-distributed-futhark", source_fp)
    inference_binary = _find_latest_binary("jaide-inference-server", source_fp)

    if distributed_binary is None:
        _log("A jelenlegi forrás-ujjlenyomathoz tartozó disztributált bináris nem található")
    else:
        _log(f"Disztributált bináris megtalálva: {distributed_binary}")

    if inference_binary is None:
        _log("A jelenlegi forrás-ujjlenyomathoz tartozó inferencia bináris nem található")
    else:
        _log(f"Inferencia bináris megtalálva: {inference_binary}")

    vocab_path = Path(CHECKPOINT_PATH)
    vocabulary_present = vocab_path.is_file() and vocab_path.stat().st_size > 0

    return {
        "run_id": run_id,
        "source_fingerprint": source_fp,
        "distributed_binary_present": distributed_binary is not None,
        "inference_binary_present": inference_binary is not None,
        "distributed_binary_path": str(distributed_binary) if distributed_binary is not None else "",
        "inference_binary_path": str(inference_binary) if inference_binary is not None else "",
        "vocabulary_present": vocabulary_present,
        "vocabulary_path": str(vocab_path),
        "phases": {"C_prep_dataset": dataset_phase},
    }


@app.function(
    image=image,
    cpu=(CPU_REQUEST, CPU_LIMIT),
    memory=(MEMORY_REQUEST_MB, MEMORY_LIMIT_MB),
    timeout=CPU_TIMEOUT_SEC,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
        str(REPORT_MOUNT_PATH): report_volume,
        str(BUILD_MOUNT_PATH): build_volume,
    },
)
def prepare_cpu(run_id: int) -> Dict[str, Any]:
    build_volume.reload()
    data_volume.reload()
    checkpoint_volume.reload()

    project_dir = str(PROJECT_MOUNT_PATH)
    env = os.environ.copy()

    report_dir = REPORT_MOUNT_PATH / f"run_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "report_dir": str(report_dir),
        "phases": {},
    }
    token_dataset_path = Path(TOKEN_DATASET_PATH)

    existing_dataset_meta = _read_token_dataset_meta(token_dataset_path)
    if existing_dataset_meta is not None:
        window_length = existing_dataset_meta["maximum_sequence_length"] + 1
        if existing_dataset_meta["total_tokens"] % window_length != 0:
            raise RuntimeError(
                f"A kész token dataset token-száma nem osztható a mintablokk méretével: {token_dataset_path}"
            )
        sample_count = existing_dataset_meta["total_tokens"] // window_length
        result["distributed_binary_present"] = False
        result["inference_binary_present"] = False
        result["phases"]["C_prep_dataset"] = {
            "duration_s": 0.0,
            "sample_count": sample_count,
            "dataset_bytes": token_dataset_path.stat().st_size,
            "dataset_path": str(token_dataset_path),
            "source_dataset_path": str(DATASET_PATH),
            "source": "modal_volume:jaide-bench-data",
            "pretokenized": True,
            "reused": True,
        }
        _log(
            f"A végleges bináris dataset már létezik, újrahasználva: "
            f"{sample_count} minta, {token_dataset_path.stat().st_size} bájt"
        )
        report_volume.commit()
        return result

    _log("=" * 70)
    _log(f"CPU ELŐKÉSZÍTÉSI FÁZIS run_id={run_id}")
    _log("=" * 70)

    tokenizer_checkpoint = _ensure_tokenizer_in_checkpoint()
    checkpoint_volume.commit()
    result["tokenizer_checkpoint"] = tokenizer_checkpoint
    if tokenizer_checkpoint["reused"]:
        _log(
            f"Modal checkpoint tokenizer használata: {tokenizer_checkpoint['destination_path']} "
            f"({tokenizer_checkpoint['bytes']} bájt, meglévő fájl)"
        )
    elif tokenizer_checkpoint["format"] == "rebuild_required":
        _log(
            "A tokenizer régi vagy hiányzó formátumú; a GPU folyamat a vocabot "
            "újraépíti a benchmark-adatkészletből"
        )
    else:
        _log(
            f"Tokenizer feltöltve a Modal checkpoint kötetbe: "
            f"{tokenizer_checkpoint['source_path']} -> {tokenizer_checkpoint['destination_path']} "
            f"({tokenizer_checkpoint['bytes']} bájt)"
        )

    dataset_path = Path(DATASET_PATH)
    if not dataset_path.is_file() or dataset_path.stat().st_size <= 0:
        raise RuntimeError(
            "A Modal jaide-bench-data volume datasetje hiányzik vagy üres: "
            f"{dataset_path}. A benchmark nem tölt le másik adathalmazt."
        )
    _log(f"Modal volume dataset használata: {dataset_path} ({dataset_path.stat().st_size} bájt)")
    source_fp = _source_fingerprint(Path(project_dir))
    result["source_fingerprint"] = source_fp
    existing_dist = _find_latest_binary("jaide-distributed-futhark", source_fp)
    existing_inf = _find_latest_binary("jaide-inference-server", source_fp)
    cached_binaries_reused = False
    if not FORCE_REBUILD and existing_dist and existing_inf:
        _log(f"Legfrissebb lefordított bináris megtalálva: {existing_dist}")
        _log(f"Legfrissebb inferencia bináris megtalálva: {existing_inf}")
        _log("A GPU binárisok újrafordítása átugorva; csak a dataset készül elő.")
        build_target_dir = BUILD_MOUNT_PATH / f"run_{run_id}"
        build_target_dir.mkdir(parents=True, exist_ok=True)
        dist_dst = build_target_dir / "jaide-distributed-futhark"
        inf_dst = build_target_dir / "jaide-inference-server"
        shutil.copy2(str(existing_dist), str(dist_dst))
        os.chmod(str(dist_dst), 0o755)
        _write_fingerprint(dist_dst, source_fp)
        shutil.copy2(str(existing_inf), str(inf_dst))
        os.chmod(str(inf_dst), 0o755)
        _write_fingerprint(inf_dst, source_fp)

        result["distributed_binary_present"] = True
        result["inference_binary_present"] = True
        cached_binaries_reused = True
        result["phases"]["B_gpu_build"] = {
            "returncode": 0,
            "inference_build_returncode": 0,
            "distributed_build_returncode": 0,
            "duration_s": 0.0,
            "reused_binary": str(existing_dist),
        }
    if not cached_binaries_reused:
        _log("A prep nem fordít GPU binárisokat; csak az előtokenizált dataset készül.")
        result["distributed_binary_present"] = False
        result["inference_binary_present"] = False

    inference_bin = Path(project_dir) / "zig-out" / "bin" / "jaide-inference-server"
    distributed_bin = Path(project_dir) / "zig-out" / "bin" / "jaide-distributed-futhark"

    build_target_dir = BUILD_MOUNT_PATH / f"run_{run_id}"
    build_target_dir.mkdir(parents=True, exist_ok=True)

    if distributed_bin.exists():
        dist_dst = build_target_dir / "jaide-distributed-futhark"
        shutil.copy2(str(distributed_bin), str(dist_dst))
        os.chmod(str(dist_dst), 0o755)
        _write_fingerprint(dist_dst, source_fp)
        result["distributed_binary_present"] = True
    else:
        result["distributed_binary_present"] = False
        _log(f"FIGYELMEZTETÉS: a disztributált bináris nem épült fel itt: {distributed_bin}")

    if inference_bin.exists():
        inf_dst = build_target_dir / "jaide-inference-server"
        shutil.copy2(str(inference_bin), str(inf_dst))
        os.chmod(str(inf_dst), 0o755)
        _write_fingerprint(inf_dst, source_fp)
        result["inference_binary_present"] = True
    else:
        result["inference_binary_present"] = False
        _log(f"FIGYELMEZTETÉS: az inferencia bináris nem épült fel itt: {inference_bin}")

    _log("=" * 70)
    _log(f"C-előkészítő fázis: {SAMPLE_CAP} minta előtokenizálása GPU tréninghez")
    _log("=" * 70)
    token_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    pretokenize_bin = Path(project_dir) / "zig-out" / "bin" / "jaide-pretokenize"
    if not pretokenize_bin.is_file():
        _run(
            ["zig", "build", "pretokenize", "-Doptimize=ReleaseSafe", "-Dskip-futhark=true"],
            cwd=project_dir,
            env=env,
            timeout=CPU_TIMEOUT_SEC,
        )
    if not pretokenize_bin.is_file():
        raise RuntimeError(f"Az előtokenizáló bináris hiányzik: {pretokenize_bin}")
    _safe_unlink(token_dataset_path)
    _safe_unlink(token_dataset_path.with_name(token_dataset_path.name + ".tmp"))
    pretokenize_tmp = token_dataset_path.with_name(token_dataset_path.name + ".tmp")
    vocab_path = Path(CHECKPOINT_PATH)
    if not vocab_path.is_file() or vocab_path.stat().st_size <= 0:
        raise RuntimeError(f"A Modal checkpoint tokenizer hiányzik vagy üres: {vocab_path}")
    sample_count, size, prep_duration = _parallel_pretokenize(
        pretokenize_bin=pretokenize_bin,
        input_path=dataset_path,
        output_tmp=pretokenize_tmp,
        vocab_path=vocab_path,
        max_seq_len=MAX_SEQ_LEN,
        sample_cap=SAMPLE_CAP,
        project_dir=project_dir,
        env=env,
        timeout=CPU_TIMEOUT_SEC,
    )
    pretokenize_tmp.replace(token_dataset_path)
    result["phases"]["C_prep_dataset"] = {
        "duration_s": round(prep_duration, 2),
        "sample_count": sample_count,
        "dataset_bytes": size,
        "dataset_path": str(token_dataset_path),
        "source_dataset_path": str(dataset_path),
        "source": "modal_volume:jaide-bench-data",
        "pretokenized": True,
    }
    _log(f"Modal volume dataset készen áll: {sample_count} minta, {size} bájt")

    data_volume.commit()
    checkpoint_volume.commit()
    build_volume.commit()
    report_volume.commit()

    _log("=" * 70)
    _log("CPU ELŐKÉSZÍTÉSI FÁZIS BEFEJEZŐDÖTT")
    _log("=" * 70)

    return result


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=(CPU_REQUEST, CPU_LIMIT),
    memory=(MEMORY_REQUEST_MB, MEMORY_LIMIT_MB),
    timeout=TIMEOUT_SEC,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
        str(REPORT_MOUNT_PATH): report_volume,
        str(BUILD_MOUNT_PATH): build_volume,
    },
)
def run_gpu_train_and_infer(
    run_id: int,
    prep_result: Dict[str, Any],
) -> Dict[str, Any]:
    project_dir = str(PROJECT_MOUNT_PATH)
    env = os.environ.copy()

    build_volume.reload()
    data_volume.reload()
    checkpoint_volume.reload()

    report_dir = REPORT_MOUNT_PATH / f"run_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "gpu_spec": GPU_SPEC,
        "model_dim": MODEL_DIM,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "sample_cap": SAMPLE_CAP,
        "max_seq_len": MAX_SEQ_LEN,
        "learning_rate": LEARNING_RATE,
        "vocab_size": VOCAB_SIZE,
        "spectral_norm_target": SPECTRAL_NORM_TARGET,
        "spectral_power_iterations": SPECTRAL_POWER_ITERATIONS,
        "seed_offset": SEED_OFFSET,
        "grad_mean": GRAD_MEAN,
        "normalized_gradient_flow": NORMALIZED_GRADIENT_FLOW,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "sfd_trust_ratio": SFD_TRUST_RATIO,
        "sfd_weight_floor": SFD_WEIGHT_FLOOR,
        "spectral_interval": SPECTRAL_INTERVAL,
        "logdet_weight": LOGDET_WEIGHT,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_version": CHECKPOINT_VERSION,
        "reasoning_cycles": REASONING_CYCLES,
        "relational_pass_interval": RELATIONAL_PASS_INTERVAL,
        "phases": {},
    }

    _log("=" * 70)
    _log(f"GPU FÁZIS INDÍTÁSA gpu={GPU_SPEC} run_id={run_id}")
    _log("=" * 70)
    gpu_phase_start = time.time()

    _run(["nvidia-smi"], cwd=project_dir, env=env, check=False, timeout=30)
    _run(["lscpu"], cwd=project_dir, env=env, check=False, timeout=10)

    build_source_dir = BUILD_MOUNT_PATH / f"run_{run_id}"
    distributed_bin_src = build_source_dir / "jaide-distributed-futhark"
    inference_bin_src = build_source_dir / "jaide-inference-server"

    source_fp = _source_fingerprint(Path(project_dir))
    result["source_fingerprint"] = source_fp
    if not distributed_bin_src.exists():
        latest_dist = _find_latest_binary("jaide-distributed-futhark", source_fp)
        if latest_dist:
            distributed_bin_src = latest_dist

    if not inference_bin_src.exists():
        latest_inf = _find_latest_binary("jaide-inference-server", source_fp)
        if latest_inf:
            inference_bin_src = latest_inf

    distributed_bin = Path("/tmp/jaide-distributed-futhark")
    inference_bin = Path("/tmp/jaide-inference-server")

    if distributed_bin_src and distributed_bin_src.exists():
        shutil.copy2(str(distributed_bin_src), str(distributed_bin))
        os.chmod(str(distributed_bin), 0o755)
        _log(f"Disztributált bináris sikeresen betöltve: {distributed_bin_src} -> {distributed_bin}")
    else:
        _log("HIBA: A disztributált bináris sehol sem található a perzisztens tárolóban")

    if inference_bin_src and inference_bin_src.exists():
        shutil.copy2(str(inference_bin_src), str(inference_bin))
        os.chmod(str(inference_bin), 0o755)
        _log(f"Inferenciaciklus binárisa sikeresen betöltve: {inference_bin_src} -> {inference_bin}")
    else:
        _log("FIGYELMEZTETÉS: Az inferenciaciklus binárisa hiányzik")

    dataset_meta = prep_result.get("phases", {}).get("C_prep_dataset", {})
    dataset_path = dataset_meta.get("dataset_path") or DATASET_PATH
    sample_count = int(dataset_meta.get("sample_count", 0) or 0)
    token_dataset_file = Path(dataset_path)
    if (
        not token_dataset_file.is_file()
        or token_dataset_file.stat().st_size < 64
        or (token_dataset_file.stat().st_size - 64) % 4 != 0
    ):
        raise RuntimeError(
            "A külön dataset-előkészítő Modal futás nem fejeződött be sikeresen: "
            f"érvénytelen vagy hiányzó token dataset: {token_dataset_file}"
        )
    if sample_count <= 0 and token_dataset_file.exists():
        sample_count = _count_nonempty_lines(Path(dataset_path))

    training_succeeded = False
    training_started_ns = 0

    if not distributed_bin.exists():
        result["phases"]["C_training_convergence"] = {"skipped": "A disztributált bináris hiányzik"}
    elif not dataset_path or not Path(dataset_path).exists() or sample_count <= 0:
        result["phases"]["C_training_convergence"] = {"skipped": "Az adathalmaz nincs előkészítve"}
    else:
        _log("=" * 70)
        _log(f"C FÁZIS: BETANÍTÁS ({sample_count} minta, {EPOCHS} epocha, dim={MODEL_DIM})")
        _log("=" * 70)

        train_env = env.copy()
        train_env["WORLD_SIZE"] = str(NUM_GPUS)
        train_env["MASTER_ADDR"] = MASTER_ADDR
        train_env["MASTER_PORT"] = MASTER_PORT
        train_env["JAIDE_EPOCHS"] = str(EPOCHS)
        train_env["JAIDE_DATASET"] = str(dataset_path)
        train_env["JAIDE_MODEL_DIM"] = str(MODEL_DIM)
        train_env["JAIDE_LAYERS"] = str(NUM_LAYERS)
        train_env["JAIDE_BATCH_SIZE"] = str(BATCH_SIZE)
        nccl_id_path = f"/tmp/jaide_nccl_id_{run_id}"
        train_env["JAIDE_NCCL_ID_PATH"] = nccl_id_path
        train_env["JAIDE_TOTAL_SAMPLES"] = str(sample_count)
        train_env["JAIDE_MAX_SAMPLES"] = str(min(sample_count, SAMPLE_CAP))
        train_env["JAIDE_MAX_SEQ_LEN"] = str(MAX_SEQ_LEN)
        train_env["JAIDE_LEARNING_RATE"] = LEARNING_RATE
        train_env["JAIDE_VOCAB_SIZE"] = str(VOCAB_SIZE)
        train_env["JAIDE_TOKENIZER_VOCAB"] = CHECKPOINT_PATH
        train_env["JAIDE_SPECTRAL_NORM_TARGET"] = SPECTRAL_NORM_TARGET
        train_env["JAIDE_SPECTRAL_POWER_ITERATIONS"] = str(SPECTRAL_POWER_ITERATIONS)
        train_env["JAIDE_SEED_OFFSET"] = str(SEED_OFFSET)
        train_env["JAIDE_GRAD_MEAN"] = GRAD_MEAN
        train_env["JAIDE_NORMALIZED_GRADIENT_FLOW"] = NORMALIZED_GRADIENT_FLOW
        train_env["JAIDE_GRADIENT_CLIP_NORM"] = GRADIENT_CLIP_NORM
        train_env["JAIDE_SFD_TRUST_RATIO"] = SFD_TRUST_RATIO
        train_env["JAIDE_SFD_WEIGHT_FLOOR"] = SFD_WEIGHT_FLOOR
        train_env["JAIDE_SPECTRAL_INTERVAL"] = str(SPECTRAL_INTERVAL)
        train_env["JAIDE_LOGDET_WEIGHT"] = LOGDET_WEIGHT
        train_env["JAIDE_CLIP_MIN"] = CLIP_MIN
        train_env["JAIDE_CLIP_MAX"] = CLIP_MAX
        train_env["JAIDE_CHECKPOINT_VERSION"] = str(CHECKPOINT_VERSION)
        train_env["JAIDE_CHECKPOINT_INTERVAL_EPOCHS"] = str(CHECKPOINT_INTERVAL_EPOCHS)
        if RESUME_CHECKPOINT:
            train_env["JAIDE_RESUME_CHECKPOINT"] = RESUME_CHECKPOINT
        train_env["JAIDE_TOKENIZER_LANGUAGE"] = "english"
        train_env["JAIDE_REASONING_CYCLES"] = str(REASONING_CYCLES)
        train_env["JAIDE_RELATIONAL_PASS_INTERVAL"] = str(RELATIONAL_PASS_INTERVAL)
        train_env["JAIDE_RECONSTRUCTION_ALPHA"] = RECONSTRUCTION_ALPHA
        train_env["JAIDE_PHASE_A_STEPS"] = str(PHASE_A_STEPS)
        train_env["JAIDE_PHASE_B_STEPS"] = str(PHASE_B_STEPS)
        train_env["JAIDE_SHUFFLE_TARGET_CONTROL"] = SHUFFLE_TARGET_CONTROL
        train_env["JAIDE_TARGET_SOURCE_FROZEN"] = TARGET_SOURCE_FROZEN
        train_env["JAIDE_SPECTRAL_DEPTH_COMPENSATION"] = SPECTRAL_DEPTH_COMPENSATION
        train_env["JAIDE_SKIP_KNOWLEDGE_GRAPH"] = SKIP_KNOWLEDGE_GRAPH
        train_env["JAIDE_GRAPH_CHUNK_SIZE"] = GRAPH_CHUNK_SIZE
        train_env["JAIDE_INIT_SPECTRAL_ITERS"] = INIT_SPECTRAL_ITERS
        train_env["JAIDE_FUTHARK_UNIFIED_MEMORY"] = "0"
        vocab_file = Path(CHECKPOINT_PATH)
        if vocab_file.is_file() and vocab_file.stat().st_size > 0:
            train_env["JAIDE_VOCAB_READY"] = "1"
            _log(
                f"Meglévő szótár megtalálva itt: {vocab_file} ({vocab_file.stat().st_size} bájt), a BPE tanítás átugorva (JAIDE_VOCAB_READY=1)"
            )
        else:
            train_env.pop("JAIDE_VOCAB_READY", None)
            _log(f"Nincs meglévő szótár itt: {vocab_file}, a BPE szótárépítés a 0. rangú folyamaton fog lefutni")
        train_env["NCCL_DEBUG"] = NCCL_DEBUG
        train_env["NCCL_IB_DISABLE"] = NCCL_IB_DISABLE
        train_env["NCCL_SOCKET_IFNAME"] = NCCL_SOCKET_IFNAME
        train_env["NCCL_P2P_DISABLE"] = "0"
        train_env["NCCL_SHM_DISABLE"] = "0"
        train_env["NCCL_NVLS_ENABLE"] = "0"
        train_env["CUDA_DEVICE_ORDER"] = CUDA_DEVICE_ORDER
        train_env["JAIDE_RELATIONAL_FAST"] = JAIDE_RELATIONAL_FAST
        cache_hasher = hashlib.sha256()
        for relative in (
            "src/hw/accel/main.fut",
            "src/hw/accel/futhark_kernels.fut",
            "src/hw/accel/futhark_abi_check.c",
        ):
            cache_hasher.update((PROJECT_MOUNT_PATH / relative).read_bytes())
        cache_hasher.update(b"futhark-0.26.4-cuda-sm100")
        cache_hasher.update(source_fp.encode("utf-8"))
        futhark_cache_path = CHECKPOINT_MOUNT_PATH / f"futhark_gpu_cache_{cache_hasher.hexdigest()[:20]}.bin"
        train_env["JAIDE_FUTHARK_CACHE"] = str(futhark_cache_path)

        _clear_rank_coordination_files(nccl_id_path)

        training_command = [str(distributed_bin)]
        if NCU_ENABLE:
            ncu_path = shutil.which("ncu")
            if not ncu_path:
                raise RuntimeError("JAIDE_BENCH_NCU=1 de az ncu eszköz nem található a rendszerben")
            training_command = [
                ncu_path,
                "--target-processes",
                "all",
                "--csv",
                "--page",
                "raw",
                "--launch-count",
                "20",
                "--metrics",
                "sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum",
                str(distributed_bin),
            ]
        monitor_path = Path("/tmp") / f"jaide_gpu_monitor_{run_id}.csv"
        monitor_file = monitor_path.open("w", encoding="utf-8")
        monitor_process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
                "-lms",
                "500",
            ],
            stdout=monitor_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        t0 = time.time()
        training_started_ns = time.time_ns()
        watchdog_error: Optional[str] = None
        try:
            rc_c, out_c, _ = _run_multirank(
                cmd=training_command,
                cwd=project_dir,
                base_env=train_env,
                num_gpus=NUM_GPUS,
                nccl_id_path=nccl_id_path,
                timeout=72000,
                init_deadline_sec=INIT_DEADLINE_SEC,
                idle_watchdog_sec=IDLE_WATCHDOG_SEC,
            )
        except RuntimeError as exc:
            rc_c = 1
            out_c = str(exc)
            watchdog_error = str(exc)
            _log(f"Betanítási watchdog: {exc}")
        finally:
            _terminate_process_group(monitor_process)
            monitor_process.wait()
            monitor_file.close()
        phase_c_duration = time.time() - t0
        training_succeeded = rc_c == 0
        training_failure_class = _classify_training_failure(out_c, watchdog_error is not None, "[Step " in out_c)
        gpu_monitor_text = monitor_path.read_text(encoding="utf-8") if monitor_path.exists() else ""
        gpu_monitor_metrics = _parse_gpu_monitor(gpu_monitor_text)
        _write_report(report_dir, "phase_c_gpu_monitor.csv", gpu_monitor_text)

        loss_curve: List[Tuple[int, float]] = []
        recon_curve: List[Tuple[int, float]] = []
        source_rms_curve: List[Tuple[int, float]] = []
        epoch_metrics: List[Dict[str, Any]] = []
        timing_keys = (
            "dataset_ms",
            "tokenizer_ms",
            "model_compile_initialization_ms",
            "graph_ms",
            "startup_total_ms",
            "spectral_ms",
            "relational_ms",
            "reduction_update_ms",
            "capture_ms",
            "write_ms",
            "step_total_ms",
        )
        timing_samples: Dict[str, List[int]] = {key: [] for key in timing_keys}
        throughput_samples: List[Tuple[int, int]] = []
        for line in out_c.splitlines():
            for timing_key in timing_keys:
                marker = timing_key + "="
                if marker in line:
                    try:
                        timing_samples[timing_key].append(int(line.split(marker, 1)[1].split()[0]))
                    except (ValueError, IndexError):
                        pass
            if "global_tokens=" in line and "step_total_ms=" in line:
                try:
                    token_count = int(line.split("global_tokens=", 1)[1].split()[0])
                    step_time_ms = int(line.split("step_total_ms=", 1)[1].split()[0])
                    if token_count > 0 and step_time_ms > 0:
                        throughput_samples.append((token_count, step_time_ms))
                except (ValueError, IndexError):
                    pass
            if "[Step " in line and "Loss:" in line:
                try:
                    s_part = line.split("[Step ")[1].split("]")[0].strip()
                    step_index = int(s_part)
                    l_part = line.split("Loss:")[1].strip().split()[0]
                    loss_curve.append((step_index, float(l_part)))
                except (ValueError, IndexError):
                    continue
                if "Recon:" in line:
                    try:
                        r_part = line.split("Recon:")[1].strip().split()[0]
                        recon_curve.append((step_index, float(r_part)))
                    except (ValueError, IndexError):
                        pass
                if "SourceRMS:" in line:
                    try:
                        rms_part = line.split("SourceRMS:")[1].strip().split()[0]
                        source_rms_curve.append((step_index, float(rms_part)))
                    except (ValueError, IndexError):
                        pass
            if "[Epoch " in line and "Loss:" in line and "Time:" in line:
                try:
                    epoch_line = line.split("[Epoch ", 1)[1]
                    after_bracket = epoch_line.split("]", 1)[1]
                    loss_str = after_bracket.split("Loss:")[1].split("|")[0].strip()
                    time_str = after_bracket.split("Time:")[1].strip().rstrip("s")
                    epoch_metrics.append(
                        {
                            "loss": float(loss_str),
                            "time_s": float(time_str),
                        }
                    )
                except (ValueError, IndexError):
                    pass

        metrics_path = CHECKPOINT_MOUNT_PATH / "training_metrics.json"
        training_metrics_json: Optional[Dict[str, Any]] = None
        if metrics_path.exists():
            try:
                metrics_text = metrics_path.read_text(encoding="utf-8")
                training_metrics_json = json.loads(metrics_text)
                _write_report(report_dir, "training_metrics.json", metrics_text)
            except (json.JSONDecodeError, OSError):
                pass

        result["phases"]["C_training_convergence"] = {
            "returncode": rc_c,
            "duration_s": round(phase_c_duration, 2),
            "sample_count": sample_count,
            "loss_curve_length": len(loss_curve),
            "first_loss": loss_curve[0][1] if loss_curve else None,
            "last_loss": loss_curve[-1][1] if loss_curve else None,
            "recon_curve_length": len(recon_curve),
            "first_recon": recon_curve[0][1] if recon_curve else None,
            "last_recon": recon_curve[-1][1] if recon_curve else None,
            "recon_converged": (len(recon_curve) >= 2 and recon_curve[-1][1] < recon_curve[0][1]) if recon_curve else False,
            "first_source_rms": source_rms_curve[0][1] if source_rms_curve else None,
            "last_source_rms": source_rms_curve[-1][1] if source_rms_curve else None,
            "source_collapse_suspected": (
                len(source_rms_curve) >= 2
                and source_rms_curve[0][1] > 0.0
                and source_rms_curve[-1][1] < source_rms_curve[0][1] * 0.5
            ) if source_rms_curve else False,
            "num_gpus": NUM_GPUS,
            "reconstruction_alpha": RECONSTRUCTION_ALPHA,
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "shuffle_target_control": SHUFFLE_TARGET_CONTROL,
            "effective_batch_size": BATCH_SIZE * NUM_GPUS,
            "epoch_metrics": epoch_metrics,
            "training_metrics_json": training_metrics_json,
            "gpu_monitor": gpu_monitor_metrics,
            "gpu_telemetry_available": bool(gpu_monitor_metrics),
            "gpu_memory_telemetry_available": bool(gpu_monitor_metrics) and all(
                "memory_used_mib" in metrics and metrics["memory_used_mib"].get("count", 0) > 0
                for metrics in gpu_monitor_metrics.values()
            ),
            "ncu_enabled": NCU_ENABLE,
            "ncu_occupancy_metric_present": ("sm__warps_active.avg.pct_of_peak_sustained_active" in out_c) if NCU_ENABLE else None,
            "sampled_tokens_per_second": (
                sum(tokens for tokens, _ in throughput_samples) * 1000.0 /
                sum(milliseconds for _, milliseconds in throughput_samples)
            ) if throughput_samples else None,
            "timing_ms": {
                key: {
                    "count": len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "mean": (sum(values) / len(values)) if values else None,
                }
                for key, values in timing_samples.items()
            },
            "converged": (len(loss_curve) >= 2 and loss_curve[-1][1] < loss_curve[0][1]) if loss_curve else False,
            "failure_class": training_failure_class,
            "watchdog_error": watchdog_error,
            "reached_step_one": any(step == 1 for step, _ in loss_curve),
        }
        _write_report(report_dir, "phase_c_training.log", out_c)
        _write_report(
            report_dir,
            "phase_c_loss_curve.jsonl",
            "\n".join(json.dumps({"step": s, "loss": l}) for s, l in loss_curve),
        )
        _write_report(
            report_dir,
            "phase_c_recon_curve.jsonl",
            "\n".join(json.dumps({"step": s, "recon": r}) for s, r in recon_curve),
        )
        _write_report(
            report_dir,
            "phase_c_source_rms_curve.jsonl",
            "\n".join(json.dumps({"step": s, "source_rms": v}) for s, v in source_rms_curve),
        )
        checkpoint_volume.commit()

    if not inference_bin.exists():
        result["phases"]["D_inference"] = {"skipped": "Az inferencia bináris hiányzik"}
    elif not training_succeeded:
        result["phases"]["D_inference"] = {
            "skipped": "A betanítás nem fejeződött be sikeresen; a régi checkpoint tesztelése elutasítva",
            "server_up": False,
        }
    else:
        _log("=" * 70)
        _log("D FÁZIS: INFERENCIA SZERVER FÜSTTESZT")
        _log("=" * 70)

        model_candidates: List[Path] = []
        for candidate in CHECKPOINT_MOUNT_PATH.rglob("model.ckpt"):
            try:
                if candidate.stat().st_mtime_ns >= training_started_ns:
                    model_candidates.append(candidate)
            except OSError:
                continue
        model_candidates.sort(key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)), reverse=True)
        model_path = str(model_candidates[0]) if model_candidates else None
        _log(f"Kiválasztott friss modellfájl: {model_path}")

        if not model_path:
            result["phases"]["D_inference"] = {
                "error": "A betanítás lefutott, de nem generált friss model.ckpt fájlt",
                "server_up": False,
            }
        else:
            inf_env = env.copy()
            inf_env["JAIDE_MODEL_PATH"] = model_path
            inf_env.setdefault("NCCL_DEBUG", "WARN")

            srv_log_path = report_dir / "phase_d_server.log"
            with open(srv_log_path, "w", encoding="utf-8") as srv_f:
                srv_proc = subprocess.Popen(
                    [str(inference_bin), "--port", "8080", "--host", "127.0.0.1", "--allow-anonymous"],
                    cwd=project_dir,
                    env=inf_env,
                    stdout=srv_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                try:
                    server_up = False
                    health_json = ""
                    health: Optional[Dict[str, Any]] = None
                    startup_attempts = max(1, INFERENCE_STARTUP_TIMEOUT_SEC * 2)
                    for _ in range(startup_attempts):
                        time.sleep(0.5)
                        rc_h, out_h, _ = _run(
                            [
                                "curl",
                                "-sS",
                                "-o",
                                "/tmp/health.json",
                                "-w",
                                "%{http_code}",
                                "http://127.0.0.1:8080/v1/health",
                            ],
                            cwd=project_dir,
                            env=inf_env,
                            check=False,
                            timeout=10,
                        )
                        if rc_h != 0 or out_h.strip() != "200" or not Path("/tmp/health.json").exists():
                            continue
                        health_json = Path("/tmp/health.json").read_text(
                            encoding="utf-8", errors="replace"
                        )
                        try:
                            parsed_health = json.loads(health_json)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(parsed_health, dict):
                            continue
                        health = parsed_health
                        if health.get("status") == "healthy" and health.get("model_loaded") is True:
                            server_up = True
                            break

                    if not server_up:
                        try:
                            server_log = srv_log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                        except OSError:
                            server_log = ""
                        result["phases"]["D_inference"] = {
                            "error": "Az állapot-végpont nem jelzett betöltött modellt",
                            "server_up": False,
                            "health": health_json,
                            "server_log_tail": server_log,
                            "model_path": model_path,
                        }
                    else:
                        _log(f"Szerver állapot rendben: {health_json}")

                        prompt = "The reversible sparse flow model demonstrates"
                        req_body = json.dumps({"text": prompt, "max_tokens": 20})
                        inference_response_path = Path("/tmp/inference.json")
                        _safe_unlink(inference_response_path)
                        t0 = time.time()
                        rc_i, out_i, _ = _run(
                            [
                                "curl",
                                "-sS",
                                "-o",
                                str(inference_response_path),
                                "-w",
                                "%{http_code}",
                                "-X",
                                "POST",
                                "-H",
                                "Content-Type: application/json",
                                "-d",
                                req_body,
                                "http://127.0.0.1:8080/v1/inference",
                            ],
                            cwd=project_dir,
                            env=inf_env,
                            check=False,
                            timeout=60,
                        )
                        inference_duration = time.time() - t0
                        response_body = (
                            inference_response_path.read_text(encoding="utf-8", errors="replace")
                            if inference_response_path.exists()
                            else ""
                        )

                        parsed: Optional[Dict[str, Any]] = None
                        try:
                            parsed_candidate = json.loads(response_body)
                            if isinstance(parsed_candidate, dict):
                                parsed = parsed_candidate
                        except json.JSONDecodeError:
                            pass

                        generated_tokens: List[int] = []
                        generated_text_value = ""
                        if isinstance(parsed, dict):
                            raw_tokens = parsed.get("tokens")
                            if isinstance(raw_tokens, list):
                                generated_tokens = [t for t in raw_tokens if isinstance(t, int)]
                            raw_text = parsed.get("text")
                            if isinstance(raw_text, str):
                                generated_text_value = raw_text

                        distinct_tokens = len(set(generated_tokens))
                        non_reserved = [t for t in generated_tokens if t >= 4]
                        inference_http_status = out_i.strip()
                        smoke_ok = rc_i == 0 and inference_http_status == "200" and parsed is not None

                        result["phases"]["D_inference"] = {
                            "returncode": rc_i,
                            "http_status": inference_http_status,
                            "duration_s": round(inference_duration, 2),
                            "health": health_json,
                            "prompt": prompt,
                            "response_body": response_body,
                            "response_parsed": parsed,
                            "server_up": True,
                            "model_path": model_path,
                            "generated_token_count": len(generated_tokens),
                            "generated_distinct_tokens": distinct_tokens,
                            "generated_non_reserved_count": len(non_reserved),
                            "generated_text_length": len(generated_text_value),
                            "generation_produced_output": len(generated_tokens) > 0,
                            "generation_is_degenerate": len(generated_tokens) > 1 and distinct_tokens <= 1,
                            "smoke_passed": smoke_ok,
                        }
                        if not smoke_ok:
                            result["phases"]["D_inference"]["error"] = "Az inferencia végpont nem adott érvényes HTTP 200 JSON választ"
                        _write_report(report_dir, "phase_d_inference.log", response_body)
                finally:
                    _terminate_process_group(srv_proc)

    training_phase = result.get("phases", {}).get("C_training_convergence", {})
    inference_phase = result.get("phases", {}).get("D_inference", {})
    result["verified_success"] = (
        training_phase.get("returncode") == 0
        and training_phase.get("gpu_telemetry_available") is True
        and training_phase.get("gpu_memory_telemetry_available") is True
        and training_phase.get("sampled_tokens_per_second") is not None
        and inference_phase.get("smoke_passed") is True
        and (not NCU_ENABLE or training_phase.get("ncu_occupancy_metric_present") is True)
    )
    gpu_phase_duration = time.time() - gpu_phase_start
    result["gpu_phase_duration_s"] = round(gpu_phase_duration, 2)
    _log("=" * 70)
    _log(f"GPU FÁZIS BEFEJEZŐDÖTT (időtartam={gpu_phase_duration:.2f}s)")
    _log("=" * 70)

    summary_json = json.dumps(result, indent=2, default=str)
    _write_report(report_dir, "gpu_phase_summary.json", summary_json)
    report_volume.commit()

    return result


@app.local_entrypoint()
def main() -> None:
    if RUN_MODE not in {"all", "prep", "train", "build"}:
        raise ValueError("A JAIDE_BENCH_RUN_MODE értéke csak all, prep, train vagy build lehet")
    if MODEL_DIM <= 0 or MODEL_DIM % 2 != 0:
        raise ValueError("A JAIDE_BENCH_MODEL_DIM értékének pozitív páros számnak kell lennie")
    if NUM_LAYERS <= 0:
        raise ValueError("A JAIDE_BENCH_LAYERS értékének pozitívnak kell lennie")
    if BATCH_SIZE <= 0:
        raise ValueError("A JAIDE_BENCH_BATCH értékének pozitívnak kell lennie")
    if EPOCHS <= 0:
        raise ValueError("A JAIDE_BENCH_EPOCHS értékének pozitívnak kell lennie")
    if SAMPLE_CAP <= 0:
        raise ValueError("A JAIDE_BENCH_SAMPLE_CAP értékének pozitívnak kell lennie")
    if MAX_SEQ_LEN <= 0:
        raise ValueError("A JAIDE_BENCH_MAX_SEQ_LEN értékének pozitívnak kell lennie")
    if REASONING_CYCLES <= 0:
        raise ValueError("A JAIDE_BENCH_REASONING_CYCLES értékének pozitívnak kell lennie")
    if RELATIONAL_PASS_INTERVAL <= 0:
        raise ValueError("A JAIDE_BENCH_RELATIONAL_PASS_INTERVAL értékének pozitívnak kell lennie")
    if NUM_GPUS <= 0:
        raise ValueError("A JAIDE_BENCH_NUM_GPUS értékének pozitív egész számnak kell lennie")
    if NUM_GPUS != ALLOCATED_GPU_COUNT:
        raise ValueError("A JAIDE_BENCH_NUM_GPUS értékének egyeznie kell a megadott GPU számmal")
    if CHECKPOINT_INTERVAL_EPOCHS < 0:
        raise ValueError("A JAIDE_BENCH_CHECKPOINT_INTERVAL_EPOCHS nem lehet negatív")
    if INFERENCE_STARTUP_TIMEOUT_SEC <= 0:
        raise ValueError("A JAIDE_INFERENCE_STARTUP_TIMEOUT értékének pozitívnak kell lennie")
    try:
        reconstruction_alpha_value = float(RECONSTRUCTION_ALPHA)
    except ValueError as exc:
        raise ValueError("A JAIDE_BENCH_RECONSTRUCTION_ALPHA értékének 0.0 és 1.0 közötti lebegőpontos számnak kell lennie") from exc
    if not 0.0 <= reconstruction_alpha_value <= 1.0:
        raise ValueError("A JAIDE_BENCH_RECONSTRUCTION_ALPHA értékének 0.0 és 1.0 közötti lebegőpontos számnak kell lennie")
    if PHASE_A_STEPS < 0:
        raise ValueError("A JAIDE_BENCH_PHASE_A_STEPS nem lehet negatív")
    if PHASE_B_STEPS < 0:
        raise ValueError("A JAIDE_BENCH_PHASE_B_STEPS nem lehet negatív")
    if SHUFFLE_TARGET_CONTROL not in ("0", "1", "true", "false"):
        raise ValueError("A JAIDE_BENCH_SHUFFLE_TARGET_CONTROL értéke csak 0, 1, true vagy false lehet")
    if TARGET_SOURCE_FROZEN not in ("0", "1", "true", "false"):
        raise ValueError("A JAIDE_BENCH_TARGET_SOURCE_FROZEN értéke csak 0, 1, true vagy false lehet")
    if SPECTRAL_DEPTH_COMPENSATION not in ("0", "1", "true", "false"):
        raise ValueError("A JAIDE_BENCH_SPECTRAL_DEPTH_COMPENSATION értéke csak 0, 1, true vagy false lehet")
    if GRAD_MEAN not in ("0", "1", "true", "false"):
        raise ValueError("A JAIDE_BENCH_GRAD_MEAN értéke csak 0, 1, true vagy false lehet")
    if NORMALIZED_GRADIENT_FLOW not in ("0", "1", "true", "false"):
        raise ValueError("A JAIDE_BENCH_NORMALIZED_GRADIENT_FLOW értéke csak 0, 1, true vagy false lehet")
    if SPECTRAL_INTERVAL <= 0:
        raise ValueError("A JAIDE_BENCH_SPECTRAL_INTERVAL értékének pozitívnak kell lennie")
    if INIT_DEADLINE_SEC <= 0:
        raise ValueError("A JAIDE_BENCH_INIT_DEADLINE értékének pozitívnak kell lennie")
    if IDLE_WATCHDOG_SEC <= 0:
        raise ValueError("A JAIDE_BENCH_IDLE_WATCHDOG értékének pozitívnak kell lennie")
    if SKIP_KNOWLEDGE_GRAPH not in ("0", "1", "true", "false"):
        raise ValueError("A JAIDE_BENCH_SKIP_KNOWLEDGE_GRAPH értéke csak 0, 1, true vagy false lehet")
    if int(GRAPH_CHUNK_SIZE) <= 0:
        raise ValueError("A JAIDE_BENCH_GRAPH_CHUNK_SIZE értékének pozitívnak kell lennie")
    if int(INIT_SPECTRAL_ITERS) < 0:
        raise ValueError("A JAIDE_BENCH_INIT_SPECTRAL_ITERS nem lehet negatív")
    for name, raw_value, lower, upper in (
        ("JAIDE_BENCH_GRADIENT_CLIP_NORM", GRADIENT_CLIP_NORM, 0.0, None),
        ("JAIDE_BENCH_SFD_TRUST_RATIO", SFD_TRUST_RATIO, 0.0, 1.0),
        ("JAIDE_BENCH_SFD_WEIGHT_FLOOR", SFD_WEIGHT_FLOOR, 0.0, None),
    ):
        value = float(raw_value)
        if not math.isfinite(value) or value <= lower or (upper is not None and value > upper):
            raise ValueError(f"A(z) {name} értéke a megengedett tartományon kívül esik")
    logdet_weight_value = float(LOGDET_WEIGHT)
    if not math.isfinite(logdet_weight_value):
        raise ValueError("A JAIDE_BENCH_LOGDET_WEIGHT értékének véges számnak kell lennie")
    learning_rate_value = float(LEARNING_RATE)
    if not math.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
        raise ValueError("A JAIDE_BENCH_LR értékének véges és pozitív számnak kell lennie")

    run_id = int(time.time())
    _log(f"Futtatás indítása run_id={run_id}")

    if RUN_MODE == "train":
        _log("A dataset és a GPU binárisok ellenőrzése a felcsatolt Modal volume-okban")
        prep_result = inspect_prepared_state.remote(run_id)
        print("\n" + "=" * 70)
        print("MEGLÉVŐ ELŐKÉSZÍTETT ÁLLAPOT")
        print("=" * 70)
        print(json.dumps(prep_result, indent=2, default=str))
    elif SKIP_PREP:
        _log("CPU előkészítési fázis átugorva (JAIDE_BENCH_SKIP_PREP=1); a meglévő állapot ellenőrzése")
        prep_result = inspect_prepared_state.remote(run_id)
        print("\n" + "=" * 70)
        print("MEGLÉVŐ ELŐKÉSZÍTETT ÁLLAPOT")
        print("=" * 70)
        print(json.dumps(prep_result, indent=2, default=str))
    else:
        _log("1. LÉPÉS: CPU előkészítés ellenőrzése / futtatása")
        prep_result = prepare_cpu.remote(run_id)
        print("\n" + "=" * 70)
        print("CPU ELŐKÉSZÍTÉS EREDMÉNYE")
        print("=" * 70)
        print(json.dumps(prep_result, indent=2, default=str))

    if RUN_MODE != "prep" and not prep_result.get("distributed_binary_present"):
        print("\n" + "=" * 70)
        print("MEGSZAKÍTÁS: a disztributált bináris nem áll rendelkezésre")
        print("=" * 70)
        raise RuntimeError("A disztributált bináris nem épült fel és meglévő verzió sem található")

    dataset_ok = prep_result.get("phases", {}).get("C_prep_dataset", {}).get("sample_count", 0) > 0
    if RUN_MODE != "build" and not dataset_ok:
        print("\n" + "=" * 70)
        print("MEGSZAKÍTÁS: az adathalmaz nincs előkészítve")
        print("=" * 70)
        raise RuntimeError("Az adathalmaz nem áll készen a betanításhoz")

    if RUN_MODE in {"prep", "build"}:
        print("\n" + "=" * 70)
        if RUN_MODE == "build":
            print("GPU BINÁRIS BOOTSTRAP MODAL FUTÁS BEFEJEZŐDÖTT")
        else:
            print("KÜLÖN DATASET-ELŐKÉSZÍTŐ MODAL FUTÁS BEFEJEZŐDÖTT")
        print("=" * 70)
        print(json.dumps(prep_result, indent=2, default=str))
        return

    _log("2. LÉPÉS: GPU betanítás és inferencia indítása")
    gpu_result = run_gpu_train_and_infer.remote(run_id, prep_result)
    print("\n" + "=" * 70)
    print("GPU FÁZIS EREDMÉNYE")
    print("=" * 70)
    print(json.dumps(gpu_result, indent=2, default=str))
    if gpu_result.get("verified_success") is not True:
        raise RuntimeError("A GPU betanítás, telemetria vagy az inferencia ellenőrzése nem felelt meg az elvárásoknak")

    final = {
        "run_id": run_id,
        "cpu_phase": prep_result,
        "gpu_phase": gpu_result,
    }
    print("\n" + "=" * 70)
    print("VÉGSŐ EREDMÉNY")
    print("=" * 70)
    print(json.dumps(final, indent=2, default=str))