"""A CPU stand-in for train_gpt.py, for testing ab_run.py: writes logs/<uuid>.txt in the cwd in the
Track 1 format (or Track 3's with FAKE_FORMAT=s), prints its path first, like the real trainers.

FAKE_VAL / FAKE_TIME set the means; TRAIN_SEED (if set) makes the noise reproducible; FAKE_FAIL=1
exits nonzero mid-run; FAKE_NAN=1 prints a nan val loss. FAKE_ARM_FILE: a file whose content is
added to FAKE_VAL (lets a git ref differ from another by a committed file)."""
import os
import random
import sys
import uuid

if os.environ.get("FAKE_REQUIRE_DATA") == "1":
    assert os.path.isfile(os.path.join(os.environ["DATA_PATH"], "data/fineweb10B/sentinel.bin"))
if os.environ.get("FAKE_EXPECT_CWD"):
    assert os.path.realpath(os.getcwd()) == os.path.realpath(os.environ["FAKE_EXPECT_CWD"])

seed = os.environ.get("TRAIN_SEED")
rng = random.Random(int(seed) if seed is not None else None)
val = float(os.environ.get("FAKE_VAL", "3.278"))
if os.path.exists("arm_offset.txt"):
    val += float(open("arm_offset.txt").read().strip())
t = float(os.environ.get("FAKE_TIME", "40.0"))
fmt = os.environ.get("FAKE_FORMAT", "ms")
total = 100
directory = os.environ.get("TRAIN_LOG_DIR") or "logs"
os.makedirs(directory, exist_ok=True)
path = os.path.join(directory, f"{uuid.uuid4()}.txt")
print(path, flush=True)
print(f"TRAIN_LOG_PATH={os.path.abspath(path)}", flush=True)
with open(path, "w") as f:
    f.write(open(__file__).read())
    f.write("\n" + "=" * 100 + "\nRunning PyTorch 2.10.0+cu128 compiled for CUDA 12.8\n" + "=" * 100 + "\n")
    f.write(f"cache={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}\n")
    for s in range(25, total + 1, 25):
        tt = t * s / total
        if fmt == "ms":
            f.write(f"step:{s}/{total} train_time:{tt*1000:.0f}ms step_avg:{tt*1000/s:.2f}ms\n")
        if s == 50 and os.environ.get("FAKE_FAIL") == "1":
            f.flush()
            sys.exit(3)
    tt = t + rng.gauss(0, 0.1)
    v = val + rng.gauss(0, 0.001)
    vs = "nan" if os.environ.get("FAKE_NAN") == "1" else (f"{v:.4f}" if fmt == "ms" else f"{v:.5f}")
    if fmt == "ms":
        line = f"step:{total}/{total} val_loss:{vs} train_time:{tt*1000:.0f}ms step_avg:{tt*1000/total:.2f}ms"
    else:
        line = f"step:{total}/{total} val_loss:{vs} train_time:{tt:.3f}s step_avg:{tt*1000/total:.2f}ms"
    f.write(line + "\npeak memory allocated: 100 MiB reserved: 200 MiB\n")
print(line)
