"""
preflight_sama.py — Verify SAMA compat patches import cleanly, no models loaded.

Checks:
  1. SAMA repo found and has expected structure
  2. Stub attack.run injection works
  3. attack.misc.models.ModelManager exists and can be patched
  4. SamaAttack can be imported
  5. SamaAttack has .run() method
  6. datasets.Dataset is available (SAMA dependency)

Run locally before starting the JarvisLabs instance:
  source venv3.11/bin/activate
  cd Logs/Run_2_Qwen_DLLM/code
  python preflight_sama.py
"""

import os
import sys
import types

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []


def check(label, fn):
    try:
        fn()
        print(f"{PASS} {label}")
    except Exception as e:
        print(f"{FAIL} {label}: {e}")
        failures.append(label)


# ------------------------------------------------------------------
# 1. Find SAMA root (mirrors _find_sama_root() in run_sama.py)
# ------------------------------------------------------------------
def _find_sama_root():
    candidates = [
        os.environ.get("SAMA_ROOT", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../SAMA")),
        "/home/SAMA",
        os.path.abspath("SAMA"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and os.path.isdir(os.path.join(p, "attack")):
            return p
    return None


sama_root = _find_sama_root()
check("SAMA repo found", lambda: (_ for _ in ()).throw(RuntimeError(f"Not found")) if not sama_root else None)

if not sama_root:
    print("\nCannot continue — SAMA repo missing.")
    sys.exit(1)

print(f"      SAMA root: {sama_root}")
sys.path.insert(0, sama_root)
# sama.py does `from attacks import AbstractAttack` (bare name) —
# needs SAMA/attack/ on path too.
sys.path.insert(0, os.path.join(sama_root, "attack"))

# ------------------------------------------------------------------
# 2. Expected files exist
# ------------------------------------------------------------------
check("attack/attacks/sama.py exists",
      lambda: (_ for _ in ()).throw(FileNotFoundError()) if not os.path.isfile(
          os.path.join(sama_root, "attack", "attacks", "sama.py")) else None)

check("attack/misc/models.py exists",
      lambda: (_ for _ in ()).throw(FileNotFoundError()) if not os.path.isfile(
          os.path.join(sama_root, "attack", "misc", "models.py")) else None)

check("attack/run.py exists",
      lambda: (_ for _ in ()).throw(FileNotFoundError()) if not os.path.isfile(
          os.path.join(sama_root, "attack", "run.py")) else None)

# ------------------------------------------------------------------
# 3. Inject stub attack.run (exact logic from run_sama.py)
# ------------------------------------------------------------------
def inject_stub():
    def _dummy_init_model(model_path, tokenizer_name, device_arg, lora=None):
        pass  # stub — no real model loaded in preflight

    if "attack.run" not in sys.modules:
        _stub = types.ModuleType("attack.run")
        _stub.init_model = _dummy_init_model
        sys.modules["attack.run"] = _stub
    else:
        sys.modules["attack.run"].init_model = _dummy_init_model

check("Stub attack.run injection", inject_stub)

# ------------------------------------------------------------------
# 4. attack.misc.models imports and ModelManager patchable
# ------------------------------------------------------------------
def patch_model_manager():
    import attack.misc.models as _models_mod
    assert hasattr(_models_mod, "ModelManager"), "ModelManager class not found"
    _models_mod.ModelManager.init_model = staticmethod(lambda *a, **k: None)

check("attack.misc.models import + ModelManager patch", patch_model_manager)

# ------------------------------------------------------------------
# 5. SamaAttack imports
# ------------------------------------------------------------------
def import_sama():
    from attack.attacks.sama import SamaAttack  # noqa: F401

check("from attack.attacks.sama import SamaAttack", import_sama)

# ------------------------------------------------------------------
# 6. SamaAttack has .run() method
# ------------------------------------------------------------------
def check_run_method():
    from attack.attacks.sama import SamaAttack
    assert callable(getattr(SamaAttack, "run", None)), "SamaAttack.run() not found"

check("SamaAttack.run() callable", check_run_method)

# ------------------------------------------------------------------
# 7. datasets available (SAMA calls HFDataset.from_dict)
# ------------------------------------------------------------------
def check_datasets():
    from datasets import Dataset  # noqa: F401

check("datasets.Dataset importable", check_datasets)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
if failures:
    print(f"PREFLIGHT FAILED — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
else:
    print("All preflight checks passed. Safe to start JarvisLabs instance.")
