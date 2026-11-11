# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the existing performance workflow locally or across CI network phases.

The measurement phase clears driver caches and locks GPU clocks, just like CI.
It writes sealed-export files; it does not submit results to BenchView.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.ci_submodules import inherit_git_config
from tools.perf_environment import prepare_environment
from tools.perf_instrumentation import apply_instrumentation, load_profiles, select_instrumentation

OVERLAY_PATHS = (
    "tools/ci.py",
    "tools/build_slang.py",
    "falcor2/testing/benchmark",
    "tests/python/benchmarks",
)


def parse_args(argv=None, environ=None):
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "checkout", "build", "measure"), default="all")
    parser.add_argument("--project", default=env.get("CI_PERF_PROJECT") or "falcor2")
    parser.add_argument("--falcor-ref", default=env.get("CI_PERF_FALCOR_REF", ""))
    parser.add_argument("--slang-version", default=env.get("CI_SLANG_VERSION", ""))
    parser.add_argument("--run-dir", type=Path, default=ROOT)
    parser.add_argument("--python", default=env.get("CI_PERF_PYTHON"))
    parser.add_argument("--lock-gpu-clocks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gpu-phases", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.gpu_phases is None:
        args.gpu_phases = env.get("CI_PERF_GPU_PHASES", "").lower() == "true"
    if args.project == "falcor2":
        if args.falcor_ref and not re.fullmatch(r"[0-9a-fA-F]{40}", args.falcor_ref):
            parser.error("falcor_ref must be a full commit SHA")
        args.falcor_ref = args.falcor_ref.lower()
    args.run_dir = args.run_dir.resolve()
    if (env.get("CI", "").lower() == "true" or env.get("CI_JOB_ID")) and args.run_dir != ROOT:
        parser.error("CI uses the tooling checkout as its run directory")
    args.work_dir = args.run_dir / "perf-checkout"
    args.state_file = args.run_dir / ".tmp/perf-run.json"
    args.export_path = args.run_dir / "perf-results/falcor2-pathtracer-scene-bench.json"
    return args


def run(command, cwd, env, capture=False):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def apply_benchmark_backport(target, env):
    """Restore the Python reporting helpers bundled with historical baselines."""
    patch = target / "tools/slangpy-benchmark-backport.patch"
    if not patch.is_file():
        return
    slangpy = target / "external/slangpy"
    # Use the baseline's patch, without replacing its native dependency pins.
    # Fail before build/measurement if it does not match the pinned checkout.
    run(["git", "apply", "--check", str(patch)], slangpy, env)
    run(["git", "apply", str(patch)], slangpy, env)


def checkout(args, env):
    if args.state_file.exists():
        raise RuntimeError(f"State already exists: {args.state_file}; choose a fresh --run-dir")
    target = args.work_dir if args.falcor_ref else ROOT
    harness = env.get("CI_PERF_HARNESS_COMMIT", "")
    selected = None
    if args.falcor_ref:
        if target.exists():
            raise RuntimeError(f"Checkout already exists: {target}")
        remote = run(["git", "remote", "get-url", "origin"], ROOT, env, capture=True)
        run(["git", "init", str(target)], ROOT, env)
        inherit_git_config(ROOT, target)
        run(["git", "remote", "add", "origin", remote], target, env)
        profiles = None
        refs = [args.falcor_ref]
        if args.project == "falcor2":
            profiles = load_profiles()
            refs.append(profiles["last"])
        run(["git", "fetch", "origin", *dict.fromkeys(refs)], target, env)
        if args.project == "falcor2":
            run(["git", "checkout", "-B", "main", args.falcor_ref], target, env)
            if profiles:
                selected = select_instrumentation(args.falcor_ref, target, profiles)
            harness = args.falcor_ref
            if run(["git", "rev-parse", "HEAD"], target, env, capture=True) != args.falcor_ref:
                raise RuntimeError("Source identity changed")
        else:
            run(["git", "checkout", "--detach", "FETCH_HEAD"], target, env)
            harness = run(["git", "rev-parse", "HEAD"], ROOT, env, capture=True)
            archive = args.state_file.parent / "perf-overlay.tar"
            archive.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "archive", f"--output={archive}", "HEAD", *OVERLAY_PATHS], ROOT, env)
            run(["tar", "-xf", str(archive), "-C", str(target)], ROOT, env)
    if harness:
        env["CI_PERF_HARNESS_COMMIT"] = harness
    run([sys.executable, str(ROOT / "tools/ci_submodules.py")], target, env)
    if args.project == "falcor2-slang" and args.falcor_ref:
        apply_benchmark_backport(target, env)
    if selected and selected["patches"] is not None:
        apply_instrumentation(target, profiles, selected)
        harness = run(["git", "rev-parse", "HEAD"], ROOT, env, capture=True)
        env["CI_PERF_HARNESS_COMMIT"] = harness
    # Preserve compiler overrides for a separate sealed measurement invocation.
    state = {
        "python": str(Path(sys.executable).absolute()),
        "instrumentation": selected,
        "checkout": str(target),
        "source": run(["git", "rev-parse", "HEAD"], target, env, capture=True),
        "tooling": run(["git", "rev-parse", "HEAD"], ROOT, env, capture=True),
        "environment": {
            key: value
            for key, value in env.items()
            if key.startswith("CI_SLANG_")
            or key
            in ("CI_PERF_HARNESS_COMMIT", "CI_PERF_PROJECT", "CI_USE_CUSTOM_SLANG", "CI_CONFIG")
        },
        "lock_gpu_clocks": args.lock_gpu_clocks is not False,
        "gpu_phases": args.gpu_phases,
        "export_path": str(args.export_path),
        "request_id": (
            "gitlab-"
            + "-".join(env[key] for key in ("CI_PROJECT_ID", "CI_PIPELINE_ID", "CI_JOB_ID"))
            if all(env.get(key) for key in ("CI_PROJECT_ID", "CI_PIPELINE_ID", "CI_JOB_ID"))
            else "local-" + uuid.uuid4().hex
        ),
        "built": False,
    }
    save_state(args.state_file, state)
    return state


def load_state(args, env):
    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    target = Path(state["checkout"])
    if run(["git", "rev-parse", "HEAD"], target, env, capture=True) != state["source"]:
        raise RuntimeError("Prepared source identity changed")
    if run(["git", "rev-parse", "HEAD"], ROOT, env, capture=True) != state["tooling"]:
        raise RuntimeError("Tooling identity changed")
    # Absence of a compiler override is also part of the prepared configuration.
    for key in list(env):
        if key.startswith("CI_SLANG_") or key in ("CI_USE_CUSTOM_SLANG", "CI_PERF_HARNESS_COMMIT"):
            del env[key]
    env.update(state["environment"])
    return state


def build(args, state, env):
    state["built"] = False
    save_state(args.state_file, state)
    for command in ("info", "setup", "configure", "build"):
        run([sys.executable, "-u", "tools/ci.py", command], state["checkout"], env)
    state["built"] = True
    save_state(args.state_file, state)


def measure(state, env):
    if not state["built"]:
        raise RuntimeError("Preparation has not completed successfully")
    target = state["checkout"]
    export = Path(state["export_path"])
    if export.exists() or export.with_suffix(".ready").exists():
        raise RuntimeError(f"Measurement output already exists: {export}; use a fresh run")
    if not state.get("lock_gpu_clocks", True) and (
        env.get("CI", "").lower() == "true" or env.get("CI_JOB_ID")
    ):
        raise RuntimeError("CI measurement requires locked GPU clocks")
    best_effort = not (env.get("CI", "").lower() == "true" or env.get("CI_JOB_ID"))
    cleanup = [sys.executable, "-u", str(ROOT / "tools/clear_benchmark_caches.py")]
    report = export.with_suffix(".cache-cleanup.json")
    if best_effort:
        cleanup.extend(["--best-effort", "--report", str(report)])
    run(cleanup, target, env)
    command = [sys.executable, "-u", "tools/ci.py", "benchmark", "--"]
    if state["gpu_phases"]:
        command.append("--benchmark-gpu-phases")
    command.append("--benchmark-iterations=32")
    if state.get("lock_gpu_clocks", True):
        command.append("--lock-gpu-clocks")
    command.extend(
        [
            f"--benchmark-request-id={state['request_id']}",
            f"--benchmark-export-submissions={export}",
            f"--benchmark-export-ready={export.with_suffix('.ready')}",
        ]
    )
    if not best_effort:
        run(command, target, env)
        return
    # Existing reporting copies this provenance into each observation unchanged.
    provenance_path = Path(target) / "reports/benchmark-provenance.json"
    original = provenance_path.read_bytes() if provenance_path.exists() else None
    provenance = (
        json.loads(original)
        if original is not None
        else {"falcor_commit": state["source"], "harness_commit": state["source"], "compiler": None}
    )
    provenance["driver_cache_cleanup"] = json.loads(report.read_text(encoding="utf-8"))
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        run(command, target, env)
    finally:
        if original is None:
            provenance_path.unlink(missing_ok=True)
        else:
            provenance_path.write_bytes(original)


def main(argv=None):
    args = parse_args(argv)
    python = args.python or sys.executable
    target = args.work_dir if args.falcor_ref else ROOT
    if args.phase in ("build", "measure"):
        prepared = json.loads(args.state_file.read_text(encoding="utf-8"))
        python = prepared.get("python")
        if not python:
            raise RuntimeError("Prepared state has no interpreter; prepare a fresh run")
        if args.python and Path(args.python).expanduser().absolute() != Path(python):
            raise RuntimeError("Interpreter differs from preparation; use the recorded Python")
        target = Path(prepared["checkout"])
    executable, env = prepare_environment(python, target)
    if executable != Path(sys.executable).absolute():
        subprocess.run(
            [
                str(executable),
                "-u",
                str(Path(__file__).resolve()),
                *(sys.argv[1:] if argv is None else argv),
                "--python",
                str(executable),
            ],
            env=env,
            check=True,
        )
        return
    env["CI_PERF_PROJECT"] = args.project
    env["CI_SLANG_VERSION"] = args.slang_version
    env.setdefault("CI_CONFIG", "Release")
    if args.phase in ("all", "checkout"):
        state = checkout(args, env)
    else:
        state = load_state(args, env)
    if args.phase in ("all", "build"):
        build(args, state, env)
    if args.phase in ("all", "measure"):
        if args.lock_gpu_clocks is not None:
            state["lock_gpu_clocks"] = args.lock_gpu_clocks
        measure(state, env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Performance run failed: {error}", file=sys.stderr)
        sys.exit(1)
