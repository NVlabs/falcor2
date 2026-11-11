# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe the compiler and scene ABI needed by the glTF specialization plan.

Run once per backend with extra-data references embedded in each material record.
The generated Slang and target assemblies are retained beside the JSON report.
An unavailable backend or an unproved layout is a failure, not a skipped check.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Any

import numpy as np
import slangpy as spy


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _ROOT / ".tmp/gltf-material-specialization"
_FEATURE_BIT = 2
_CUDA_CAPABILITY = "cuda_sm_8_0"
_TARGETS = {
    "d3d12": ("dxil-asm", "sm_6_6", "dxil.txt"),
    "vulkan": ("spirv-asm", "glsl_460", "spirv.txt"),
    "cuda": ("ptx", None, "ptx.txt"),
}
_LOAD_PATTERNS = {
    "d3d12": r"dx\.op\.(?:rawBufferLoad|bufferLoad)",
    "vulkan": r"\bOpLoad\b",
    "cuda": r"\bld\.(?:global|const)\b",
}


def _lut_access_present(backend: str, assembly: str) -> bool:
    """Find the actual LUT access path in each standalone target assembly.

    The CUDA parameter index follows probe_main's declared argument order:
    masks, expensive_lut, output. The PTX kernel ABI keeps unused parameters,
    so the parameter declaration alone is not evidence that the LUT is used.
    """
    if backend == "d3d12":
        return "entryPointParams_expensive_lut_0" in assembly
    if backend == "vulkan":
        return bool(re.search(r"OpAccessChain[^\n]*%entryPointParams_expensive_lut\b", assembly))
    return bool(re.search(r"ld\.param\.u64[^\n]*\[probe_main_param_1\]", assembly))


# This source is deliberately small enough for the standalone Slang compiler.
# Its three functions preserve the exact source forms proposed for production.
_MASK_SOURCE = """\
public struct ProbeMask<let CompileMask : uint>
{
    public uint run_time_mask;

    public uint nested(ByteAddressBuffer expensive_lut, uint address)
    {
        if ((CompileMask & 2u) != 0u) {
            if ((run_time_mask & 2u) != 0u)
                return expensive_lut.Load(address);
        }
        return 0u;
    }

    public uint combined(ByteAddressBuffer expensive_lut, uint address)
    {
        uint full_mask = CompileMask & run_time_mask;
        if ((full_mask & 2u) != 0u)
            return expensive_lut.Load(address);
        return 0u;
    }

    public uint booleans(ByteAddressBuffer expensive_lut, uint address)
    {
        bool compile_time_enabled = (CompileMask & 2u) != 0u;
        bool run_time_enabled = (run_time_mask & 2u) != 0u;
        if (compile_time_enabled && run_time_enabled)
            return expensive_lut.Load(address);
        return 0u;
    }
};

[shader("compute")]
[numthreads(1, 1, 1)]
void probe_main(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<uint> masks,
    ByteAddressBuffer expensive_lut,
    RWStructuredBuffer<uint> output)
{
    ProbeMask<__FEATURES__> probe = { masks[tid.x] };
    output[tid.x] = probe.__FORM__(expensive_lut, tid.x * 4u);
}
"""


# This second standalone source forces both Conditional fields through a
# dynamically dispatched call. It complements the full Scene/IMaterial run.
_DYNAMIC_SOURCE = """\
public interface IProbeMaterial
{
    public uint evaluate(ByteAddressBuffer expensive_lut, uint run_time_mask);
};

public struct ProbeMultiscatterState
{
    public float4 weight = float4(1.0);
    public float4 proposal = float4(0.5);
    public uint lut_address = 0u;
};

public struct ProbeBSDF<let Features : uint>
{
    public Conditional<ProbeMultiscatterState, (Features & 2u) != 0u> multiscatter
        = Conditional<ProbeMultiscatterState, (Features & 2u) != 0u>(ProbeMultiscatterState());

    public uint evaluate(ByteAddressBuffer expensive_lut, uint run_time_mask)
    {
        if (let state = multiscatter.get()) {
            if ((run_time_mask & 2u) != 0u)
                return expensive_lut.Load(state.lut_address);
        }
        return 0u;
    }
};

public struct ProbeInstance<let Features : uint>
{
    public Conditional<ProbeMultiscatterState, (Features & 2u) != 0u> multiscatter
        = Conditional<ProbeMultiscatterState, (Features & 2u) != 0u>(ProbeMultiscatterState());

    public uint evaluate(ByteAddressBuffer expensive_lut, uint run_time_mask)
    {
        if (let state = multiscatter.get()) {
            if ((run_time_mask & 2u) != 0u) {
                ProbeBSDF<Features> bsdf = {};
                return bsdf.evaluate(expensive_lut, run_time_mask)
                    * uint(state.weight.x);
            }
        }
        return 0u;
    }
};

public struct ProbePayload { public uint data; };

public struct ProbeMaterial<let Features : uint> : IProbeMaterial
{
    public uint data;

    public uint evaluate(ByteAddressBuffer expensive_lut, uint run_time_mask)
    {
        ProbeInstance<Features> instance = {};
        return instance.evaluate(expensive_lut, run_time_mask);
    }
};

[shader("compute")]
[numthreads(1, 1, 1)]
void probe_main(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<uint> masks,
    ByteAddressBuffer expensive_lut,
    RWStructuredBuffer<uint> output)
{
    ProbePayload payload = { 0u };
    IProbeMaterial material = createDynamicObject<IProbeMaterial, ProbePayload>(0u, payload);
    output[tid.x] = material.evaluate(expensive_lut, masks[tid.x]);
}
"""


# The real render module supplies IMaterial, MaterialData, g_scene, and
# BufferHandle. The generic material is intentionally ordinary Type<let ...>.
_SCENE_SOURCE = """\
import slangpy;
import falcor2.render;
import falcor2.utils;

public struct ProbeFeatureRecord
{
    public float4 first;
    public float4 second;
};

public struct ProbeTransformRecord
{
    public float2 axis_x;
    public float2 axis_y;
    public float2 offset;
};

public struct ProbeMultiscatterState
{
    public float4 weight = float4(1.0);
    public float4 proposal = float4(0.5);
    public float4 lut_parameters = float4(0.25);
    public float resolved_extra = 0.0;
};

public struct ProbeBSDF<let Features : uint>
{
    public uint feature_mask;
    public Conditional<ProbeMultiscatterState, (Features & 2u) != 0u> multiscatter
        = Conditional<ProbeMultiscatterState, (Features & 2u) != 0u>(ProbeMultiscatterState());

    public float eval()
    {
        if (let state = multiscatter.get()) {
            if ((feature_mask & 2u) != 0u)
                return state.weight.x * state.resolved_extra;
        }
        return 0.0;
    }
};

public struct ProbeMaterialInstance<let Features : uint> : IMaterialInstance
{
    public uint feature_mask;
    public Conditional<ProbeMultiscatterState, (Features & 2u) != 0u> multiscatter
        = Conditional<ProbeMultiscatterState, (Features & 2u) != 0u>(ProbeMultiscatterState());

    public BSDFEval eval_with_pdf<let Hints : BSDFEvalHints = BSDFEvalHints::both>(
        const SurfaceInteraction si, const float3 wo_ws)
    {
        BSDFEval result = {};
        if (let state = multiscatter.get()) {
            if ((feature_mask & 2u) != 0u) {
                ProbeBSDF<Features> bsdf = {};
                bsdf.feature_mask = feature_mask;
                bsdf.multiscatter.set(state);
                result.value = float3(bsdf.eval() * state.weight.x);
            }
        }
        return result;
    }

    public bool sample(
        const SurfaceInteraction si, const MaterialSampleInput input, out MaterialSample sample)
    {
        sample = {};
        return false;
    }

    public MaterialProperties collect_properties(const SurfaceInteraction si) { return {}; }
    public BSDFFlags get_lobe_types(const SurfaceInteraction si) { return BSDFFlags::none; }
};

public struct ProbeGLTFMaterial<let F : uint, let T : uint, let X : uint, let U : uint> : IMaterial
{
    typedef ProbeMaterialInstance<F> MaterialInstance;
    public MaterialHeader header;
    public uint feature_mask;
    public BufferOffsetPair extra_data;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si, const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none)
    {
        MaterialInstance instance = {};
        instance.feature_mask = feature_mask;
        if ((F & 2u) != 0u) {
            if ((feature_mask & 2u) != 0u) {
                ProbeMultiscatterState state = {};
                state.resolved_extra = extra_data.load<float>(192u);
                instance.multiscatter.set(state);
            }
        }
        return instance;
    }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si, const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none)
    {
        if ((F & 2u) != 0u) {
            if ((feature_mask & 2u) != 0u) {
                ProbeFeatureRecord feature
                    = extra_data.load<ProbeFeatureRecord>(0u);
                ProbeTransformRecord transform
                    = extra_data.load<ProbeTransformRecord>(32u);
                return float3(feature.first.x, transform.offset.x, feature.second.w);
            }
        }
        return float3(0.0);
    }
};

public float3 probe_material(uint material_id)
{
    SurfaceInteraction si = {};
    let material = g_scene.get_material(MaterialID(material_id));
    return material.eval_emission(si, ExplicitLodSampler(0.0));
}

public float3 probe_instance(uint material_id)
{
    SurfaceInteraction si = {};
    let material = g_scene.get_material(MaterialID(material_id));
    let instance = material.setup_material_instance(si, ExplicitLodSampler(0.0));
    return instance.eval(si, float3(0.0, 0.0, 1.0));
}
"""


def _slangc_path(override: Path | None) -> Path:
    if override is not None:
        result = override
    else:
        found = shutil.which("slangc") or shutil.which("slangc.exe")
        result = (
            Path(found) if found else _ROOT / "build/windows-msvc/_deps/slang-src/bin/slangc.exe"
        )
    if not result.is_file():
        raise FileNotFoundError(f"Slang compiler not found: {result}")
    return result.resolve()


def _run_codegen(backend: str, directory: Path, slangc: Path) -> dict[str, Any]:
    target, profile, suffix = _TARGETS[backend]
    compiler_env = os.environ.copy()
    if backend == "d3d12":
        dxc_directory = _ROOT / "build/windows-msvc/_deps/dxc-src/bin/x64"
        if dxc_directory.is_dir():
            compiler_env["PATH"] = str(dxc_directory) + os.pathsep + compiler_env.get("PATH", "")
    cases: dict[str, Any] = {}
    for form in ("nested", "combined", "booleans", "conditional_dispatch"):
        for feature_mask in (0, _FEATURE_BIT):
            name = f"{form}_{'on' if feature_mask else 'off'}"
            source = (
                _DYNAMIC_SOURCE
                if form == "conditional_dispatch"
                else _MASK_SOURCE.replace("__FEATURES__", f"{feature_mask}u").replace(
                    "__FORM__", form
                )
            )
            source_path = directory / f"{name}.slang"
            assembly_path = directory / f"{name}.{suffix}"
            source_path.write_text(source, encoding="utf-8")
            command = [
                str(slangc),
                str(source_path),
                "-entry",
                "probe_main",
                "-target",
                target,
                "-o",
                str(assembly_path),
            ]
            if profile:
                command.extend(("-profile", profile))
            if backend == "cuda":
                command.extend(("-capability", _CUDA_CAPABILITY))
            if form == "conditional_dispatch":
                command.extend(("-conformance", f"ProbeMaterial<{feature_mask}>:IProbeMaterial=0"))
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False, env=compiler_env
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{name} Slang compilation failed ({completed.returncode}): "
                    f"{completed.stdout}\n{completed.stderr}"
                )
            assembly = assembly_path.read_text(encoding="utf-8", errors="replace")
            if backend == "cuda" and not re.search(r"(?m)^[ \t]*\.target[ \t]+sm_80\b", assembly):
                raise RuntimeError(f"{name} PTX does not target sm_80")
            cases[name] = {
                "source": str(source_path),
                "generated_code": str(assembly_path),
                "generated_bytes": assembly_path.stat().st_size,
                "load_instruction_count": len(re.findall(_LOAD_PATTERNS[backend], assembly)),
                "expensive_lut_mentions": assembly.count("expensive_lut"),
                "targeted_lut_access_present": _lut_access_present(backend, assembly),
            }
    comparisons = {}
    for form in ("nested", "combined", "booleans", "conditional_dispatch"):
        off = cases[f"{form}_off"]["load_instruction_count"]
        on = cases[f"{form}_on"]["load_instruction_count"]
        off_lut = cases[f"{form}_off"]["targeted_lut_access_present"]
        on_lut = cases[f"{form}_on"]["targeted_lut_access_present"]
        comparisons[form] = {
            "disabled_load_instructions": off,
            "enabled_load_instructions": on,
            "additional_loads_when_enabled": on - off,
            "load_count_reduced": on > off,
            "disabled_lut_access_present": off_lut,
            "enabled_lut_access_present": on_lut,
            "lut_elimination_proved_in_standalone": not off_lut and on_lut,
        }
    if not all(item["load_count_reduced"] for item in comparisons.values()):
        raise RuntimeError(
            "At least one mask form lacks additional enabled load instructions; "
            "inspect saved target assembly before accepting this backend"
        )
    preferred_forms = ("nested", "booleans", "conditional_dispatch")
    if not all(
        comparisons[form]["lut_elimination_proved_in_standalone"] for form in preferred_forms
    ):
        raise RuntimeError(
            "The preferred nested-if, two-bool, or Conditional form retained its LUT "
            "access in the disabled standalone shader"
        )
    return {
        "compiler": str(slangc),
        "target": target,
        "target_capability": _CUDA_CAPABILITY if backend == "cuda" else None,
        "cases": cases,
        "comparisons": comparisons,
        "production_shader_review_required": (
            "Targeted access checks prove the standalone shader result only. Inspect "
            "the linked production material shader before claiming its LUT path was eliminated."
        ),
    }


def _element_layout(layout: Any, type_name: str) -> Any:
    layout = getattr(layout, "program_layout", layout)
    buffer_type = layout.find_type_by_name(f"StructuredBuffer<{type_name}>")
    if buffer_type is None:
        raise RuntimeError(f"Cannot resolve StructuredBuffer<{type_name}>")
    return layout.get_type_layout(buffer_type).element_type_layout


def _reflect_layouts(module: spy.Module) -> dict[str, Any]:
    layout = module.layout
    names = {
        "material_off": "ProbeGLTFMaterial<0,0,0,0>",
        "material_on": "ProbeGLTFMaterial<2,0,0,0>",
        "instance_off": "ProbeMaterialInstance<0>",
        "instance_on": "ProbeMaterialInstance<2>",
        "bsdf_off": "ProbeBSDF<0>",
        "bsdf_on": "ProbeBSDF<2>",
        "feature_record": "ProbeFeatureRecord",
        "transform_record": "ProbeTransformRecord",
    }
    result = {}
    for label, name in names.items():
        reflected = _element_layout(layout, name)
        result[label] = {
            "type": name,
            "size": reflected.size,
            "stride": reflected.stride,
            "alignment": reflected.alignment,
            "fields": [field.name for field in reflected.fields],
        }
    if result["feature_record"]["stride"] != 32:
        raise RuntimeError("Feature record is not 32 bytes")
    if result["transform_record"]["stride"] != 24:
        raise RuntimeError("Transform record is not 24 bytes")
    for kind in ("instance", "bsdf"):
        if result[f"{kind}_off"]["stride"] >= result[f"{kind}_on"]["stride"]:
            raise RuntimeError(f"Disabled Conditional did not shrink {kind} storage")
    for material_kind in ("material_off", "material_on"):
        if result[material_kind]["stride"] > 128:
            raise RuntimeError(f"{material_kind} exceeds MaterialData's 128-byte record")
    return result


def _field_word(layout: Any, *path: str) -> int:
    offset = 0
    for name in path:
        field = next(field for field in layout.fields if field.name == name)
        offset += field.offset
        layout = field.type_layout
    if offset % 4 != 0 or offset >= 128:
        raise RuntimeError(f"Invalid material word offset for {'.'.join(path)}: {offset}")
    return offset // 4


def _material_words(module: spy.Module) -> dict[str, int]:
    names = ("ProbeGLTFMaterial<0,0,0,0>", "ProbeGLTFMaterial<2,0,0,0>")
    words = []
    for name in names:
        layout = _element_layout(module.layout, name)
        words.append(
            {
                "feature_mask": _field_word(layout, "feature_mask"),
                "extra_handle": _field_word(layout, "extra_data", "buffer", "data"),
                "extra_offset": _field_word(layout, "extra_data", "offset"),
            }
        )
    if words[0] != words[1]:
        raise RuntimeError(f"Probe material field offsets differ by specialization: {words}")
    if words[0]["extra_handle"] + 1 >= 32:
        raise RuntimeError("Probe material buffer handle exceeds MaterialData")
    return words[0]


def _pack_materials(
    on_type_id: int, off_type_id: int, words: dict[str, int], extra_handle: int
) -> np.ndarray:
    # MaterialData is a 16-byte header followed by 112 payload bytes.
    records = np.zeros((5, 32), dtype=np.uint32)
    records[0, 0] = 0  # InvalidMaterial remains the first conformance.
    materials = (
        (off_type_id, 2, 0xFFFFFFFF),
        (on_type_id, 0, 0xFFFFFFFF),
        (on_type_id, 2, 0),
        (on_type_id, 2, 64),
    )
    for material_id, (type_id, feature_mask, extra_offset) in enumerate(materials, start=1):
        handle = extra_handle if type_id == on_type_id and feature_mask else 0xFFFFFFFFFFFFFFFF
        records[material_id, 0] = type_id
        records[material_id, words["feature_mask"]] = feature_mask
        records[material_id, words["extra_handle"]] = handle & 0xFFFFFFFF
        records[material_id, words["extra_handle"] + 1] = handle >> 32
        records[material_id, words["extra_offset"]] = extra_offset
    return records


def _pack_extra(second_feature: float, size: int = 256) -> np.ndarray:
    data = bytearray(size)
    for offset, first, transform, last in (
        (0, 1.25, 0.125, 2.5),
        (64, second_feature, 0.625, 4.5),
    ):
        struct.pack_into("<8f", data, offset, first, 0, 0, 0, 0, 0, 0, last)
        struct.pack_into("<6f", data, offset + 32, 1, 0, 0, 1, transform, 0)
    struct.pack_into("<f", data, 192, 3.0)
    return np.frombuffer(data, dtype=np.uint8).copy()


def _replace_material_conformance(
    requirements: Any, material_name: str
) -> tuple[list[spy.TypeConformance], int, int]:
    conformances = []
    probe_id = None
    for item in requirements.type_conformances:
        if item.interface_name == "IMaterial" and item.type_name.startswith("GLTFMaterial<"):
            probe_id = item.id
            conformances.append(spy.TypeConformance("IMaterial", material_name, item.id))
        else:
            conformances.append(item)
    if probe_id is None:
        raise RuntimeError("Scene requirements lack the expected GLTFMaterial specialization")
    off_id = (
        max(
            item.id for item in requirements.type_conformances if item.interface_name == "IMaterial"
        )
        + 1
    )
    conformances.append(spy.TypeConformance("IMaterial", "ProbeGLTFMaterial<0,0,0,0>", off_id))
    return conformances, probe_id, off_id


def _run_scene_probe(backend: str, directory: Path) -> dict[str, Any]:
    import falcor2 as f2
    from falcor2.editor import SceneShaderHelper, create_device

    device = create_device(getattr(spy.DeviceType, backend))
    try:
        scene = f2.Scene.create(device)
        scene.create_material(f2.GLTFMaterial)
        scene.update()

        source_path = directory / "scene_probe.slang"
        source_path.write_text(_SCENE_SOURCE, encoding="utf-8")
        source_module = spy.Module(
            device.load_module_from_source("gltf_material_specialization_probe", _SCENE_SOURCE)
        )
        helper = SceneShaderHelper(device)
        module, requirements = helper.prepare(scene, source_module)
        layouts = _reflect_layouts(module)
        material_words = _material_words(module)
        material_name = "ProbeGLTFMaterial<2,0,0,0>"
        conformances, on_type_id, off_type_id = _replace_material_conformance(
            requirements, material_name
        )

        def read(extra_data: np.ndarray, material_id: int, function_name: str) -> list[float]:
            extra_buffer = device.create_buffer(
                usage=spy.BufferUsage.shader_resource, data=extra_data
            )
            material_buffer = device.create_buffer(
                element_count=5,
                struct_size=128,
                usage=spy.BufferUsage.shader_resource,
                data=_pack_materials(
                    on_type_id, off_type_id, material_words, f2.to_handle(extra_buffer).data
                ),
            )

            def bind(cursor: spy.ShaderCursor) -> None:
                helper.bind_scene(cursor)
                cursor["g_scene"]["material_system"]["material_data"] = material_buffer

            func = module[function_name].as_func().type_conformances(conformances).write(bind)
            return np.asarray(func(material_id), dtype=np.float32).tolist()

        initial = _pack_extra(3.25)
        observed = {
            "compile_disabled": read(initial, 1, "probe_material"),
            "runtime_disabled_no_allocation": read(initial, 2, "probe_material"),
            "first_material": read(initial, 3, "probe_material"),
            "second_material": read(initial, 4, "probe_material"),
            "instance_compile_disabled": read(initial, 1, "probe_instance"),
            "instance_runtime_disabled": read(initial, 2, "probe_instance"),
            "instance_enabled": read(initial, 3, "probe_instance"),
            "growth_rebind": read(_pack_extra(6.25, size=512), 4, "probe_material"),
        }
        expected = {
            "compile_disabled": [0.0, 0.0, 0.0],
            "runtime_disabled_no_allocation": [0.0, 0.0, 0.0],
            "first_material": [1.25, 0.125, 2.5],
            "second_material": [3.25, 0.625, 4.5],
            "instance_compile_disabled": [0.0, 0.0, 0.0],
            "instance_runtime_disabled": [0.0, 0.0, 0.0],
            "instance_enabled": [3.0, 3.0, 3.0],
            "growth_rebind": [6.25, 0.625, 4.5],
        }
        for name, values in observed.items():
            if not np.allclose(values, expected[name], rtol=0, atol=1e-6):
                raise RuntimeError(f"{name}: expected {expected[name]}, got {values}")
        return {
            "layouts": layouts,
            "material_words": material_words,
            "type_conformances": [
                {"type": material_name, "id": on_type_id},
                {"type": "ProbeGLTFMaterial<0,0,0,0>", "id": off_type_id},
            ],
            "record_values": observed,
            "expected_record_values": expected,
            "device": str(device.info),
            "source": str(source_path),
        }
    finally:
        device.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=tuple(_TARGETS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--slangc", type=Path)
    parser.add_argument(
        "--codegen-only",
        action="store_true",
        help="Compile standalone mask and Conditional probes without creating a GPU device.",
    )
    args = parser.parse_args()
    output = args.output or _DEFAULT_OUTPUT / f"probe-{args.backend}.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.parent / f"probe-{args.backend}-code"
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"backend": args.backend, "status": "failed"}
    try:
        report["codegen"] = _run_codegen(args.backend, directory, _slangc_path(args.slangc))
        if not args.codegen_only:
            report["scene"] = _run_scene_probe(args.backend, directory)
        report["status"] = "standalone_passed" if args.codegen_only else "scene_probe_passed"
        report["production_codegen_review_required"] = not args.codegen_only
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if report["status"] in ("standalone_passed", "scene_probe_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
