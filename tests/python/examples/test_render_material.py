# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from examples.render_material.materialx import provider as materialx_suite
from examples.render_material.materialx.discovery import make_materialx_entry
from examples.render_material.mdl import provider as mdl_suite
from examples.render_material.mdl.discovery import make_mdl_entry
from examples.render_material.pathtracer_backend import MATERIALX_DEFAULT_SCREEN_COLOR_SRGB8
from examples.render_material.render_backend import (
    MATERIAL_VISUALIZER_BACKEND,
    PATH_TRACER_BACKEND,
    image_suffix_for_backend,
)
from examples.render_material.render_material_manifest import (
    RenderEntry,
    RenderManifest,
    RenderOutput,
    RenderResult,
    RenderSettings,
    SCHEMA_VERSION,
    read_manifest,
    read_results,
    write_manifest,
)
from examples.render_material.suite import (
    exit_code_for_result,
    progress_message_for_entry,
    run_provider,
)
from examples.render_material.manifest_runner import RenderRunResult, render_entries
from tools.material_image_comparison import compare_image_sets


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MATERIALX_ROOT = PROJECT_ROOT / "external" / "MaterialX"
MDL_ROOT = PROJECT_ROOT / "data" / "assets" / "mdl_sdk_examples"


RENDER_BACKENDS = (PATH_TRACER_BACKEND, MATERIAL_VISUALIZER_BACKEND)


def test_materialx_default_background_matches_glsl() -> None:
    assert MATERIALX_DEFAULT_SCREEN_COLOR_SRGB8 == (76, 76, 82)


def test_materialx_default_options_do_not_warn() -> None:
    options = materialx_suite.read_materialx_options(materialx_suite.DEFAULT_OPTIONS)
    assert materialx_suite.warning_messages(options) == []


def test_materialx_representative_options_do_not_warn() -> None:
    options = materialx_suite.read_materialx_options(
        materialx_suite.DEFAULT_OPTIONS.with_name("_options_representative.mtlx")
    )
    assert materialx_suite.warning_messages(options) == []


def test_render_run_exit_code_reports_failures(tmp_path: Path) -> None:
    result = RenderRunResult(
        output_dir=tmp_path,
        results=(
            RenderResult(
                entry_id="ok",
                status="pass",
                image="ok.png",
                message="Rendered",
            ),
            RenderResult(
                entry_id="bad",
                status="fail",
                image=None,
                message="failed",
            ),
        ),
    )
    assert exit_code_for_result(result) == 1


def test_material_image_compare_writes_html_and_assets_only(tmp_path: Path) -> None:
    entry = RenderEntry(
        entry_id="material:tiny",
        label="tiny",
        material_class="MaterialXMaterial",
        properties={"mtlx_path": "tiny.mtlx"},
        output=RenderOutput(kind="material"),
        provider="materialx",
        artifact_name="tiny",
    )
    settings = RenderSettings(width=2, height=2, samples_per_pixel=1, image_extension="png")
    for root, suffix, value in (
        (tmp_path / "reference", "_visualizer", 128),
        (tmp_path / "tested", "_falcor2", 96),
    ):
        image = Path("tiny" + suffix + ".png")
        _write_png(root / image, value)
        write_manifest(
            root / "manifest.json",
            RenderManifest(entries=(entry,), settings=settings),
            (
                RenderResult(
                    entry_id=entry.entry_id,
                    status="pass",
                    image=image.as_posix(),
                    message="Rendered",
                ),
            ),
        )

    output_dir = tmp_path / "comparison"
    stale_asset = output_dir / "html_assets" / "stale.png"
    _write_png(stale_asset, 255)
    report = compare_image_sets(tmp_path / "reference", tmp_path / "tested", output_dir)

    assert len(report.rows) == 1
    assert (output_dir / "comparison.html").exists()
    assert (output_dir / "html_assets").is_dir()
    assert not stale_asset.exists()
    assert {path.name for path in output_dir.iterdir()} == {"comparison.html", "html_assets"}


def test_material_image_compare_rejects_images_outside_roots(tmp_path: Path) -> None:
    entry = RenderEntry(
        entry_id="material:outside",
        label="outside",
        material_class="MaterialXMaterial",
        properties={"mtlx_path": "outside.mtlx"},
        output=RenderOutput(kind="material"),
        provider="materialx",
        artifact_name="outside",
    )
    settings = RenderSettings(width=2, height=2, samples_per_pixel=1, image_extension="png")
    _write_png(tmp_path / "outside.png", 64)
    for root in (tmp_path / "reference", tmp_path / "tested"):
        write_manifest(
            root / "manifest.json",
            RenderManifest(entries=(entry,), settings=settings),
            (
                RenderResult(
                    entry_id=entry.entry_id,
                    status="pass",
                    image="../outside.png",
                    message="Rendered",
                ),
            ),
        )

    output_dir = tmp_path / "comparison"
    report = compare_image_sets(tmp_path / "reference", tmp_path / "tested", output_dir)

    assert report.rows[0]["status"] == "missing_reference_image"
    assert not list((output_dir / "html_assets").glob("*.png"))


@pytest.mark.parametrize("render_backend", RENDER_BACKENDS)
def test_material_entries_require_radiance_ibl_path(tmp_path: Path, render_backend: str) -> None:
    options_path = tmp_path / "_options.mtlx"
    _write_materialx_smoke_options(
        options_path,
        tmp_path / "out",
        render_backend,
        radiance_ibl_path="",
    )
    provider = materialx_suite.MaterialXProvider(
        materialx_suite.read_materialx_options(options_path),
        MATERIALX_ROOT,
        "automatic",
    )

    with pytest.raises(ValueError, match="Material renders require radianceIBLPath"):
        run_provider(provider)


def test_material_entries_require_existing_radiance_ibl_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.hdr"
    options_path = tmp_path / "_options.mtlx"
    _write_materialx_smoke_options(
        options_path,
        tmp_path / "out",
        PATH_TRACER_BACKEND,
        radiance_ibl_path=missing_path.as_posix(),
    )
    provider = materialx_suite.MaterialXProvider(
        materialx_suite.read_materialx_options(options_path),
        MATERIALX_ROOT,
        "automatic",
    )

    with pytest.raises(FileNotFoundError, match="radianceIBLPath does not exist"):
        run_provider(provider)


def test_mdl_output_directory_is_cwd_relative(tmp_path: Path) -> None:
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, Path(".tmp/mdl-cwd-relative"), PATH_TRACER_BACKEND)

    options = mdl_suite.read_mdl_options(options_path)

    assert options.output_directory == Path(".tmp/mdl-cwd-relative")


def test_mdl_empty_output_directory_uses_default(tmp_path: Path) -> None:
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, Path(".tmp/ignored"), PATH_TRACER_BACKEND)
    payload = _read_json_options(options_path)
    payload["outputDirectory"] = ""
    _write_json_options(options_path, payload)

    options = mdl_suite.read_mdl_options(options_path)

    assert options.output_directory == Path(".tmp/render_material_mdl")


def test_mdl_default_radiance_ibl_path_comes_from_library(tmp_path: Path) -> None:
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, tmp_path / "out", PATH_TRACER_BACKEND)
    payload = _read_json_options(options_path)
    del payload["radianceIBLPath"]
    _write_json_options(options_path, payload)

    options = mdl_suite.read_mdl_options(options_path)

    assert options.radiance_ibl_path == MDL_ROOT / "resources" / "environment.hdr"


def test_materialx_output_subdirectory_preserves_relative_path_context() -> None:
    first = make_materialx_entry(
        label="first",
        material_path=MATERIALX_ROOT / "resources" / "Materials" / "First" / "gold.mtlx",
        materialx_root=MATERIALX_ROOT,
        basepath=MATERIALX_ROOT / "resources" / "Materials" / "First",
        element_name="gold",
        element_type="material",
        layering_mode="bsdf_mix",
    )
    second = make_materialx_entry(
        label="second",
        material_path=MATERIALX_ROOT / "resources" / "Materials" / "Second" / "gold.mtlx",
        materialx_root=MATERIALX_ROOT,
        basepath=MATERIALX_ROOT / "resources" / "Materials" / "Second",
        element_name="gold",
        element_type="material",
        layering_mode="bsdf_mix",
    )

    assert first.output_subdirectory == "resources/Materials/First/gold"
    assert second.output_subdirectory == "resources/Materials/Second/gold"


def test_mdl_discovery_normalizes_dot_output_names() -> None:
    entry = make_mdl_entry(
        {
            "mdl_material_name": "::package::material",
            "module_simple_name": "..",
            "label": "..",
        },
        MDL_ROOT,
        mdl_class_compilation=False,
    )

    assert entry.output_subdirectory == "mdl"
    assert entry.artifact_name
    assert entry.artifact_name not in {".", ".."}


@pytest.mark.parametrize("render_backend", RENDER_BACKENDS)
@pytest.mark.parametrize("radiance_ibl_path", [None, ""])
def test_mdl_material_entries_require_radiance_ibl_path(
    tmp_path: Path, render_backend: str, radiance_ibl_path: object
) -> None:
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, tmp_path / "out", render_backend)
    payload = _read_json_options(options_path)
    payload["radianceIBLPath"] = radiance_ibl_path
    _write_json_options(options_path, payload)
    provider = mdl_suite.MDLProvider(
        mdl_suite.read_mdl_options(options_path),
        "automatic",
    )

    with pytest.raises(ValueError, match="Material renders require radianceIBLPath"):
        run_provider(provider)


@pytest.mark.parametrize("render_backend", RENDER_BACKENDS)
def test_mdl_material_entries_require_existing_radiance_ibl_path(
    tmp_path: Path, render_backend: str
) -> None:
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, tmp_path / "out", render_backend)
    payload = _read_json_options(options_path)
    payload["radianceIBLPath"] = str(tmp_path / "missing.hdr")
    _write_json_options(options_path, payload)
    provider = mdl_suite.MDLProvider(
        mdl_suite.read_mdl_options(options_path),
        "automatic",
    )

    with pytest.raises(FileNotFoundError, match="radianceIBLPath does not exist"):
        run_provider(provider)


def test_mdl_reference_quality_and_cli_override_select_samples(tmp_path: Path) -> None:
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, tmp_path / "out", PATH_TRACER_BACKEND)
    options = mdl_suite.read_mdl_options(options_path)

    assert mdl_suite.MDLProvider(options, "automatic").render_settings().samples_per_pixel == 32

    payload = _read_json_options(options_path)
    payload["enableReferenceQuality"] = True
    _write_json_options(options_path, payload)
    options = mdl_suite.read_mdl_options(options_path)

    assert mdl_suite.MDLProvider(options, "automatic").render_settings().samples_per_pixel == 256
    assert (
        mdl_suite.MDLProvider(
            options,
            "automatic",
            samples_per_pixel_override=7,
        )
        .render_settings()
        .samples_per_pixel
        == 7
    )


def test_removed_manifest_fields_are_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [
            {
                "entry_id": "material:legacy",
                "label": "legacy",
                "material_class": "MaterialXMaterial",
                "properties": {},
                "output": {"kind": "material"},
                "provider": "materialx",
                "provider_metadata": {},
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError):
        read_manifest(manifest_path)


def test_removed_result_fields_are_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [],
        "results": [
            {
                "entry_id": "material:legacy",
                "status": "pass",
                "image": "legacy.png",
                "message": "Rendered",
                "duration_seconds": 1.0,
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError):
        read_results(manifest_path)


def test_old_manifest_schema_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION - 1, "entries": [], "results": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported material image schema version"):
        read_manifest(manifest_path)
    with pytest.raises(ValueError, match="Unsupported material image schema version"):
        read_results(manifest_path)


def test_render_entries_prints_materialx_style_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = RenderEntry(
        entry_id="materialx:progress",
        label="progress",
        material_class="MaterialXMaterial",
        properties={"mtlx_path": "materials/progress.mtlx"},
        output=RenderOutput(kind="material"),
        provider="materialx",
        artifact_name="progress",
    )
    settings = RenderSettings(width=2, height=2, samples_per_pixel=1, image_extension="png")

    def render_func(_entry: RenderEntry, _settings: RenderSettings, image_path: Path) -> None:
        _write_png(image_path, 16)

    provider = materialx_suite.MaterialXProvider(
        materialx_suite.MaterialXOptions(),
        MATERIALX_ROOT,
        "d3d12",
    )
    result = render_entries(
        (entry,),
        settings,
        tmp_path,
        render_func=render_func,
        progress_message_func=lambda row: progress_message_for_entry(
            PATH_TRACER_BACKEND, provider.progress_subject(row)
        ),
    )

    assert result.results[0].status == "pass"
    assert capsys.readouterr().out == (
        "Validating Falcor2 PathTracer rendering for: materials/progress.mtlx\n"
    )


def _assert_smoke_manifest_has_pass(
    output_dir: Path,
    image_suffix: str,
    expected_samples_per_pixel: int,
) -> list[RenderResult]:
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    assert not (output_dir / "manifest.tsv").exists()

    manifest = read_manifest(manifest_path)
    assert manifest.settings is not None
    assert manifest.settings.samples_per_pixel == expected_samples_per_pixel

    results = read_results(manifest_path)
    assert results
    assert [result.entry_id for result in results] == [entry.entry_id for entry in manifest.entries]
    assert not [result for result in results if result.status == "fail"]

    passing_results = [
        result
        for result in results
        if result.status == "pass" and result.image and result.image.endswith(f"{image_suffix}.png")
    ]

    assert len(passing_results) == len(results)
    for result in passing_results:
        assert result.image is not None
        assert (output_dir / result.image).exists()
    return passing_results


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((2, 2, 3), value, dtype=np.uint8)
    Image.fromarray(data, mode="RGB").save(path)


def _write_materialx_smoke_options(
    path: Path,
    output_dir: Path,
    render_backend: str,
    radiance_ibl_path: str = "resources/Lights/san_giuseppe_bridge.hdr",
) -> None:
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<materialx version="1.39">',
                '  <nodedef name="TestSuiteOptions">',
                '    <input name="shaderInterfaces" type="integer" value="2" />',
                '    <input name="renderSize" type="vector2" value="64, 64" />',
                f'    <input name="renderBackend" type="string" value="{render_backend}" />',
                '    <input name="dumpGeneratedCode" type="boolean" value="false" />',
                '    <input name="renderGeometry" type="string" value="sphere.obj" />',
                '    <input name="enableDirectLighting" type="boolean" value="false" />',
                f'    <input name="radianceIBLPath" type="string" value="{radiance_ibl_path}" />',
                '    <input name="renderTestPaths" type="string" '
                'value="resources/Materials/Examples/StandardSurface/'
                "standard_surface_brass_tiled.mtlx,resources/Materials/TestSuite/stdlib/"
                'convolution/heighttonormal.mtlx" />',
                f'    <input name="outputDirectory" type="string" value="{output_dir.as_posix()}" />',
                "  </nodedef>",
                "</materialx>",
            ]
        ),
        encoding="utf-8",
    )


def _write_mdl_smoke_options(path: Path, output_dir: Path, render_backend: str) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "mdl-smoke",
                "overrideFiles": ["gun_metal.mdl"],
                "shaderInterfaces": 2,
                "renderBackend": render_backend,
                "renderSize": [64, 64],
                "dumpGeneratedCode": False,
                "renderGeometry": str(MATERIALX_ROOT / "resources" / "Geometry" / "sphere.obj"),
                "enableDirectLighting": False,
                "radianceIBLPath": str(MDL_ROOT / "resources" / "environment.hdr"),
                "renderTestPaths": [str(MDL_ROOT)],
                "outputDirectory": str(output_dir),
            }
        ),
        encoding="utf-8",
    )


def _read_json_options(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json_options(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.slow
@pytest.mark.parametrize("render_backend", RENDER_BACKENDS)
def test_materialx_smoke_writes_pass_manifest(tmp_path: Path, render_backend: str) -> None:
    try:
        device_type = materialx_suite.select_materialx_device_type("automatic")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    output_dir = tmp_path / f"render_material_mtlx_{render_backend}"
    options_path = tmp_path / "_options.mtlx"
    _write_materialx_smoke_options(options_path, output_dir, render_backend)
    shutil.rmtree(output_dir, ignore_errors=True)

    exit_code = materialx_suite.main(
        [
            "--options",
            str(options_path),
            "--device",
            device_type,
            "--render-backend",
            render_backend,
            "--samples-per-pixel",
            "1",
        ]
    )
    assert exit_code == 0

    passing_results = _assert_smoke_manifest_has_pass(
        output_dir,
        image_suffix_for_backend(render_backend),
        expected_samples_per_pixel=1,
    )
    assert len(passing_results) >= 2


@pytest.mark.slow
@pytest.mark.parametrize("render_backend", RENDER_BACKENDS)
def test_mdl_smoke_writes_pass_manifest(tmp_path: Path, render_backend: str) -> None:
    output_dir = tmp_path / f"render_material_mdl_{render_backend}"
    options_path = tmp_path / "_options.jsonc"
    _write_mdl_smoke_options(options_path, output_dir, render_backend)
    shutil.rmtree(output_dir, ignore_errors=True)

    exit_code = mdl_suite.main(
        [
            "--options",
            str(options_path),
            "--render-backend",
            render_backend,
            "--samples-per-pixel",
            "1",
        ]
    )
    assert exit_code == 0

    _assert_smoke_manifest_has_pass(
        output_dir,
        image_suffix_for_backend(render_backend),
        expected_samples_per_pixel=1,
    )
