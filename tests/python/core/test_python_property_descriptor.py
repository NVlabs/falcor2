# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PythonPropertyDescriptor and PythonClassReflection (C++ bridge)."""

import enum

import pytest
import slangpy as spy

from falcor2.reflection import (
    reflected,
    Property,
    PythonPropertyDescriptor,
    PythonClassReflection,
    UIFlags,
)
import falcor2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Color(enum.IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


@reflected
class SampleObject:
    flag = Property(True, doc="A boolean flag")
    count = Property(42, doc="An integer count", value_range=(0.0, 100.0))
    roughness = Property(
        0.5,
        doc="Surface roughness",
        value_range=(0.0, 1.0),
        ui_flags=UIFlags.advanced,
        ui_label="Roughness",
        ui_group="Appearance",
        ui_drag_speed=0.01,
    )
    name = Property("default", doc="Object name")
    color = Property(Color.RED, enum_type=Color, doc="Object color")


@reflected
class VectorSampleObject:
    position = Property(spy.float3(0, 0, 0), doc="Position")
    rotation = Property(spy.float4(0, 0, 0, 1), doc="Rotation")
    pixel = Property(spy.int2(0, 0), doc="Pixel coord")
    resolution = Property(spy.uint2(0, 0), doc="Resolution")
    transform = Property(
        spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        doc="Transform",
    )
    normal_matrix = Property(
        spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        doc="Normal matrix",
    )


@reflected
class HalfSampleObject:
    roughness = Property(spy.float16_t(0.5), doc="Half roughness")
    color = Property(spy.float16_t3(0, 0, 0), doc="Half color")


@reflected
class MetadataSampleObject:
    enabled = Property(True, doc="Enabled")
    color_value = Property(
        spy.float3(1, 0, 0),
        doc="Color",
        ui_flags=UIFlags.display_as_color,
    )
    debug_mode = Property(False, doc="Debug mode", ui_flags=UIFlags.hidden)
    detail_level = Property(
        0,
        doc="Detail level",
        value_range=(0.0, 10.0),
        ui_enable_if=lambda self: self.enabled,
    )
    combo_flags = Property(
        0.5,
        doc="Combo",
        ui_flags=UIFlags.advanced | UIFlags.hidden,
    )


@reflected
class ReadOnlyObject:
    @Property(doc="A read-only value")
    def derived(self) -> float:
        return 3.14


@reflected
class ComputedObject:
    def __init__(self):
        super().__init__()
        self._x = 0.0

    @Property(doc="X value", value_range=(0.0, 10.0))
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self._x = value


# ---------------------------------------------------------------------------
# PythonPropertyDescriptor tests
# ---------------------------------------------------------------------------


class TestPythonPropertyDescriptor:
    def test_create_from_info(self):
        info = SampleObject._reflected_properties[0]  # flag
        desc = PythonPropertyDescriptor(info)
        assert desc.name == "flag"
        assert desc.is_read_only is False
        assert desc.has_default_value is True

    def test_bool_round_trip(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "flag"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()

        val = desc.get_any_value(obj)
        assert val is True

        desc.set_any_value(obj, False)
        assert obj.flag is False
        assert desc.get_any_value(obj) is False

    def test_int_round_trip(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "count"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()

        val = desc.get_any_value(obj)
        assert val == 42

        desc.set_any_value(obj, 99)
        assert obj.count == 99

    def test_float_round_trip(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()

        val = desc.get_any_value(obj)
        assert val == pytest.approx(0.5)

        desc.set_any_value(obj, 0.8)
        assert obj.roughness == pytest.approx(0.8)

    def test_string_round_trip(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "name"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()

        val = desc.get_any_value(obj)
        assert val == "default"

        desc.set_any_value(obj, "changed")
        assert obj.name == "changed"

    def test_enum_round_trip(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "color"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()

        val = desc.get_enum_as_int64(obj)
        assert val == 0

        desc.set_enum_from_int64(obj, 2)
        assert obj.color == Color.BLUE

    def test_metadata_doc(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        assert desc.doc == "Surface roughness"

    def test_metadata_value_range(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        vr = desc.value_range
        assert vr is not None
        assert vr[0] == pytest.approx(0.0)
        assert vr[1] == pytest.approx(1.0)

    def test_metadata_group(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        assert desc.ui_group == "Appearance"

    def test_metadata_label(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        assert desc.ui_label == "Roughness"

    def test_metadata_drag_speed(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        assert desc.ui_drag_speed == pytest.approx(0.01)

    def test_is_default(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        assert desc.is_default(obj) is True
        obj.roughness = 0.9
        assert desc.is_default(obj) is False

    def test_reset(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        obj.roughness = 0.9
        desc.reset(obj)
        assert obj.roughness == pytest.approx(0.5)

    def test_read_only(self):
        info = ReadOnlyObject._reflected_properties[0]
        desc = PythonPropertyDescriptor(info)
        assert desc.is_read_only is True

        obj = ReadOnlyObject()
        val = desc.get_any_value(obj)
        assert val == pytest.approx(3.14)

    def test_computed_round_trip(self):
        info = ComputedObject._reflected_properties[0]
        desc = PythonPropertyDescriptor(info)
        obj = ComputedObject()

        val = desc.get_any_value(obj)
        assert val == pytest.approx(0.0)

        desc.set_any_value(obj, 5.5)
        assert obj.x == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Vector/matrix descriptor round-trip tests
# ---------------------------------------------------------------------------


class TestVectorMatrixDescriptor:
    def _get_desc(self, name: str) -> PythonPropertyDescriptor:
        info = [p for p in VectorSampleObject._reflected_properties if p.name == name][0]
        return PythonPropertyDescriptor(info)

    def test_float3_round_trip(self):
        desc = self._get_desc("position")
        obj = VectorSampleObject()
        val = desc.get_any_value(obj)
        assert val == spy.float3(0, 0, 0)
        desc.set_any_value(obj, spy.float3(1, 2, 3))
        assert desc.get_any_value(obj) == spy.float3(1, 2, 3)

    def test_float4_round_trip(self):
        desc = self._get_desc("rotation")
        obj = VectorSampleObject()
        val = desc.get_any_value(obj)
        assert val == spy.float4(0, 0, 0, 1)
        desc.set_any_value(obj, spy.float4(1, 0, 0, 0))
        assert desc.get_any_value(obj) == spy.float4(1, 0, 0, 0)

    def test_int2_round_trip(self):
        desc = self._get_desc("pixel")
        obj = VectorSampleObject()
        val = desc.get_any_value(obj)
        assert val == spy.int2(0, 0)
        desc.set_any_value(obj, spy.int2(10, 20))
        assert desc.get_any_value(obj) == spy.int2(10, 20)

    def test_uint2_round_trip(self):
        desc = self._get_desc("resolution")
        obj = VectorSampleObject()
        val = desc.get_any_value(obj)
        assert val == spy.uint2(0, 0)
        desc.set_any_value(obj, spy.uint2(1920, 1080))
        assert desc.get_any_value(obj) == spy.uint2(1920, 1080)

    def test_float4x4_round_trip(self):
        desc = self._get_desc("transform")
        obj = VectorSampleObject()
        new_val = spy.float4x4([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])
        desc.set_any_value(obj, new_val)
        assert desc.get_any_value(obj) == new_val

    def test_float3x3_round_trip(self):
        desc = self._get_desc("normal_matrix")
        obj = VectorSampleObject()
        new_val = spy.float3x3([0, 1, 0, 1, 0, 0, 0, 0, 1])
        desc.set_any_value(obj, new_val)
        assert desc.get_any_value(obj) == new_val


# ---------------------------------------------------------------------------
# Properties serialization tests
# ---------------------------------------------------------------------------


class TestPropertiesSerialization:
    def test_bool_write_read(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "flag"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        obj.flag = False

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)
        assert props.has_property("flag")

        obj2 = SampleObject()
        assert obj2.flag is True
        result = desc.read_from_properties(obj2, props)
        assert result is True
        assert obj2.flag is False

    def test_int_write_read(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "count"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        obj.count = 77

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)

        obj2 = SampleObject()
        desc.read_from_properties(obj2, props)
        assert obj2.count == 77

    def test_float_write_read(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        obj.roughness = 0.75

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)

        obj2 = SampleObject()
        desc.read_from_properties(obj2, props)
        assert obj2.roughness == pytest.approx(0.75)

    def test_string_write_read(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "name"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        obj.name = "test_name"

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)

        obj2 = SampleObject()
        desc.read_from_properties(obj2, props)
        assert obj2.name == "test_name"

    def test_is_serializable(self):
        for info in SampleObject._reflected_properties:
            desc = PythonPropertyDescriptor(info)
            assert desc.is_serializable_to_properties is True

    def test_read_missing_key(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "count"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        props = falcor2.Properties()
        result = desc.read_from_properties(obj, props)
        assert result is False
        assert obj.count == 42  # unchanged

    def test_enum_write_read(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "color"][0]
        desc = PythonPropertyDescriptor(info)
        obj = SampleObject()
        obj.color = Color.BLUE

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)
        assert props.has_property("color")

        obj2 = SampleObject()
        assert obj2.color == Color.RED
        result = desc.read_from_properties(obj2, props)
        assert result is True
        assert obj2.color == Color.BLUE


# ---------------------------------------------------------------------------
# Vector/matrix serialization tests
# ---------------------------------------------------------------------------


class TestVectorMatrixSerialization:
    def _roundtrip(self, name: str, new_value: object) -> object:
        info = [p for p in VectorSampleObject._reflected_properties if p.name == name][0]
        desc = PythonPropertyDescriptor(info)
        obj = VectorSampleObject()
        setattr(obj, name, new_value)

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)
        assert props.has_property(name)

        obj2 = VectorSampleObject()
        result = desc.read_from_properties(obj2, props)
        assert result is True
        return getattr(obj2, name)

    def test_float3_write_read(self):
        assert self._roundtrip("position", spy.float3(1, 2, 3)) == spy.float3(1, 2, 3)

    def test_float4_write_read(self):
        assert self._roundtrip("rotation", spy.float4(1, 0, 0, 0)) == spy.float4(1, 0, 0, 0)

    def test_int2_write_read(self):
        assert self._roundtrip("pixel", spy.int2(10, 20)) == spy.int2(10, 20)

    def test_uint2_write_read(self):
        assert self._roundtrip("resolution", spy.uint2(1920, 1080)) == spy.uint2(1920, 1080)

    def test_float4x4_write_read(self):
        val = spy.float4x4([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])
        assert self._roundtrip("transform", val) == val

    def test_float3x3_write_read(self):
        val = spy.float3x3([0, 1, 0, 1, 0, 0, 0, 0, 1])
        assert self._roundtrip("normal_matrix", val) == val


class TestHalfSerialization:
    def test_float16_write_read(self):
        info = [p for p in HalfSampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        obj = HalfSampleObject()
        obj.roughness = spy.float16_t(0.75)

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)

        obj2 = HalfSampleObject()
        result = desc.read_from_properties(obj2, props)
        assert result is True
        assert float(obj2.roughness) == pytest.approx(0.75)

    def test_float16_vector_write_read(self):
        info = [p for p in HalfSampleObject._reflected_properties if p.name == "color"][0]
        desc = PythonPropertyDescriptor(info)
        obj = HalfSampleObject()
        obj.color = spy.float16_t3(0.25, 0.5, 0.75)

        props = falcor2.Properties()
        desc.write_to_properties(obj, props)

        obj2 = HalfSampleObject()
        result = desc.read_from_properties(obj2, props)
        assert result is True
        assert obj2.color == spy.float16_t3(0.25, 0.5, 0.75)


# ---------------------------------------------------------------------------
# Metadata verification tests
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_flags_advanced(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "roughness"][0]
        desc = PythonPropertyDescriptor(info)
        assert UIFlags.advanced in desc.ui_flags

    def test_flags_display_as_color(self):
        info = [p for p in MetadataSampleObject._reflected_properties if p.name == "color_value"][0]
        desc = PythonPropertyDescriptor(info)
        assert UIFlags.display_as_color in desc.ui_flags

    def test_flags_hidden(self):
        info = [p for p in MetadataSampleObject._reflected_properties if p.name == "debug_mode"][0]
        desc = PythonPropertyDescriptor(info)
        assert UIFlags.hidden in desc.ui_flags

    def test_flags_combined(self):
        info = [p for p in MetadataSampleObject._reflected_properties if p.name == "combo_flags"][0]
        desc = PythonPropertyDescriptor(info)
        flags = desc.ui_flags
        assert UIFlags.advanced in flags
        assert UIFlags.hidden in flags

    def test_enable_if_present(self):
        info = [p for p in MetadataSampleObject._reflected_properties if p.name == "detail_level"][
            0
        ]
        desc = PythonPropertyDescriptor(info)
        assert desc.has_ui_enable_if is True

    def test_enable_if_absent(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "flag"][0]
        desc = PythonPropertyDescriptor(info)
        assert desc.has_ui_enable_if is False

    def test_no_flags_returns_empty(self):
        info = [p for p in SampleObject._reflected_properties if p.name == "flag"][0]
        desc = PythonPropertyDescriptor(info)
        assert desc.ui_flags == UIFlags.none


# ---------------------------------------------------------------------------
# PythonClassReflection tests
# ---------------------------------------------------------------------------


class TestPythonClassReflection:
    def test_create(self):
        refl = PythonClassReflection(SampleObject)
        assert refl.class_name == "SampleObject"

    def test_descriptor_count(self):
        refl = PythonClassReflection(SampleObject)
        assert len(refl) == 5

    def test_descriptor_access(self):
        refl = PythonClassReflection(SampleObject)
        desc = refl[0]
        assert isinstance(desc, PythonPropertyDescriptor)
        assert desc.name == "flag"

    def test_all_descriptors(self):
        refl = PythonClassReflection(SampleObject)
        names = [refl[i].name for i in range(len(refl))]
        assert "flag" in names
        assert "count" in names
        assert "roughness" in names
        assert "name" in names
        assert "color" in names

    def test_round_trip_through_reflection(self):
        refl = PythonClassReflection(SampleObject)
        obj = SampleObject()

        # Find roughness descriptor
        desc = None
        for i in range(len(refl)):
            if refl[i].name == "roughness":
                desc = refl[i]
                break
        assert desc is not None

        desc.set_any_value(obj, 0.9)
        assert desc.get_any_value(obj) == pytest.approx(0.9)

    def test_index_out_of_range(self):
        refl = PythonClassReflection(SampleObject)
        with pytest.raises(RuntimeError):
            refl[100]

    def test_find_property(self):
        refl = PythonClassReflection(SampleObject)
        desc = refl.find_property("roughness")
        assert desc is not None
        assert desc.name == "roughness"

    def test_find_property_missing(self):
        refl = PythonClassReflection(SampleObject)
        desc = refl.find_property("nonexistent")
        assert desc is None

    def test_generation_initial(self):
        PythonClassReflection.clear_cache()
        refl = PythonClassReflection(SampleObject)
        assert refl.generation == 0

    def test_generation_increments_on_rebuild(self):
        PythonClassReflection.clear_cache()
        refl = PythonClassReflection(SampleObject)
        assert refl.generation == 0
        refl.rebuild()
        assert refl.generation == 1
        refl.rebuild()
        assert refl.generation == 2

    def test_per_class_caching(self):
        """Two PythonClassReflection instances for the same class share descriptors."""
        PythonClassReflection.clear_cache()
        refl1 = PythonClassReflection(SampleObject)
        refl2 = PythonClassReflection(SampleObject)
        # Both should see the same generation (shared cached data).
        assert refl1.generation == refl2.generation
        assert len(refl1) == len(refl2)

    def test_rebuild_propagates_to_shared_instances(self):
        """rebuild() on one replaces the cache; new instances see the new generation."""
        PythonClassReflection.clear_cache()
        refl1 = PythonClassReflection(SampleObject)
        gen_before = refl1.generation
        refl1.rebuild()
        # A new instance should pick up the rebuilt (newer) cache entry.
        refl2 = PythonClassReflection(SampleObject)
        assert refl2.generation == gen_before + 1

    def test_construct_from_instance(self):
        """PythonClassReflection can be constructed from an instance (resolves to type)."""
        PythonClassReflection.clear_cache()
        obj = SampleObject()
        refl = PythonClassReflection(obj)
        assert refl.class_name == "SampleObject"
        assert len(refl) == 5

    def test_ensure_up_to_date_noop(self):
        """ensure_up_to_date is a no-op when properties haven't changed."""
        PythonClassReflection.clear_cache()
        refl = PythonClassReflection(SampleObject)
        gen = refl.generation
        refl.ensure_up_to_date()
        assert refl.generation == gen  # No rebuild happened.

    def test_clear_cache(self):
        """clear_cache purges the global cache; new instances rebuild from scratch."""
        refl1 = PythonClassReflection(SampleObject)
        refl1.rebuild()
        gen_before = refl1.generation
        PythonClassReflection.clear_cache()
        refl2 = PythonClassReflection(SampleObject)
        # After clearing, the new instance starts at generation 0 again.
        assert refl2.generation == 0

    def test_different_classes_independent(self):
        """Separate Python classes get independent cached descriptors."""
        PythonClassReflection.clear_cache()
        refl_sample = PythonClassReflection(SampleObject)
        refl_readonly = PythonClassReflection(ReadOnlyObject)
        assert refl_sample.class_name == "SampleObject"
        assert refl_readonly.class_name == "ReadOnlyObject"
        assert len(refl_sample) != len(refl_readonly)

    def test_find_property_by_name(self):
        """find_property returns the correct descriptor."""
        PythonClassReflection.clear_cache()
        refl = PythonClassReflection(SampleObject)
        desc = refl.find_property("roughness")
        assert desc is not None
        assert desc.name == "roughness"
        assert refl.find_property("nonexistent") is None


# ---------------------------------------------------------------------------
# OnChange metadata tests
# ---------------------------------------------------------------------------


class TestOnChangeMetadata:
    def test_on_change_present(self):
        """on_change callback is accessible as metadata on the C++ descriptor."""

        @reflected
        class Obj:
            val = Property(1.0, on_change=lambda self: None)

        desc = PythonPropertyDescriptor(Obj._reflected_properties[0])
        # The on_change callback is stored as OnChange metadata, which is
        # extracted by the C++ build_metadata. We can verify the descriptor
        # was constructed without error and the property round-trips.
        obj = Obj()
        assert desc.get_any_value(obj) == pytest.approx(1.0)

    def test_on_change_absent(self):
        """Properties without on_change still work normally."""

        @reflected
        class Obj:
            val = Property(1.0)

        desc = PythonPropertyDescriptor(Obj._reflected_properties[0])
        obj = Obj()
        desc.set_any_value(obj, 2.0)
        assert obj.val == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dynamic schema tests
# ---------------------------------------------------------------------------


class TestDynamicSchema:
    def test_ensure_up_to_date_detects_change(self):
        """ensure_up_to_date detects when _reflected_properties is replaced."""

        @reflected
        class Obj:
            x = Property(1)

        PythonClassReflection.clear_cache()
        refl = PythonClassReflection(Obj)
        assert len(refl) == 1
        gen_before = refl.generation

        # Simulate adding a property by re-running @reflected on a subclass
        # that replaces the property list.
        Obj.y = Property(2)
        Obj.y.__set_name__(Obj, "y")
        Obj._reflected_properties = list(Obj._reflected_properties) + [Obj.y._make_info("y", Obj)]

        refl.ensure_up_to_date()
        assert refl.generation == gen_before + 1
        assert len(refl) == 2

    def test_ensure_up_to_date_noop_when_unchanged(self):
        """ensure_up_to_date is a no-op when nothing changed."""

        @reflected
        class Obj:
            x = Property(1)

        PythonClassReflection.clear_cache()
        refl = PythonClassReflection(Obj)
        gen = refl.generation
        refl.ensure_up_to_date()
        assert refl.generation == gen


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
