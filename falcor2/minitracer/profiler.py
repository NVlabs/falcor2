# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import slangpy
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from enum import Flag
from weakref import WeakKeyDictionary


@dataclass
class ProfilerEvent:
    name: str
    cpu_start: float
    cpu_end: float
    cpu_duration: float
    gpu_start: float
    gpu_end: float
    gpu_duration: float


ProfilerFrame = list[ProfilerEvent]


class Profiler:

    class FrameState(Flag):
        idle = 0
        recording = 1
        pending = 2
        resolved = 3

    @dataclass
    class Event:
        name: str = ""
        nesting: int = 0
        query_start: int = 0
        query_end: int = 0
        cpu_start: float = 0
        cpu_end: float = 0
        gpu_start: float = 0
        gpu_end: float = 0

    @dataclass
    class Frame:
        query_pool: slangpy.QueryPool
        state: "Profiler.FrameState"
        events: list["Profiler.Event"]
        event_count: int

    def __init__(
        self, device: slangpy.Device, frames_in_flight: int = 1, max_events_per_frame: int = 4096
    ):
        super().__init__()
        self._device = device
        self._max_events_per_frame = max_events_per_frame * 2
        self._enabled = True
        self._frames = [
            Profiler.Frame(
                query_pool=self._device.create_query_pool(
                    slangpy.QueryType.timestamp, max_events_per_frame * 2
                ),
                state=Profiler.FrameState.idle,
                events=[Profiler.Event() for _ in range(max_events_per_frame)],
                event_count=0,
            )
            for _ in range(frames_in_flight)
        ]
        self._frame_index = -1
        self._nesting = 0

    @property
    def enabled(self) -> bool:
        return True

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def begin_frame(self):
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        frame = self._frames[self._frame_index]
        assert frame.state in (Profiler.FrameState.idle, Profiler.FrameState.resolved)
        frame.state = Profiler.FrameState.recording
        frame.event_count = 0
        self._nesting = 0

    def end_frame(self) -> Optional[ProfilerFrame]:
        frame = self._frames[self._frame_index]
        assert frame.state == Profiler.FrameState.recording
        frame.state = Profiler.FrameState.pending

        last_frame_index = (self._frame_index - 1) % len(self._frames)
        last_frame = self._frames[last_frame_index]
        if last_frame.state == Profiler.FrameState.idle:
            return None
        if last_frame.state == Profiler.FrameState.pending:
            self._resolve_frame(last_frame)
        assert last_frame.state == Profiler.FrameState.resolved
        return [
            ProfilerEvent(
                name=event.name,
                cpu_start=event.cpu_start,
                cpu_end=event.cpu_end,
                cpu_duration=event.cpu_end - event.cpu_start,
                gpu_start=event.gpu_start,
                gpu_end=event.gpu_end,
                gpu_duration=event.gpu_end - event.gpu_start,
            )
            for i, event in enumerate(last_frame.events)
            if i < last_frame.event_count
        ]

    @contextmanager
    def profile(self, name: str, command_buffer: slangpy.CommandEncoder):

        if not self._enabled:
            yield
            return
        frame = self._frames[self._frame_index]
        assert frame.state == Profiler.FrameState.recording
        if frame.event_count >= self._max_events_per_frame:
            raise RuntimeError(
                "Profiler reached maximum events! Increase max_events_per_frame (or disable profiling)."
            )
        event_index = frame.event_count
        event = frame.events[event_index]
        event.name = name
        event.nesting = self._nesting
        event.query_start = event_index * 2
        event.query_end = event_index * 2 + 1
        event.cpu_start = self._cpu_time()
        command_buffer.write_timestamp(frame.query_pool, event.query_start)
        frame.event_count += 1
        self._nesting += 1
        yield
        self._nesting -= 1
        command_buffer.write_timestamp(frame.query_pool, event.query_end)
        event.cpu_end = self._cpu_time()

    def _resolve_frame(self, frame: Frame):
        assert frame.state == Profiler.FrameState.pending
        timestamps = frame.query_pool.get_timestamp_results(0, frame.event_count * 2)
        for i in range(frame.event_count):
            frame.events[i].gpu_start = timestamps[i * 2]
            frame.events[i].gpu_end = timestamps[i * 2 + 1]
        frame.state = Profiler.FrameState.resolved

    def _cpu_time(self) -> float:
        return time()


PROFILER_BY_DEVICE: WeakKeyDictionary[slangpy.Device, Profiler] = WeakKeyDictionary()


def get_profiler_for_device(device: slangpy.Device) -> Profiler:
    if device not in PROFILER_BY_DEVICE:
        PROFILER_BY_DEVICE[device] = Profiler(device)
    return PROFILER_BY_DEVICE[device]


@contextmanager
def profile(name: str, command_buffer: slangpy.CommandEncoder):
    profiler = get_profiler_for_device(command_buffer.device)
    yield profiler.profile(name, command_buffer)
