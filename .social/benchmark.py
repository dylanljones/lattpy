# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Generates the lattice build time plot."""
import time
import threading
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from lattpy import Lattice

MEGABYTE = 1024 * 1024
GIGABYTE = 1024 * MEGABYTE


class ProfileThread(threading.Thread):
    def __init__(self, profiler, interval=0.5):
        super().__init__()
        self._running = False
        self._interval = interval
        self.profiler = profiler
        self.data = list()

    def stop(self):
        self._running = False

    def update(self):
        self.profiler.take_snapshot()
        t = self.profiler.seconds
        m = self.profiler.memory
        self.data.append([t, m])

    def run(self) -> None:
        self._running = True
        while self._running:
            self.update()
            time.sleep(self._interval)


class Profiler:
    """Profiler object for measuring time, memory and other stats."""

    __slots__ = ["_time", "_t0", "_snapshot", "_thread"]

    def __init__(self, method=None):
        self._time = method or time.perf_counter
        self._snapshot = None
        self._t0 = 0
        self._thread = None
        self.start()

    @property
    def seconds(self) -> float:
        """Returns the time since the timer has been started in seconds."""
        return self.time() - self._t0

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def memory(self):
        stats = self.statistics("filename")
        return sum(stat.size for stat in stats)

    def time(self) -> float:
        """Returns the current time as a timestamp."""
        return self._time()

    def start(self) -> None:
        """Start the profiler."""
        tracemalloc.start()
        self._t0 = self._time()

    @staticmethod
    def stop() -> None:
        """Stop the profiler."""
        tracemalloc.stop()

    def take_snapshot(self):
        snapshot = tracemalloc.take_snapshot()
        snapshot = snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )
        self._snapshot = snapshot
        return snapshot

    def statistics(self, group="lineno"):
        return self._snapshot.statistics(group)

    def start_watching(self, interval=0.5):
        self._thread = ProfileThread(self, interval)
        self._thread.start()
        return self._thread

    def stop_watching(self):
        self._thread.stop()
        self._thread.join()
        data = self._thread.data
        self._thread = None
        return np.asarray(data).T


def benchmark_time(func, sizes, calls=1, **kwargs):
    counts = np.zeros_like(sizes)
    times = np.zeros_like(sizes, dtype=np.float64)
    for i, size in enumerate(sizes):
        print(f"\rBenchmarking time: {i + 1}/{len(sizes)}", end="", flush=True)
        shape = (size, size)
        n, t = 0, 0
        for call in range(calls):
            p = Profiler()
            # t0 = time.perf_counter()
            latt = func(shape, **kwargs)
            t += p.seconds  # time.perf_counter() - t0

            n = latt.num_sites
        counts[i] = n
        times[i] = t / calls
    print()
    return counts, times


def benchmark_memory(func, sizes, dt=0.01, calls=1, **kwargs):
    counts = np.zeros_like(sizes)

    max_memory = np.zeros_like(sizes, dtype=np.float64)
    memory = np.zeros_like(sizes, dtype=np.float64)
    for i, size in enumerate(sizes):
        print(f"\rBenchmarking Memory: {i + 1}/{len(sizes)}", end="", flush=True)
        shape = (size, size)
        n, mem, max_mem = 0, 0, 0
        for call in range(calls):
            p = Profiler()
            p.start_watching(dt)
            latt = func(shape, **kwargs)
            times, _memory = p.stop_watching()
            n = latt.num_sites
            mem += _memory[-1]
            max_mem += np.max(_memory)
        counts[i] = n
        memory[i] = mem / calls
        max_memory[i] = max_mem / calls
    print()
    return counts, memory, max_memory


def profile_build_memory(func, shape, args=(), dt=0.01):
    p = Profiler()
    p.start_watching(dt)
    latt = func(shape, *args)
    times, memory = p.stop_watching()
    num_sites = latt.num_sites
    return num_sites, times, memory


def plot_benchmark_time(func, sys_sizes, calls=5):

    sizes = np.sqrt(sys_sizes).astype(np.int64)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("System size")
    ax.set_ylabel("Time (s)")
    ax.set_axisbelow(True)
    ax.grid()

    # Single thread
    x, y = benchmark_time(func, sizes, calls, num_jobs=+1)
    ax.plot(x, y, marker="o", label="Single-thread")

    # Multi thread
    x, y = benchmark_time(func, sizes, calls, num_jobs=-1)
    ax.plot(x, y, marker="o", label="Multi-thread")

    # ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_benchmark_memory(func, sys_sizes, dt=0.01, calls=1):
    sizes = np.sqrt(sys_sizes).astype(np.int64)
    counts, memory, max_memory = benchmark_memory(func, sizes, dt, calls, num_jobs=-1)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("System size")
    ax.set_ylabel("Memory (MiB)")

    ax.plot(counts, memory / MEGABYTE, marker="o", label="Stored data")
    ax.plot(counts, max_memory / MEGABYTE, marker="o", label="Max. memory")

    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_profile_memory(func, shape, args=(), dt=0.01):
    num_sites, times, memory = profile_build_memory(func, shape, args, dt)

    fig, ax = plt.subplots()
    ax.plot(times, memory / GIGABYTE)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (GiB)")
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    return fig, ax


def build(shape, num_jobs=-1):
    latt = Lattice(np.eye(2))
    latt.add_atom()
    latt.add_connections(1)
    latt.build(shape, relative=True, num_jobs=num_jobs)
    return latt


def main():
    sys_sizes = [10, 35, 100, 350, 1e3, 3.5e3, 1e4, 3.5e4, 1e5, 3.5e5, 1e6, 3.5e6, 1e7]

    fig, ax = plot_benchmark_memory(build, sys_sizes)
    fig.savefig("benchmark_memory.png")

    fig, ax = plot_benchmark_time(build, sys_sizes, calls=2)
    fig.savefig("benchmark_time.png")

    # shape = (10000, 1000)
    # fig, ax = plot_profile_memory(build, shape)
    # fig.savefig("build_memory.png")

    plt.show()


if __name__ == "__main__":
    main()
