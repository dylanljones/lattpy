# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2022, Dylan Jones

import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import lattpy as lp

MB = 1024 * 1024
MAX_SITES = 10_000_000
RUNS = 3
MAX_POINTS = 30

overwrite = False
latts = {
    "chain": lp.simple_chain(),
    "square": lp.simple_square(),
    "hexagonal": lp.simple_hexagonal(),
    "cubic": lp.simple_cubic(),
}


class Profiler:
    """Profiler object for measuring time, memory and other stats."""

    def __init__(self):
        self._timefunc = time.perf_counter
        self._snapshot = None
        self._t0 = 0
        self._m0 = 0
        self._thread = None

    @property
    def seconds(self) -> float:
        """Returns the time since the timer has been started in seconds."""
        return self._timefunc() - self._t0

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def memory(self):
        return self.get_memory() - self._m0

    def start(self):
        """Start the profiler."""
        tracemalloc.start()
        self._m0 = self.get_memory()
        self._t0 = self._timefunc()

    def stop(self) -> None:
        """Stop the profiler."""
        self._t0 = self._timefunc()
        tracemalloc.stop()

    def get_memory(self):
        self.take_snapshot()
        stats = self.statistics("filename")
        return sum(stat.size for stat in stats)

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


def _benchmark_build_periodic(latt: lp.Lattice, sizes, runs, **kwargs):
    profiler = Profiler()
    data = np.zeros((len(sizes), 4))
    for i, size in enumerate(sizes):
        t, mem, per = 0.0, 0.0, 0.0
        for _ in range(runs):
            profiler.start()
            latt.build(np.ones(latt.dim) * size, **kwargs)
            t += profiler.seconds
            mem += profiler.memory
            profiler.start()
            latt.set_periodic(True)
            per += profiler.seconds

        data[i, 0] = latt.num_sites
        data[i, 1] = t / runs
        data[i, 2] = mem / runs
        data[i, 3] = per / runs
    return data


def bench_build_periodic(latt, runs=RUNS):
    total_sizes = np.geomspace(10, MAX_SITES, MAX_POINTS).astype(np.int64)
    if latt.dim > 1:
        sizes = np.unique(np.power(total_sizes, 1 / latt.dim).astype(np.int64))
    else:
        sizes = total_sizes
    return _benchmark_build_periodic(latt, sizes, runs, primitive=True)


def plot_benchmark_build(data):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    for name, arr in data.items():
        num_sites = arr[:, 0]
        line = ax1.plot(num_sites, arr[:, 1], label=name)[0]
        ax1.plot(num_sites, arr[:, 1], ls="--", color=line.get_color())
        ax2.plot(num_sites[1:], arr[1:, 2] / MB, label=name)
        ax3.plot(num_sites, arr[:, 3], label=name)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Memory (MB)")
    ax3.set_ylabel("Time (s)")
    ax1.set_xlabel("N")
    ax2.set_xlabel("N")
    ax3.set_xlabel("N")
    ax1.set_ylim(0.001, 100)
    ax3.set_ylim(0.001, 100)
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig("bench_build_time.png")
    fig2.savefig("bench_build_memory.png")
    fig3.savefig("bench_periodic_time.png")


def main():
    file = "benchmark_build_periodic.npz"
    if overwrite or not os.path.exists(file):
        data = dict()
        for name, latt in latts.items():
            print("Benchmarking build:", name)
            values = bench_build_periodic(latt)
            data[name] = values
            np.savez(file, **data)
    else:
        data = np.load(file)

    plot_benchmark_build(data)
    plt.show()


if __name__ == "__main__":
    main()
