"""Atomic files, process checks, resource guards and portable detach primitives."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON via a same-directory replace to survive process interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def alive(pid: int | None) -> bool:
    """Check a process without sending a destructive signal."""
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def free_disk_gb(path: Path) -> float:
    """Return free bytes on the campaign volume in GiB."""
    return shutil.disk_usage(path).free / (1024**3)


def available_memory_gb() -> float:
    """Read available physical memory without an optional dependency."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MemoryStatus()
        state.dwLength = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):  # type: ignore[attr-defined]
            return state.ullAvailPhys / (1024**3)
        raise OSError("Could not read memory status")
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return 2.0


def process_peak_rss_gb() -> float:
    """Return this process's peak resident footprint for preflight sizing."""
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
        get_current_process.restype = ctypes.c_void_p
        process = get_current_process()
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
        get_memory_info.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        ok = get_memory_info(process, ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("Could not read Windows process peak RSS")
        return counters.PeakWorkingSetSize / (1024**3)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def safe_worker_count(
    cpu_count: int,
    available_gb: float,
    reserve_gb: float,
    per_worker_gb: float,
    configured_max: int,
) -> int:
    """Cap workers by CPU, observed memory and the frozen configuration."""
    memory_safe = max(0, int((available_gb - reserve_gb) // per_worker_gb))
    return min(configured_max, max(1, cpu_count // 2), memory_safe)


def keep_awake(active: bool) -> None:
    """Hold or release a Windows system sleep inhibition only while watchdog runs."""
    if sys.platform == "win32":
        value = 0x80000000 | 0x1 if active else 0x80000000
        result = ctypes.windll.kernel32.SetThreadExecutionState(value)  # type: ignore[attr-defined]
        if result == 0:
            raise OSError("Windows could not set the system sleep inhibition state")


def detached_process(command: list[str], cwd: Path, log: Path) -> subprocess.Popen[bytes]:
    """Spawn a child independent of the caller's terminal and redirect both streams."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab", buffering=0)
    flags = 0
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            creationflags=flags,
            close_fds=True,
            **kwargs,
        )
    finally:
        stream.close()


def wait_for(path: Path, predicate: Any, timeout_s: float = 20) -> bool:
    """Wait briefly for a child-produced heartbeat or ready marker."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            if path.exists() and predicate(json.loads(path.read_text(encoding="utf-8"))):
                return True
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    return False
