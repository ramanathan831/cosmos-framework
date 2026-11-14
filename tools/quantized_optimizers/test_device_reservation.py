"""CPU process-lifecycle tests only; native CUDA pressure is a separate experiment."""

import sys

import pytest

from tools.quantized_optimizers import device_reservation


def stand_in(device, size, ready, *, corrupt=False):
    # No CUDA claim: these workers test supervision, identity and ownership only.
    script = """
import json, os, sys, time
from pathlib import Path
ready, device, size = sys.argv[1:]
p = Path(ready)
t = p.with_suffix('.tmp')
t.write_text(json.dumps({'pid':os.getpid(),'device':int(device),'bytes':int(size)}))
os.replace(t,p)
while True: time.sleep(60)
"""
    return [sys.executable, "-c", script, str(ready), str(device), str(size + int(corrupt))]


def test_disabled_reservation_has_no_worker_or_cuda_requirement(tmp_path, monkeypatch):
    def forbidden(*_):
        raise AssertionError("default path must not allocate pressure workers")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(device_reservation, "worker_command", forbidden)
    with device_reservation.ResidentDeviceMemory(0, tmp_path) as reservation:
        reservation.check()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [None, "worker_exit", "receipt"])
def test_pressure_lifetime_and_failure_cleanup(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        device_reservation,
        "worker_command",
        lambda *args: stand_in(*args, corrupt=failure == "receipt"),
    )
    reservation = device_reservation.ResidentDeviceMemory(1, tmp_path, timeout=10)

    def run():
        with reservation:
            assert len(reservation.processes) == 2
            reservation.check()
            if failure == "worker_exit":
                reservation.processes[0].terminate()
                reservation.processes[0].wait(timeout=10)
                reservation.check()

    if failure:
        with pytest.raises(RuntimeError, match="identity differs|pressure run is invalid") as error:
            run()
        assert error.value.__traceback__ is not None
    else:
        run()
    assert all(process.poll() is not None for process in reservation.processes)
    assert all(log.closed for log in reservation.logs)
