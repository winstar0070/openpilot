# USB GPU PCIe power-cycle bench test

This tool is manual and is not connected to launcher, manager, SCons, or modeld.
It runs only when `IsOffroad=true`, `deviceState.started=false`, fresh panda states
report ignition off, and the target is the custom ASM2464 device at exactly 12 or
5000 Mbps.

Check all gates without sending F3:

```sh
cd /data/openpilot
python3 openpilot/tools/usbgpu/pcie_power_cycle_test.py \
  --dry-run
```

Run one 500 ms F3 off/on cycle:

```sh
cd /data/openpilot
python3 openpilot/tools/usbgpu/pcie_power_cycle_test.py \
  --confirm-risk
```

The tool automatically detects the `ADD1:0001` device when exactly one is
present. If multiple matching devices exist, it refuses to choose; pass an
explicit identity such as `--device usb:4-10`. A failed ON may leave the GPU off
until external power is cycled, so keep physical power
recovery available during testing. `SIGTERM`, `SIGHUP`, `SIGQUIT`, and normal
keyboard interruption trigger one best-effort recovery ON; `SIGKILL`, host power
loss, or a failed USB control channel cannot be recovered by the script.

Each stage is atomically recorded under
`/data/community/egpu_logs/power_cycle/`. `latest.json` points to the most recent
attempt and the most recent 20 attempt logs are retained. An ON result of
`NO_DEVICE` is recorded as `detached_ambiguous`; the test passes only if the same
sysfs path subsequently remains configured at exactly 5000 Mbps for two seconds.
