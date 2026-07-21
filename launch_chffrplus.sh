#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$DIR/launch_env.sh"

USBGPU_USB_SYSFS_ROOT="${USBGPU_USB_SYSFS_ROOT:-/sys/bus/usb/devices}"
USBGPU_PLATFORM_SYSFS_ROOT="${USBGPU_PLATFORM_SYSFS_ROOT:-/sys/bus/platform/devices}"
USBGPU_ATTACH_TIMEOUT_SEC="${USBGPU_ATTACH_TIMEOUT_SEC:-10}"
USBGPU_INITIAL_STABILITY_TIMEOUT_SEC="${USBGPU_INITIAL_STABILITY_TIMEOUT_SEC:-12}"
USBGPU_RECOVERY_TIMEOUT_SEC="${USBGPU_RECOVERY_TIMEOUT_SEC:-30}"
USBGPU_STABLE_SECONDS="${USBGPU_STABLE_SECONDS:-8}"
USBGPU_LOW_SPEED_GRACE_SEC="${USBGPU_LOW_SPEED_GRACE_SEC:-5}"
USBGPU_RECOVERY_ATTEMPTS="${USBGPU_RECOVERY_ATTEMPTS:-3}"
USBGPU_REBIND_DELAY_SEC="${USBGPU_REBIND_DELAY_SEC:-3}"
USBGPU_RETRY_DELAY_SEC="${USBGPU_RETRY_DELAY_SEC:-5}"

function agnos_init {
  # TODO: move this to agnos
  sudo rm -f /data/etc/NetworkManager/system-connections/*.nmmeta
  rm -f /data/scons_cache/config.lock

  # set success flag for current boot slot
  sudo abctl --set_success

  # TODO: do this without udev in AGNOS
  # udev does this, but sometimes we startup faster
  sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
  sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0

  # Check if AGNOS update is required
  if [ $(< /VERSION) != "$AGNOS_VERSION" ]; then
    AGNOS_PY="$DIR/openpilot/common/hardware/tici/agnos.py"
    MANIFEST="$DIR/openpilot/system/hardware/tici/agnos.json"
    if $AGNOS_PY --verify $MANIFEST; then
      sudo reboot
    fi
    $DIR/openpilot/common/hardware/tici/updater $AGNOS_PY $MANIFEST
  fi
}

function usbgpu_speed {
  local usbgpu_device usbgpu_speed_candidate usbgpu_speed_value=0
  for usbgpu_device in "$USBGPU_USB_SYSFS_ROOT"/*; do
    [ -f "$usbgpu_device/idVendor" ] || continue
    [ -f "$usbgpu_device/idProduct" ] || continue
    [ "$(< "$usbgpu_device/idVendor")" = "add1" ] || continue
    [ "$(< "$usbgpu_device/idProduct")" = "0001" ] || continue
    [ -f "$usbgpu_device/speed" ] || continue
    usbgpu_speed_candidate="$(< "$usbgpu_device/speed")"
    usbgpu_speed_candidate="${usbgpu_speed_candidate%%.*}"
    if [ "$usbgpu_speed_candidate" -gt "$usbgpu_speed_value" ] 2>/dev/null; then
      usbgpu_speed_value="$usbgpu_speed_candidate"
    fi
  done
  echo "$usbgpu_speed_value"
}

function wait_for_usbgpu_stable {
  local usbgpu_timeout="$1"
  local usbgpu_break_on_low_speed="$2"
  local usbgpu_current_speed usbgpu_elapsed usbgpu_stable_seconds=0 usbgpu_slow_seconds=0

  for ((usbgpu_elapsed = 0; usbgpu_elapsed < usbgpu_timeout; usbgpu_elapsed++)); do
    usbgpu_current_speed="$(usbgpu_speed)"
    if [ "$usbgpu_current_speed" -ge 5000 ] 2>/dev/null; then
      usbgpu_stable_seconds=$((usbgpu_stable_seconds + 1))
      usbgpu_slow_seconds=0
      [ "$usbgpu_stable_seconds" -ge "$USBGPU_STABLE_SECONDS" ] && return 0
    elif [ "$usbgpu_current_speed" -gt 0 ] 2>/dev/null; then
      usbgpu_stable_seconds=0
      usbgpu_slow_seconds=$((usbgpu_slow_seconds + 1))
      if [ "$usbgpu_break_on_low_speed" = "1" ] && [ "$usbgpu_slow_seconds" -ge "$USBGPU_LOW_SPEED_GRACE_SEC" ]; then
        return 1
      fi
    else
      usbgpu_stable_seconds=0
      usbgpu_slow_seconds=0
    fi
    sleep 1
  done
  return 1
}

function prepare_usbgpu {
  local usbgpu_xhci_device="xhci-hcd.1.auto"
  local usbgpu_xhci_path="$USBGPU_PLATFORM_SYSFS_ROOT/$usbgpu_xhci_device"
  local usbgpu_xhci_driver usbgpu_current_speed
  local usbgpu_attempt usbgpu_waited

  [ -L "$usbgpu_xhci_path/driver" ] || return

  for ((usbgpu_waited = 0; usbgpu_waited < USBGPU_ATTACH_TIMEOUT_SEC; usbgpu_waited++)); do
    usbgpu_current_speed="$(usbgpu_speed)"
    [ "$usbgpu_current_speed" -gt 0 ] 2>/dev/null && break
    sleep 1
  done
  if ! [ "$usbgpu_current_speed" -gt 0 ] 2>/dev/null; then
    echo "USB GPU not detected within ${USBGPU_ATTACH_TIMEOUT_SEC}s; modeld will fall back to QCOM"
    return
  fi

  usbgpu_xhci_driver="$(readlink -f "$usbgpu_xhci_path/driver")"
  [ -d "$usbgpu_xhci_driver" ] || return
  echo "USB GPU detected at ${usbgpu_current_speed}M"

  # The custom ASM2464PD firmware intentionally enumerates at low speed first.
  # tinygrad sends F3 to power PCIe and owns the resulting SuperSpeed
  # re-enumeration. Resetting xHCI here races that firmware state machine.
  if [ "$usbgpu_current_speed" -lt 5000 ] 2>/dev/null; then
    echo "USB GPU bootstrap device detected; handing off to tinygrad"
    return
  fi

  if [ "$usbgpu_current_speed" -ge 5000 ] 2>/dev/null; then
    if wait_for_usbgpu_stable "$USBGPU_INITIAL_STABILITY_TIMEOUT_SEC" 0; then
      echo "USB GPU ready at ${usbgpu_current_speed}M"
      return
    fi
    echo "USB GPU SuperSpeed link was not stable; resetting controller"
  fi

  for ((usbgpu_attempt = 1; usbgpu_attempt <= USBGPU_RECOVERY_ATTEMPTS; usbgpu_attempt++)); do
    echo "Resetting $usbgpu_xhci_device (attempt $usbgpu_attempt/$USBGPU_RECOVERY_ATTEMPTS)"

    if ! echo "$usbgpu_xhci_device" | sudo tee "$usbgpu_xhci_driver/unbind" > /dev/null; then
      echo "Failed to unbind $usbgpu_xhci_device"
      return
    fi
    sleep "$USBGPU_REBIND_DELAY_SEC"
    if ! echo "$usbgpu_xhci_device" | sudo tee "$usbgpu_xhci_driver/bind" > /dev/null; then
      echo "Failed to bind $usbgpu_xhci_device"
      return
    fi

    # Require eight continuous seconds at SuperSpeed after controller recovery.
    if wait_for_usbgpu_stable "$USBGPU_RECOVERY_TIMEOUT_SEC" 1; then
      usbgpu_current_speed="$(usbgpu_speed)"
      echo "USB GPU ready at ${usbgpu_current_speed}M"
      return
    fi

    usbgpu_current_speed="$(usbgpu_speed)"
    if [ "$usbgpu_attempt" -lt "$USBGPU_RECOVERY_ATTEMPTS" ]; then
      echo "USB GPU remained at ${usbgpu_current_speed}M; retrying in ${USBGPU_RETRY_DELAY_SEC} seconds"
      sleep "$USBGPU_RETRY_DELAY_SEC"
    fi
  done
  echo "USB GPU did not remain at SuperSpeed; modeld will fall back to QCOM"
}

function launch {
  # Remove orphaned git lock if it exists on boot
  [ -f "$DIR/.git/index.lock" ] && rm -f $DIR/.git/index.lock

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The DIR init file has to exist, with a newer modtime than anything in
  #    the DIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${DIR}/.overlay_init" ]; then
    find ${DIR}/.git -newer ${DIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${DIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $DIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $DIR
          cd $DIR

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          unset AGNOS_VERSION
          exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # submodule package symlinks for PYTHONPATH imports on device.
  # on PC these come from editable installs via pyproject.toml / uv.
  ln -sfn msgq_repo/msgq msgq
  ln -sfn opendbc_repo/opendbc opendbc
  ln -sfn rednose_repo/rednose rednose
  ln -sfn teleoprtc_repo/teleoprtc teleoprtc
  ln -sfn tinygrad_repo/tinygrad tinygrad

  # hardware specific init
  if [ -f /AGNOS ]; then
    agnos_init
    prepare_usbgpu
  fi

  # write tmux scrollback to a file
  tmux capture-pane -pq -S-1000 > /tmp/launch_log

  # start manager
  cd openpilot/system/manager
  if [ ! -f $DIR/prebuilt ]; then
    ./build.py
  fi
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  launch
fi
