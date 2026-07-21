#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$DIR/launch_env.sh"

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
  for usbgpu_device in /sys/bus/usb/devices/*; do
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

function prepare_usbgpu {
  local usbgpu_xhci_device="xhci-hcd.1.auto"
  local usbgpu_xhci_path="/sys/bus/platform/devices/$usbgpu_xhci_device"
  local usbgpu_xhci_driver usbgpu_current_speed
  local usbgpu_attempt usbgpu_stable_seconds usbgpu_slow_seconds

  [ -L "$usbgpu_xhci_path/driver" ] || return
  usbgpu_current_speed="$(usbgpu_speed)"
  [ "$usbgpu_current_speed" -gt 0 ] 2>/dev/null || return
  [ "$usbgpu_current_speed" -lt 5000 ] 2>/dev/null || return

  usbgpu_xhci_driver="$(readlink -f "$usbgpu_xhci_path/driver")"
  [ -d "$usbgpu_xhci_driver" ] || return
  echo "USB GPU detected at ${usbgpu_current_speed}M"

  for usbgpu_attempt in {1..3}; do
    usbgpu_stable_seconds=0
    usbgpu_slow_seconds=0
    echo "Resetting $usbgpu_xhci_device (attempt $usbgpu_attempt/3)"

    if ! echo "$usbgpu_xhci_device" | sudo tee "$usbgpu_xhci_driver/unbind" > /dev/null; then
      echo "Failed to unbind $usbgpu_xhci_device"
      return
    fi
    sleep 3
    if ! echo "$usbgpu_xhci_device" | sudo tee "$usbgpu_xhci_driver/bind" > /dev/null; then
      echo "Failed to bind $usbgpu_xhci_device"
      return
    fi

    # Require eight continuous seconds at SuperSpeed. A stable low-speed
    # enumeration cannot upgrade in place, so retry it without waiting longer.
    for _ in {1..30}; do
      usbgpu_current_speed="$(usbgpu_speed)"
      if [ "$usbgpu_current_speed" -ge 5000 ] 2>/dev/null; then
        usbgpu_stable_seconds=$((usbgpu_stable_seconds + 1))
        usbgpu_slow_seconds=0
        if [ "$usbgpu_stable_seconds" -ge 8 ]; then
          echo "USB GPU ready at ${usbgpu_current_speed}M"
          return
        fi
      elif [ "$usbgpu_current_speed" -gt 0 ] 2>/dev/null; then
        usbgpu_stable_seconds=0
        usbgpu_slow_seconds=$((usbgpu_slow_seconds + 1))
        [ "$usbgpu_slow_seconds" -ge 5 ] && break
      else
        usbgpu_stable_seconds=0
        usbgpu_slow_seconds=0
      fi
      sleep 1
    done

    if [ "$usbgpu_attempt" -lt 3 ]; then
      echo "USB GPU remained at ${usbgpu_current_speed}M; retrying in 5 seconds"
      sleep 5
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

launch
