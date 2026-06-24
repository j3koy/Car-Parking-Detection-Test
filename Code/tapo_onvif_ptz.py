"""
Tapo PTZ test tool - ONVIF edition
==================================

pytapo's native moveMotor() does NOT work on newer Tapo models like the
C520WS: the camera rejects the legacy "motor" API with error -40107.
This version drives Pan/Tilt over ONVIF instead, which the C520WS DOES
support (ONVIF Profile S, service port 2020).

Setup
-----
    pip install onvif-zeep

Credentials are the SAME "Camera Account" you use with pytapo
(Tapo app -> Camera Settings -> Advanced Settings -> Camera Account),
NOT your TP-Link cloud login.

Run it in a REAL terminal (VS Code integrated terminal, not the Output
panel) so keypresses are captured:
    python tapo_onvif_ptz.py

Controls
--------
  W / S : tilt up / down
  A / D : pan left / right
  Q     : save current position as a preset
  R     : recall the saved preset
  + / - : increase / decrease nudge duration (how far per press)
  H     : show help
  X / Esc / Ctrl-C : quit
"""

import sys
import os
import glob
import site
import time

try:
    from onvif import ONVIFCamera
    import onvif as _onvif
except ImportError:
    sys.exit("onvif-zeep is not installed.  Run:  pip install onvif-zeep")


def find_wsdl_dir():
    """Locate the ONVIF WSDL files.

    onvif-zeep frequently installs its WSDL files OUTSIDE the directory its
    own code expects (especially with 'pip install --user'), producing:
        No such file: ...\\site-packages\\wsdl\\devicemgmt.wsdl
    We search the likely locations and return whichever folder actually
    contains devicemgmt.wsdl, so we can hand it to ONVIFCamera directly.
    """
    pkg = os.path.dirname(_onvif.__file__)
    candidates = [
        os.path.join(pkg, "wsdl"),                    # onvif/wsdl
        os.path.join(os.path.dirname(pkg), "wsdl"),   # site-packages/wsdl (the default it tries)
    ]
    bases = []
    try:
        bases.append(site.getuserbase())              # e.g. AppData\Roaming\Python\Python313
    except Exception:
        pass
    bases += [sys.prefix, sys.base_prefix]
    for b in bases:
        if b:
            candidates.append(os.path.join(b, "wsdl"))

    for c in candidates:
        if os.path.isfile(os.path.join(c, "devicemgmt.wsdl")):
            return c

    # last resort: brute-force search under each base
    for b in bases:
        if b and os.path.isdir(b):
            hits = glob.glob(os.path.join(b, "**", "devicemgmt.wsdl"), recursive=True)
            if hits:
                return os.path.dirname(hits[0])
    return None

# ----------------------------------------------------------------------
# CONFIG  -- edit these
# ----------------------------------------------------------------------
HOST     = "192.168.254.115"      # camera IP address
PORT     = 2020                # Tapo ONVIF service port (default 2020)
USERNAME = "johelenmedz"             # Tapo Camera Account username
PASSWORD = "october2028"     # Tapo Camera Account password

SPEED       = 0.5              # pan/tilt velocity, 0.0 - 1.0
MOVE_TIME   = 0.4             # seconds the camera moves per key press
PRESET_NAME = "pytapo_test"   # name used when saving with Q

# If a direction feels reversed on your camera, flip the matching flag.
INVERT_PAN  = False
INVERT_TILT = False
# ----------------------------------------------------------------------


# ---- cross-platform single-key reader (no extra deps) ----------------
def get_key():
    """Read one keypress and return it as a lowercase string."""
    try:  # Windows
        import msvcrt
        ch = msvcrt.getch()
        try:
            return ch.decode("utf-8", "ignore").lower()
        except Exception:
            return ""
    except ImportError:  # Unix / macOS
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()


HELP = """
  W/S = tilt up/down     A/D = pan left/right
  Q   = save preset      R   = recall preset
  +/- = nudge duration   H   = help      X = quit
"""


def main():
    print(f"Connecting to {HOST}:{PORT} as '{USERNAME}' over ONVIF ...")
    wsdl_dir = find_wsdl_dir()
    if wsdl_dir:
        print(f"Using WSDL dir: {wsdl_dir}")
    else:
        print("WARNING: could not auto-locate ONVIF WSDL files; trying onvif-zeep default.")
    try:
        if wsdl_dir:
            cam = ONVIFCamera(HOST, PORT, USERNAME, PASSWORD, wsdl_dir)
        else:
            cam = ONVIFCamera(HOST, PORT, USERNAME, PASSWORD)
        media = cam.create_media_service()
        ptz = cam.create_ptz_service()
        profile = media.GetProfiles()[0]      # first profile = main stream
        token = profile.token
    except Exception as e:
        sys.exit(
            f"\nONVIF connect/login failed: {e}\n\n"
            "Checklist:\n"
            "  * PORT must be 2020 (Tapo ONVIF service port)\n"
            "  * USERNAME/PASSWORD must be the *Camera Account* creds\n"
            "    (Tapo app -> Advanced Settings -> Camera Account)\n"
            "  * Camera and PC must be on the same subnet"
        )

    print(f"Connected. Profile token: {token}")

    # list any presets already stored on the camera
    try:
        presets = ptz.GetPresets({'ProfileToken': token})
        listing = [(p.token, p.Name) for p in presets] if presets else []
        print(f"  Existing presets: {listing or '[]'}")
    except Exception as e:
        print("  Could not list presets:", e)

    move_time = MOVE_TIME
    saved_token = None
    latencies = []                       # ContinuousMove command latencies (ms)
    inv_pan  = -1 if INVERT_PAN  else 1
    inv_tilt = -1 if INVERT_TILT else 1

    # key -> (pan velocity, tilt velocity, label)
    moves = {
        "w": (0,                inv_tilt * SPEED,  "tilt up"),
        "s": (0,               -inv_tilt * SPEED,  "tilt down"),
        "a": (-inv_pan * SPEED, 0,                 "pan left"),
        "d": (inv_pan * SPEED,  0,                 "pan right"),
    }

    print(HELP)
    print(f"Speed={SPEED}  nudge={move_time}s.  Ready - press a key.\n")

    while True:
        k = get_key()

        # quit
        if k in ("x", "\x1b", "\x03", ""):
            if latencies:
                lo, hi = min(latencies), max(latencies)
                avg = sum(latencies) / len(latencies)
                print(f"\n  Move-command latency over {len(latencies)} moves: "
                      f"min={lo:.1f}  avg={avg:.1f}  max={hi:.1f} ms")
            print("\nBye.")
            break

        # movement
        elif k in moves:
            x, y, label = moves[k]
            ms = nudge(ptz, token, x, y, move_time, label)
            if ms is not None:
                latencies.append(ms)

        # presets
        elif k == "q":
            saved_token = save_preset(ptz, token, PRESET_NAME, saved_token)
        elif k == "r":
            recall_preset(ptz, token, saved_token)

        # nudge duration
        elif k in ("+", "="):
            move_time = round(move_time + 0.1, 2)
            print(f"  nudge -> {move_time}s")
        elif k in ("-", "_"):
            move_time = max(0.1, round(move_time - 0.1, 2))
            print(f"  nudge -> {move_time}s")

        elif k == "h":
            print(HELP)


def nudge(ptz, token, x, y, duration, label):
    """ContinuousMove for `duration` seconds, then Stop.

    Times the ONVIF commands and returns the ContinuousMove latency in
    milliseconds -- i.e. how long the camera takes to ACK the 'start
    moving' command (network round-trip + camera processing). Returns
    None on failure.

    Note: `duration` is your own MOVE_TIME pause while the camera sweeps;
    that is a value you set, not latency, so it is NOT counted here.
    """
    try:
        req = ptz.create_type('ContinuousMove')
        req.ProfileToken = token
        req.Velocity = {'PanTilt': {'x': x, 'y': y}, 'Zoom': {'x': 0.0}}

        t0 = time.perf_counter()
        ptz.ContinuousMove(req)
        move_ms = (time.perf_counter() - t0) * 1000.0

        time.sleep(duration)

        t1 = time.perf_counter()
        ptz.Stop({'ProfileToken': token, 'PanTilt': True, 'Zoom': True})
        stop_ms = (time.perf_counter() - t1) * 1000.0

        print(f"  move x={x} y={y}  [{label}]  "
              f"start-cmd={move_ms:.1f} ms  stop-cmd={stop_ms:.1f} ms")
        return move_ms
    except Exception as e:
        print(f"  move failed [{label}]: {e}")
        return None


def save_preset(ptz, token, name, prev_token):
    """Save current position as a preset; return its PresetToken."""
    try:
        new_token = ptz.SetPreset({'ProfileToken': token, 'PresetName': name})
        print(f"  Preset saved as '{name}' (token={new_token}). Press R to return.")
        return new_token
    except Exception as e:
        print(f"  save preset failed: {e}")
        return prev_token


def recall_preset(ptz, token, preset_token):
    if preset_token is None:
        print("  No preset saved yet - press Q first.")
        return
    try:
        ptz.GotoPreset({'ProfileToken': token, 'PresetToken': preset_token})
        print(f"  Returning to preset token={preset_token} ...")
    except Exception as e:
        print(f"  recall failed: {e}")


if __name__ == "__main__":
    main()