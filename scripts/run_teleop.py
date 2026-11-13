"""IGRIS-C video teleoperation (robot PC).

    pixi run teleop --no-robot                 # no robot: check the whole loop (keys, ramps, logs)
    pixi run teleop                            # robot connected, DRY RUN (reads state, sends nothing)
    pixi run teleop --enable-publish           # REAL robot (Low-Level Walkmode required)

Options: --config config/teleop.yaml  --addr tcp://host:5556  --no-hand  --max-duration S
"""

import argparse
import sys

from igris_teleop.config import TeleopConfig
from igris_teleop.teleop import Teleop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--addr", default=None, help="override stream.addr")
    ap.add_argument("--enable-publish", action="store_true", help="really command the robot")
    ap.add_argument("--no-robot", action="store_true", help="run without DDS (pipeline test)")
    ap.add_argument("--no-hand", action="store_true", help="do not command the grippers")
    ap.add_argument("--max-duration", type=float, default=0.0)
    args = ap.parse_args()
    cfg = TeleopConfig.load(args.config)
    return Teleop(cfg, publish=args.enable_publish, no_robot=args.no_robot,
                  use_hand=not args.no_hand).run(addr=args.addr, max_duration=args.max_duration)


if __name__ == "__main__":
    sys.exit(main())
