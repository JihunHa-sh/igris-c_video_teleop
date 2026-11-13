"""Print the robot control mode and the current upper-body joint angles (read-only)."""

import time

from igris_teleop.config import TeleopConfig
from igris_teleop.robot_io import RobotIO
from igris_teleop.robot_spec import JOINT_GROUPS, JOINT_NAMES

cfg = TeleopConfig.load()
io = RobotIO(cfg.robot["domain_id"], cfg.robot["namespace"], use_hand=True)
io.start()
ok = io.wait_ready(5.0)
time.sleep(0.5)
print(f"ready={ok}  control state={io.control_state()}  (teleop needs {cfg.robot['required_control_state']})")
if ok:
    q = io.state()["joint_pos_pjs"]
    for g in ("waist", "left_arm", "right_arm", "neck"):
        print(f"  {g:9s} " + "  ".join(f"{JOINT_NAMES[i]}={q[i]:+.2f}" for i in JOINT_GROUPS[g]))
    h = io.state().get("hand_joint_pos")
    if h is not None:
        print(f"  grippers  left(id21)={h[6]:.2f}  right(id11)={h[0]:.2f}")
io.stop()
