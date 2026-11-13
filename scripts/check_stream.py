"""Check the SMPL stream and the retargeting, without touching the robot.

    pixi run check-stream                      # stream rate/latency + retargeted joint angles
    pixi run check-stream --viz                # live 3D view: person (SMPL) | IGRIS (retargeted)
    pixi run check-stream --addr tcp://localhost:5556 --viz   # against scripts/fake_stream.py

Both skeletons are drawn in the robot torso frame (x forward, y left, z up), so they
should look the same. Use this before every real-robot session.
"""

import argparse
import time

import numpy as np

from igris_teleop.config import TeleopConfig
from igris_teleop.retarget import C, Retargeter
from igris_teleop.robot_spec import JOINT_NAMES
from igris_teleop.smpl import BONES, J, JOINT_NAMES as SMPL_NAMES, chain, forward_kinematics, local_rotations
from igris_teleop.smpl_stream import SmplStream
from igris_teleop.urdf_fk import ROBOT_DRAW, UrdfFK


def person_in_torso_frame(pose):
    p, _ = forward_kinematics(pose)
    T = chain(local_rotations(pose), "spine1", "spine2", "spine3")   # torso orientation (pelvis frame)
    pts = (T.T @ (p - p[J["spine3"]]).T).T           # SMPL canonical, torso frame
    return (C @ pts.T).T                              # robot axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--addr", default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--snapshot", default=None, help="save one person|robot frame to this PNG and exit")
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    cfg = TeleopConfig.load(args.config)
    addr = args.addr or cfg.stream["addr"]
    stream = SmplStream(addr, cfg.stream["topic"]).start()
    rt = Retargeter()
    from igris_teleop.robot_spec import JOINT_GROUPS
    legs = JOINT_GROUPS["left_leg"] + JOINT_GROUPS["right_leg"]
    legs_driven = set(legs) <= set(cfg.active)
    show = sorted(set(cfg.active) | set(legs))   # legs are always shown (retargeted), sent only if driven
    print(f"[check_stream] {addr}", flush=True)

    if not args.viz and not args.snapshot:
        t0 = time.monotonic()
        try:
            while not args.duration or time.monotonic() - t0 < args.duration:
                time.sleep(1.0)
                fr = stream.latest()
                n, err = stream.counts()
                if fr is None:
                    print(f"  waiting... (msgs {n}, bad {err})", flush=True)
                    continue
                q = cfg.shape(rt(fr.smpl_pose).q)
                hc = fr.extra.get("hand_closure")
                hs = fr.extra.get("hand_straightness")
                hands = ("no hand_closure in stream" if hc is None else
                         f"hand closure L {np.ravel(hc)[0]:.2f} R {np.ravel(hc)[1]:.2f}  (0 open .. 1 fist"
                         + (f", raw straightness {np.ravel(hs)[0]:.2f}/{np.ravel(hs)[1]:.2f})" if hs is not None else ")"))
                print(f"  {hands}")
                print(f"  {stream.rate():5.1f} Hz  age {stream.age() * 1000:5.0f} ms  frame {fr.frame_index}  bad {err}\n    "
                      + "  ".join(f"{JOINT_NAMES[i]}={q[i]:+.2f}" for i in show), flush=True)
        except KeyboardInterrupt:
            pass
        stream.stop()
        return

    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fk = UrdfFK()
    fig = plt.figure(figsize=(11, 5.5))
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    for ax, title in zip(axes, ("person (Fast SAM 3D Body SMPL)", "IGRIS-C (retargeted, URDF FK)")):
        ax.set_title(title)
    status = fig.suptitle("")

    def setup(ax):
        ax.cla()
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-1.1, 0.5)
        ax.set_xlabel("x fwd"); ax.set_ylabel("y left"); ax.set_zlabel("z up")
        ax.view_init(elev=12, azim=200)  # look at the front of the body

    def update(_):
        fr = stream.latest()
        if fr is None:
            status.set_text(f"waiting for {addr} ...")
            return
        pts = person_in_torso_frame(fr.smpl_pose)
        q = cfg.shape(rt(fr.smpl_pose).q)
        q_draw = np.where(np.isin(np.arange(len(q)), show), q, 0.0)
        setup(axes[0]); setup(axes[1])
        axes[0].set_title("person (Fast SAM 3D Body SMPL)")
        axes[1].set_title("IGRIS-C (retargeted, URDF FK)")
        for a, b in BONES:
            col = "tab:blue" if "left" in SMPL_NAMES[b] else "tab:red" if "right" in SMPL_NAMES[b] else "k"
            axes[0].plot(*np.stack([pts[a], pts[b]]).T, color=col, lw=2)
        poses = fk.link_poses(q_draw)
        for name, links in ROBOT_DRAW.items():
            P = np.array([poses[l][1] for l in links])
            col = "tab:blue" if name.startswith("left") else "tab:red" if name.startswith("right") else "k"
            ls = "--" if name.endswith("_leg") and not legs_driven else "-"
            axes[1].plot(*P.T, color=col, lw=3, ls=ls, marker="o", ms=3)
        if not legs_driven:
            axes[1].text2D(0.02, 0.02, "legs dashed: retargeted, not sent (not in drive:)",
                           transform=axes[1].transAxes, color="0.4", fontsize=8)
        status.set_text(f"{stream.rate():.1f} Hz   age {stream.age() * 1000:.0f} ms   "
                        f"WY {q[0]:+.2f} WP {q[2]:+.2f}   L: SP {q[15]:+.2f} SR {q[16]:+.2f} EP {q[18]:+.2f}   "
                        f"R: SP {q[22]:+.2f} SR {q[23]:+.2f} EP {q[25]:+.2f}\n"
                        f"legs  L: HP {q[3]:+.2f} HR {q[4]:+.2f} KN {q[6]:+.2f}   "
                        f"R: HP {q[9]:+.2f} HR {q[10]:+.2f} KN {q[12]:+.2f}   (blue = left)")

    if args.snapshot:
        t_end = time.monotonic() + 5.0
        while stream.latest() is None and time.monotonic() < t_end:
            time.sleep(0.1)
        update(0)
        fig.savefig(args.snapshot, dpi=90)
        print(f"[check_stream] saved {args.snapshot}")
        stream.stop()
        return

    anim = FuncAnimation(fig, update, interval=66, cache_frame_data=False)  # noqa: F841
    if args.duration:
        fig.canvas.new_timer(interval=int(args.duration * 1000)).add_callback(plt.close, fig)
    plt.show()
    stream.stop()


if __name__ == "__main__":
    main()
