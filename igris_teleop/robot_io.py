"""Thin, thread-safe wrapper around the igris_c_sdk Python bindings (from igris_demo_ws, cameras removed).

DDS callbacks run on SDK threads; they only copy data into numpy arrays under a lock.
"""

from __future__ import annotations

import threading
import time
import numpy as np

import igris_c_sdk as igc

from .robot_spec import NUM_HAND, NUM_JOINTS

LOWSTATE_TOPIC = "rt/lowstate"
LOWCMD_TOPIC = "rt/lowcmd"
HANDSTATE_TOPIC = "rt/handstate"
HANDCMD_TOPIC = "rt/handcmd"
ROBOTSTATE_TOPIC = "rt/robotstate"
CMDVEL_TOPIC = "rt/cmd_vel"            # walk controller velocity (igris_c_sdk twist_publisher.py)

# int -> "WALK_LOW", "LOW", ... (enum names without the ROBOT_STATE_ prefix)
CONTROL_STATE_NAMES = {
    int(v): k.replace("ROBOT_STATE_", "") for k, v in igc.RobotControlState.__members__.items()
}


class RobotIO:
    def __init__(
        self,
        domain_id: int,
        namespace: str,
        use_hand: bool = True,
    ):
        self.domain_id = domain_id
        self.namespace = namespace
        self.use_hand = use_hand

        self._lock = threading.Lock()
        self._low: dict[str, np.ndarray] | None = None
        self._low_stamp = 0.0
        self._low_count = 0
        self._hand: dict[str, np.ndarray] | None = None
        self._hand_stamp = 0.0
        self._ctrl_state: str | None = None
        self.hand_state_len = 0
        self._ctrl_stamp = 0.0

        self.channel = None
        self.client = None
        self.lowcmd_pub = None
        self.handcmd_pub = None
        self._subs: list[object] = []

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        self.channel = igc.ChannelFactory.Instance()
        self.channel.Init(self.domain_id, self.namespace)
        if not self.channel.IsInitialized():
            raise RuntimeError("ChannelFactory init failed (check CYCLONEDDS_URI / domain / namespace)")

        self.client = igc.IgrisC_Client()
        self.client.Init()
        self.client.SetTimeout(5.0)

        sensor_qos = igc.QosProfile.SensorData()

        low_sub = igc.LowStateSubscriber(LOWSTATE_TOPIC, sensor_qos)
        if not low_sub.init(self._on_lowstate):
            raise RuntimeError("LowStateSubscriber init failed")
        self._subs.append(low_sub)

        rs_sub = igc.RobotStateSubscriber(ROBOTSTATE_TOPIC)
        if not rs_sub.init(self._on_robotstate):
            raise RuntimeError("RobotStateSubscriber init failed")
        self._subs.append(rs_sub)

        if self.use_hand:
            hand_sub = igc.HandStateSubscriber(HANDSTATE_TOPIC, sensor_qos)
            if not hand_sub.init(self._on_handstate):
                raise RuntimeError("HandStateSubscriber init failed")
            self._subs.append(hand_sub)

    def open_publishers(self, walk: bool = False) -> None:
        """Only called when commands will really be sent (not in dry-run)."""
        if walk:
            # Default QoS, matching the bridge's cmd_vel subscriber (SDK twist_publisher.py)
            self.twist_pub = igc.WalkTwistPublisher(CMDVEL_TOPIC, igc.QosProfile.Default())
            if not self.twist_pub.init():
                raise RuntimeError("WalkTwistPublisher init failed")
        qos = igc.QosProfile.SensorData()
        self.lowcmd_pub = igc.LowCmdPublisher(LOWCMD_TOPIC, qos)
        if not self.lowcmd_pub.init():
            raise RuntimeError("LowCmdPublisher init failed")
        if self.use_hand:
            self.handcmd_pub = igc.HandCmdPublisher(HANDCMD_TOPIC, qos)
            if not self.handcmd_pub.init():
                raise RuntimeError("HandCmdPublisher init failed")

    def publish_twist(self, vx: float, vy: float, wz: float) -> None:
        if getattr(self, "twist_pub", None) is None:
            return
        msg = igc.WalkTwist()
        tw = msg.twist()
        tw.linear().x(float(vx))
        tw.linear().y(float(vy))
        tw.angular().z(float(wz))
        self.twist_pub.write(msg)

    def stop(self, watchdog_s: float = 5.0) -> None:
        """Tear down DDS entities.

        The SDK's native stop() can deadlock on the GIL (main thread holds the GIL
        inside publisher.stop() while a DDS thread waits for it), observed
        intermittently. A Python-level timeout cannot fire in that state, so arm
        faulthandler's C watchdog: if teardown + interpreter exit take longer than
        `watchdog_s`, the process is terminated (stacks go to logs/shutdown_watchdog.txt).
        Call this last, after logs are saved and command streaming has stopped.
        """
        if watchdog_s > 0:
            import faulthandler
            from pathlib import Path
            path = Path(__file__).resolve().parents[1] / "logs" / "shutdown_watchdog.txt"
            path.parent.mkdir(exist_ok=True)
            self._watchdog_file = open(path, "w")
            faulthandler.dump_traceback_later(watchdog_s, exit=True, file=self._watchdog_file)
        for s in self._subs:
            try:
                s.stop()
            except Exception:
                pass
        for p in (self.lowcmd_pub, self.handcmd_pub, getattr(self, "twist_pub", None)):
            if p is not None:
                try:
                    p.stop()
                except Exception:
                    pass
        if self.channel is not None:
            try:
                self.channel.Release()
            except Exception:
                pass

    # -------------------------------------------------------------- callbacks
    def _on_lowstate(self, msg) -> None:
        ms = msg.motor_state()
        js = msg.joint_state()
        imu = msg.imu_state()
        d = {
            "joint_pos": np.array([ms[i].q() for i in range(NUM_JOINTS)]),
            "joint_vel": np.array([ms[i].dq() for i in range(NUM_JOINTS)]),
            "joint_torque": np.array([ms[i].tau_est() for i in range(NUM_JOINTS)]),
            "joint_pos_pjs": np.array([js[i].q() for i in range(NUM_JOINTS)]),
            "joint_vel_pjs": np.array([js[i].dq() for i in range(NUM_JOINTS)]),
            "joint_torque_pjs": np.array([js[i].tau_est() for i in range(NUM_JOINTS)]),
            "imu_quaternion": np.array(list(imu.quaternion()), dtype=np.float64),
            "imu_gyroscope": np.array(list(imu.gyroscope()), dtype=np.float64),
            "imu_accelerometer": np.array(list(imu.accelerometer()), dtype=np.float64),
        }
        with self._lock:
            self._low = d
            self._low_stamp = time.monotonic()
            self._low_count += 1

    def _on_handstate(self, msg) -> None:
        ms = msg.motor_state()
        n = len(ms)
        q = np.zeros(NUM_HAND)
        dq = np.zeros(NUM_HAND)
        if n >= NUM_HAND:
            slots = list(range(NUM_HAND))
        elif n == 2:
            # 1-DoF grippers reporting only (id 11, id 21): ASSUMED order -> slots 0, 6
            slots = [0, 6]
        else:
            slots = list(range(n))
        for k, slot in enumerate(slots):
            q[slot] = ms[k].q()
            dq[slot] = ms[k].dq()
        with self._lock:
            self.hand_state_len = n
            # ROBOT order (= HDF5 observation/hand_joint_pos order), no swap.
            self._hand = {"hand_joint_pos": q, "hand_joint_vel": dq}
            self._hand_stamp = time.monotonic()

    def _on_robotstate(self, msg) -> None:
        name = CONTROL_STATE_NAMES.get(int(msg.state()), f"UNKNOWN({int(msg.state())})")
        with self._lock:
            self._ctrl_state = name
            self._ctrl_stamp = time.monotonic()

    # ---------------------------------------------------------------- queries
    def lowstate_age(self) -> float:
        with self._lock:
            return float("inf") if self._low is None else time.monotonic() - self._low_stamp

    def hand_age(self) -> float:
        with self._lock:
            return float("inf") if self._hand is None else time.monotonic() - self._hand_stamp

    def control_state(self, max_age: float = 2.0) -> str | None:
        """Robot control state ("WALK_LOW", "LOW", ...) or None if not received recently."""
        with self._lock:
            if self._ctrl_state is None or time.monotonic() - self._ctrl_stamp > max_age:
                return None
            return self._ctrl_state

    def lowstate_count(self) -> int:
        with self._lock:
            return self._low_count

    def wait_ready(self, timeout: float) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            ok = self.lowstate_age() < 0.5 and (not self.use_hand or self.hand_age() < 0.5)
            if ok:
                return True
            time.sleep(0.05)
        return False

    def state(self) -> dict[str, np.ndarray]:
        """Copy of the latest LowState (+ HandState) arrays."""
        with self._lock:
            if self._low is None:
                raise RuntimeError("No LowState received yet")
            out = {k: v.copy() for k, v in self._low.items()}
            if self._hand is not None:
                out.update({k: v.copy() for k, v in self._hand.items()})
        return out
