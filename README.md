# igris_video_teleop_ws — 카메라 영상으로 IGRIS-C 텔레옵

VR 장비 없이 카메라 2대로 사람 자세를 추정해서 IGRIS-C 상체(허리 + 양팔)를 따라 움직이게 합니다.

```
[GPU PC: ailab]                                         [로봇 PC: 이 노트북]
RealSense x2 ─► Fast SAM 3D Body (multi-view) ─ZMQ:5556─► SMPL 수신 ─► 리타게팅 ─► 필터 ─► 300Hz LowCmd ─DDS─► IGRIS-C
               gpu_server/run_fastsam_multiview.py   SMPL   igris_teleop/        (One Euro)  (속도·관절 제한)
                                                   (24관절)  retarget.py                      다리 = Walkmode 제어기
```

## SONIC/GEM-X 구성과의 관계

| | SONIC 예제 (Unitree G1) | 이 작업공간 (IGRIS-C) |
|---|---|---|
| 사람 자세 추정 | GEM-X | **Fast SAM 3D Body** (카메라 2대 fusion) |
| 전송 | ZMQ, SMPL protocol v2/v3 | **같은 형식** (Fast-SAM publisher 수정 없이 사용) |
| 사람 → 로봇 | SONIC 학습 정책 (G1 전용) | **명시적 리타게팅** (`retarget.py`, IGRIS URDF 기반) |
| 균형·다리 | SONIC 정책 | IGRIS **Low-Level Walkmode** 제어기 (VR 텔레옵과 같음) |

SONIC 정책은 G1 전용으로 학습돼 있어 IGRIS를 움직일 수 없습니다. 그래서 SONIC 자리에 IGRIS용 리타게터를 넣었습니다.

## 폴더 구조

```
igris_teleop/
  smpl_stream.py   Fast-SAM ZMQ 수신 (헤더를 읽어 v2/v3 모두 처리, 최신 프레임만 유지)
  smpl.py          SMPL 관절 정의
  retarget.py      SMPL -> IGRIS 관절각 (허리 3, 팔 7x2, 목 2)
  urdf_fk.py       IGRIS URDF forward kinematics (검증과 뷰어용)
  filters.py       One Euro 필터
  commander.py     300Hz LowCmd / 50Hz HandCmd, 관절별 속도 제한 (igris_demo_ws에서 가져옴)
  robot_io.py      DDS 상태 구독 (igris_demo_ws에서 가져옴)
  gripper.py       손 쥐기 -> 그리퍼 열기/닫기 (히스테리시스 + 유지 시간)
  teleop.py        메인 루프: HOLD -> ENGAGE -> TRACK, 안전 검사, 로그
config/teleop.yaml      기본 설정 (허리 + 양팔)
config/first_test.yaml  첫 실제 로봇 시험용 (양팔만, 속도 낮춤, 블렌드 5초)
scripts/  run_teleop.py · check_stream.py (3D 뷰어) · fake_stream.py (카메라 없이 시험) · check_robot.py
gpu_server/  GPU PC에 복사하는 파일 (아래 참고)
assets/urdf/igris_c.urdf   igris_c_isaacsim_public의 igris_c_v2_wrist_hand_isaac.urdf
tests/    리타게팅·그리퍼 검증 (pixi run test)
```

## 실행 순서

**1. GPU PC (ailab)**: D555 2대가 PoE로 연결돼 있고, 시리얼은 `cameras.txt`에 들어 있습니다 (`254822300448`, `254922303155`, 첫 번째 = 정면).
```bash
ssh ailab
cd ~/igris_video_teleop_gpu
./start_publisher.sh            # 시작 후 몇 초는 카메라를 건드리지 마세요 (IMU로 중력 보정)
```
- 정면 카메라가 `254922303155`이면 `cameras.txt`에서 순서를 바꾸세요. 카메라를 바꿨으면 `~/miniconda3/envs/fast_sam_3d_body/bin/python run_fastsam_multiview.py --list`로 확인합니다.
- 사람이 전신이 다 보이게 2~3m 떨어져 서고, 화면에는 한 명만 있어야 합니다. 두 카메라는 서로 다른 각도(예: 정면과 45° 옆)에서 사람을 보게 두세요.

**2. 로봇 PC: 로봇 없이 확인**
```bash
pixi run check-stream --viz        # 3D 뷰어: 왼쪽 = 사람, 오른쪽 = 리타게팅된 IGRIS (파란색 = 왼쪽)
pixi run check-stream --snapshot out.png   # 화면 없이 한 프레임만 저장
```
팔을 들고 굽혀 보면서 오른쪽 로봇 그림이 똑같이 움직이는지 먼저 확인하세요.

**3. 로봇 준비**: 텔레옵 때와 같은 순서로 **Low-Level Walkmode**까지 진행합니다 (`pixi run check-robot`에서 `WALK_LOW` 확인).

**4. 텔레옵**
```bash
pixi run teleop                                                    # dry-run: 로봇 상태만 읽고 명령은 안 보냄
pixi run teleop --config config/first_test.yaml --enable-publish   # 첫 실제 시험 (양팔만, 느리게)
pixi run teleop --enable-publish                                   # 허리 + 양팔
```

| 키 | 동작 |
|---|---|
| `ENTER` | 따라 하기 시작 (현재 로봇 자세에서 사람 자세로 3초 동안 블렌드) / 일시정지 |
| `g` | 그리퍼 모드 전환: **hand** (기본, 주먹 = 닫기 / 손 펴기 = 열기) ↔ **key** |
| `z` / `x` | key 모드에서 왼쪽 / 오른쪽 그리퍼 토글 (hand 모드에서 누르면 key 모드로 바뀜) |
| `q` 또는 `Ctrl+C` | 시작 자세로 돌아간 뒤 종료 (`Ctrl+C` 두 번 = 즉시 정지) |

### 걷기 (사람 위치 → `rt/cmd_vel`, Low-Level Walkmode)
- `ENTER`를 누른 자리와 몸 방향이 기준(home)입니다. 기준에서 **앞/뒤/옆으로 20cm 넘게** 옮기면 그 방향으로 걷고, **골반을 20° 넘게 돌리면** 회전합니다. 기준 자리로 돌아오면 멈춥니다 (조이스틱처럼 동작).
- 최대 0.2 m/s(앞뒤), 0.1 m/s(옆), 0.3 rad/s(회전). 0.3초 이상 기준 밖에 있어야 시작합니다. 카메라가 사람을 놓치거나 일시정지하면 0으로 감속합니다.
- `w` 걷기 켜기/끄기, `h` 지금 자리를 새 기준으로. 설정은 `config/teleop.yaml`의 `walk:`.
- GPU 쪽 publisher가 `root_pos`, `root_heading`(정면 카메라 기준 사람 위치·방향)을 보내야 합니다.

### 행거 시험: 다리까지 직접 (`config/hanger_legs.yaml`)
- **Low Level Joint Control**(걷기 제어기 꺼짐)에서 31개 관절 전부를 사람 자세로 움직입니다. 균형 제어가 없으므로 **반드시 행거에 매달고 발이 바닥에 닿지 않게** 합니다.
- 다리 gain은 SDK GUI 기본값(`sdk_gui_client.py`). 목은 정면으로 고정합니다.

### 그리퍼 (1-DoF, SDK id 21 = 왼쪽, 11 = 오른쪽)
- GPU 쪽 publisher가 정면 카메라의 양손을 잘라 SAM 3D Body **손 디코더**로 손가락을 추정하고, 손 쥐기 정도 `hand_closure` (0 = 펴짐 .. 1 = 주먹)를 SMPL 메시지에 같이 보냅니다 (`gpu_server/hand_closure.py`). 몸 디코더만 쓰던 처음 버전은 주먹을 인식하지 못했습니다(실제 시험에서 최대 0.54).
- 로봇 쪽은 이 값을 **열기/닫기 두 가지 명령**으로 바꿉니다 (`igris_teleop/gripper.py`):
  0.6을 넘는 상태가 0.15초 유지되면 CLOSE(q=1.0), 0.35 아래로 0.15초 유지되면 OPEN(q=0.0).
  두 기준 사이(반쯤 쥔 손)에서는 상태를 유지하므로 떨리지 않습니다. 기준값은 `config/*.yaml`의 `gripper:`에서 바꿉니다.
- 손은 팔이 사람을 따라갈 때(ENGAGE/TRACK)만 반영합니다. 일시정지 중에는 그리퍼도 그대로입니다.
- **기준값 맞추기**: `pixi run check-stream`이 손 쥐기 값과 원본 값(straightness)을 출력합니다. 손을 폈을 때와 주먹을 쥐었을 때 값을 보고 `close_above`/`open_below`를 조정하세요. 테스트 사진(손 디코더): 꽉 말린 손 1.00, 반쯤 쥔 손 0.54. 실제 카메라에서 주먹과 편 손 값은 아직 확인하지 않았습니다.
- 그리퍼 초기화(hand init)는 기존 텔레옵 준비 순서대로 해 두어야 합니다.

## 안전장치
- `--enable-publish`는 로봇이 `WALK_LOW`일 때만 동작합니다. 실행 중 모드가 바뀌면 즉시 명령 송신을 멈춥니다.
- 시작하면 현재 자세를 유지(HOLD)하고, `ENTER`를 눌러야 따라 하기 시작합니다. 시작할 때 사람 자세로 천천히 블렌드합니다.
- **사람 한 명만 고정**: 여러 명이 보이면 GPU 쪽이 가장 크게 보이는 한 명을 고정해서 따라갑니다. 고정한 사람이 안 보이면 스트림이 끊긴 것과 같게 처리하고, 다른 사람으로 바뀌면(`target_id` 변경) 즉시 일시정지합니다. 조작자가 다른 사람보다 카메라에 확실히 가까이 서세요.
- 카메라 스트림이 0.25초 끊기면 목표를 고정하고, 1초 끊기면 일시정지합니다. 다시 시작하려면 `ENTER`를 누릅니다.
- 관절별 속도 제한(허리 0.8 / 팔 2.5 / 손목 3.0 rad/s), 관절 한계에서 0.05rad 여유를 둡니다.
- 명령과 실제 자세 차이가 0.8rad를 넘으면 실제 자세에서 정지합니다. LowState가 0.2초 이상 끊겨도 일시정지합니다.
- 다리 관절은 kp=kd=0이라 Walkmode 제어기가 계속 담당합니다.
- 로그: `logs/teleop_*.npz` (사람 목표, 명령, 실제 관절각, 상태).

## 리타게팅 방식 (`retarget.py`)
- **허리**: 골반 대비 몸통 회전(spine1·2·3) → yaw/roll/pitch. URDF에서 base_link가 몸통이라 관절은 역회전이 됩니다.
- **팔**: 위팔과 아래팔 **방향**이 사람과 같아지도록 기하학적으로 계산합니다(로봇 팔꿈치는 1자유도). 팔꿈치가 거의 펴졌을 때만 SMPL의 위팔 비틀림을 씁니다. 어깨 roll 90°(T자세) 부근의 특이점은 이전 값을 유지해서 연속성을 지킵니다.
- **손목**: 아래팔 대비 손 방향 → yaw/roll/pitch.
- **다리**: 팔과 같은 방식. 골반 대비 허벅지·정강이 방향 → hip pitch/roll/yaw + knee, 정강이 대비 발 방향 → ankle pitch/roll. 무릎이 앞으로 꺾이는 추정 오차는 무시합니다.
  다리 각도는 항상 계산해서 뷰어에 보여 주지만(`drive`에 없으면 점선), **로봇에는 보내지 않습니다**. Low-Level Walkmode에서는 걷기 제어기가 다리를 잡고 있고, 다리를 직접 움직이면 균형을 잡는 주체가 없어서 넘어지기 때문입니다.
- 관절별 조정은 `config/teleop.yaml`의 `shaping`에서 합니다: `{scale, offset, sign, limit_min, limit_max}`.

검증 (`pixi run test`, 15개 통과 — 팔·허리 7 + 다리 5 + 그리퍼 3):
- 기준 자세(T자세, 팔 내림, 앞으로 뻗기, 팔꿈치 90°, 상체 숙이기·비틀기)가 예상 관절각과 일치합니다.
- 무작위 팔 자세 400개에서, URDF로 계산한 로봇 위팔·아래팔 방향이 사람과 5° 이내로 일치합니다.
- 손 방향은 0.5° 이내로 일치합니다.
- 무작위 다리 자세 300개(600 다리)에서 허벅지·정강이 방향과 발 방향이 0.5° 이내로 일치합니다.
- 실제 Fast-SAM 출력으로 계산한 뼈 방향이 Fast-SAM이 보낸 관절 위치와 일치합니다 (위팔 4°, 아래팔 7° 이내).

## 아직 확인하지 않은 것 / 한계
- **허리 yaw 부호**: URDF 기준으로 맞췄지만 실제 로봇에서는 아직 확인하지 않았습니다. 그래서 `first_test.yaml`은 팔만 움직입니다. 반대로 움직이면 `shaping: {Waist_Yaw: {sign: -1}}`를 넣으면 됩니다.
- **주먹 기준값**: 주먹 사진이 없어 닫힘 쪽 값(`R_FIST = 0.55`)은 추정입니다. 실제 카메라에서 확인이 필요합니다. 손이 카메라에 작게 찍히거나 가려지면 손 추정이 부정확합니다.
- **목**: 검증된 gain이 없어서 기본으로 끕니다. 쓰려면 `drive`에 `neck`을 넣고 `gains`에 kp/kd를 지정하세요.
- **지연**: Fast-SAM 추론 약 85ms + publisher 보간 140ms + 와이파이 + 필터. 처음에는 천천히 움직이세요.
