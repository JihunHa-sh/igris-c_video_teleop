# GPU PC 쪽 (Fast SAM 3D Body publisher)

ailab에 설치된 `fast_sam_3d_body` conda 환경과 `~/third_party/Fast-SAM-3D-Body`를 그대로 사용합니다. upstream 코드는 수정하지 않습니다.
이 폴더의 파일을 `ailab:~/igris_video_teleop_gpu/`에 복사해 두었습니다.

| 파일 | 역할 |
|---|---|
| `run_fastsam_multiview.py` | upstream `run_multiview_publisher.py`에 이 PC의 경로(SMPL, MHR assets, 체크포인트)를 채워서 실행합니다. 카메라 모델에 맞는 해상도를 자동으로 고릅니다 (D555 = 896x504, 그 외 848x480). `--list`로 연결된 카메라와 연결 방식(USB / PoE)을 확인합니다. 카메라 소스를 자체 구현해서 **PoE로 연결한 D555**도 동작합니다 (upstream은 PoE에 없는 `accel` 스트림을 요청해서 실패함. 여기서는 통합 `motion` 스트림으로 중력 보정). |
| `hand_closure.py` | 손가락 3D 위치(MHR70)로 손 쥐기 정도 계산 (0 = 펴짐 .. 1 = 주먹). 검지~새끼 손가락의 `끝-손목 거리 / 마디 길이 합`을 평균. |
| (손 디코더) | upstream은 속도 때문에 **몸 디코더만** 돌려서 손가락이 거의 움직이지 않습니다(주먹도 펴진 손으로 나옴). 그래서 정면 카메라에서 `HAND_EVERY`(기본 3) 프레임마다, 몸 디코더가 찾은 양손 박스를 잘라 SAM 3D Body **손 디코더**를 돌립니다(왼손은 좌우 반전, 원래 SAM 3D Body 전체 추론과 같은 방식). 손 디코더는 한 번에 약 95ms가 걸려 자세 추정 속도가 약 9fps에서 6.8fps로 줄어듭니다. `HAND_EVERY=5 ./start_publisher.sh`처럼 간격을 늘리면 덜 느려집니다. 결과는 `hand_closure`, `hand_straightness` 필드로 보냅니다. |
| (사람 고정) | 여러 명이 보여도 **한 명만** 따라갑니다. 처음에 가장 크게 보이는 사람(다음 사람보다 1.5배 이상, 옆 카메라는 2.5배)을 고정하고, 이후에는 그 사람 박스만 따라갑니다. 0.7초 동안 안 보이면 다시 고릅니다(`target_id` 증가 → 로봇은 일시정지). 비슷한 크기의 사람이 여럿이면 고정하지 않고 기다립니다. |
| `start_publisher.sh` | `cameras.txt`의 시리얼로 시작합니다. 5556 포트가 사용 중이면 바로 알려 줍니다. |
| `cameras.txt` | `SN_MAIN,SN_SIDE` 형식입니다. 첫 번째(정면) 카메라가 몸 방향 기준이 됩니다. |
| `make_test_video.py` | 사진 한 장으로 카메라 2대짜리 테스트 영상과 intrinsics를 만듭니다 (카메라 없이 시험할 때). |

```bash
cd ~/igris_video_teleop_gpu
PY=~/miniconda3/envs/fast_sam_3d_body/bin/python
$PY run_fastsam_multiview.py --list
./start_publisher.sh                    # --record: 영상과 SMPL을 output/records에 저장
# 카메라 없이 시험
$PY make_test_video.py ~/third_party/Fast-SAM-3D-Body/notebook/images/dancing.jpg test_input
$PY run_fastsam_multiview.py --source video --videos test_input/cam_main.mp4 test_input/cam_side.mp4 \
    --intrinsics test_input/cam_main.json test_input/cam_side.json
```

시험 결과 (2026-11-12, RTX 4090, 테스트 영상 2개): 백본+디코더 약 84ms, 노트북 수신 22~29Hz.
YOLO TensorRT 엔진(`checkpoints/yolo/yolo11m-pose.engine`)이 없어서 `.pt`를 쓰고 있습니다. Fast-SAM의 `convert_yolo_pose_trt.py`로 엔진을 만들면 더 빨라집니다.

ailab 방화벽은 5556 포트를 막지 않습니다 (노트북에서 수신 확인).

현재 카메라 (2026-11-12): D555 2대, PoE 연결. `254822300448`, `254922303155` (`cameras.txt`, 첫 번째 = 정면).
두 대 동시에 896x504 컬러 28.8fps, 중력 보정 정상. 카메라 2대 기준 자세 추정 속도는 약 9fps입니다 (4090, 테스트 영상 기준).
USB로 연결할 때는 메인보드 뒷면 포트에 직접 꽂으세요. 앞면 포트·허브에서는 `error -71`과 과전류(over-current)로 끊긴 기록이 있습니다.
