# PromptVFusion

PromptVFusion fuses synchronized visible (VIS) and infrared (IR) video when one modality is missing. It completes missing features using the current observation and previously observed features, then produces a fused image for every frame. The default inference path includes **coherent motion and background candidates** before the learned feature correction.

## Setup

Run the commands from the repository root. The original experiments used Python 3.9, PyTorch 2.4.1 and CUDA 12.1. Install a PyTorch and torchvision build for your CUDA version, then install the remaining requirements.

```bash
python -m pip install -r requirements.txt
```

Provide these checkpoints locally.

| Path | Purpose |
| --- | --- |
| `checkpoint/model.pth` | Original MAVFusion weights for the complete-input teacher and frozen structure encoders |
| `checkpoint/Tartan-C-T-TSKH-spring540x960-S.pth` | SEA-RAFT optical flow weights specified by `config/module/spring-S.json` |
| `checkpoint/prompt_gate_init.pth` | Prompt and Gate warm start for reproducing the motion-guided training run |
| `checkpoint/epoch_2.pth` | Motion-guided student checkpoint containing `backbone` and `prompt` state dicts |

Use checkpoints from sources you trust. See [upstream credits and weight sources](NOTICE.md). Set the dataset root and checkpoint paths in `config/train.json` for your machine.

## Inference

```bash
python -m promptvfusion.infer --config config/train.json \
  --checkpoint checkpoint/epoch_2.pth --manifest manifests/inference.jsonl \
  --output outputs/fused
```

Each line in the inference manifest describes one frame. Frame indices for a sequence start at zero and are consecutive. The model starts from a synchronized pair. Inputs are ordered VIS, IR. The example shows one missing IR frame.

```json
{"sequence":"clip01","frame_index":0,"available_mask":[1,1],"input_paths":{"visible":"visible/clip01/00000.png","infrared":"infrared/clip01/00000.png"}}
{"sequence":"clip01","frame_index":1,"available_mask":[1,0],"input_paths":{"visible":"visible/clip01/00001.png","infrared":null}}
```

Paths are relative to `data_root`. Use one missing modality at a time. The training protocol starts with ten complete frames and supports directly adjacent outages of opposite modalities. For that switch, the newly missing modality needs at least two consecutive preceding real observations. The inference command writes complete 640×480 frames by default. Pass `--no-coherent` for the same-weight comparison with motion and background candidates disabled.

The Prompt reads real history at each frame. Short-step transport updates the mapping back to that history. The fusion network uses the current completed features and the previous two feature states. Predicted features do not overwrite the real Prompt anchors.

## Training

The training JSONL contains full pairs. Each row has `sequence`, `frame_index` and `target_paths` with `visible` and `infrared` paths. The trainer constructs its own input masks. The original configuration checks for 170 training videos and 117,088 paired frames. Set `expected_train_videos` and `expected_train_frames` to match your manifest if using another split.

```bash
python -m promptvfusion.fit --config config/train.json --output runs/motion_guided
```

The two stages cap outages at four and seven frames. Available observations drive the student. Hidden true features and the complete-input teacher's fused image provide supervision. The VIS/IR encoders, temporal alignment and fusion modules, and Prompt update jointly. The teacher and flow estimator stay frozen. Coherent candidates are added at inference using the same trained weights.

```bash
python -m pip install pytest
python -m pytest -q
```

## Validation

The native paired comparison evaluated 392 missing frames from 14 independent videos, plus the `0111_1753` showcase. At the end of a seven-frame VIS outage immediately after a modality switch, both old-location and current-location edge errors decreased in 4 of 14 videos. Eight showed no material change. In the first VIS outage, this candidate path was not activated under the evaluated protocol. Multiple moving vehicles can still be displaced, broken up or duplicated. The [validation record](VALIDATION.md) describes the comparison protocol and its scope.
