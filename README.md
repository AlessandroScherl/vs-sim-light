# vs-sim-light

A lightweight simulator for **image-based visual servoing (IBVS)**: a free-flying camera
(Intel RealSense D435i model, colour + depth, 1920 x 1080) over textured scenes, rendered by
[MuJoCo](https://mujoco.org). Drive the camera with your own code, or fly your method from fixed
start poses and compare it with published baselines (ViT-VS, ART-VS, SIFT IBVS).

New to IBVS? Read Chaumette & Hutchinson, ["Visual servo control, Part I: Basic approaches"](https://doi.org/10.1109/MRA.2006.250573)
(IEEE RAM, 2006), then [`docs/IBVS.md`](docs/IBVS.md) for how it works in this simulator.

![The goal view (left) and the live camera (right), with the matched points](docs/media/demo.gif)

*`sift_ransac` on `industrial1_drill`, starting 743 mm and 97° from the goal view; 0.3 mm after 600 steps.*

## Install

Linux, Python 3.10+:

```bash
git clone https://github.com/AlessandroScherl/vs-sim-light.git && cd vs-sim-light
python -m venv .venv && source .venv/bin/activate
pip install torch && pip install -e ".[baselines]"
```

Then put the scene pictures in place ([`docs/ASSETS.md`](docs/ASSETS.md)).

**GPU.** Used automatically when present (rendering and the DINO baselines); `--device cpu`
keeps torch on the CPU. On a shared machine, `source tools/cpu_env.sh` forces CPU-only.

## Drive the camera yourself

`render()` gives colour, depth and the true pose; `apply(twist, dt)` moves the camera. The twist
`[vx, vy, vz, wx, wy, wz]` (m/s, rad/s) is in the camera frame: x right, y down, z forward.

```python
import numpy as np
from vsbench.methods import make_method
from vsbench.vs.scene import Pose
from vsbench.world.mujoco_scene import MujocoScene

with MujocoScene("poster2") as scene:
    goal_rgb, goal_depth = scene.render_goal()
    g = scene.goal_pose
    scene.reset(Pose(g.position + [0.01, 0.0, 0.0], g.rotation))   # 1 cm to the side
    method = make_method("sift_ransac")                            # or your own method
    K = scene.intrinsics
    method.configure(K.K, K.width, K.height, dt=1 / 60)
    method.set_target(goal_rgb, goal_depth)
    for _ in range(100):
        rgb, depth, _ = scene.render()
        twist, info = method.get_control_rate(rgb, depth)
        scene.apply(twist, 1 / 60)
    print(f"{1000 * np.linalg.norm(scene.pose.position - g.position):.2f} mm from the goal")
```

## Fly the benchmark

```bash
python -m vsbench list                       # every method name
python -m vsbench fly --scene poster2 --method sift_ransac vitvs_dinov3 --start-set near60 --limit 1 --film
```

Each episode writes a record to `runs/<scene>_<method>.jsonl` (`final_translation_mm`,
`final_rotation_deg`, `success`, ...) and, with `--film`, a video. Useful flags: `--perturb`
(`light`, `occluder`, `distractors`, `noise`, `ground:...`, `swap:...`), `--mask objects|file.png`
(keep only matches on the goal's objects), `--debug-gif` (the view below). Every baseline,
variant, mode and flag: [`docs/BASELINES.md`](docs/BASELINES.md).

![The debug view of the flight above](docs/media/debug.gif)

## Plug in your method

A method is one class with a `@register` line. It gets the goal picture, the live picture,
depth and the intrinsics - never the true pose - and returns a twist. A complete example is
[`examples/my_method.py`](examples/my_method.py); fly it with
`python -m vsbench fly --scene poster2 --method my_ibvs --plugin examples/my_method.py --start-set near60 --limit 1`.

```python
import numpy as np
from vsbench.methods import register

@register("my_ibvs")
class MyIBVS:
    wants_depth = "dense"            # "none", "scalar" or "dense"
    privileged = False
    def configure(self, K, width, height, dt): ...
    def reset(self): ...
    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None): ...
    def get_control_rate(self, current_rgb, current_depth=None):
        return np.zeros(6), {}       # the twist; optionally {"kp_cur": ..., "kp_des": ...}
```

## Scenes

![The six scenes, seen from their goal pose](docs/media/scenes.png)

Three posters (`hollywood`, `poster2`, `poster3`) and three rooms (`industrial1_drill`,
`cluttered2`, `cluttered3`), each with frozen start poses so every method flies from the same
places. You can also build your own scenes ([`docs/ASSETS.md`](docs/ASSETS.md)).

## Baselines

Copied unchanged from the Isaac Sim benchmark vs-sim ([`PROVENANCE`](vsbench/vs/PROVENANCE.md)):

- **ViT-VS** (`vitvs_dinov3`, `vitvs_dinov2`): [arXiv 2503.04545](https://arxiv.org/abs/2503.04545)
- **ART-VS** (`artvs`): [art-vs.github.io](https://art-vs.github.io/)
- **SIFT / ORB / AKAZE IBVS** (`sift_ransac`, `orb_ransac`, `akaze_ransac`)

`vitvs_dinov3` needs the gated [DINOv3 weights](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
(`hf download facebook/dinov3-vits16-pretrain-lvd1689m`); DINOv2 downloads itself.

MIT licence.
