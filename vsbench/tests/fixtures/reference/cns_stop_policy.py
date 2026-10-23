"""VENDORED, UNMODIFIED CNS SOURCE — the reference `refinevs.control.CnsPlateau` is tested against.

Upstream: https://github.com/hhcaz/CNS  `cns/benchmark/stop_policy.py`
          (CNS: Correspondence Encoded Neural Image Servo Policy, Chen/Yu/Wang/Xiong)
Obtained: 2026-09-03, tarball of `main` (no git metadata, so there is no commit hash to
          quote). The pin is the md5 of the upstream file as fetched:
          `cc90739c52393a92a1bc991265505f20` — re-download and `md5sum` to verify.
Licence:  MIT, reproduced in full below. This is third-party code: do not reformat it, do
          not "fix" it, do not lint it (`pyproject.toml` excludes this directory). If it
          ever disagrees with `refinevs.control.CnsPlateau`, the port is what changes.
Why it is here: protocol v2 makes CNS's plateau rule the **scored** stop for every arm
          (`../../../../docs/eval_protocol_v2_2026-09-02.md` §5). U4 first implemented it
          from a prose description and got five things wrong (review R12, 2026-09-03);
          `tests/fast/test_control.py::test_cns_plateau_matches_the_vendored_policy` now
          runs both on the same traces and requires the same firing step.

TWO EDITS, and only these two — both stated so a reader can diff against upstream:
  1. dropped `from skimage.color import rgb2gray` / `from skimage.metrics import
     structural_similarity` (upstream lines 2-3) and the `SSIMStopPolicy` class
     (upstream lines 129-145) that needs them: scikit-image is not a dependency of this
     repo and is absent from both environments, and CNS's benchmark uses SSIM only for
     the two image-error baselines we do not run.
  2. nothing else. `ErrorHoldingStopPolicy` (upstream 6-66, including its debug `print`)
     and `PixelStopPolicy` (upstream 86-126) are byte-for-byte, as is the commented-out
     earlier `PixelStopPolicy` (upstream 69-83), which is kept because it documents the
     95th-percentile variant the published one replaced.

MIT License

Copyright (c) 2023 hhcaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


class ErrorHoldingStopPolicy:
    def __init__(self, waiting_time, conduct_thresh):
        self.waiting_time = waiting_time
        self.conduct_thresh = conduct_thresh
        self.min_err = None
        self.cur_err = float("inf")
        self.prev_t = None
        self.prev_err = None
        self.accum_time = 0
    
    def reset(self):
        self.min_err = None
        self.prev_t = None
        self.prev_err = None
        self.accum_time = 0
    
    def calculate_error(self, data):
        raise NotImplementedError
    
    def __call__(self, data, timestamp: float) -> bool:
        if data is None:
            self.reset()
            self.cur_err = float("inf")
            return False
        
        cur_err = self.calculate_error(data)
        self.cur_err = cur_err

        if cur_err > self.conduct_thresh:
            self.reset()
            return False
        
        if self.min_err is None or cur_err < 0.9 * self.min_err:
            self.min_err = cur_err
            self.prev_t = timestamp
            self.prev_err = cur_err
            self.accum_time = 0
            return False
        
        thresh = 1.1 * self.min_err + 5e-4
        # thresh = 1.1 * self.min_err + 2e-4
        a, b = self.prev_err, cur_err
        t0, t1 = self.prev_t, timestamp
        self.prev_t = timestamp
        self.prev_err = cur_err

        if a < thresh and thresh <= b:
            dt = (thresh - a) / (b - a) * (t1 - t0)
        elif a >= thresh and thresh > b:
            dt = (thresh - b) / (a - b) * (t1 - t0)
        elif a <= thresh and b <= thresh:
            dt = t1 - t0
        else:
            dt = 0
            # dt = (t1 - t0) * 0.01  # avoid waiting for too much time
        self.accum_time += dt

        print(f"[INFO] min_err = {self.min_err:.3e}, cur_err = {cur_err:.3e}, dt = {dt:.3f}, accum = {self.accum_time:.3f}"
              )

        return self.accum_time > self.waiting_time


# class PixelStopPolicy(ErrorHoldingStopPolicy):
#     def __init__(self, waiting_time, conduct_thresh=0.01):
#         super().__init__(waiting_time, conduct_thresh)
    
#     def calculate_error(self, data):
#         delta_pos = getattr(data, "pos_cur") - getattr(data, "pos_tar")
#         delta_pos = torch.abs(delta_pos[getattr(data, "node_mask")])
#         delta_pos = delta_pos.squeeze(0).cpu().numpy()
#         delta_pos = np.linalg.norm(delta_pos, axis=-1)
#         # mask = stats.zscore(delta_pos) < 2
#         # mask = delta_pos < np.percentile(delta_pos, 90)
#         mask = delta_pos < np.percentile(delta_pos, 95)
#         delta_pos = delta_pos[mask]
#         mean_pix_err = delta_pos.mean()
#         return mean_pix_err


class PixelStopPolicy(ErrorHoldingStopPolicy):
    def __init__(self, waiting_time, conduct_thresh=0.01):
        super().__init__(waiting_time, conduct_thresh)
        self.weight = None
        self.witness_count = None
        self.decay = 0.95
        self.maximum = 1.0 / (1 - self.decay)  # solve from: x = decay * x + 1
    
    def reset(self):
        if isinstance(self.weight, np.ndarray):
            self.weight.fill(0)
            self.witness_count.fill(0)
        return super().reset()
    
    def calculate_error(self, data):
        delta_pos = (data.pos_cur - data.pos_tar).cpu().numpy()
        node_mask = data.node_mask.cpu().numpy()

        if self.weight is None or len(self.weight) != len(node_mask):
            # initialize weight and witness count
            self.weight = np.zeros(len(node_mask), dtype=np.float32)
            self.witness_count = np.zeros(len(node_mask), dtype=np.int64)
        
        pre_is_0 = self.witness_count == 0
        self.witness_count[node_mask] += 1
        now_is_1 = self.witness_count == 1

        mask = pre_is_0 & now_is_1  # first witness mask
        self.weight[mask] = self.maximum  # set weight of first witness point to maximum 
        self.weight[~mask] = self.decay * self.weight[~mask] + node_mask[~mask]

        weight = self.weight[node_mask]
        delta_pos = np.linalg.norm(delta_pos[node_mask], axis=-1)

        error = weight * delta_pos
        mask = error < np.percentile(error, 90)
        weight = weight[mask]
        weight = weight / (weight.sum() + 1e-7)
        mean_err = np.sum(weight * delta_pos[mask])

        return mean_err
