| scene | arm | identity (n / \|v\| m/s / \|w\| deg/s) | near start -> final mm, deg (MuJoCo) | near | Isaac near final mm, deg [closer] | capped30 start -> final mm (MuJoCo) | Isaac capped30 start -> final mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hollywood | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.009, 2.69 -> 0.004 | closer | 0.171, 0.018 [yes] | w1_034: 479 -> 0.0 | w1_034: 479 -> 0.2 |
| hollywood | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.592, 2.69 -> 0.096 | closer | 0.427, 0.070 [yes] | w1_034: 479 -> 1.0 | - |
| hollywood | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 5.755, 2.69 -> 0.486 | closer | 3.962, 0.510 [yes] | w1_034: 479 -> 6.0 | w1_034: 479 -> 9.8 |
| hollywood | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 20.111, 2.69 -> 2.542 | NOT closer | 1.615, 0.021 [yes] | w1_034: 479 -> 3.5 | w1_034: 479 -> 2.9 |
| poster2 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.029, 2.69 -> 0.006 | closer | 0.146, 0.022 [yes] | w1_034: 479 -> 0.0 | - |
| poster2 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.661, 2.69 -> 0.121 | closer | 0.415, 0.039 [yes] | w1_034: 479 -> 1.1 | - |
| poster2 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 19.560, 2.69 -> 2.713 | NOT closer | 14.890, 2.105 [no] | w1_034: 479 -> 8.5 | - |
| poster2 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 9.515, 2.69 -> 2.436 | NOT closer | 17.155, 3.282 [no] | w1_034: 479 -> 6.1 | - |
| poster3 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.159, 2.69 -> 0.022 | closer | 0.165, 0.023 [yes] | w1_076: 293 -> 0.0 | - |
| poster3 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.363, 2.69 -> 0.037 | closer | 0.210, 0.010 [yes] | w1_076: 293 -> 4.0 | - |
| poster3 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 8.971, 2.69 -> 0.727 | NOT closer | 8.839, 0.917 [no] | w1_076: 293 -> 15.5 | - |
| poster3 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 2.158, 2.69 -> 0.204 | closer | 24.053, 2.769 [no] | w1_076: 293 -> 2.3 | - |
| industrial1_drill | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.157, 2.69 -> 0.022 | closer | 0.175, 0.019 [yes] | w1_034: 479 -> 2982.0 | - |
| industrial1_drill | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.401, 2.69 -> 0.033 | closer | 1.011, 0.107 [yes] | w1_034: 479 -> 11397.2 | - |
| industrial1_drill | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 25.051, 2.69 -> 2.883 | NOT closer | 19.802, 2.261 [no] | w1_034: 479 -> 415.3 | - |
| industrial1_drill | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 35.351, 2.69 -> 4.041 | NOT closer | 29.710, 3.106 [no] | w1_034: 479 -> 97.8 | - |
| cluttered2 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.121, 2.69 -> 0.007 | closer | 0.662, 0.026 [yes] | w1_034: 479 -> 0.5 | w1_034: 479 -> 20815.1 |
| cluttered2 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.649, 2.69 -> 0.056 | closer | 1.701, 0.094 [yes] | w1_034: 479 -> 2.0 | - |
| cluttered2 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 57.352, 2.69 -> 2.492 | NOT closer | 43.754, 1.939 [no] | w1_034: 479 -> 1052.4 | w1_034: 479 -> 1100.6 |
| cluttered2 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 39.235, 2.69 -> 2.532 | NOT closer | 99.233, 4.809 [no] | w1_034: 479 -> 3144.9 | w1_034: 479 -> 2078.1 |
| cluttered3 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.225, 2.69 -> 0.007 | closer | 2.186, 0.082 [yes] | w1_076: 293 -> 0.4 | - |
| cluttered3 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 2.401, 2.69 -> 0.122 | closer | 1.814, 0.056 [yes] | w1_076: 293 -> 3.3 | - |
| cluttered3 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 42.899, 2.69 -> 1.895 | NOT closer | 47.670, 2.061 [no] | w1_076: 293 -> 41.1 | - |
| cluttered3 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 7.506, 2.69 -> 0.218 | closer | 2.234, 0.196 [yes] | w1_076: 293 -> 2546.1 | - |
