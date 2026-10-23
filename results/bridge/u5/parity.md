# U5 parity - hollywood: the copied mock vs MuJoCo (raw look)

Landing = the error at the cap (step 150). Curve: median over the last third (steps 101-150) and max over the second half (steps 76-150) of the true error of the frame each step steered on.

| arm | start | world | start mm/deg | landing mm | landing deg | med t last 1/3 | max t 2nd half | med r last 1/3 | plateau step | frame-1 n | switch step | cpu s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vitvs_dinov3 | narrow20/start_00 | mock | 107.67/8.04 | 10.457 | 0.692 | 10.875 | 12.227 | 0.710 | 127 | 24 | None | 267.31 |
| vitvs_dinov3 | narrow20/start_00 | mujoco | 107.67/8.04 | 9.784 | 0.708 | 9.514 | 11.576 | 0.548 | 131 | 24 | None | 227.84 |
| vitvs_dinov3 | narrow20/start_00 | mujoco960 | 107.67/8.04 | 12.408 | 0.980 | 12.322 | 12.807 | 0.921 | 125 | 24 | None | 241.21 |
| vitvs_dinov3 | narrow20/start_01 | mock | 86.77/6.92 | 12.666 | 0.994 | 12.940 | 15.171 | 1.006 | 121 | 24 | None | 268.56 |
| vitvs_dinov3 | narrow20/start_01 | mujoco | 86.77/6.92 | 10.372 | 0.762 | 10.791 | 17.516 | 0.803 | 131 | 24 | None | 229.33 |
| vitvs_dinov3 | narrow20/start_01 | mujoco960 | 86.77/6.92 | 9.954 | 0.701 | 10.600 | 13.936 | 0.762 | 124 | 24 | None | 239.6 |
| vitvs_dinov3 | near60/n_000 | mock | 5.58/2.98 | 4.528 | 0.425 | 3.991 | 6.223 | 0.446 | 121 | 24 | None | 259.06 |
| vitvs_dinov3 | near60/n_000 | mujoco | 5.58/2.98 | 5.118 | 0.396 | 4.379 | 5.413 | 0.426 | 119 | 24 | None | 250.05 |
| vitvs_dinov3 | near60/n_000 | mujoco960 | 5.58/2.98 | 4.122 | 0.604 | 3.663 | 4.182 | 0.521 | 128 | 24 | None | 236.9 |
| vitvs_dinov3 | near60/n_001 | mock | 3.84/2.19 | 4.276 | 0.219 | 4.096 | 5.007 | 0.207 | 117 | 24 | None | 290.19 |
| vitvs_dinov3 | near60/n_001 | mujoco | 3.84/2.19 | 8.162 | 0.873 | 6.735 | 8.254 | 0.733 | 124 | 24 | None | 233.89 |
| vitvs_dinov3 | near60/n_001 | mujoco960 | 3.84/2.19 | 4.128 | 0.482 | 4.912 | 5.876 | 0.608 | 117 | 24 | None | 231.27 |
| vitvs_dinov3 | near60/n_002 | mock | 5.99/1.33 | 7.930 | 0.826 | 8.031 | 8.506 | 0.874 | 62 | 24 | None | 270.19 |
| vitvs_dinov3 | near60/n_002 | mujoco | 5.99/1.33 | 8.569 | 0.903 | 8.188 | 8.656 | 0.898 | 69 | 24 | None | 227.85 |
| vitvs_dinov3 | near60/n_002 | mujoco960 | 5.99/1.33 | 4.988 | 0.620 | 5.309 | 5.924 | 0.679 | 60 | 24 | None | 236.25 |
| vitvs_dinov2 | narrow20/start_00 | mock | 107.67/8.04 | 6.969 | 0.326 | 8.041 | 12.597 | 0.460 | 126 | 24 | None | 322.43 |
| vitvs_dinov2 | narrow20/start_00 | mujoco | 107.67/8.04 | 7.108 | 0.533 | 6.977 | 8.901 | 0.359 | 128 | 24 | None | 275.9 |
| vitvs_dinov2 | narrow20/start_00 | mujoco960 | 107.67/8.04 | 7.227 | 0.402 | 8.666 | 10.284 | 0.521 | 125 | 24 | None | 342.7 |
| vitvs_dinov2 | narrow20/start_01 | mock | 86.77/6.92 | 9.763 | 0.935 | 10.485 | 13.212 | 1.013 | 123 | 24 | None | 378.82 |
| vitvs_dinov2 | narrow20/start_01 | mujoco | 86.77/6.92 | 9.575 | 0.805 | 10.266 | 14.567 | 0.886 | 124 | 24 | None | 279.97 |
| vitvs_dinov2 | narrow20/start_01 | mujoco960 | 86.77/6.92 | 8.459 | 0.744 | 10.480 | 15.139 | 1.016 | 126 | 24 | None | 333.1 |
| vitvs_dinov2 | near60/n_000 | mock | 5.58/2.98 | 5.117 | 0.491 | 4.998 | 5.286 | 0.483 | 122 | 24 | None | 347.39 |
| vitvs_dinov2 | near60/n_000 | mujoco | 5.58/2.98 | 2.774 | 0.297 | 2.775 | 4.115 | 0.322 | 117 | 24 | None | 327.04 |
| vitvs_dinov2 | near60/n_000 | mujoco960 | 5.58/2.98 | 4.191 | 0.615 | 3.961 | 4.439 | 0.659 | 120 | 24 | None | 349.35 |
| vitvs_dinov2 | near60/n_001 | mock | 3.84/2.19 | 1.533 | 0.309 | 1.438 | 2.745 | 0.333 | 120 | 24 | None | 355.98 |
| vitvs_dinov2 | near60/n_001 | mujoco | 3.84/2.19 | 5.334 | 0.748 | 3.300 | 5.334 | 0.531 | 115 | 24 | None | 301.32 |
| vitvs_dinov2 | near60/n_001 | mujoco960 | 3.84/2.19 | 5.473 | 0.500 | 4.804 | 5.957 | 0.417 | 115 | 24 | None | 348.95 |
| vitvs_dinov2 | near60/n_002 | mock | 5.99/1.33 | 8.637 | 0.881 | 8.430 | 8.729 | 0.881 | 70 | 24 | None | 323.0 |
| vitvs_dinov2 | near60/n_002 | mujoco | 5.99/1.33 | 8.428 | 0.890 | 8.332 | 8.432 | 0.884 | 69 | 24 | None | 287.34 |
| vitvs_dinov2 | near60/n_002 | mujoco960 | 5.99/1.33 | 8.672 | 0.899 | 8.310 | 8.744 | 0.869 | 67 | 24 | None | 359.01 |
| artvs | narrow20/start_00 | mock | 107.67/8.04 | 5.181 | 0.380 | 6.128 | 11.961 | 0.478 | None | 24 | 17 | 1543.69 |
| artvs | narrow20/start_00 | mujoco | 107.67/8.04 | 5.533 | 0.444 | 5.773 | 9.368 | 0.427 | None | 24 | 14 | 1600.6 |
| artvs | narrow20/start_01 | mock | 86.77/6.92 | 5.722 | 0.499 | 6.409 | 11.718 | 0.561 | None | 24 | 17 | 1401.59 |
| artvs | narrow20/start_01 | mujoco | 86.77/6.92 | 6.174 | 0.578 | 7.282 | 12.481 | 0.689 | None | 24 | 17 | 1466.65 |
| artvs | near60/n_000 | mock | 5.58/2.98 | 1.339 | 0.159 | 1.264 | 1.501 | 0.197 | 138 | 24 | 4 | 1571.94 |
| artvs | near60/n_000 | mujoco | 5.58/2.98 | 1.735 | 0.115 | 1.740 | 1.918 | 0.150 | 138 | 24 | 5 | 1571.23 |
| artvs | near60/n_001 | mock | 3.84/2.19 | 2.103 | 0.104 | 2.049 | 2.747 | 0.113 | 132 | 24 | 2 | 1841.39 |
| artvs | near60/n_001 | mujoco | 3.84/2.19 | 2.077 | 0.133 | 2.105 | 2.284 | 0.138 | 134 | 24 | 3 | 1805.21 |
| artvs | near60/n_002 | mock | 5.99/1.33 | 8.878 | 1.266 | 8.373 | 8.777 | 1.241 | 50 | 24 | None | 162.52 |
| artvs | near60/n_002 | mujoco | 5.99/1.33 | 9.668 | 1.302 | 9.597 | 9.668 | 1.302 | 50 | 24 | None | 118.78 |
| sift_ransac | narrow20/start_00 | mock | 107.67/8.04 | 0.580 | 0.041 | 1.327 | 7.267 | 0.096 | None | 24 | None | 308.14 |
| sift_ransac | narrow20/start_00 | mujoco | 107.67/8.04 | 0.566 | 0.030 | 1.323 | 7.259 | 0.088 | None | 24 | None | 245.67 |
| sift_ransac | narrow20/start_00 | mujoco960 | 107.67/8.04 | 0.592 | 0.040 | 1.337 | 7.285 | 0.099 | None | 24 | None | 282.29 |
| sift_ransac | narrow20/start_01 | mock | 86.77/6.92 | 0.500 | 0.036 | 1.133 | 6.181 | 0.082 | None | 22 | None | 342.72 |
| sift_ransac | narrow20/start_01 | mujoco | 86.77/6.92 | 0.449 | 0.018 | 1.101 | 6.172 | 0.077 | None | 24 | None | 245.08 |
| sift_ransac | narrow20/start_01 | mujoco960 | 86.77/6.92 | 0.535 | 0.038 | 1.155 | 6.173 | 0.085 | None | 22 | None | 283.69 |
| sift_ransac | near60/n_000 | mock | 5.58/2.98 | 0.055 | 0.018 | 0.081 | 0.448 | 0.044 | None | 24 | None | 341.27 |
| sift_ransac | near60/n_000 | mujoco | 5.58/2.98 | 0.114 | 0.022 | 0.103 | 0.455 | 0.047 | None | 24 | None | 245.29 |
| sift_ransac | near60/n_000 | mujoco960 | 5.58/2.98 | 0.088 | 0.018 | 0.088 | 0.451 | 0.044 | None | 24 | None | 290.62 |
| sift_ransac | near60/n_001 | mock | 3.84/2.19 | 0.031 | 0.013 | 0.093 | 0.292 | 0.032 | None | 24 | None | 311.46 |
| sift_ransac | near60/n_001 | mujoco | 3.84/2.19 | 0.066 | 0.011 | 0.114 | 0.289 | 0.031 | None | 24 | None | 246.53 |
| sift_ransac | near60/n_001 | mujoco960 | 3.84/2.19 | 0.066 | 0.008 | 0.089 | 0.324 | 0.028 | None | 24 | None | 285.31 |
| sift_ransac | near60/n_002 | mock | 5.99/1.33 | 0.100 | 0.015 | 0.164 | 0.486 | 0.029 | None | 24 | None | 349.44 |
| sift_ransac | near60/n_002 | mujoco | 5.99/1.33 | 0.150 | 0.021 | 0.192 | 0.509 | 0.030 | None | 24 | None | 247.26 |
| sift_ransac | near60/n_002 | mujoco960 | 5.99/1.33 | 0.053 | 0.010 | 0.093 | 0.479 | 0.021 | None | 24 | None | 282.36 |

## World pairs

| arm | start | pair | d landing mm | d landing deg | pose distance mm | pose angle deg | d med t | d med r | corr \|v\| | corr t | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vitvs_dinov3 | narrow20/start_00 | mock-mujoco | 0.673 | 0.015 | 4.598 | 0.584 | 1.361 | 0.162 | 0.9238 | 0.9964 | pass |
| vitvs_dinov3 | narrow20/start_00 | mock-mujoco960 | 1.951 | 0.288 | 3.803 | 0.474 | 1.447 | 0.211 | 0.9113 | 0.9979 | pass |
| vitvs_dinov3 | narrow20/start_00 | mujoco960-mujoco | 2.624 | 0.272 | 3.676 | 0.434 | 2.808 | 0.373 | 0.9043 | 0.9982 | pass |
| vitvs_dinov3 | narrow20/start_01 | mock-mujoco | 2.294 | 0.233 | 3.699 | 0.431 | 2.148 | 0.203 | 0.8243 | 0.9963 | pass |
| vitvs_dinov3 | narrow20/start_01 | mock-mujoco960 | 2.711 | 0.293 | 3.251 | 0.388 | 2.340 | 0.243 | 0.8419 | 0.9993 | pass |
| vitvs_dinov3 | narrow20/start_01 | mujoco960-mujoco | 0.417 | 0.061 | 2.580 | 0.284 | 0.192 | 0.040 | 0.8296 | 0.9973 | pass |
| vitvs_dinov3 | near60/n_000 | mock-mujoco | 0.590 | 0.029 | 1.306 | 0.199 | 0.388 | 0.020 | 0.2815 | 0.3621 | pass |
| vitvs_dinov3 | near60/n_000 | mock-mujoco960 | 0.406 | 0.179 | 1.270 | 0.204 | 0.328 | 0.075 | 0.3715 | -0.0677 | pass |
| vitvs_dinov3 | near60/n_000 | mujoco960-mujoco | 0.996 | 0.208 | 2.556 | 0.306 | 0.716 | 0.095 | 0.3709 | 0.4428 | pass |
| vitvs_dinov3 | near60/n_001 | mock-mujoco | 3.885 | 0.654 | 8.488 | 1.069 | 2.639 | 0.526 | 0.362 | 0.385 | FAIL |
| vitvs_dinov3 | near60/n_001 | mock-mujoco960 | 0.148 | 0.263 | 5.218 | 0.626 | 0.817 | 0.401 | 0.2487 | 0.3459 | pass |
| vitvs_dinov3 | near60/n_001 | mujoco960-mujoco | 4.034 | 0.391 | 4.199 | 0.537 | 1.823 | 0.125 | 0.3578 | 0.858 | FAIL |
| vitvs_dinov3 | near60/n_002 | mock-mujoco | 0.639 | 0.077 | 2.709 | 0.304 | 0.157 | 0.024 | 0.2637 | 0.7315 | pass |
| vitvs_dinov3 | near60/n_002 | mock-mujoco960 | 2.941 | 0.206 | 4.285 | 0.461 | 2.723 | 0.195 | 0.1274 | -0.4005 | pass |
| vitvs_dinov3 | near60/n_002 | mujoco960-mujoco | 3.581 | 0.284 | 5.978 | 0.664 | 2.879 | 0.218 | 0.1135 | -0.793 | FAIL |
| vitvs_dinov2 | narrow20/start_00 | mock-mujoco | 0.138 | 0.208 | 5.414 | 0.648 | 1.065 | 0.102 | 0.9458 | 0.998 | pass |
| vitvs_dinov2 | narrow20/start_00 | mock-mujoco960 | 0.258 | 0.077 | 1.993 | 0.190 | 0.625 | 0.061 | 0.9247 | 0.9991 | pass |
| vitvs_dinov2 | narrow20/start_00 | mujoco960-mujoco | 0.120 | 0.131 | 4.590 | 0.571 | 1.689 | 0.162 | 0.9317 | 0.9992 | pass |
| vitvs_dinov2 | narrow20/start_01 | mock-mujoco | 0.188 | 0.130 | 2.370 | 0.331 | 0.219 | 0.127 | 0.8924 | 0.9991 | pass |
| vitvs_dinov2 | narrow20/start_01 | mock-mujoco960 | 1.304 | 0.192 | 1.401 | 0.201 | 0.004 | 0.003 | 0.8696 | 0.9978 | pass |
| vitvs_dinov2 | narrow20/start_01 | mujoco960-mujoco | 1.117 | 0.061 | 2.107 | 0.265 | 0.214 | 0.130 | 0.8654 | 0.9988 | pass |
| vitvs_dinov2 | near60/n_000 | mock-mujoco | 2.343 | 0.194 | 2.622 | 0.329 | 2.224 | 0.162 | 0.4146 | -0.1309 | pass |
| vitvs_dinov2 | near60/n_000 | mock-mujoco960 | 0.926 | 0.124 | 2.451 | 0.285 | 1.038 | 0.176 | 0.327 | 0.4798 | pass |
| vitvs_dinov2 | near60/n_000 | mujoco960-mujoco | 1.417 | 0.318 | 3.010 | 0.407 | 1.186 | 0.338 | 0.3699 | 0.5519 | FAIL |
| vitvs_dinov2 | near60/n_001 | mock-mujoco | 3.801 | 0.440 | 5.464 | 0.682 | 1.862 | 0.198 | 0.2618 | -0.2331 | FAIL |
| vitvs_dinov2 | near60/n_001 | mock-mujoco960 | 3.940 | 0.192 | 4.823 | 0.582 | 3.366 | 0.084 | 0.3281 | -0.3795 | FAIL |
| vitvs_dinov2 | near60/n_001 | mujoco960-mujoco | 0.139 | 0.248 | 2.775 | 0.362 | 1.504 | 0.114 | 0.3442 | 0.573 | pass |
| vitvs_dinov2 | near60/n_002 | mock-mujoco | 0.209 | 0.009 | 1.015 | 0.134 | 0.098 | 0.004 | 0.371 | 0.8653 | pass |
| vitvs_dinov2 | near60/n_002 | mock-mujoco960 | 0.035 | 0.019 | 0.750 | 0.095 | 0.120 | 0.012 | 0.1885 | 0.8581 | pass |
| vitvs_dinov2 | near60/n_002 | mujoco960-mujoco | 0.244 | 0.009 | 0.551 | 0.044 | 0.022 | 0.015 | 0.1877 | 0.9563 | pass |
| artvs | narrow20/start_00 | mock-mujoco | 0.351 | 0.064 | 1.471 | 0.187 | 0.355 | 0.051 | 0.9153 | 0.9972 | pass |
| artvs | narrow20/start_01 | mock-mujoco | 0.452 | 0.079 | 0.712 | 0.092 | 0.873 | 0.127 | 0.8962 | 0.9994 | pass |
| artvs | near60/n_000 | mock-mujoco | 0.396 | 0.045 | 0.474 | 0.059 | 0.475 | 0.047 | 0.794 | 0.8146 | pass |
| artvs | near60/n_001 | mock-mujoco | 0.026 | 0.030 | 0.428 | 0.043 | 0.057 | 0.025 | 0.473 | 0.3523 | pass |
| artvs | near60/n_002 | mock-mujoco | 0.790 | 0.037 | 1.151 | 0.136 | 1.224 | 0.060 | 0.333 | 0.9846 | pass |
| sift_ransac | narrow20/start_00 | mock-mujoco | 0.015 | 0.011 | 0.120 | 0.015 | 0.004 | 0.008 | 0.9998 | 1.0 | pass |
| sift_ransac | narrow20/start_00 | mock-mujoco960 | 0.012 | 0.001 | 0.025 | 0.003 | 0.011 | 0.003 | 0.9998 | 1.0 | pass |
| sift_ransac | narrow20/start_00 | mujoco960-mujoco | 0.026 | 0.010 | 0.128 | 0.016 | 0.015 | 0.011 | 0.9998 | 1.0 | pass |
| sift_ransac | narrow20/start_01 | mock-mujoco | 0.051 | 0.018 | 0.166 | 0.022 | 0.032 | 0.006 | 0.9995 | 1.0 | pass |
| sift_ransac | narrow20/start_01 | mock-mujoco960 | 0.035 | 0.001 | 0.037 | 0.002 | 0.022 | 0.002 | 0.9995 | 1.0 | pass |
| sift_ransac | narrow20/start_01 | mujoco960-mujoco | 0.087 | 0.020 | 0.177 | 0.022 | 0.054 | 0.008 | 0.9996 | 1.0 | pass |
| sift_ransac | near60/n_000 | mock-mujoco | 0.059 | 0.004 | 0.078 | 0.010 | 0.022 | 0.003 | 0.9993 | 0.9999 | pass |
| sift_ransac | near60/n_000 | mock-mujoco960 | 0.033 | 0.000 | 0.039 | 0.005 | 0.007 | 0.000 | 0.9994 | 1.0 | pass |
| sift_ransac | near60/n_000 | mujoco960-mujoco | 0.026 | 0.004 | 0.048 | 0.006 | 0.015 | 0.003 | 0.9991 | 1.0 | pass |
| sift_ransac | near60/n_001 | mock-mujoco | 0.035 | 0.001 | 0.035 | 0.004 | 0.021 | 0.001 | 0.9949 | 0.9998 | pass |
| sift_ransac | near60/n_001 | mock-mujoco960 | 0.035 | 0.004 | 0.045 | 0.006 | 0.004 | 0.004 | 0.9929 | 0.9998 | pass |
| sift_ransac | near60/n_001 | mujoco960-mujoco | 0.000 | 0.003 | 0.040 | 0.005 | 0.025 | 0.003 | 0.9886 | 0.9998 | pass |
| sift_ransac | near60/n_002 | mock-mujoco | 0.050 | 0.006 | 0.050 | 0.006 | 0.028 | 0.001 | 0.9997 | 0.9999 | pass |
| sift_ransac | near60/n_002 | mock-mujoco960 | 0.047 | 0.006 | 0.053 | 0.007 | 0.070 | 0.008 | 0.9997 | 0.9998 | pass |
| sift_ransac | near60/n_002 | mujoco960-mujoco | 0.097 | 0.011 | 0.101 | 0.013 | 0.099 | 0.009 | 0.9998 | 0.9998 | pass |

## Verdict (gate pair mock-mujoco)

```
{
  "landing": {
    "vitvs_dinov3": {
      "n": 5,
      "pass": false,
      "failing": [
        "near60/n_001"
      ],
      "failing_but_curve_medians_agree": []
    },
    "vitvs_dinov2": {
      "n": 5,
      "pass": false,
      "failing": [
        "near60/n_001"
      ],
      "failing_but_curve_medians_agree": [
        "near60/n_001"
      ]
    },
    "artvs": {
      "n": 5,
      "pass": true,
      "failing": [],
      "failing_but_curve_medians_agree": []
    },
    "sift_ransac": {
      "n": 5,
      "pass": true,
      "failing": [],
      "failing_but_curve_medians_agree": []
    }
  },
  "artvs_switch": {
    "pass": false,
    "not_fired": {
      "mock": [
        "near60/n_002"
      ],
      "mujoco": [
        "near60/n_002"
      ]
    },
    "worlds_agree_on_every_start": true,
    "steps": {
      "mock": {
        "narrow20/start_00": 17,
        "narrow20/start_01": 17,
        "near60/n_000": 4,
        "near60/n_001": 2,
        "near60/n_002": null
      },
      "mujoco": {
        "narrow20/start_00": 14,
        "narrow20/start_01": 17,
        "near60/n_000": 5,
        "near60/n_001": 3,
        "near60/n_002": null
      }
    }
  }
}
```

CPU seconds per world: {'mock': 11257.1, 'mujoco': 10432.8, 'mujoco960': 4342.6}; GL renderers: ['llvmpipe (LLVM 20.1.2, 256 bits)']
