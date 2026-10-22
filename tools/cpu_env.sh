# tools/cpu_env.sh - source it in every shell that builds or tests the vs-sim port.
#
#   source tools/cpu_env.sh            # from the repository root
#
# The clone's tier must run with VSSIM_ASSET_ROOT / VSSIM_YCB_ROOT UNSET: three tests in
# test_assets.py assert the built-in defaults. The full command:
#   PYTHONPATH=external/vs-sim env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT \
#     $VSBENCH_PY -m pytest external/vs-sim/tests/fast -q
# (results/bridge/u0_clone_fast_tier.txt is that run).
#
# The station rules (docs/PORT_PLAN.md section 7), scripted once so nobody retypes them:
# no GPU ever, MuJoCo on Mesa llvmpipe, at most two threads, lowest CPU priority, weights
# read from our own copy (never the shared HF / torch.hub caches), never a download.

export CUDA_VISIBLE_DEVICES=""            # no GPU, ever
export VSBENCH_SOFTWARE_GL=1              # Mesa llvmpipe for MuJoCo (mujoco-egl; no libOSMesa here)
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCH_NUM_THREADS=2                # the bridge calls torch.set_num_threads(2) too
export HF_HOME="$HOME/genvs_ws/vsbench_assets/weights/hf"  HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$HOME/genvs_ws/vsbench_assets/weights/torch"
export VSBENCH_WORKSPACE_ROOT="$HOME/genvs_ws"   # where /workspace/... textures and USDs really are
export VSSIM_ASSET_ROOT="$HOME/genvs_ws/genvs-sim"    # the curated object USDs (read only)
export VSSIM_YCB_ROOT="$HOME/genvs_ws/assets/ycb"     # the YCB scans (read only)
# pip: never compile in parallel (and the install commands use --only-binary=:all:)
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1

# The interpreter, at nice 19. `py` for interactive shells; "$VSBENCH_PY" in scripts
# (aliases are not expanded in non-interactive shells).
export VSBENCH_PY="nice -n 19 $HOME/genvs_ws/vs_sim_light/.venv/bin/python"
alias py='nice -n 19 ~/genvs_ws/vs_sim_light/.venv/bin/python'
