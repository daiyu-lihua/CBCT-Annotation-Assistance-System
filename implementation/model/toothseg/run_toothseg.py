# -*- coding: utf-8 -*-
"""
ToothSeg 端到端推理入口(Windows) —— CBCT-Annotation-Assistance-System 集成版。

改编自 ToothSegWork/scripts/windows/run_toothseg.py(2026-09-02 交接版),差异仅:
  1. 路径不再硬编码,按本脚本位置自动定位,并支持环境变量覆盖:
       TOOTHSEG_HOME           代码根(含 memsafe_inference.py / postprocess_predictions/
                               / toothseg 包)。默认 = 本脚本所在目录。
       TOOTHSEG_NNUNET_RESULTS 权重根(含 Dataset121... / Dataset123...)。
                               默认 = <代码根>/../weights, 即 implementation/model/weights。
  2. 后处理子进程注入 PYTHONPATH=TOOTHSEG_HOME, 使 `from toothseg.datasets...
       import copy_geometry` 解析到本目录内的迷你 toothseg 包(自包含,不再依赖
       ToothSegWork 或 site-packages 安装)。
  3. nnUNet_raw / nnUNet_preprocessed 默认放 <代码根>/../work/ 下。

流程(与原版一致,对应官方 inference_generel.sh):
  1. 语义分支  nnUNetv2_predict (Dataset121)
  2. 实例分支  nnUNetv2_predict (Dataset123)  -- 输入同一批原始 imagesTs,由 nnU-Net 自动 resample 到 0.2mm
  3. border_core -> instances
  4. resize 回原始 spacing (以 raw imagesTs 作为 ref)
  5. assign 牙位编号 (mincost 自纠正)

SimpleITK 的 NIfTI I/O 由 nninteractive 环境的 sitecustomize.py 自动用 nibabel 兜底
(中文路径/损坏的 sitk 均可用),故后处理脚本可正常运行。
"""
import os
import sys
import argparse
import subprocess
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA 显存分配策略: 必须在 torch 启动 caching allocator 之前设置。
# 真实根因(经 faulthandler + pytorch 自身诊断确认): 语义分支单个滑窗 tile
# 前向峰值本身贴近 8GB 卡(allocated 6.01GiB, reserved-unallocated 仅 31MB),
# 解码器 instance_norm 层再要 1GB 时, 分配器碎片化拼不出空闲块 → CUDA OOM(早期
# 随机表现为原生 segfault 0xC0000005)。
# 修法: expandable_segments:True(可增长段大幅减少碎片, pytorch 报错点名建议)
#      + garbage_collection_threshold 兜底 + max_split_size_mb 限制大块碎片。
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.5,max_split_size_mb:128'
# ---------------------------------------------------------------------------

# ---- 路径配置(自动定位, 可用环境变量覆盖) ----
CODE_DIR = Path(os.environ.get('TOOTHSEG_HOME', Path(__file__).resolve().parent))
NNUNET_RESULTS = Path(os.environ.get('TOOTHSEG_NNUNET_RESULTS', CODE_DIR.parent / 'weights'))
WORK_DIR = NNUNET_RESULTS.parent / 'work'
PYTHON = sys.executable

SEMSEG_DATASET = 'Dataset121_ToothFairy2_Teeth'
SEMSEG_TRAINER = 'nnUNetTrainer_onlyMirror01_DASegOrd0'
SEMSEG_CONFIG = '3d_fullres_resample_torch_256_bs8_ctnorm'
SEMSEG_MODEL_DIR = NNUNET_RESULTS / SEMSEG_DATASET / f'{SEMSEG_TRAINER}__nnUNetPlans__{SEMSEG_CONFIG}'

INSTSEG_DATASET = 'Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px'
INSTSEG_TRAINER = 'nnUNetTrainer'
INSTSEG_CONFIG = '3d_fullres_resample_torch_192_bs8_ctnorm'
INSTSEG_MODEL_DIR = NNUNET_RESULTS / INSTSEG_DATASET / f'{INSTSEG_TRAINER}__nnUNetPlans__{INSTSEG_CONFIG}'

DP_PATH = CODE_DIR / 'postprocess_predictions'


def check():
    import torch
    ok_cuda = torch.cuda.is_available()
    print('torch', torch.__version__, '| cuda:', ok_cuda)
    for p in [SEMSEG_MODEL_DIR, INSTSEG_MODEL_DIR]:
        cp = p / 'fold_5' / 'checkpoint_final.pth'
        print(' ', 'OK ' if cp.exists() else 'MISSING', cp)
    return ok_cuda


def check_input(indir: Path) -> Path:
    """校验输入并定位到 nnU-Net 需要的 imagesTs 目录。
    支持四种传法:
      1. 单个 *_0000.nii.gz 文件   -> 用其所在目录
      2. 含 imagesTs 子目录的目录   -> 用该子目录
      3. 目录本身直接含 .nii.gz    -> 直接用该目录
      4. 空父目录                  -> 自动创建 imagesTs 子目录
    """
    indir = Path(indir)
    if not indir.exists():
        raise SystemExit(f'[ERROR] 输入路径不存在: {indir}')

    # 情况1: 传入的是单个图像文件 -> 定位到父目录
    if indir.is_file():
        imagesTs = indir.parent
        print('[输入] 检测到单个文件,自动使用其所在目录:', imagesTs, flush=True)
    else:
        imagesTs = indir / 'imagesTs'
        # 情况2: 已有 imagesTs 子目录且包含图像
        if imagesTs.exists() and any(imagesTs.glob('*.nii.gz')):
            pass
        elif any(indir.glob('*.nii.gz')):
            # 情况3: 目录本身就直接含图像
            imagesTs = indir
        else:
            # 情况4: 空目录 -> 创建 imagesTs
            try:
                imagesTs.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise SystemExit(f'[ERROR] 无法在 {indir} 下创建 imagesTs 目录: {e}')

    files = list(imagesTs.glob('*.nii*'))
    if not files:
        raise SystemExit(f'[ERROR] {imagesTs} 下没有任何 .nii.gz 输入文件')
    bad = [f for f in files if '_0000.' not in f.name]
    if bad:
        raise SystemExit(f'[ERROR] 文件名必须带 "_0000"(如 xxx_0000.nii.gz)。发现不符合的文件: {[b.name for b in bad]}')
    return imagesTs


def run_predict_branch(input_dir: Path, out_dir: Path, model_dir: Path, folds, chk='checkpoint_final.pth',
                       step_size=0.5, use_tta=False, save_probs_blocks=True, probs_cache_dir=None):
    """MemSafe 低内存推理(单进程 Windows 稳定):
    - 滑窗阶段按 fp16 logits 大小自动选择显存/内存累积, 避免 OOM 整轮重跑;
    - 导出阶段主进程流式 + z 轴分块重采样/argmax/概率 fp16 落盘, 峰值内存 ~5GB;
    - 修复原版 pred.step_size 无效 bug(父类属性实为 tile_step_size, 构造时传入)。"""
    sys.path.insert(0, str(CODE_DIR))
    from memsafe_inference import MemSafePredictor
    print(f'\n=== 预测分支: {model_dir.parent.name} '
          f'(滑窗步长 tile_step_size={step_size}, TTA={use_tta}) ===', flush=True)
    pred = MemSafePredictor(tile_step_size=float(step_size), use_mirroring=bool(use_tta),
                            verbose=True, probs_cache_dir=probs_cache_dir,
                            save_probs_blocks=bool(save_probs_blocks))
    pred.initialize_from_trained_model_folder(str(model_dir), use_folds=folds, checkpoint_name=chk)
    os.makedirs(str(out_dir), exist_ok=True)
    pred.predict_from_files(str(input_dir), str(out_dir),
                            save_probabilities=(model_dir.parent.name == SEMSEG_DATASET),
                            num_processes_preprocessing=1, num_processes_segmentation_export=1)
    print('  完成 ->', out_dir, flush=True)


def postprocess(raw_imagesTs: Path, instseg_out: Path, semseg_out: Path, out_root: Path, np_proc: int = 4):
    import subprocess
    def run(pyfile, *args):
        cmd = [PYTHON, str(DP_PATH / pyfile)] + [str(a) for a in args]
        print('  $', ' '.join(cmd), flush=True)
        # 注入 PYTHONPATH: 让 postprocess 脚本的 `from toothseg.datasets...` 解析到
        # 本目录内的迷你 toothseg 包(与 ToothSegWork 解耦)。PYTHONPATH 优先级高于
        # site-packages, 因此即便环境中另装了 toothseg 也不会引用错位。
        env = os.environ.copy()
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = str(CODE_DIR) + (os.pathsep + existing if existing else '')
        subprocess.check_call(cmd, env=env)

    border_core = instseg_out          # 实例分支原始输出(border-core 语义)
    instances = out_root / 'step3_instances'
    instances_resized = out_root / 'step4_instances_resized'
    final = out_root / 'final_prediction'

    print('\n=== 步骤3: border-core -> 实例 ===', flush=True)
    run('border_core_to_instances.py', '-i', border_core, '-o', instances, '-np', np_proc)
    print('\n=== 步骤4: 重采样回原始 spacing ===', flush=True)
    run('resize_predictions.py', '-i', instances, '-o', instances_resized, '-ref', raw_imagesTs, '-np', np_proc)
    print('\n=== 步骤5: 牙位编号(mincost 自纠正) ===', flush=True)
    # 显式传绝对路径: 该脚本默认值是相对 cwd 的 'toothseg/datasets/...', 只有
    # 恰好在 TOOTHSEG_HOME 下启动才解析正确; 传绝对路径后对 cwd 无假设。
    run('assign_mincost_tooth_labels.py', '-ifolder', instances_resized, '-sfolder', semseg_out,
        '-o', final, '-np', np_proc,
        '--distributions', CODE_DIR / 'toothseg' / 'datasets' / 'toothfairy2' / 'fdi_pair_distrs.json')
    print(f'\n[完成] 最终结果目录: {final}', flush=True)


def main():
    ap = argparse.ArgumentParser(description='ToothSeg 端到端牙齿分割(带编号)')
    ap.add_argument('-i', required=False, default=None, help='输入目录(内含 imagesTs 或直接用图像目录); 仅 --only-check 时可省略')
    ap.add_argument('-o', required=False, default=None, help='结果输出目录(默认: <输入目录父>/toothseg_output)')
    ap.add_argument('-f', default='5', help='fold,默认 5')
    ap.add_argument('--mode', default='full', choices=['full', 'sem', 'inst'],
                    help='运行模式: full(完整流程) / sem(仅语义) / inst(仅实例)')
    ap.add_argument('--np', type=int, default=2, help='后处理并行进程数,默认 2(内存有限时建议 1)')
    ap.add_argument('--step-size', type=float, default=0.5,
                    help='滑动窗口步长(小=更精确但慢,大=快但略降精度),默认 0.5')
    ap.add_argument('--tta', action='store_true', help='启用镜像 TTA(更准但约慢 2-8 倍),默认关闭')
    ap.add_argument('--no-probs-cache', action='store_true',
                    help='不保存 fp16 概率分块缓存(省磁盘, 但 assign 牙位编号将失去概率自纠正)')
    ap.add_argument('--only-check', action='store_true', help='仅检查环境与权重')
    args = ap.parse_args()

    os.environ['nnUNet_results'] = str(NNUNET_RESULTS)
    os.environ['nnUNet_raw'] = str(WORK_DIR / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(WORK_DIR / 'nnUNet_preprocessed')
    os.makedirs(WORK_DIR / 'nnUNet_raw', exist_ok=True)
    os.makedirs(WORK_DIR / 'nnUNet_preprocessed', exist_ok=True)

    folds = tuple(int(x) for x in args.f.split(',') if x)
    if args.only_check:
        check()
        return
    if args.i is None:
        raise SystemExit('[ERROR] 缺少必需参数 -i(输入目录); 仅环境检查可省略 -i')

    indir = Path(args.i).resolve()
    if not indir.exists():
        raise SystemExit(f'[ERROR] 输入目录不存在: {indir}')
    imagesTs = check_input(indir)
    out_root = Path(args.o or (indir.parent / 'toothseg_output'))

    semseg_out = out_root / 'step1_semseg_branch'
    instseg_out = out_root / 'step2_instseg_branch'
    os.makedirs(semseg_out, exist_ok=True)
    os.makedirs(instseg_out, exist_ok=True)

    script = str(Path(__file__).resolve())
    if args.mode == 'full':
        # 调度父进程故意不初始化 CUDA / 不 import torch: 让两个推理子进程独占
        # GPU。若父进程先执行 check() 初始化 cuda context, Windows 上父子进程
        # 竞争同一 GPU 会偶发原生访问违例(0xC0000005, 见语义 Step2 崩溃)。
        # 环境与权重由第一个推理子进程自行检查。
        # 两个全系数分支拆成独立子进程依次运行: 语义进程跑完即退出, 其物理
        # 内存/显存全部归还系统, 使实例分支的预处理(windows spawn worker 做
        # 0.2mm 重采样, 内存峰值高)不再叠加语义残留而 OOM。每分支独占整机
        # 资源, 也避免同一进程内两套模型/缓存相互拖累。
        env_map = os.environ.copy()
        base = [sys.executable, script, '-i', str(imagesTs), '-o', str(out_root),
                '-f', args.f, '--step-size', str(args.step_size)]
        if args.tta:
            base.append('--tta')
        cmd_sem = base + ['--mode', 'sem']
        if args.no_probs_cache:
            cmd_sem.append('--no-probs-cache')
        print(f"\n### [子进程] 语义分支 (独立进程, tile_step_size={args.step_size}) ###", flush=True)
        subprocess.check_call(cmd_sem, env=env_map)
        cmd_inst = base + ['--mode', 'inst']
        print(f"\n### [子进程] 实例分支 (独立进程, tile_step_size={args.step_size}) ###", flush=True)
        subprocess.check_call(cmd_inst, env=env_map)
        print('\n### 后处理 ###')
        postprocess(imagesTs, instseg_out, semseg_out, out_root, np_proc=args.np)
    else:
        print('### Checking environment ###')
        check()
        if args.mode == 'sem':
            print(f'\n### 语义分支 (tile_step_size={args.step_size}) ###')
            run_predict_branch(imagesTs, semseg_out, SEMSEG_MODEL_DIR, folds, step_size=args.step_size,
                               use_tta=args.tta, save_probs_blocks=not args.no_probs_cache)
        else:
            print(f'\n### 实例分支 (tile_step_size={args.step_size}) ###')
            run_predict_branch(imagesTs, instseg_out, INSTSEG_MODEL_DIR, folds, step_size=args.step_size,
                               use_tta=args.tta, save_probs_blocks=False)
        print(f'\n[完成] 已按模式 [{args.mode}] 完成预测,跳过后续步骤')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
