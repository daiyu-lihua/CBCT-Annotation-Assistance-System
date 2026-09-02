# -*- coding: utf-8 -*-
"""
MemSafe 推理补丁：让 ToothSeg(nnU-Net v2) 在 32GB 内存 + 8GB 显存的本地电脑上稳定运行。

================ 内存问题根因（原版） ================
1) 导出阶段(爆内存主因)：
   convert_predicted_logits_to_segmentation_with_correct_shape 一次性执行
   fp16 logits -> fp32 全卷(~4-8GB) -> F.interpolate 全卷输出(~4-8GB)
   -> softmax 概率 fp32 全卷(~4-8GB) -> revert_cropping 回原图 fp32(12-37GB)
   -> np.savez_compressed 压缩缓冲再翻倍。峰值轻松超过 30GB。
2) 滑窗阶段：全卷 fp16 logits 在显存/内存累积（1-2GB，可承受，保留）。
3) export_pool 子进程通过 pickle 复制一份完整 logits（主进程一份+子进程一份）。
4) run_toothseg.py 中 pred.step_size 属性无效(父类用 tile_step_size)，步长从未生效。

================ MemSafe 方案（本文件） ================
- 子类化 nnUNetPredictor：滑窗推理完全复用父类(显存 fp16 累积，8GB 显存足够)；
- 导出阶段完全重写为“分块流水线”：
  * 沿 z 轴分块，块内 fp32 三线性重采样（自实现 1D 插值，全局坐标精确一致，
    与 F.interpolate(align_corners=False) 数值对齐，分块不引入任何精度损失）；
  * seg 直接在块内 argmax（softmax 单调，argmax(softmax(x))==argmax(x)，省一份全卷）；
  * 概率按块 softmax 后以 fp16 存为无压缩 .npy 分块缓存 + meta.json
    （替代原版 12-37GB 的 fp32 全卷 npz；fp16 精度 ~5e-4，对 0.1/0.95 阈值判断无影响）；
  * 回填 bbox、转置、写 nii.gz 全部在 uint8 小数组上进行；
- 导出从多进程 Pool 改为主进程流式（消除 pickle 复制；预处理子进程保留，队列限流）；
- 内存峰值从 ~30GB 降到 ~5GB，且与输入尺寸基本无关。

概率缓存接口(供后处理 assign_mincost 低内存版使用, 见 meta.json)：
  <name>_probs_cache/
    meta.json      # 块划分、crop bbox、shape、dtype 等
    block_0000.npy # fp16 (C, X, Y, z_blk)，模型轴序(transpose 后)，crop 内坐标系
"""
import itertools
import json
import shutil
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from tqdm import tqdm

try:
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
except Exception:  # 兼容环境缺 batchgenerators 的情况(仅影响迭代器收尾)
    MultiThreadedAugmenter = None

try:
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
except Exception:
    from nnunetv2.inference.export_prediction import insert_crop_into_image  # 老版本回退

# fp32 分块预算(元素数)：约 1.2e8 元素 = ~0.5GB fp32，留足中间数组余量
_BLOCK_ELEM_BUDGET = 1.2e8
# RAM 常驻 logits 预算(GB)：超过则警告(仍可运行，但建议减小输入或分块)
_RAM_LOGITS_WARN_GB = 8.0


def _compute_input_window(out_start: int, out_len: int, scale: float, in_len: int):
    """计算输出区间 [out_start, out_start+out_len) 在输入轴上依赖的窗口与插值权重。
    语义与 PyTorch F.interpolate(mode='linear', align_corners=False) 完全一致：
      src(k) = clamp((k + 0.5) * scale - 0.5, 0, in_len - 1)   (k 为全局输出索引)
      i0 = floor(src), i1 = i0 + 1, w = src - i0
    返回 (win_lo, win_hi_exclusive, w)，w 为每个输出位置的小数权重(fp64)。
    """
    k = np.arange(out_start, out_start + out_len, dtype=np.float64)
    src = (k + 0.5) * scale - 0.5
    src = np.clip(src, 0.0, float(in_len - 1))
    i0 = np.floor(src)
    w = src - i0
    lo = int(i0.min())
    hi = int(i0.max()) + 2  # 窗口必须覆盖 i1 = i0 + 1
    lo = max(0, lo)
    hi = min(in_len, hi)
    return lo, hi, w


def _linear_interp_window(x: torch.Tensor, dim: int, scale: float, out_start: int, out_len: int,
                          win_start: int, win_len: int, in_len: int) -> torch.Tensor:
    """沿 dim 对输入窗口 x(长 win_len，对应全局输入坐标 [win_start, win_start+win_len))
    做线性插值，输出长 out_len，输出全局区间 [out_start, out_start+out_len)。
    数值语义与 F.interpolate(align_corners=False) 一致（全局坐标系精确，无分块误差）。
    PyTorch 语义: src<0 -> 0(权重 0); src>=in-1 -> 用 in-1(右邻=自身, 权重归零)。
    """
    dev = x.device
    src = (np.arange(out_start, out_start + out_len, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.clip(src, 0.0, float(in_len - 1))
    i0g = np.floor(src)
    w = torch.from_numpy((src - i0g).astype(np.float32)).to(dev)
    # 窗口内索引: i0 可落在窗口最后一位(此时 w=0, i1 取自身即可)
    i0l = torch.from_numpy(np.clip((i0g - win_start).astype(np.int64), 0, max(win_len - 1, 0))).to(dev)
    i1l = (i0l + 1).clamp_(max=max(win_len - 1, 0))
    x0 = x.index_select(dim, i0l)
    x1 = x.index_select(dim, i1l)
    shape = [1] * x.dim()
    shape[dim] = out_len
    w0 = (1.0 - w).view(shape)
    w1 = w.view(shape)
    return x0 * w0 + x1 * w1


class MemSafePredictor(nnUNetPredictor):
    """低内存版 nnUNetPredictor。

    额外参数(在 nnUNetPredictor 基础上)：
      probs_cache_dir: 概率分块缓存根目录(默认与输出文件同目录)。
      save_probs_blocks: 预测时是否把概率写成 fp16 分块缓存(替代原版巨型 fp32 npz)。
      verbose_log: 打印分块进度与内存日志。
    """

    def __init__(self, *args, probs_cache_dir=None, save_probs_blocks=True, verbose_log=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs_cache_dir = Path(probs_cache_dir) if probs_cache_dir else None
        self.save_probs_blocks = save_probs_blocks
        self.verbose_log = verbose_log

    # ------------------------------------------------------------------
    # 滑窗：覆写父类，把"全卷 isinf 检查"改为分块流式，
    # 避免为 8GB 级 fp16 logits 分配同尺寸 bool 数组导致 OOM。
    # 其余逻辑(累积/高斯)与父类完全一致。
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self, data, slicers,
                                                       do_on_device: bool = True):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = threading.Thread(target=producer, args=(data, slicers, queue))
            t.start()

            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # 分块 isinf 检查：任一 z 块出现 inf 即中止(逐块即时释放 bool 缓冲)。
            z = predicted_logits.shape[-1]
            blk_z = max(1, int(np.ceil(z / max(1, int(np.ceil(
                predicted_logits.numel() * predicted_logits.element_size() /
                (256 * 1024 ** 2)))))))  # 每片目标位预算 ~256MB
            for dz0 in range(0, z, blk_z):
                if torch.isinf(predicted_logits[..., dz0:dz0 + blk_z]).any():
                    raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                       'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                       'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    # ------------------------------------------------------------------
    # 主入口：流式导出(替代父类的 Pool + pickle 方案)
    # ------------------------------------------------------------------
    def predict_from_data_iterator(self, data_iterator, save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 1):
        assert num_processes_segmentation_export in (0, 1), \
            'MemSafe 模式下导出在主进程流式执行(num_processes_segmentation_export 必须 <= 1)'
        save_probabilities = bool(save_probabilities and self.save_probs_blocks)
        results = []
        for preprocessed in data_iterator:
            data = preprocessed['data']
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                import os as _os
                _os.remove(delfile)

            ofile = preprocessed['ofile']
            properties = preprocessed['data_properties']
            if ofile is not None:
                print(f'\nPredicting {Path(ofile).name}:')
            else:
                print(f'\nPredicting image of shape {tuple(data.shape)}:')

            self._tune_perform_everything_on_device(data)
            t0 = time.time()
            prediction = self.predict_logits_from_preprocessed_data(data)  # torch fp16 CPU (C,X,Y,Z)
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach().to('cpu')
            if self.verbose_log:
                gb = prediction.numel() * prediction.element_size() / 1024 ** 3
                print(f'  [MemSafe] 滑窗完成: logits fp16 {tuple(prediction.shape)} = {gb:.2f}GB, '
                      f'耗时 {time.time() - t0:.1f}s')

            if ofile is not None:
                self._export_prediction_memsafe(prediction, properties, ofile, save_probabilities)
                results.append(None)
            else:
                seg = self._segmentation_from_logits_memsafe(prediction, properties)
                results.append(seg)

            del prediction, data
            empty_cache(self.device)

        if MultiThreadedAugmenter is not None and isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return results

    # ------------------------------------------------------------------
    # 数据迭代器：关闭 pin_memory。
    # 原因: 同一进程连续跑语义/实例两个分支时, 上一个分支遗留的 pinned
    # host 内存仍处于 cudaHostRegister 映射状态, 下一个分支 preprocessing
    # 再次 pin_memory() 触发 "CUDA error: resource already mapped" 崩溃。
    # 本机分块预处理数据量很小, 关闭 pin_memory 无性能损失。
    # ------------------------------------------------------------------
    def _internal_get_data_iterator_from_lists_of_filenames(self, input_list_of_lists,
                                                            seg_from_prev_stage_files,
                                                            output_filenames_truncated, num_processes):
        from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, False,
                                                self.verbose_preprocessing)

    # ------------------------------------------------------------------
    # 显存/内存累积位置：GPU 优先 + 动态预估 + 无 OOM 兜底。
    #   * 网络 forward 始终在 GPU 执行(算力主要用显存);
    #   * 滑窗累积数组(fp16 logits)优先放显存(最大化利用空闲显存),
    #     仅当"fp16 logits + 瞬态工作区"超出空闲显存才放到内存(fp16);
    #   * 保证绝不爆显存: 父类 predict_sliding_window 用 try/except 包裹
    #     本方法, OOM(RuntimeError)会自动以 CPU 累积整轮重试再退出。
    # 说明: 本机测试首次采用 GPU 累积跑 0.2mm 实例分支时, 曾在"多进程 Pool 导出"
    # 阶段触发 c10.dll 原生崩溃(0xC0000005); 现导出改为主进程 CPU 流式, logits 在
    # 滑窗结束立即 .to('cpu'), 已根除该崩溃面, GPU 累积路径可安全地用更紧预算。
    # ------------------------------------------------------------------
    def _tune_perform_everything_on_device(self, data: torch.Tensor) -> None:
        self.perform_everything_on_device = False
        if self.device.type != 'cuda':
            return
        try:
            c_out = self.label_manager.num_segmentation_heads
        except Exception:
            c_out = len(self.dataset_json.get('labels', {})) + 1
        logits_gb = c_out * float(np.prod(data.shape[1:])) * 2 / 1024 ** 3  # fp16
        try:
            free_gb, _ = torch.cuda.mem_get_info(self.device)
            free_gb /= 1024 ** 3
        except Exception:
            free_gb = 0.0
        # 动态显存分配(GPU 优先): 已占用(内存 reserved 已含 网络权重+CUDA 上下文)
        # 不是瓶颈, 决定是否上 GPU 的是 "fp16 累积 + 瞬态工作区" 是否装得下空闲显存。
        # 瞬态工作区 = 单 tile forward 激活峰值 + workon 输入 + 高斯/计数数组 ≈ 1.5GB。
        # 留 4% 安全头。更重要: 即使估错, 父类 predict_sliding_window 的 try/except
        # RuntimeError 会自动把整轮滑窗 OOM->CPU 重试, 因此极限下绝不爆显存。
        # （导出阶段已在 CPU 流式执行, logits 在滑窗结束即 .to('cpu'), 无 c10 崩溃面。）
        transient_gb = 1.5
        use_gpu = free_gb > 0 and (logits_gb + transient_gb) <= free_gb * (1.0 - 0.04)
        self.perform_everything_on_device = use_gpu
        if self.verbose_log:
            print(f'  [MemSafe] fp16 logits 预计 {logits_gb:.2f}GB / 显存空闲 {free_gb:.1f}GB -> '
                  f'滑窗累积于 {"显存(GPU优先)" if use_gpu else "内存(fp16,防爆显存)"}')

    # ------------------------------------------------------------------
    # 分块重采样：输出 z 区间 [dz0, dz1) 的 fp32 logits（三线性，全局坐标精确）
    # ------------------------------------------------------------------
    def _resample_z_block(self, logits: torch.Tensor, target_shape, dz0: int, dz1: int) -> torch.Tensor:
        Ci, Xi, Yi, Zi = logits.shape
        Xo, Yo, Zo = target_shape
        # z 输入窗口
        z_lo, z_hi, _ = _compute_input_window(dz0, dz1 - dz0, Zi / Zo, Zi)
        sub = logits[:, :, :, z_lo:z_hi].float()          # fp32 (C,Xi,Yi,win)
        # 1) z 插值
        sub = _linear_interp_window(sub, 3, Zi / Zo, dz0, dz1 - dz0, z_lo, z_hi - z_lo, Zi)
        # 2) y 插值(全轴窗口)
        sub = _linear_interp_window(sub, 2, Yi / Yo, 0, Yo, 0, Yi, Yi)
        # 3) x 插值(全轴窗口)
        sub = _linear_interp_window(sub, 1, Xi / Xo, 0, Xo, 0, Xi, Xi)
        return sub                                        # (C, Xo, Yo, dz1-dz0) fp32

    @staticmethod
    def _choose_z_block(target_shape, num_channels: int) -> int:
        elem_per_slice = max(1, num_channels * target_shape[0] * target_shape[1])
        zb = int(_BLOCK_ELEM_BUDGET // elem_per_slice)
        return max(1, min(zb, target_shape[2]))

    # ------------------------------------------------------------------
    # MemSafe 导出：分块 重采样->argmax->(softmax 概率分块落盘) -> 回填 -> 写 nii.gz
    # ------------------------------------------------------------------
    def _export_prediction_memsafe(self, predicted_logits, properties_dict: dict,
                                   output_file: str, save_probabilities: bool):
        t0 = time.time()
        ofile_truncated = str(output_file)[:-len(self.dataset_json['file_ending'])] \
            if str(output_file).endswith(self.dataset_json['file_ending']) else str(output_file)

        logits = predicted_logits
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().to('cpu')
        else:
            logits = torch.from_numpy(np.ascontiguousarray(logits))
        # 反转置回 xyz 轴序(与 target_shape / 回填 bbox 对齐)。permute 仅视图, 不复制;
        # 分块内 .float() 时才发生小规模复制, 全程无 8GB 级拷贝。
        transpose_backward = list(self.plans_manager.transpose_backward)
        if transpose_backward != [0, 1, 2]:
            logits = logits.permute(0, *[i + 1 for i in transpose_backward])
        # crop 内 xyz 轴序的目标形状(即概率重采样目标)
        target_shape = list(properties_dict['shape_after_cropping_and_before_resampling'])
        assert len(target_shape) == 3, target_shape

        C = logits.shape[0]
        z_block = self._choose_z_block(target_shape, C)
        seg_crop = np.empty(target_shape, dtype=np.uint8)

        # 概率分块缓存
        cache_meta = None
        block_files = []
        probs_dir = None
        if save_probabilities:
            probs_dir = self._probs_dir_for(ofile_truncated)
            probs_dir.mkdir(parents=True, exist_ok=True)
            cache_meta = {
                'format': 'toothseg-memsafe-probs-v1',
                'dtype': 'float16',
                'channels': int(C),
                'transpose_backward': transpose_backward,
                'crop_shape': target_shape,
                'crop_bbox': [list(map(int, b)) for b in properties_dict['bbox_used_for_cropping']],
                'full_shape': [int(v) for v in properties_dict['shape_before_cropping']],
                'blocks': [],
            }

        if self.verbose_log:
            print(f'  [MemSafe] 分块导出: 目标 {target_shape}, z 块大小 {z_block}, '
                  f'概率缓存={"fp16分块" if save_probabilities else "关闭"}')

        for dz0 in range(0, target_shape[2], z_block):
            dz1 = min(dz0 + z_block, target_shape[2])
            blk = self._resample_z_block(logits, target_shape, dz0, dz1)      # fp32 (C,Xo,Yo,nz)
            seg_crop[:, :, dz0:dz1] = blk.argmax(dim=0).to(torch.uint8).numpy()
            if save_probabilities:
                probs = torch.softmax(blk, dim=0).to(torch.float16).numpy()   # fp16 块
                f = probs_dir / f'block_{dz0:05d}.npy'
                np.save(f, probs)
                block_files.append({'file': f.name, 'z0': int(dz0), 'z1': int(dz1),
                                    'shape': list(probs.shape)})
                del probs
            del blk
            if self.verbose_log and target_shape[2] // z_block <= 12:
                print(f'  [MemSafe]   z 块 [{dz0}:{dz1}) 完成')

        del logits
        empty_cache(self.device)

        if save_probabilities:
            cache_meta['blocks'] = block_files
            with open(probs_dir / 'meta.json', 'w', encoding='utf-8') as f:
                json.dump(cache_meta, f, ensure_ascii=False, indent=1)

        # 回填 bbox -> 反转置 -> 写 nii.gz（uint8 小数组，内存无压力）
        seg_t = seg_crop.transpose(*transpose_backward)
        full = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
        full = insert_crop_into_image(full, np.ascontiguousarray(seg_t),
                                      properties_dict['bbox_used_for_cropping'])
        del seg_crop, seg_t

        rw = self.plans_manager.image_reader_writer_class()
        rw.write_seg(full, ofile_truncated + self.dataset_json['file_ending'], properties_dict)

        if self.verbose_log:
            print(f'  [MemSafe] 导出完成, 耗时 {time.time() - t0:.1f}s -> {ofile_truncated}'
                  f'{self.dataset_json["file_ending"]}')

    def _segmentation_from_logits_memsafe(self, predicted_logits, properties_dict: dict) -> np.ndarray:
        """ofile 为 None 时: 分块重采样并返回最终分割数组(原图轴序)。"""
        logits = predicted_logits
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().to('cpu')
        else:
            logits = torch.from_numpy(np.ascontiguousarray(logits))
        transpose_backward = list(self.plans_manager.transpose_backward)
        if transpose_backward != [0, 1, 2]:
            logits = logits.permute(0, *[i + 1 for i in transpose_backward])
        target_shape = list(properties_dict['shape_after_cropping_and_before_resampling'])
        z_block = self._choose_z_block(target_shape, logits.shape[0])
        seg_crop = np.empty(target_shape, dtype=np.uint8)
        for dz0 in range(0, target_shape[2], z_block):
            dz1 = min(dz0 + z_block, target_shape[2])
            blk = self._resample_z_block(logits, target_shape, dz0, dz1)
            seg_crop[:, :, dz0:dz1] = blk.argmax(dim=0).to(torch.uint8).numpy()
            del blk
        seg_t = seg_crop.transpose(*transpose_backward)
        full = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
        full = insert_crop_into_image(full, np.ascontiguousarray(seg_t),
                                      properties_dict['bbox_used_for_cropping'])
        return full

    def _probs_dir_for(self, ofile_truncated: str) -> Path:
        base = Path(ofile_truncated).name
        if self.probs_cache_dir is not None:
            return Path(self.probs_cache_dir) / (base + '_probs_cache')
        return Path(ofile_truncated).parent / (base + '_probs_cache')


def cleanup_probs_cache(folder) -> None:
    """删除概率分块缓存目录(assign_mincost 用完后可调用以释放磁盘)。"""
    folder = Path(folder)
    if folder.exists() and folder.name.endswith('_probs_cache'):
        shutil.rmtree(folder, ignore_errors=True)
        print(f'[MemSafe] 已清理概率缓存: {folder}')


def verify_interp_against_torch(max_err=2e-3, n_cases: int = 6, seed: int = 0) -> None:
    """数值自检：自实现的分块三线性插值 vs torch F.interpolate（全卷一次性）。
    用于保证“分块 == 全卷”，防止导出引入插值误差。"""
    rng = np.random.default_rng(seed)
    # 每项: (C, Xi, Yi, Zi, Xo, Yo, Zo) —— 放大/缩小/混合比例均有覆盖
    shapes = [(8, 20, 20, 26, 33, 24, 30), (4, 13, 33, 21, 7, 18, 26),
              (3, 40, 40, 40, 50, 51, 49), (2, 19, 31, 44, 5, 22, 33),
              (5, 64, 64, 64, 32, 48, 56), (6, 23, 17, 29, 4, 30, 21)]
    for i in range(min(n_cases, len(shapes))):
        C, Xi, Yi, Zi, Xo, Yo, Zo = shapes[i]
        x = torch.from_numpy(rng.standard_normal((C, Xi, Yi, Zi)).astype(np.float16))
        ref = torch.nn.functional.interpolate(
            x[None].float(), size=(Xo, Yo, Zo), mode='trilinear', align_corners=False)[0]
        # 分块复现
        zb = max(1, Zo // 3)
        outs = []
        for dz0 in range(0, Zo, zb):
            dz1 = min(dz0 + zb, Zo)
            z_lo, z_hi, _ = _compute_input_window(dz0, dz1 - dz0, Zi / Zo, Zi)
            sub = x[:, :, :, z_lo:z_hi].float()
            sub = _linear_interp_window(sub, 3, Zi / Zo, dz0, dz1 - dz0, z_lo, z_hi - z_lo, Zi)
            sub = _linear_interp_window(sub, 2, Yi / Yo, 0, Yo, 0, Yi, Yi)
            sub = _linear_interp_window(sub, 1, Xi / Xo, 0, Xo, 0, Xi, Xi)
            outs.append(sub)
        got = torch.cat(outs, dim=3)
        err = (got - ref).abs().max().item()
        status = 'OK' if err <= max_err else 'FAIL'
        print(f'  [{status}] case{i}: {(C,Xi,Yi,Zi)} -> {(Xo,Yo,Zo)}  max_err={err:.2e}')
        assert err <= max_err, '分块插值与 F.interpolate 不一致!'


if __name__ == '__main__':
    print('=== MemSafe 插值数值自检 ===')
    verify_interp_against_torch()
    print('=== 全部通过 ===')
