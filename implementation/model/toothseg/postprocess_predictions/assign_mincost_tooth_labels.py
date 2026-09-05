import argparse
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path

import nibabel
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm


def _load_cached_sem_probs(sem_dir: Path, name: str):
    """读取 MemSafe 推理产生的 fp16 概率分块缓存(MemSafePredictor 契约)。

    返回 (probs, crop_bbox, full_shape) 或 None(缓存不存在/格式不符)：
      probs     : fp16 (C, zc, yc, xc)，数组轴序(nnU-Net 模型轴序)，crop 空间
      crop_bbox : 数组序 [[z0,z1],[y0,y1],[x0,x1]]
      full_shape: 数组序 [Z, Y, X] 全图(裁剪前)形状
    """
    cache_dir = Path(sem_dir) / f'{name}_probs_cache'
    meta_file = cache_dir / 'meta.json'
    if not meta_file.exists():
        return None
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    if meta.get('format') != 'toothseg-memsafe-probs-v1':
        return None
    C = int(meta['channels'])
    zc, yc, xc = [int(v) for v in meta['crop_shape']]
    probs = np.empty((C, zc, yc, xc), dtype=np.float16)
    for b in meta['blocks']:
        blk = np.load(cache_dir / b['file'])
        probs[:, :, :, int(b['z0']):int(b['z1'])] = blk
    crop_bbox = [tuple(int(v) for v in bb) for bb in meta['crop_bbox']]
    full_shape = [int(v) for v in meta['full_shape']]
    return probs, crop_bbox, full_shape


def prepare_instances(
    sem_dir: Path,
    inst_file: Path,
    out_dir: Path,
    overwrite: bool,
):
    if not overwrite and (out_dir / (inst_file.name[:-7] + '.npz')).exists():
        out = np.load(out_dir / (inst_file.name[:-7] + '.npz'))

        return out['centroids'], out['probs'], out['seg']

    # load and orient instances
    inst_nii = nibabel.load(inst_file)
    inst_seg = np.asarray(inst_nii.dataobj)
    orientation = nibabel.io_orientation(inst_nii.affine)
    assert np.all(orientation[:, 0] == [0, 1, 2]), inst_file.name + str(orientation)
    spacing = np.array(inst_nii.header.get_zooms())

    # orient instance predictions
    inst_seg = nibabel.apply_orientation(inst_seg, orientation)
    instances, inverse = np.unique(inst_seg, return_inverse=True)
    inst_seg = inverse.reshape(inst_seg.shape).astype(inst_seg.dtype)

    # load and orient semantic predictions
    # 优先走 MemSafe 概率分块缓存(低内存, 只在 crop 区) → inst_seg 对齐并切到 crop, 不回填全图;
    # 缓存缺失/probs 覆盖不全时回退原版全图 fp32 npz 路径(兼容旧产物)。
    use_memsafe = False
    cached = _load_cached_sem_probs(sem_dir, inst_file.name[:-7])
    if cached is not None:
        probs_crop, crop_bbox, full_shape = cached
        # 文件序(与 dataobj 一致)的 crop 范围与全长: 数组序 [z,y,x] -> 文件序 [x,y,z]
        fcrop = [crop_bbox[2], crop_bbox[1], crop_bbox[0]]
        full_file = [full_shape[2], full_shape[1], full_shape[0]]
        orit_slices, orit_offsets = [], []
        for i in range(3):
            j = int(orientation[i, 0])
            flip = int(orientation[i, 1]) < 0
            lo, hi = fcrop[j]
            Nf = full_file[j]
            if not flip:
                orit_slices.append(slice(lo, hi))
                orit_offsets.append(lo)
            else:
                orit_slices.append(slice(Nf - hi, Nf - lo))
                orit_offsets.append(Nf - hi)
        # 保险检查: 实例分割的前景必须全部落在语义 crop 区域内
        if np.count_nonzero(inst_seg[tuple(orit_slices)]) == np.count_nonzero(inst_seg):
            sem_seg = nibabel.apply_orientation(probs_crop.transpose(0, 3, 2, 1), np.concatenate((
                np.array([[0, 1]]),
                np.column_stack((orientation[:, 0] + 1, orientation[:, 1])),
            )))
            sem_seg[sem_seg == 0] = 1e-6
            inst_crop = inst_seg[tuple(orit_slices)]       # oriented crop(与 sem_seg 空间对齐)
            use_memsafe = True
            print(f'  [MemSafe] 语义概率来自分块缓存(低内存): crop {tuple(sem_seg.shape[1:])}, '
                  f'轴序 {tuple(int(o) for o in orientation[:, 0])}')
        else:
            print('WARNING: 实例前景超出语义 crop 区域, 回退原版全图(npz)路径:', inst_file.name)
    if not use_memsafe:
        sem_seg = np.load(sem_dir / f'{inst_file.name[:-7]}.npz')['probabilities']
        sem_seg[sem_seg == 0] = 1e-6
        sem_seg = nibabel.apply_orientation(sem_seg.transpose(0, 3, 2, 1), np.concatenate((
            np.array([[0, 1]]),
            np.column_stack((orientation[:, 0] + 1, orientation[:, 1])),
        )))
        inst_crop = inst_seg

    # determine centroid and distribution of each instance
    inst_centroids = np.zeros((0, 3))
    inst_probs = np.zeros((0, 33))
    out_seg = np.zeros_like(inst_seg)
    for inst_idx in range(1, instances.shape[0]):
        inst_mask = inst_crop == inst_idx

        voxel_probs = sem_seg[:, inst_mask]
        class_idxs = voxel_probs.argmax(0)
        scores = np.zeros(33)
        for class_idx in np.nonzero(voxel_probs.mean(1) >= 0.1)[0]:
            if not np.any(class_idxs == class_idx):
                print('Empty slice:', inst_file.name, 'instance', inst_idx)
                continue
            score = voxel_probs[class_idx, class_idxs == class_idx].mean()
            scores[class_idx] = score

        if (scores[1:] >= 0.95).sum() <= 1:
            split_idxs = np.zeros(inst_mask.sum(), dtype=int)
        else:
            print('Splitting:', inst_file.name, 'instance', inst_idx)
            class_idxs = np.nonzero(scores[1:] >= 0.95)[0] + 1
            split_idxs = sem_seg[class_idxs][:, inst_mask].argmax(0)

        voxel_idxs = np.column_stack(np.nonzero(inst_mask))
        if use_memsafe:
            voxel_idxs = voxel_idxs + np.array(orit_offsets, dtype=np.int64)  # crop -> 全图 oriented 坐标
        for split_idx in np.unique(split_idxs):
            inst_centroid = voxel_idxs[split_idxs == split_idx].mean(0) * spacing
            inst_centroids = np.concatenate((inst_centroids, [inst_centroid]))

            prob_dist = voxel_probs[:, split_idxs == split_idx].mean(1)
            inst_probs = np.concatenate((inst_probs, [prob_dist]))

            out_seg[tuple(voxel_idxs[split_idxs == split_idx].T)] = out_seg.max() + 1

    out_seg = nibabel.apply_orientation(
        out_seg, nibabel.io_orientation(np.linalg.inv(inst_nii.affine)),
    )

    # save intermediate result to storage as cache
    np.savez(
        out_dir / (inst_file.name[:-7] + '.npz'),
        centroids=inst_centroids,
        probs=inst_probs,
        seg=out_seg,
    )

    return inst_centroids, inst_probs, out_seg


def determine_sequence(centroids):
    idxs, inverse = np.full((2, centroids.shape[0]), -1)

    first_idx = centroids[:, 1].argmin()
    idxs[0] = first_idx
    inverse[first_idx] = 0
    for i in range(1, centroids.shape[0]):
        dists = np.linalg.norm(centroids[inverse == -1] - centroids[idxs[i - 1]], axis=-1)
        next_idx = np.nonzero(inverse == -1)[0][dists.argmin()]
        idxs[i] = next_idx
        inverse[next_idx] = i

    return idxs, inverse

    
def determine_transition_probabilities(
    normals,
    centroids,
    is_arch_lower,
    seq_idxs,
):
    index = np.arange(16, 32) if is_arch_lower else np.arange(16)

    trans_log_probs = np.zeros((centroids.shape[0] - 1, 16, 16))
    for i, (idx1, idx2) in enumerate(zip(seq_idxs[:-1], seq_idxs[1:])):
        offsets = centroids[idx2] - centroids[idx1]
        for j in range(16):
            for k in range(16):
                normal = normals[index[j]][index[k]]
                trans_log_probs[i, j, k] = normal.logpdf(offsets[:2])

    return trans_log_probs

    
def dynamic_programming(
    tooth_probs,
    seq_idxs,
    trans_log_probs,
    tooth_factor: float=4.0,
):
    tooth_log_probs = np.log(tooth_probs)

    q = np.zeros_like(tooth_log_probs)
    q[0] = -tooth_factor * tooth_log_probs[seq_idxs[0]]

    up = np.arange(16)
    p = np.zeros_like(q, dtype=int)
    p[0] = up

    for i in range(1, tooth_probs.shape[0]):
        for j in range(16):
            prev_costs = q[i - 1]
            trans_costs = -trans_log_probs[i - 1, :, j].copy()

            costs = prev_costs + trans_costs
            m = costs.min()
            q[i, j] = m - tooth_factor * tooth_log_probs[seq_idxs[i], j]
            p[i, j] = costs.argmin()

    path = q[-1].argmin(keepdims=True)
    for i in range(tooth_probs.shape[0] - 1):
        path = np.concatenate(([p[-1 - i, path[0]]], path))

    return path, q[-1].min()


def process_case(sem_dir: Path, out_dir: Path, normals, overwrite: bool, inst_file: Path):
    inst_centroids, inst_probs, inst_seg = prepare_instances(sem_dir, inst_file, out_dir, overwrite)

    # remove background instances
    is_background = inst_probs[:, 0] >= 0.95
    if np.any(is_background):
        bg_idxs = np.nonzero(is_background)[0]
        print('Ignoring:', inst_file.name, 'instances', 1 + bg_idxs, 'probs', inst_probs[bg_idxs, 0])
    inst_centroids = inst_centroids[~is_background]
    inst_probs = inst_probs[~is_background]
    
    # process each arch independently
    is_inst_lower = inst_probs[:, 17:].sum(-1) > inst_probs[:, 1:17].sum(-1)
    inst_fdis = np.zeros(inst_centroids.shape[0], dtype=int)
    for is_arch_lower in [False, True]:
        if not np.any(is_arch_lower == is_inst_lower):
            continue

        arch_idxs = np.nonzero(is_arch_lower == is_inst_lower)[0]
        arch_centroids = inst_centroids[arch_idxs]
        arch_probs = inst_probs[arch_idxs]
        arch_probs = arch_probs[:, 17:] if is_arch_lower else arch_probs[:, 1:17]
        arch_probs /= arch_probs.sum(axis=1, keepdims=True)

        seq_idxs, seq_inverse = determine_sequence(arch_centroids)
        trans_probs = determine_transition_probabilities(
            normals, arch_centroids, is_arch_lower, seq_idxs,
        )
        path, cost = dynamic_programming(arch_probs, seq_idxs, trans_probs)

        argmax_fdis = np.argmax(arch_probs[seq_idxs], axis=-1) + 16 * is_arch_lower + 1
        path_fdis = path + 16 * is_arch_lower + 1
        if not np.all(argmax_fdis == path_fdis):
            print('Updating:', inst_file.name, 'from', argmax_fdis, 'to', path_fdis)

        inst_fdis[arch_idxs[seq_idxs]] = path_fdis

    # incorporate background voxels and instances
    inst_map = np.zeros(is_background.shape[0] + 1)
    inst_map[np.nonzero(~is_background)[0] + 1] = inst_fdis

    # save result to storage
    inst_nii = nibabel.load(inst_file)
    fdi_seg = inst_map[inst_seg].astype(np.uint8)
    fdi_nii = nibabel.Nifti1Image(fdi_seg, inst_nii.affine, dtype=np.uint8)
    nibabel.save(fdi_nii, out_dir / inst_file.name)
    
    return inst_file.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser("This script takes a folder with instance segmentations and a folder wtih "
                                     "semantic segmentations and assigns the tooth labels as predicted by the "
                                     "semantic segmentation model to the instances.")
    parser.add_argument('-ifolder', type=str, required=True,
                        help='Input folder. Must contain files with instance predictions')
    parser.add_argument('-sfolder', type=str, required=True,
                        help='Input folder. Must contain files with semantic segmentations')
    parser.add_argument('-o', type=str, required=True,
                        help="Output folder. Must be empty! If it doesn't exist it will be created")
    parser.add_argument('--distributions', type=str, required=False,
                        default='toothseg/datasets/toothfairy2/fdi_pair_distrs.json',
                        help='JSON file containing the means and covariancematrices of tooth pair offsets.')
    parser.add_argument('--overwrite', action='store_true',
                        help='By default the script will skip existing results. Set this flag to overwrite (recompute) '
                             'them instead.')
    parser.add_argument('-np', type=int, required=False, default=8,
                        help='Number of processes used for multiprocessing. Default: 8. Make this a lot higher if you '
                             'can!')
    args = parser.parse_args()

    # load distributions of tooth pair distances
    print('Using distribution file:', args.distributions)
    with open(args.distributions, 'r') as f:
        pair_dists = json.load(f)

    normals = []
    for i in range(32):
        normals.append([])
        for j in range(32):
            if i // 16 != j // 16:
                normals[-1].append(None)
                continue

            normal = multivariate_normal(
                mean=pair_dists['means'][i][j][:2],
                cov=np.array(pair_dists['covs'][i][j])[:2, :2],
            )
            normals[-1].append(normal)
    
    # run post-processing for each case
    Path(args.o).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.ifolder).glob('*.nii.gz'))
    # files = [f for f in files if '024' in f.name]
    with mp.Pool(args.np) as p:
        f = partial(process_case, Path(args.sfolder), Path(args.o), normals, args.overwrite)
        t = tqdm(p.imap_unordered(f, files), total=len(files))
        for file in t:
            t.set_description(file)
