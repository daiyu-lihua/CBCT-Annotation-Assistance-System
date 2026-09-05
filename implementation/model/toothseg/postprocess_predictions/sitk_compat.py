# -*- coding: utf-8 -*-
"""
SimpleITK NIfTI I/O 兼容补丁。

当前环境中 SimpleITK 的底层 nifti C 引擎无法读写文件(内存操作正常),
而 ToothSeg 的后处理脚本依赖 sitk.ReadImage / sitk.WriteImage。
本模块用 numpy + nibabel 实现这两个函数的等价格式,允许其它脚本无缝替换。
"""
import numpy as np
import SimpleITK as sitk

import nibabel as nib


def sitk_read_nibabel(filename):
    """等价于 sitk.ReadImage(filename), 返回一个带几何信息的 SimpleITK Image。
    通过先将 nibabel 数据灌入内存的 sitk Image 实现(内存操作正常)。"""
    n = nib.load(filename)
    arr = np.asanyarray(n.dataobj)
    itk = sitk.GetImageFromArray(arr)
    # 设定点阵几何
    spacing = tuple(float(x) for x in n.header.get_zooms()[:3])
    origin = tuple(float(x) for x in n.affine[:3, 3])
    itk.SetSpacing(spacing)
    itk.SetOrigin(origin)
    # 方向:仿射矩阵左上 3x3 归一到坐标轴(axis aligned 多数情况对角/置换)
    R = n.affine[:3, :3]
    # 通过方向余弦构造 SimpleITK 可用的 9 元素(行主序 xyz)
    dir9 = np.zeros(9)
    rot = R / np.linalg.norm(R, axis=0)  # 单位向量,每列一个轴
    dir9[0:3] = rot[:, 0]
    dir9[3:6] = rot[:, 1]
    dir9[6:9] = rot[:, 2]
    try:
        itk.SetDirection(tuple(float(x) for x in dir9))
    except Exception:
        pass
    return itk


def sitk_write_nibabel(image, filename):
    """等价于 sitk.WriteImage(image, filename)。导出为 NIfTI(nibabel)。"""
    arr = sitk.GetArrayFromImage(image)
    ndim = arr.ndim
    spacing = np.array(image.GetSpacing())[::-1][:ndim] if ndim <= 3 else np.array(image.GetSpacing())
    # 计算仿射
    affine = np.diag(list(spacing) + [1.0]) if ndim == 3 else np.diag(list(spacing[:ndim]) + [1.0])
    # 方向
    try:
        d = np.array(image.GetDirection())
        if d.size == 9:
            rot = d.reshape(3, 3)
            M = rot * spacing * np.sign(np.diag(np.diag(rot)))
            affine[:3, :3] = rot @ np.diag(spacing)
    except Exception:
        pass
    affine[:3, 3] = np.array(image.GetOrigin())[:3] if ndim == 3 else np.array(image.GetOrigin())[:ndim]
    out = nib.Nifti1Image(arr, affine)
    if filename.endswith('.nii.gz'):
        nib.save(out, filename)
    else:
        nib.save(out, filename)
    return True


def patch_sitk_io():
    """将 sitk.ReadImage / sitk.WriteImage 替换为 nibabel 实现,仅在 .nii/.nii.gz 上生效。
    其它格式仍走 SimpleITK 原生(若它可用)。"""
    _orig_read = sitk.ReadImage
    _orig_write = sitk.WriteImage

    def read_new(filename, *a, **k):
        if isinstance(filename, str) and (filename.endswith('.nii.gz') or filename.endswith('.nii')):
            return sitk_read_nibabel(filename)
        return _orig_read(filename, *a, **k)

    def write_new(image, filename, *a, **k):
        if isinstance(filename, str) and (filename.endswith('.nii.gz') or filename.endswith('.nii')):
            return sitk_write_nibabel(image, filename)
        return _orig_write(image, filename, *a, **k)

    sitk.ReadImage = read_new
    sitk.WriteImage = write_new


if __name__ == '__main__':
    import SimpleITK as sitk
    print("sitk native read/write test:")
    try:
        import tempfile, os
        d = tempfile.mkdtemp()
        p = os.path.join(d, 't.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(np.random.rand(8, 8, 8).astype('float32')), p)
        print("   native write OK")
    except Exception as e:
        print("   native write FAILED:", str(e)[-60:])

    patch_sitk_io()
    print("patched write/read roundtrip:")
    arr = np.random.rand(8, 8, 8).astype('float32')
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing([0.3, 0.3, 0.3])
    itk.SetOrigin([10, 20, 30])
    p = r'E:\trae_file\大创\ToothSegWork\nnUNet_raw\inference_ts\imagesTs\_patch_test.nii.gz'
    sitk.WriteImage(itk, p)
    back = sitk.ReadImage(p)
    print("   shape", sitk.GetArrayFromImage(back).shape, "spacing", back.GetSpacing(), "origin", back.GetOrigin())
    print("PAT_OK")