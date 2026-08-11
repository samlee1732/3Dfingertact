from datetime import datetime
import json
from pathlib import Path

import cv2
import matplotlib
# 使用无窗口绘图后端，保证从 VS Code 或终端运行都能直接保存图片。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import yaml


# 使用其他图片时只需修改下面三个路径；输入必须是同一传感器校正、裁剪后的图片。
ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / 'examples/figure3_logo/reference.png'
INPUT_PATH = ROOT / 'examples/figure3_logo/input.png'
OUTPUT_PATH = ROOT.parent / 'results'


def reconstruct(ref, img, Pixel_to_Depth, pixel_per_mm, cfg):
    """按原项目算法返回从灰度图到三维点云的全部中间结果。"""
    ref_GRAY = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lighting_threshold = cfg['lighting_threshold']

    # 保留原算法的 uint8 减法：负差值会回绕为高值，再由阈值掩膜抑制。
    diff_raw = ref_GRAY - img_GRAY - lighting_threshold
    diff_mask = (diff_raw < 100).astype(np.uint8)
    diff_masked = diff_raw * diff_mask + lighting_threshold

    # 将差分限制在标定表范围内，平滑后取整数作为查表索引。
    diff_clipped = diff_masked.copy()
    max_index = len(Pixel_to_Depth) - 1
    diff_clipped[diff_clipped > max_index] = max_index
    diff_index = cv2.GaussianBlur(
        diff_clipped.astype(np.float32), (7, 7), 0).astype(int)

    # Pixel_to_Depth 把灰度差索引转换为标定高度，并减去无接触基准高度。
    height_map_lut = Pixel_to_Depth[diff_index] - \
        Pixel_to_Depth[lighting_threshold]

    # 两次高斯平滑与原 sensor.py 的 kernel_list 循环等价。
    kernel_1, kernel_2 = cfg['kernel_list']
    height_map_blur_1 = cv2.GaussianBlur(
        height_map_lut.astype(np.float32), (kernel_1, kernel_1), 0)
    height_map = cv2.GaussianBlur(
        height_map_blur_1.astype(np.float32), (kernel_2, kernel_2), 0)

    # depth_map 是便于查看的灰度图；几何高度仍保存在 height_map 中。
    contact_mask = (height_map > 0).astype(np.uint8)
    depth_map = (height_map * cfg['depth_k'] +
                 contact_mask * cfg['contact_gray_base']).astype(np.uint8)

    # 扩展到约 28 mm x 21 mm 的传感器坐标范围，空白区域高度为 0。
    expand_x = int(28.0 / pixel_per_mm) + 2
    expand_y = int(21.0 / pixel_per_mm) + 2
    height_map_expand = np.zeros([expand_y, expand_x])
    height, width = height_map.shape
    begin_y = int((expand_y - height) / 2)
    begin_x = int((expand_x - width) / 2)
    height_map_expand[begin_y:begin_y + height,
                      begin_x:begin_x + width] = height_map

    # 点云每行是 [X, Y, Z]；Y 轴方向与原项目保持一致，Z 为标定高度。
    X, Y = np.meshgrid(np.arange(expand_x), np.arange(expand_y))
    points = np.column_stack((X.ravel() * pixel_per_mm,
                              -Y.ravel() * pixel_per_mm,
                              height_map_expand.ravel()))

    return {
        'ref_GRAY': ref_GRAY,
        'img_GRAY': img_GRAY,
        'diff_raw': diff_raw,
        'diff_mask': diff_mask,
        'diff_masked': diff_masked,
        'diff_clipped': diff_clipped,
        'diff_index': diff_index,
        'height_map_lut': height_map_lut,
        'height_map_blur_1': height_map_blur_1,
        'height_map': height_map,
        'contact_mask': contact_mask,
        'depth_map': depth_map,
        'height_map_expand': height_map_expand,
        'points': points,
    }


def save_results(ref, img, result, Pixel_to_Depth):
    """把本次运行的逐步图片、统计和精确数组保存到独立时间戳目录。"""
    output = OUTPUT_PATH / datetime.now().strftime('%y%m%d-%H%M%S')
    output.mkdir(parents=True)

    cv2.imwrite(str(output / '00-reference.png'), ref)
    cv2.imwrite(str(output / '01-input.png'), img)

    max_index = len(Pixel_to_Depth) - 1
    max_height = max(result['height_map_lut'].max(), result['height_map'].max())
    # 文件名前缀就是处理顺序；同类高度图共用色标，便于比较平滑影响。
    images = [
        ('02-reference-gray.png', 'ref_GRAY', 'gray', 0, 255),
        ('03-input-gray.png', 'img_GRAY', 'gray', 0, 255),
        ('04-uint8-difference.png', 'diff_raw', 'magma', 0, 255),
        ('05-threshold-mask.png', 'diff_mask', 'gray', 0, 1),
        ('06-masked-difference.png', 'diff_masked', 'magma', 0, None),
        ('07-clipped-index.png', 'diff_clipped', 'magma', 0, max_index),
        ('08-smoothed-index.png', 'diff_index', 'magma', 0, max_index),
        ('09-pixel-to-depth.png', 'height_map_lut', 'viridis', 0, max_height),
        ('10-height-smoothing-1.png', 'height_map_blur_1', 'viridis', 0, max_height),
        ('11-height-smoothing-2.png', 'height_map', 'viridis', 0, max_height),
        ('12-contact-mask.png', 'contact_mask', 'gray', 0, 1),
        ('13-depth-display.png', 'depth_map', 'gray', 0, 255),
        ('14-expanded-canvas.png', 'height_map_expand', 'viridis', 0, max_height),
    ]
    for filename, key, cmap, vmin, vmax in images:
        plt.imsave(output / filename, result[key], cmap=cmap,
                   vmin=vmin, vmax=vmax)

    # 保留原来的彩色散点图，0.08 与官方 Visualizer 的显示阈值一致。
    points = result['points']
    contact_points = points[points[:, 2] > 0.08]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    graph = ax.scatter(contact_points[:, 0], contact_points[:, 1],
                       contact_points[:, 2], c=contact_points[:, 2],
                       cmap='viridis', s=1)
    ax.set_xlabel('X / mm')
    ax.set_ylabel('Y / mm')
    ax.set_zlabel('Height')
    fig.colorbar(graph, ax=ax, shrink=0.7, label='Height')
    fig.savefig(output / '15-3d-point-cloud.png', dpi=160,
                bbox_inches='tight')
    plt.close(fig)

    # 论文风格灰色高度表面；阈值只清理预览噪声，不修改点云数据。
    surface_height = np.where(
        result['height_map_expand'] > 0.08,
        result['height_map_expand'], 0)
    surface_shape = surface_height.shape
    X = points[:, 0].reshape(surface_shape)
    Y = points[:, 1].reshape(surface_shape)
    light = LightSource(azdeg=315, altdeg=40)
    illumination = light.hillshade(
        surface_height, vert_exag=5,
        dx=abs(X[0, 1] - X[0, 0]),
        dy=abs(Y[1, 0] - Y[0, 0]), fraction=0.7)
    facecolors = plt.cm.gray(0.35 + 0.55 * illumination)

    fig = plt.figure(figsize=(8, 5), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, surface_height, facecolors=facecolors,
                    rstride=2, cstride=2, linewidth=0,
                    antialiased=False, shade=False)
    max_surface_height = max(float(surface_height.max()), 0.1)
    ax.view_init(elev=58, azim=-88)
    ax.set_proj_type('persp', focal_length=0.9)
    ax.set_box_aspect((X.max() - X.min(), Y.max() - Y.min(),
                       max_surface_height * 3))
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(0, max_surface_height * 1.2)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output / '16-3d-point-cloud-paper-style.png', dpi=160,
                bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)

    # JSON 记录各阶段的数值变化，NPZ 保留未归一化的原始数组。
    metrics = {
        name: {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'min': float(value.min()),
            'max': float(value.max()),
            'mean': float(value.mean()),
            'std': float(value.std()),
        }
        for name, value in result.items()
    }
    with (output / '17-stage-metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(output / '18-reconstruction.npz', **result)
    print('Results saved to:', output)


if __name__ == '__main__':
    with (ROOT / 'shape_config.yaml').open(encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    sensor_dir = ROOT / 'calibration' / ('sensor_' + str(cfg['sensor_id']))
    camera_calibration = sensor_dir / 'camera_calibration'
    depth_calibration = sensor_dir / 'depth_calibration'

    ref = cv2.imread(str(REFERENCE_PATH))
    img = cv2.imread(str(INPUT_PATH))
    if ref is None or img is None:
        raise FileNotFoundError('Reference or input image cannot be read.')

    # 标定表负责灰度到高度映射，position_scale[2] 是像素坐标缩放值。
    Pixel_to_Depth = np.load(depth_calibration / 'Pixel_to_Depth.npy')
    pixel_per_mm = np.load(camera_calibration / 'position_scale.npy')[2]
    result = reconstruct(ref, img, Pixel_to_Depth, pixel_per_mm,
                         cfg['sensor_reconstruction'])
    save_results(ref, img, result, Pixel_to_Depth)
