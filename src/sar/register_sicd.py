"""
Rigorous registration of the PFA reconstruction to Umbra's SICD, using
normalised cross-correlation (skimage.match_template, returns NCC in [-1,1]).

Both are RGAZIM slant-plane images.  We resample both to a common pixel size,
cut a distinctive template chip from the SICD (the pit), then test all 8
dihedral orientations of my image and a small rotation refinement to find where
the template best matches.  A high NCC at a consistent orientation proves the
reconstruction reproduces Umbra's own image.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom, rotate
from skimage.feature import match_template
from sarpy.io.complex.sicd import SICDReader
from viz import sar_stretch


def feat(a):
    a = a.astype(np.float32)
    return np.log1p(a / (a.mean() + 1e-9))


ORIENTS = {
    "identity":  lambda x: x,
    "flipud":    lambda x: x[::-1],
    "fliplr":    lambda x: x[:, ::-1],
    "rot180":    lambda x: x[::-1, ::-1],
    "T":         lambda x: x.T,
    "T+flipud":  lambda x: x.T[::-1],
    "T+fliplr":  lambda x: x.T[:, ::-1],
    "T+rot180":  lambda x: x.T[::-1, ::-1],
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--complex', default='out/full_complex.npy')
    ap.add_argument('--geom', default='out/full_geom.npz')
    ap.add_argument('--sicd', default='data/sar/SICD.nitf')
    ap.add_argument('--out', default='out/register_sicd.png')
    args = ap.parse_args()
    COMMON = 1.5  # metres/pixel working grid

    mine_full = np.abs(np.asarray(np.load(args.complex, mmap_mode='r')))
    geom = np.load(args.geom)
    my_px = (float(geom['dy_cross']), float(geom['dx_range']))  # (axis0=azimuth, axis1=range)
    mine = zoom(mine_full, (my_px[0] / COMMON, my_px[1] / COMMON), order=1)

    sr = SICDReader(args.sicd)
    s = sr.sicd_meta
    sic = np.abs(sr[:, :]).astype(np.float32)          # [range, az]
    sic_px = (s.Grid.Row.SS, s.Grid.Col.SS)
    sic = zoom(sic, (sic_px[0] / COMMON, sic_px[1] / COMMON), order=1)
    print(f"@ {COMMON} m/px : mine {mine.shape}  SICD {sic.shape}")

    # template from SICD centre (the pit), sized well inside my image
    T = 260
    r0, c0 = sic.shape[0] // 2, sic.shape[1] // 2
    tmpl0 = feat(sic[r0 - T:r0 + T, c0 - T:c0 + T])

    best = (-1, None, 0.0, None, None)
    for name, fn in ORIENTS.items():
        img = feat(fn(mine))
        if img.shape[0] <= tmpl0.shape[0] or img.shape[1] <= tmpl0.shape[1]:
            continue
        res = match_template(img, tmpl0)
        pk = res.max()
        idx = np.unravel_index(np.argmax(res), res.shape)
        if pk > best[0]:
            best = (pk, name, 0.0, idx, img)
    pk0, name, _, idx, img = best
    print(f"coarse best orientation = {name}   NCC = {pk0:.3f}   at {idx}")

    # small rotation refinement around the winning orientation
    best_ang, best_pk, best_idx = 0.0, pk0, idx
    for ang in np.arange(-4, 4.01, 0.5):
        t = rotate(tmpl0, ang, order=1, reshape=False)
        res = match_template(img, t)
        if res.max() > best_pk:
            best_pk = res.max(); best_ang = ang
            best_idx = np.unravel_index(np.argmax(res), res.shape)
    print(f"refined: orientation={name}  rotation={best_ang:+.1f} deg  NCC={best_pk:.3f}")

    # overlay checkerboard where the template landed
    t = rotate(tmpl0, best_ang, order=1, reshape=False)
    oy, ox = best_idx
    chip = img[oy:oy + t.shape[0], ox:ox + t.shape[1]]
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tile = 40
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    ck = ((yy // tile + xx // tile) % 2).astype(bool)
    board = np.where(ck, sar_stretch(np.expm1(chip)), sar_stretch(np.expm1(t)))
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    axs[0].imshow(sar_stretch(np.expm1(chip)), cmap='gray'); axs[0].set_title(f'my PFA  [{name}]')
    axs[1].imshow(sar_stretch(np.expm1(t)), cmap='gray'); axs[1].set_title('Umbra SICD template')
    axs[2].imshow(board, cmap='gray'); axs[2].set_title(f'checkerboard  NCC={best_pk:.3f}')
    for ax in axs: ax.axis('off')
    fig.suptitle('CPHD → PFA reconstruction vs Umbra SICD (same collect, no autofocus)')
    fig.tight_layout(); fig.savefig(args.out, dpi=140); plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
