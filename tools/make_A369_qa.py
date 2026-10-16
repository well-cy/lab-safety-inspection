# -*- coding: utf-8 -*-
"""渲染 A档369 某批「最终标签」的 QA 目检拼图（12 张/张，4×3）。

用法：
  python tools/make_A369_qa.py --batch m1                 # 只渲染导出改动过的图
  python tools/make_A369_qa.py --batch m1 --all            # 渲染整批
"""
import argparse, collections, glob, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = 'outputs/_A369'
SRC = f'{ROOT}/A档_6类预标注包'
CLSN = {0: 'person', 1: 'mask', 2: 'gloves', 3: 'lab_coat', 4: 'goggles', 5: 'coverall'}
CLSC = {0: (255, 140, 0), 1: (0, 220, 220), 2: (255, 0, 200), 3: (0, 190, 0), 4: (255, 80, 0), 5: (160, 0, 255)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def load(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            pp = l.split()
            if len(pp) == 5:
                out.append([int(float(pp[0]))] + [float(x) for x in pp[1:]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--export', default=None, help='默认 outputs/_A369/exports/<batch>_export.txt')
    a = ap.parse_args()

    exp = a.export or f'{ROOT}/exports/{a.batch}_export.txt'
    staged = f'{ROOT}/staged/{a.batch}'
    ws = f'{ROOT}/editor/ws/ws_a369_{a.batch}/labels'
    idx = f'{ROOT}/editor/a369_{a.batch}/index.txt'

    dirty = set()
    if not a.all and os.path.exists(exp):
        for l in open(exp, encoding='utf-8'):
            if l.startswith('### '):
                dirty.add(l[4:].strip())

    rows = [l.rstrip('\n').split('\t') for l in open(idx, encoding='utf-8')
            if not l.startswith('#') and l.strip()]
    rows = [r for r in rows if len(r) >= 6]
    if dirty:
        rows = [r for r in rows if r[1].rsplit('.', 1)[0] in dirty]

    imgs_dir = f'{SRC}/images'
    name_map = {os.path.splitext(f)[0]: f for f in os.listdir(imgs_dir)}
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    except Exception:
        font = ImageFont.load_default()

    tiles = []
    for r in rows:
        fn = r[1]
        stem = fn.rsplit('.', 1)[0]
        real = name_map.get(stem)
        if not real:
            print('缺图:', stem)
            continue
        im = imread_u(os.path.join(imgs_dir, real))
        H, W = im.shape[:2]
        boxes = load(os.path.join(staged, stem + '.txt'))
        old = load(os.path.join(ws, stem + '.txt'))
        lab = im.copy()
        for b in boxes:
            x1, y1, x2, y2 = int((b[1] - b[3] / 2) * W), int((b[2] - b[4] / 2) * H), \
                             int((b[1] + b[3] / 2) * W), int((b[2] + b[4] / 2) * H)
            col = CLSC.get(b[0], (255, 255, 255))
            cv2.rectangle(lab, (x1, y1), (x2, y2), col, 2)
            cv2.putText(lab, CLSN.get(b[0], str(b[0])), (x1, max(13, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        tiles.append((stem, Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)), len(old), len(boxes)))

    CW, CH, TOP = 470, 400, 38
    per, cols = 12, 4
    nsh = max(1, (len(tiles) + per - 1) // per)
    sheets = [Image.new('RGB', (CW * cols, (CH + TOP) * 3), 'white') for _ in range(nsh)]
    for i, (stem, pil, n_old, n_new) in enumerate(tiles):
        pil.thumbnail((CW - 8, CH - 8))
        sh = sheets[i // per]
        rr, cc = divmod(i % per, cols)
        x0, y0 = cc * CW, rr * (CH + TOP)
        sh.paste(pil, (x0 + (CW - pil.width) // 2, y0 + TOP))
        d = ImageDraw.Draw(sh)
        d.text((x0 + 5, y0 + 3), f"{i+1}. {stem.replace('SafetyLab__','S__')[:30]}  {n_old}→{n_new}框",
               fill=(0, 0, 0), font=font)
        d.text((x0 + 5, y0 + 20), "橙person 青mask 洋红gloves 绿lab_coat 蓝goggles", fill=(120, 120, 120), font=font)
    os.makedirs(staged, exist_ok=True)
    for i, sh in enumerate(sheets):
        p = f'{staged}/QA_{i+1:02d}.png'
        sh.save(p)
        print('saved', p, sh.size)
    print(f'共 {len(tiles)} 张')


if __name__ == '__main__':
    main()
