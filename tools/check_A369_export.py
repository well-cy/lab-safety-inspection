# -*- coding: utf-8 -*-
"""核对 A档369 审核编辑器导出的改动，并落暂存标签。

用法：
  python tools/check_A369_export.py --batch m1 --export outputs/_A369/exports/m1_export.txt
  python tools/check_A369_export.py --batch u8 --export <file> --stage

说明（与 p0 批的 apply_person_review 不同）：
  - A369 用的是 make_box_editor 的**全类别**编辑器，导出只含「改动过」的图（dirty）。
  - 未出现在导出里的图 → 沿用工作区初始标签（= 原标注 + 模型候选），视为通过。
  - 导出行格式：`### <无扩展名文件名>` + 可选 `!REJECT [原因]` + `类别 cx cy w h`。
  - 暂存目录 outputs/_A369/staged/<batch>/，不写 data/。
"""
import argparse, collections, os, sys

CLSN = {0: 'person', 1: 'mask', 2: 'gloves', 3: 'lab_coat', 4: 'goggles', 5: 'coverall'}
ROOT = 'outputs/_A369'


def read_index(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        p = line.rstrip('\n').split('\t')
        if len(p) >= 6:
            rows.append(p)
    return rows


def load_labels(path):
    out = []
    if os.path.exists(path):
        for l in open(path):
            pp = l.split()
            if len(pp) == 5:
                out.append([int(float(pp[0]))] + [float(x) for x in pp[1:]])
    return out


def parse_export(path):
    blocks, cur, rej = collections.OrderedDict(), None, []
    for raw in open(path, encoding='utf-8'):
        s = raw.rstrip('\n')
        if s.startswith('### '):
            cur = s[4:].strip()
            blocks[cur] = []
        elif s.startswith('!REJECT'):
            rej.append((cur, s[7:].strip()))
        elif s.strip():
            p = s.split()
            if len(p) == 5:
                if cur is None:
                    sys.exit('导出格式异常：出现无归属的标签行')
                blocks[cur].append([int(float(p[0]))] + [float(x) for x in p[1:]])
    return blocks, rej


def key(b):
    return (b[0], round(b[1], 3), round(b[2], 3), round(b[3], 3), round(b[4], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, choices=['u8', 'c35', 'm1', 'm2'])
    ap.add_argument('--export', required=True)
    ap.add_argument('--stage', action='store_true', help='把结果写入 outputs/_A369/staged/<batch>/')
    a = ap.parse_args()

    idx = f'{ROOT}/editor/a369_{a.batch}/index.txt'
    ws = f'{ROOT}/editor/ws/ws_a369_{a.batch}/labels'
    rows = read_index(idx)
    valid = {r[1].rsplit('.', 1)[0] for r in rows}
    blocks, rej = parse_export(a.export)

    print(f'批 {a.batch}: 共 {len(rows)} 张 | 导出改动 {len(blocks)} 张 | 剔除 {len(rej)} 张')
    bad = [k for k in blocks if k not in valid]
    if bad:
        sys.exit(f'❌ 导出含批外文件（疑似错贴）：{bad}')
    miss = [k for k in blocks if not os.path.exists(os.path.join(ws, k + '.txt'))]
    if miss:
        sys.exit(f'❌ 工作区缺初始标签：{miss}')
    print('文件名合法性 ✅')

    total_add = total_del = 0
    for name in blocks:
        old = load_labels(os.path.join(ws, name + '.txt'))
        new = blocks[name]
        oc, nc = collections.Counter(map(key, old)), collections.Counter(map(key, new))
        add, rem = list((nc - oc).elements()), list((oc - nc).elements())
        total_add += len(add)
        total_del += len(rem)
        if add or rem:
            print(f'\n{name[:56]}  {len(old)}→{len(new)} 框')
            for x in add:
                print(f'   + {CLSN.get(x[0], x[0]):9s} c=({x[1]:.3f},{x[2]:.3f}) wh=({x[3]:.3f},{x[4]:.3f})')
            for x in rem:
                print(f'   - {CLSN.get(x[0], x[0]):9s} c=({x[1]:.3f},{x[2]:.3f}) wh=({x[3]:.3f},{x[4]:.3f})')
    print(f'\n合计 新增 {total_add} 框 / 删除 {total_del} 框')

    # 迷你废框检查（skill 流程：导出里 w/h<=0.03 的行需目检）
    tiny = [(n, b) for n, bs in blocks.items() for b in bs if b[3] <= 0.03 or b[4] <= 0.03]
    if tiny:
        print(f'\n⚠️ 导出含 {len(tiny)} 个极小框（w 或 h <= 0.03），需目检是否误拖：')
        for n, b in tiny[:15]:
            print(f'   {n[:44]} {CLSN.get(b[0], b[0])} c=({b[1]:.3f},{b[2]:.3f}) wh=({b[3]:.3f},{b[4]:.3f})')
    else:
        print('无 w/h<=0.03 的新拖迷你框 ✅')

    if a.stage:
        out = f'{ROOT}/staged/{a.batch}'
        os.makedirs(out, exist_ok=True)
        n_keep = 0
        for r in rows:
            stem = r[1].rsplit('.', 1)[0]
            if stem in blocks:
                lines = [' '.join([str(b[0])] + [f'{v:.6f}' for v in b[1:]]) for b in blocks[stem]]
            else:
                lines = [l.rstrip('\n') for l in open(os.path.join(ws, stem + '.txt')) if len(l.split()) == 5]
                n_keep += 1
            open(os.path.join(out, stem + '.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        print(f'\n已暂存 {len(rows)} 个标签 → {out}（其中 {n_keep} 张沿用初始）')
        if rej:
            open(f'{ROOT}/staged/{a.batch}_rejected.txt', 'w', encoding='utf-8').write(
                '\n'.join(f'{n}\t{r}' for n, r in rej) + '\n')
            print(f'剔除清单已存 {ROOT}/staged/{a.batch}_rejected.txt')


if __name__ == '__main__':
    main()
