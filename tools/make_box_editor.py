# -*- coding: utf-8 -*-
"""生成手工框选页 editor.html —— 让她自己拖框/改框/删框/加框。

自包含离线页面：内嵌 19 张图的当前标签 + 原图 file:// 路径。
操作：
  空白拖拽 = 新建框（当前类别）
  拖框内部 = 移动
  拖右下角手柄 = 缩放
  点框右上角 × = 删除
  数字键 1-5 = 切换类别（选中框会跟着改）
  Delete = 删除选中框
右侧「导出」区：复制全部改动标签 / 逐个下载 txt。

用法: .venv/Scripts/python.exe tools/make_box_editor.py --batch r2
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass  # 已是 utf-8 或被宿主接管

ROOT = Path(__file__).resolve().parents[1]


def imsize(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#16181c;color:#e6e6e6;
      display:flex;flex-direction:column;height:100vh;overflow:hidden}
 header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;background:#22252a;
        border-bottom:1px solid #333;flex:0 0 auto}
 .cat{padding:4px 10px;border-radius:5px;border:2px solid #444;background:#2c3037;color:#ccc;
      cursor:pointer;font-size:13px}
 .cat.on{color:#fff;font-weight:700}
 select,input[type=range]{background:#2c3037;color:#e6e6e6;border:1px solid #444;border-radius:5px;padding:3px 6px}
 button{background:#0a84ff;color:#fff;border:0;border-radius:5px;padding:5px 12px;cursor:pointer;font-size:13px}
 button.gray{background:#3a3f47}
 button:disabled{opacity:.45;cursor:default}
 main{flex:1 1 auto;display:flex;min-height:0}
 #stage{flex:1 1 auto;overflow:auto;background:#101215;display:flex;align-items:flex-start;
        justify-content:center;padding:14px}
 #wrap{position:relative;flex:0 0 auto;box-shadow:0 0 0 1px #333}
 #bg{display:block;user-select:none;-webkit-user-drag:none}
 #layer{position:absolute;inset:0}
 .bx{position:absolute;border:2px solid #0f0;cursor:move;box-sizing:border-box}
 .bx.sel{border-width:3px;box-shadow:0 0 0 1px #fff inset}
 .bx .lb{position:absolute;top:-19px;left:-2px;font-size:11px;color:#000;padding:0 4px;border-radius:3px 3px 0 0;
         white-space:nowrap;font-weight:700}
 .bx .x{position:absolute;top:-19px;right:-2px;font-size:13px;line-height:16px;width:16px;height:16px;
        text-align:center;background:#ff3b30;color:#fff;border-radius:50%;cursor:pointer;font-style:normal}
 .bx .h{position:absolute;right:-6px;bottom:-6px;width:12px;height:12px;background:#fff;border:2px solid #0a84ff;
        border-radius:2px;cursor:nwse-resize;display:none}
 .bx.sel .h{display:block}
 aside{flex:0 0 340px;border-left:1px solid #333;background:#1c1f24;display:flex;flex-direction:column;
        padding:10px;gap:8px;overflow:auto}
 aside h3{margin:0;font-size:13px;color:#9aa0a6;font-weight:600}
 #out{flex:1 1 auto;min-height:180px;background:#121417;color:#9fe89f;border:1px solid #333;border-radius:5px;
      padding:8px;font-family:Consolas,monospace;font-size:11px;white-space:pre;overflow:auto}
 .hint{font-size:12px;color:#8b9099;line-height:1.7}
 .hint b{color:#c9ced6}
 #dirty{font-size:12px;color:#ffd479}
</style></head><body>
<header>
  <span id="cats"></span>
  <select id="picker"></select>
  <button id="prev" class="gray">‹ 上一张</button>
  <button id="next" class="gray">下一张 ›</button>
  <label style="font-size:12px;color:#9aa0a6">缩放 <input type="range" id="zoom" min="30" max="220" value="100" style="width:110px;vertical-align:middle"></label>
  <button id="reset" class="gray">撤销本图改动</button>
  <button id="clear" class="gray">清空本图</button>
  <button id="fit" class="gray">适应窗口</button>
  <button id="reload" class="gray">重新载入</button>
  <span id="dirty"></span>
</header>
<div id="err" style="display:none;background:#ff3b30;color:#fff;padding:6px 12px;font-size:12px"></div>
<main>
  <div id="stage"><div id="wrap"><img id="bg"><div id="layer"></div></div></div>
  <aside>
    <h3>导出（改完把下面内容贴给我）</h3>
    <button id="copy">复制全部改动标签</button>
    <div id="dls"></div>
    <h3>改动内容</h3>
    <textarea id="out" readonly placeholder="改完后点上面的「复制全部改动标签」"></textarea>
    <div class="hint">
      <b>操作</b><br>
      · 空白处拖拽 = 新建框（当前类别）<br>
      · 拖框内部 = 移动<br>
      · 拖右下角白手柄 = 缩放<br>
      · 点框右上角红 × = 删除<br>
      · 数字键 <b>1-5</b> = 切类别（选中框一起改）<br>
      · <b>Delete</b> = 删除选中框<br>
      · 类别：1 person · 2 mask · 3 gloves · 4 lab_coat · 5 goggles
    </div>
  </aside>
</main>
<script>
const COLORS={0:'#00c800',1:'#eb0000',2:'#008ceb',3:'#c800c8',4:'#ffc800'};
const NAMES={0:'person',1:'mask',2:'gloves',3:'lab_coat',4:'goggles'};
const DATA=__DATA__;

let imgs=DATA.imgs.map(o=>({name:o.name,path:o.path,nw:o.nw,nh:o.nh,
  boxes:o.boxes.map(b=>({c:b[0],x:b[1],y:b[2],w:b[3],h:b[4]}))}));
const orig=JSON.parse(JSON.stringify(imgs));
const dirty={};
let cur=DATA.focus||0, cls=1, z=1, sel=-1, drag=null;

const bg=document.getElementById('bg'), layer=document.getElementById('layer'),
      wrap=document.getElementById('wrap'), stage=document.getElementById('stage'),
      picker=document.getElementById('picker'), out=document.getElementById('out'),
      dls=document.getElementById('dls'), dirtyEl=document.getElementById('dirty'),
      errEl=document.getElementById('err');

function showErr(msg){ errEl.style.display='block'; errEl.textContent='⚠ '+msg+'（把这句发我）'; }
window.onerror=function(m,s,l,c){ showErr('脚本错误: '+m+' @'+l+':'+c); };
bg.onerror=function(){ showErr('图片加载失败: '+bg.getAttribute('src')); };

// 类别按钮
const catsEl=document.getElementById('cats');
[0,1,2,3,4].forEach(c=>{
  const b=document.createElement('button');
  b.className='cat'; b.dataset.c=c; b.textContent=(c+1)+' '+NAMES[c];
  b.style.borderColor=COLORS[c];
  b.onclick=()=>{cls=c; if(sel>=0){imgs[cur].boxes[sel].c=c; touch();} render();};
  catsEl.appendChild(b);
});

// 图片下拉
function shortName(n){ return n.replace(/^stock_/,'').replace(/^(r3[su]|webs|websr2)_/,'').replace(/\.jpg$/,''); }
function refreshPicker(){
  [...picker.options].forEach((o,i)=>{
    const im=imgs[i];
    o.textContent='#'+im.no+' '+shortName(im.name)+'（'+im.boxes.length+' 框'+(dirty[im.name]?' · 已改':'')+'）';
  });
}
imgs.forEach((im,i)=>{
  const o=document.createElement('option'); o.value=i;
  picker.appendChild(o);
});
refreshPicker();
picker.onchange=()=>{cur=+picker.value; sel=-1; render();};
document.getElementById('prev').onclick=()=>go(cur-1);
document.getElementById('next').onclick=()=>go(cur+1);
function go(i){ if(i<0||i>=imgs.length) return; cur=i; sel=-1; render(); }
document.getElementById('zoom').oninput=e=>{z=e.target.value/100; render();};
document.getElementById('reset').onclick=()=>{
  imgs[cur]=JSON.parse(JSON.stringify(orig[cur])); sel=-1;
  delete dirty[imgs[cur].name]; render();
};
document.getElementById('reload').onclick=()=>location.reload();
document.getElementById('clear').onclick=()=>{
  if(!confirm('清空本图的所有框？（可在右侧导出后重画）')) return;
  imgs[cur].boxes=[]; sel=-1; touch(); render();
};
document.getElementById('fit').onclick=()=>{
  const im=imgs[cur];
  let zz=Math.min((stage.clientWidth-40)/im.nw,(stage.clientHeight-40)/im.nh,1.5);
  if(!isFinite(zz)||zz<=0) zz=1;
  z=zz; document.getElementById('zoom').value=Math.round(zz*100); render();
};
document.getElementById('copy').onclick=()=>{
  const txt=buildText(); out.value=txt;
  out.select();
  try{navigator.clipboard.writeText(txt);}catch(e){document.execCommand('copy');}
  alert(txt? '已复制 '+Object.keys(dirty).length+' 张的标签，直接粘给我即可'
           : '还没有任何改动');
};

function touch(){ dirty[imgs[cur].name]=1; dirtyEl.textContent='已改 '+Object.keys(dirty).length+' 张'; }
function clamp(v){return Math.max(0,Math.min(1,v));}

function render(){
  const im=imgs[cur];
  picker.value=cur;
  wrap.style.width=(im.nw*z)+'px';
  wrap.style.height=(im.nh*z)+'px';
  bg.style.width=(im.nw*z)+'px';
  bg.style.height=(im.nh*z)+'px';
  if(bg.getAttribute('src')!==im.path) bg.src=im.path;
  layer.innerHTML='';
  im.boxes.forEach((b,i)=>{
    const d=document.createElement('div');
    d.className='bx'+(i===sel?' sel':'');
    d.dataset.i=i;
    d.style.borderColor=COLORS[b.c];
    d.innerHTML='<span class="lb" style="background:'+COLORS[b.c]+'">'+NAMES[b.c]+'#'+i+
                '</span><b class="x" title="删除">×</b><i class="h"></i>';
    layer.appendChild(d);
    place(d,b,im);
  });
  [...catsEl.children].forEach(el=>el.classList.toggle('on',+el.dataset.c===cls));
  dirtyEl.textContent='已改 '+Object.keys(dirty).length+'/'+imgs.length+' 张';
  refreshPicker();
}
function place(d,b,im){
  d.style.left=(b.x*im.nw*z)+'px'; d.style.top=(b.y*im.nh*z)+'px';
  d.style.width=(b.w*im.nw*z)+'px'; d.style.height=(b.h*im.nh*z)+'px';
}
function ptr(e){
  const r=layer.getBoundingClientRect(), im=imgs[cur];
  return {x:clamp((e.clientX-r.left)/(im.nw*z)), y:clamp((e.clientY-r.top)/(im.nh*z))};
}

layer.addEventListener('pointerdown',e=>{
  const im=imgs[cur], p=ptr(e), t=e.target;
  if(t.classList.contains('x')){
    im.boxes.splice(+t.parentNode.dataset.i,1); sel=-1; touch(); render(); e.preventDefault(); return;
  }
  if(t.classList.contains('h')){
    drag={mode:'resize',i:+t.parentNode.dataset.i}; layer.setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  if(t.classList.contains('bx')){
    sel=+t.dataset.i;
    const b=im.boxes[sel];
    drag={mode:'move',i:sel,ox:p.x-b.x,oy:p.y-b.y};
    render(); layer.setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  // 空白 → 新建
  im.boxes.push({c:cls,x:p.x,y:p.y,w:0.02,h:0.02});
  sel=im.boxes.length-1;
  drag={mode:'resize',i:sel,new:true};
  render(); layer.setPointerCapture(e.pointerId); e.preventDefault();
});

window.addEventListener('pointermove',e=>{
  if(!drag) return;
  const im=imgs[cur], p=ptr(e), b=im.boxes[drag.i], d=layer.querySelector('.bx[data-i="'+drag.i+'"]');
  if(drag.mode==='move'){
    b.x=clamp(p.x-drag.ox); b.y=clamp(p.y-drag.oy);
    b.w=Math.min(b.w,1-b.x); b.h=Math.min(b.h,1-b.y);
  }else{
    b.w=Math.max(0.004,p.x-b.x); b.h=Math.max(0.004,p.y-b.y);
    if(b.x+b.w>1) b.w=1-b.x;
    if(b.y+b.h>1) b.h=1-b.y;
  }
  if(d) place(d,b,im);
  touch();
});

window.addEventListener('pointerup',e=>{
  if(!drag) return;
  const im=imgs[cur], b=im.boxes[drag.i];
  if(drag.new && (b.w<0.012||b.h<0.012)){ im.boxes.splice(drag.i,1); sel=-1; }
  drag=null; render();
});

window.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='Delete'||e.key==='Backspace'){
    if(sel>=0){ imgs[cur].boxes.splice(sel,1); sel=-1; touch(); render(); e.preventDefault(); }
  }
  if(e.key==='ArrowLeft'){ go(cur-1); e.preventDefault(); }
  if(e.key==='ArrowRight'){ go(cur+1); e.preventDefault(); }
  if(e.key>='1'&&e.key<='5'){
    cls=+e.key-1;
    if(sel>=0){ imgs[cur].boxes[sel].c=cls; touch(); }
    render();
  }
});

function buildText(){
  const keys=Object.keys(dirty);
  if(!keys.length) return '';
  const im=imgs.find(o=>o.name===keys[0]);
  let s='';
  keys.forEach(name=>{
    const o=imgs.find(x=>x.name===name);
    s+='### '+name.replace(/\.jpg$/,'')+'\n';
    o.boxes.forEach(b=>{
      s+=b.c+' '+((b.x+b.w/2).toFixed(6))+' '+((b.y+b.h/2).toFixed(6))+' '
         +(b.w.toFixed(6))+' '+(b.h.toFixed(6))+'\n';
    });
  });
  return s;
}

// 下载区（每次 render 时重建）
const _r=render;
render=function(){
  _r();
  dls.innerHTML='';
  Object.keys(dirty).forEach(name=>{
    const o=imgs.find(x=>x.name===name);
    let s='';
    o.boxes.forEach(b=>{
      s+=b.c+' '+((b.x+b.w/2).toFixed(6))+' '+((b.y+b.h/2).toFixed(6))+' '
        +(b.w.toFixed(6))+' '+(b.h.toFixed(6))+'\n';
    });
    const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([s],{type:'text/plain'}));
    a.download=name.replace(/\.jpg$/,'')+'.txt';
    a.textContent='下载 '+name.replace(/\.jpg$/,'')+'.txt';
    a.style.cssText='display:block;color:#7cc4ff;font-size:12px;margin:3px 0';
    dls.appendChild(a);
  });
};

render();
window.addEventListener('load',()=>{ document.getElementById('fit').click(); });
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="r2")
    ap.add_argument("--focus", type=int, default=9, help="打开时默认定位的图片编号")
    ap.add_argument("--max-side", type=int, default=2400,
                    help="预览图最长边超过此值则生成缩图（坐标仍按原图归一化，不影响标签精度）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    _b = f"_{args.batch}" if args.batch else ""

    REVIEW = ROOT / "outputs" / f"web_lab_scene_review{_b}"
    IMG = ROOT / "data" / "raw" / "web_lab_scene" / f"images{_b}"
    LBL = ROOT / "data" / "_prelabel" / f"webscene{_b}" / "labels"
    out = Path(args.out) if args.out else REVIEW / "editor.html"
    VIEW = REVIEW / "_view"

    idx = {}
    for line in (REVIEW / "index.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            idx[int(p[0])] = p[1].strip()

    imgs = []
    focus = 0
    for n in sorted(idx):
        name = idx[n]
        ip = IMG / name
        w, h = imsize(ip)
        # 超长边 -> 生成预览缩图（归一化坐标与原图一致，标签精度不受影响）
        vp, vw, vh = ip, w, h
        if max(w, h) > args.max_side:
            s = args.max_side / max(w, h)
            vw, vh = max(1, int(w * s)), max(1, int(h * s))
            VIEW.mkdir(parents=True, exist_ok=True)
            vp = VIEW / (Path(name).stem + ".jpg")
            if not vp.exists():
                im = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imencode(".jpg", cv2.resize(im, (vw, vh), interpolation=cv2.INTER_AREA),
                             [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(vp))
        rows = []
        lp = LBL / (Path(name).stem + ".txt")
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    q = ln.split()
                    rows.append([int(q[0])] + [round(float(v), 6) for v in q[1:5]])
        imgs.append({
            "no": n, "name": name,
            "path": "file:///" + str(vp).replace("\\", "/"),
            "nw": vw, "nh": vh, "boxes": rows,
        })
        if n == args.focus:
            focus = len(imgs) - 1
        tag = "" if (vw, vh) == (w, h) else f" (view {vw}x{vh})"
        print(f"  #{n} {name} {w}x{h} boxes={len(rows)}{tag}")

    html = (HTML
            .replace("__TITLE__", f"手工框选 · {len(imgs)} 张（batch {args.batch}）")
            .replace("__DATA__", json.dumps(
                {"imgs": imgs, "focus": focus}, ensure_ascii=False)))
    out.write_text(html, encoding="utf-8")
    print(f"完成 -> {out}")


if __name__ == "__main__":
    main()
