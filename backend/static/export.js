/*
 * ============================================================================
 * 违规记录导出功能 —— 前端入口（成员5 · 艾柯代）
 * ============================================================================
 * 挂载点：index.html 中 <div id="export-panel"></div>
 * 设计约定：本文件动态渲染导出 UI 到 #export-panel 容器，
 *           index.html 不包含任何导出按钮代码，仅保留挂载点。
 *           （因此本文件不改动 index.html，也不会与前端负责人冲突）
 *
 * 依赖的后端接口（backend/export.py，成员5 负责）：
 *   GET /api/export/violations/count  导出前预览条数
 *       ?date_from=&date_to=&severity=&ppe_type=&lab_id=
 *       → {total, max_export_rows, exceed_limit, formats}
 *   GET /api/export/violations        导出文件（CSV / Excel）
 *       ?fmt=csv|xlsx&date_from=&date_to=&severity=&ppe_type=&lab_id=&sort=&order=
 *       → 文件流（Content-Disposition 附件下载；X-Total-Count 返回条数）
 *
 * 注意：本文件在 index.html 主脚本【之前】被加载，
 *       因此初始化必须等到 DOMContentLoaded，否则 showToast 尚不存在。
 * ============================================================================
 */
(function () {
  'use strict';

  var PANEL_ID = 'export-panel';
  var CN_TYPE_OPTIONS = ['未佩戴口罩', '未佩戴手套', '未穿实验服', '未佩戴护目镜', '未佩戴安全帽'];

  /* ---------------------------------------------------------------- 工具 */

  function el(id) { return document.getElementById(id); }

  // 优先用主页的 showToast；若主页脚本异常则降级为 alert，保证提示不丢
  function toast(msg, type, duration) {
    if (typeof window.showToast === 'function') {
      window.showToast(msg, type || 'info', duration || 3500);
    } else {
      window.alert(msg);
    }
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function pad2(n) { return String(n).padStart(2, '0'); }

  function dateStr(d) {
    return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
  }

  function daysAgo(n) {
    var d = new Date();
    d.setDate(d.getDate() - n);
    return dateStr(d);
  }

  function stamp() {
    var d = new Date();
    return String(d.getFullYear()) + pad2(d.getMonth() + 1) + pad2(d.getDate()) +
           '_' + pad2(d.getHours()) + pad2(d.getMinutes()) + pad2(d.getSeconds());
  }

  /* ------------------------------------------------------- 样式（注入，不动 index.html） */

  function injectStyle() {
    if (el('ex-style')) return;
    var css = [
      '.ex-wrap{border:1px dashed var(--border);border-radius:9px;padding:14px 16px;margin:6px 0 14px;background:#fafbfc}',
      '.ex-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:10px}',
      '.ex-title{font-size:14px;font-weight:600;color:#24292f}',
      '.ex-title small{font-weight:400;color:var(--muted);font-size:12px;margin-left:8px}',
      '.ex-quick{display:flex;gap:6px;align-items:center}',
      '.ex-mini{background:#fff;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer}',
      '.ex-mini:hover{border-color:var(--primary);color:var(--primary)}',
      '.ex-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}',
      '.ex-msg{font-size:12px;line-height:1.7;color:var(--muted);min-height:18px}',
      '.ex-msg.ok{color:var(--ok)}',
      '.ex-msg.err{color:var(--danger)}',
      '.ex-msg.warn{color:var(--warn)}',
      '.ex-count{font-weight:600;color:var(--primary)}'
    ].join('');
    var s = document.createElement('style');
    s.id = 'ex-style';
    s.textContent = css;
    document.head.appendChild(s);
  }

  /* ------------------------------------------------------------- 渲染 UI */

  function render() {
    var panel = el(PANEL_ID);
    if (!panel) return;

    panel.innerHTML = [
      '<div class="ex-wrap">',
      '  <div class="ex-head">',
      '    <div class="ex-title">违规记录导出<small>按下方条件筛选后导出为 CSV 或 Excel 文件</small></div>',
      '    <div class="ex-quick">',
      '      <span class="ex-mini" data-range="7">近 7 天</span>',
      '      <span class="ex-mini" data-range="30">近 30 天</span>',
      '      <span class="ex-mini" data-range="0">全部时间</span>',
      '    </div>',
      '  </div>',
      '  <div class="ex-row">',
      '    <label>导出格式</label>',
      '    <select id="ex-fmt">',
      '      <option value="csv">CSV（通用，Excel 可直接打开）</option>',
      '      <option value="xlsx">Excel 工作簿 (.xlsx)</option>',
      '    </select>',
      '    <label>起始日期</label><input type="date" id="ex-from">',
      '    <label>结束日期</label><input type="date" id="ex-to">',
      '    <label>严重程度</label>',
      '    <select id="ex-sev"><option value="">全部</option><option>一般违规</option><option>严重违规</option></select>',
      '    <label>违规类型</label>',
      '    <select id="ex-type"><option value="">全部</option>' +
             CN_TYPE_OPTIONS.map(function (t) { return '<option>' + t + '</option>'; }).join('') +
           '</select>',
      '    <label>实验室</label>',
      '    <select id="ex-lab"><option value="">全部实验室</option></select>',
      '  </div>',
      '  <div class="ex-row">',
      '    <button class="btn" id="ex-btn">导出记录</button>',
      '    <button class="btn ghost" id="ex-reset">重置条件</button>',
      '    <span class="ex-msg" id="ex-msg"></span>',
      '  </div>',
      '</div>'
    ].join('');

    bind();
  }

  /* ------------------------------------------------------------- 事件绑定 */

  function bind() {
    var fmt = el('ex-fmt'), from = el('ex-from'), to = el('ex-to');
    var sev = el('ex-sev'), type = el('ex-type'), lab = el('ex-lab');

    [fmt, from, to, sev, type, lab].forEach(function (node) {
      if (node) node.addEventListener('change', refreshCount);
    });

    // 快捷时间区间
    document.querySelectorAll('#export-panel .ex-mini').forEach(function (b) {
      b.addEventListener('click', function () {
        var n = parseInt(b.getAttribute('data-range'), 10);
        if (n > 0) {
          from.value = daysAgo(n - 1);
          to.value = daysAgo(0);
        } else {
          from.value = '';
          to.value = '';
        }
        refreshCount();
      });
    });

    el('ex-btn').addEventListener('click', doExport);

    el('ex-reset').addEventListener('click', function () {
      from.value = ''; to.value = '';
      sev.value = ''; type.value = ''; lab.value = '';
      fmt.value = 'csv';
      setMsg('');
      refreshCount();
    });
  }

  /* ------------------------------------------------------- 实验室下拉（动态加载） */

  var labsLoaded = false;

  function loadLabs() {
    var sel = el('ex-lab');
    if (!sel) return;
    fetch(window.location.origin + '/api/settings')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.labs) return;
        var cur = sel.value;
        sel.innerHTML = '<option value="">全部实验室</option>' +
          d.labs.map(function (l) {
            return '<option value="' + l.id + '">' + escapeHtml(l.name) + '</option>';
          }).join('');
        sel.value = cur;
        labsLoaded = true;
      })
      .catch(function () { /* 实验室列表加载失败不影响导出主流程 */ });
  }

  /* ------------------------------------------------------------ 条数预览 */

  var countTimer = null;

  function buildParams() {
    var p = new URLSearchParams();
    var from = el('ex-from').value, to = el('ex-to').value;
    var sev = el('ex-sev').value, type = el('ex-type').value, lab = el('ex-lab').value;
    if (from) p.set('date_from', from);
    if (to) p.set('date_to', to);
    if (sev) p.set('severity', sev);
    if (type) p.set('ppe_type', type);
    if (lab) p.set('lab_id', lab);
    return p;
  }

  function setMsg(text, cls) {
    var m = el('ex-msg');
    if (!m) return;
    m.className = 'ex-msg' + (cls ? ' ' + cls : '');
    m.innerHTML = text || '';
  }

  function refreshCount() {
    if (countTimer) clearTimeout(countTimer);
    countTimer = setTimeout(doRefreshCount, 350);
  }

  function doRefreshCount() {
    var p = buildParams();
    fetch(window.location.origin + '/api/export/violations/count?' + p.toString())
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        if (!res.ok) {
          setMsg(res.body && res.body.error ? res.body.error : '筛选条件有误', 'err');
          return;
        }
        var d = res.body;
        // 后端未安装 openpyxl 时，禁用 Excel 选项，而不是让用户点出一堆 500
        var fmtSel = el('ex-fmt');
        var xlsxOpt = fmtSel.querySelector('option[value="xlsx"]');
        var hasXlsx = (d.formats || []).indexOf('xlsx') !== -1;
        if (xlsxOpt) {
          xlsxOpt.disabled = !hasXlsx;
          xlsxOpt.textContent = hasXlsx
            ? 'Excel 工作簿 (.xlsx)'
            : 'Excel 工作簿 (.xlsx) —— 服务器未安装 openpyxl';
        }
        if (!hasXlsx && fmtSel.value === 'xlsx') fmtSel.value = 'csv';

        if (d.exceed_limit) {
          setMsg('符合条件 <span class="ex-count">' + d.total + '</span> 条，' +
                 '超过单次导出上限 ' + d.max_export_rows + ' 条，请缩小范围', 'warn');
        } else {
          setMsg('符合条件 <span class="ex-count">' + d.total + '</span> 条记录可导出');
        }
      })
      .catch(function () { setMsg(''); });
  }

  /* -------------------------------------------------------------- 执行导出 */

  function parseFilename(disposition) {
    if (!disposition) return '';
    var m = /filename\*?=(?:UTF-8''|")?([^";]+)/i.exec(disposition);
    return m ? decodeURIComponent(m[1].replace(/^"|"$/g, '')) : '';
  }

  function triggerDownload(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }

  function doExport() {
    var fmtSel = el('ex-fmt');
    var fmt = fmtSel.value;
    var from = el('ex-from').value, to = el('ex-to').value;

    if (from && to && from > to) {
      toast('起始日期不能晚于结束日期', 'warn');
      setMsg('起始日期不能晚于结束日期', 'err');
      return;
    }
    if (fmt === 'xlsx') {
      var opt = fmtSel.querySelector('option[value="xlsx"]');
      if (opt && opt.disabled) {
        toast('服务器未安装 Excel 导出依赖，请改用 CSV', 'warn');
        return;
      }
    }

    var p = buildParams();
    p.set('fmt', fmt);

    var btn = el('ex-btn');
    var oldText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '导出中…';

    fetch(window.location.origin + '/api/export/violations?' + p.toString())
      .then(function (r) {
        if (!r.ok) {
          return r.json().catch(function () { return {}; }).then(function (j) {
            var msg = (j && j.error) ? j.error : ('导出失败（HTTP ' + r.status + '）');
            setMsg(msg, 'err');
            toast(msg, 'error', 6000);
          });
        }
        var count = r.headers.get('X-Total-Count');
        var fname = parseFilename(r.headers.get('Content-Disposition')) ||
                    ('violations_' + stamp() + '.' + fmt);
        return r.blob().then(function (blob) {
          if (blob.size === 0) {
            setMsg('导出内容为空，请检查筛选条件', 'warn');
            toast('导出内容为空，请检查筛选条件', 'warn');
            return;
          }
          triggerDownload(blob, fname);
          var tip = (count !== null ? '已导出 ' + count + ' 条记录' : '导出完成') +
                    '（' + (blob.size / 1024).toFixed(1) + ' KB）';
          setMsg(tip + ' → ' + escapeHtml(fname), 'ok');
          toast('导出成功：' + (count !== null ? count + ' 条记录' : fname), 'success');
        });
      })
      .catch(function (e) {
        setMsg('导出失败：' + e.message, 'err');
        toast('导出失败：' + e.message, 'error');
      })
      .then(function () {
        btn.disabled = false;
        btn.textContent = oldText;
      });
  }

  /* ---------------------------------------------------------------- 初始化 */

  function init() {
    if (!el(PANEL_ID)) return;   // 挂载点不存在则静默退出，不影响其他功能
    injectStyle();
    render();
    loadLabs();
    refreshCount();

    // 切到「违规记录」页时刷新实验室下拉，保证新建的实验室能立刻出现在导出筛选里
    var navBtn = document.querySelector('nav button[data-tab="violations"]');
    if (navBtn) {
      navBtn.addEventListener('click', function () {
        loadLabs();
        refreshCount();
      });
    }

    // 供调试：控制台可用 ExportPanel.refresh() 手动刷新条数与实验室
    window.ExportPanel = {
      refresh: function () { loadLabs(); doRefreshCount(); },
      labsLoaded: function () { return labsLoaded; }
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
