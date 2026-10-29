/* -*- coding: utf-8 -*-
 * 实时摄像头巡检（新增独立脚本，不修改既有页面逻辑）
 *
 * 渲染挂载点：index.html 的 <section id="tab-realtime">
 * 后端接口：POST /api/realtime/detect、POST /api/realtime/stop
 *
 * 设计要点
 *   1. 摄像头用浏览器原生 navigator.mediaDevices.getUserMedia()，不引入第三方库
 *   2. 抓帧间隔默认 400ms（约 2.5 FPS），可在页面下拉框调整
 *   3. **不会堆积请求**：同一时刻只允许一个检测请求在飞；上一帧未返回则直接
 *      跳过本次抓帧（skip），请求返回后再调度下一帧
 *   4. 检测框/ROI/状态画在 canvas 覆盖层上，不改动 <video> 本身，画面保持流畅
 */
(function () {
  'use strict';

  var RT_CN = {
    person: '人员', mask: '口罩', gloves: '手套', lab_coat: '实验服',
    goggles: '护目镜', coverall: '连体防护服', helmet: '安全帽'
  };
  var COLOR = {
    ok: '#2ea043', minor: '#f59e0b', major: '#e5484d',
    ppe: '#00c8c8', roi: '#00b4ff', none: '#9aa4b2'
  };

  // 模块内部状态（不污染全局）
  var S = {
    running: false, stream: null, sessionId: 0, inflight: false, timer: null,
    video: null, cap: null, capCtx: null, overlay: null, octx: null,
    samples: [], frames: 0, sessionVio: 0, lastErr: '', errStreak: 0
  };

  function $(id) { return document.getElementById(id); }

  function rtToast(msg, type) {
    if (typeof window.showToast === 'function') { window.showToast(msg, type || 'info'); }
    else { console.warn('[realtime]', msg); }
  }

  function setText(id, text) { var el = $(id); if (el) el.textContent = text; }

  /* ---------------- 摄像头错误提示（不抛裸异常） ---------------- */
  function describeMediaError(err) {
    var n = err && err.name ? err.name : '';
    if (n === 'NotAllowedError' || n === 'PermissionDeniedError') {
      return '摄像头权限被拒绝。请点击浏览器地址栏左侧的摄像头图标，允许本站点使用摄像头后重试。';
    }
    if (n === 'NotFoundError' || n === 'DevicesNotFoundError') {
      return '未检测到可用的摄像头设备。请确认电脑已连接摄像头（或外接摄像头已插好）。';
    }
    if (n === 'NotReadableError' || n === 'TrackStartError') {
      return '摄像头被其它程序占用（如系统相机、会议软件）。请关闭占用程序后重试。';
    }
    if (n === 'OverconstrainedError' || n === 'ConstraintNotSatisfiedError') {
      return '摄像头不支持所请求的分辨率，请尝试更换摄像头或降低分辨率设置。';
    }
    if (n === 'SecurityError') {
      return '当前页面不是安全上下文，浏览器禁止调用摄像头。请使用 http://127.0.0.1 或 https 访问。';
    }
    if (n === 'AbortError') { return '摄像头打开被中断，请重试。'; }
    return '打开摄像头失败：' + n + (err && err.message ? ' - ' + err.message : '');
  }

  function checkSupport() {
    if (!window.isSecureContext) {
      return '当前不是安全上下文，浏览器禁止调用摄像头。请使用 http://127.0.0.1:8000 或 https 访问本系统。';
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return '当前浏览器不支持摄像头调用（需较新版本 Chrome / Edge，且通过 localhost 或 https 访问）。';
    }
    return '';
  }

  /* ---------------- 实验室下拉框 ---------------- */
  function loadRtLabs() {
    fetch(window.location.origin + '/api/settings')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var sel = $('rt-lab'); if (!sel) return;
        var cur = sel.value;
        sel.innerHTML = '';
        (d.labs || []).forEach(function (l) {
          var o = document.createElement('option');
          o.value = l.id; o.textContent = l.name;
          sel.appendChild(o);
        });
        if (cur) { sel.value = cur; }
        if (!sel.value && sel.options.length) { sel.value = sel.options[0].value; }
      })
      .catch(function () { /* 静默：下拉框为空时后端会返回可读错误 */ });
  }

  /* ---------------- 覆盖层绘制 ---------------- */
  function drawOverlay(res) {
    var ov = S.overlay, ctx = S.octx;
    if (!ov || !ctx) return;
    var w = ov.width, h = ov.height;
    ctx.clearRect(0, 0, w, h);

    // 1) ROI 实验操作区（归一化坐标 → 像素）
    (res.areas || []).forEach(function (a) {
      var x = a.x1 * w, y = a.y1 * h, bw = (a.x2 - a.x1) * w, bh = (a.y2 - a.y1) * h;
      ctx.strokeStyle = COLOR.roi; ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(x, y, bw, bh);
      ctx.setLineDash([]);
      ctx.fillStyle = COLOR.roi;
      ctx.font = '14px "Microsoft YaHei",sans-serif';
      var req = (a.required_ppe || []).map(function (p) { return RT_CN[p] || p; }).join('+');
      ctx.fillText(a.name + (req ? '（需 ' + req + '）' : ''), x + 4, y + 18);
    });

    // 2) PPE 检测框
    (res.detections || []).forEach(function (d) {
      if (d.cls === 'person') return;
      var b = d.bbox || [];
      ctx.strokeStyle = COLOR.ppe; ctx.lineWidth = 2;
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.fillStyle = COLOR.ppe;
      ctx.font = '13px "Microsoft YaHei",sans-serif';
      ctx.fillText((RT_CN[d.cls] || d.cls) + ' ' + d.conf, b[0], Math.max(12, b[1] - 4));
    });

    // 3) 人员框（颜色=稳定判定后的违规等级）
    (res.persons || []).forEach(function (p) {
      if (p.virtual) return;
      var b = p.bbox || [];
      var color = p.severity === '严重违规' ? COLOR.major
        : p.severity === '一般违规' ? COLOR.minor
          : (p.in_roi ? COLOR.ok : COLOR.none);
      ctx.strokeStyle = color; ctx.lineWidth = 3;
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      var tag = p.track_id || ('P' + p.person_id);
      var label = tag + (p.stable ? '' : '*') + ' ' +
        (p.in_roi ? p.severity : '区域外');
      ctx.fillStyle = color;
      ctx.font = 'bold 14px "Microsoft YaHei",sans-serif';
      ctx.fillText(label, b[0], Math.max(14, b[1] - 6));
      if (p.in_roi && p.violation_types && p.violation_types.length) {
        ctx.font = '13px "Microsoft YaHei",sans-serif';
        ctx.fillText(p.violation_types.join('、'), b[0], b[3] + 16);
      }
    });
  }

  function renderPersons(res) {
    var tb = document.querySelector('#rt-persons tbody');
    if (!tb) return;
    tb.innerHTML = '';
    (res.persons || []).forEach(function (p) {
      var tr = document.createElement('tr');
      var ppeText = Object.keys(p.ppe || {}).map(function (k) { return RT_CN[k] || k; }).join('、');
      var missText = (p.missing_ppe || []).map(function (k) { return RT_CN[k] || k; }).join('、');
      function cell(t) { var d = document.createElement('td'); d.textContent = t || '—'; return d; }
      var sevCell = document.createElement('td');
      var sev = p.in_roi ? (p.severity || '正常') : '不评估';
      sevCell.innerHTML = sev === '严重违规' ? '<span class="badge major">严重违规</span>'
        : sev === '一般违规' ? '<span class="badge minor">一般违规</span>'
          : sev === '正常' ? '<span class="badge ok">正常</span>' : '<span class="badge ok">区域外</span>';
      if (p.in_roi && !p.stable) {
        sevCell.insertAdjacentHTML('beforeend',
          ' <span class="loading">建立中(' + (p.window || 0) + '/5)</span>');
      }
      var tag = p.track_id ? ('人员 ' + p.track_id) : ('人员 ' + p.person_id);
      // 同一次连续违规只对应数据库里的 1 条记录；恢复后再次违规才会记第 2 条
      if (p.continuous_violation && p.episode_seconds) {
        tag += '（持续违规 ' + p.episode_seconds + 's · 本次已记 ' +
          (p.episode_events || 1) + ' 条）';
      }
      tr.append(cell(tag), cell(p.area), cell(ppeText), cell(missText), sevCell);
      tb.appendChild(tr);
    });
  }

  /* ---------------- 主循环 ---------------- */
  function schedule() {
    clearTimeout(S.timer);
    if (!S.running) return;
    var iv = parseInt(($('rt-interval') || {}).value, 10) || 400;
    S.timer = setTimeout(tick, iv);
  }

  function pushSample(ms) {
    S.samples.push(ms);
    if (S.samples.length > 10) S.samples.shift();
    var avg = S.samples.reduce(function (a, b) { return a + b; }, 0) / S.samples.length;
    // 单帧往返耗时（含网络与推理），反映"AI 有多快"
    setText('rt-detect-status', Math.round(avg) + ' ms/帧');
  }

  function updateFps() {
    // 真实检测帧率 = 会话内已完成帧数 / 已用时
    // （不等于 1000/耗时：实际发送受"上一帧未返回就跳过"限制，
    //   显示推算值会让人误以为在跑十几 FPS）
    if (!S.t0) return;
    var secs = (performance.now() - S.t0) / 1000;
    setText('rt-fps', secs > 0 ? (S.frames / secs).toFixed(1) : '-');
  }

  function tick() {
    if (!S.running) return;
    clearTimeout(S.timer);
    // 上一帧请求仍未返回 → 跳过本次抓帧（避免请求堆积）
    if (S.inflight) { schedule(); return; }
    var v = S.video;
    if (!v || v.readyState < 2 || !v.videoWidth) { schedule(); return; }

    S.inflight = true;
    var t0 = performance.now();
    var w = v.videoWidth, h = v.videoHeight;
    try {
      if (S.cap.width !== w || S.cap.height !== h) {
        S.cap.width = w; S.cap.height = h;
        S.overlay.width = w; S.overlay.height = h;
      }
      S.capCtx.drawImage(v, 0, 0, w, h);
    } catch (e) {
      S.inflight = false; schedule(); return;
    }

    new Promise(function (resolve) {
      S.cap.toBlob(resolve, 'image/jpeg', 0.7);
    }).then(function (blob) {
      if (!blob || !S.running) { return null; }
      var fd = new FormData();
      fd.append('frame', blob, 'frame.jpg');
      fd.append('lab_id', ($('rt-lab') || {}).value || '1');
      fd.append('session_id', String(S.sessionId || 0));
      return fetch(window.location.origin + '/api/realtime/detect',
        { method: 'POST', body: fd });
    }).then(function (r) {
      if (!r) return null;
      if (!r.ok) {
        var msg = '检测失败（HTTP ' + r.status + '）';
        return r.json().catch(function () { return {}; }).then(function (e) {
          throw new Error(e.error || msg);
        });
      }
      return r.json();
    }).then(function (res) {
      if (!res || !S.running) return;
      S.sessionId = res.session_id || S.sessionId;
      S.frames += 1;
      S.errStreak = 0;
      if (res.cooldown_s) S.cooldown = res.cooldown_s;
      updateFps();
      drawOverlay(res);
      renderPersons(res);
      setText('rt-person-count', res.person_count || 0);
      var vios = (res.persons || []).filter(function (p) { return p.in_roi && p.severity !== '正常'; });
      setText('rt-violation-status', vios.length ? (vios.length + ' 人违规') : '正常');
      var st = $('rt-violation-status');
      if (st) st.style.color = vios.length ? 'var(--danger)' : 'var(--ok)';
      if (res.session) setText('rt-session-vio', res.session.violation_count || 0);
      if (res.smooth) {
        setText('rt-smooth-status', res.smooth.ready
          ? '已建立（' + res.smooth.window + ' 帧窗口 · ≥' + res.smooth.vote + ' 帧确认）'
          : '正在建立检测状态……');
      }
      if (res.new_events && res.new_events.length) {
        rtToast('已记录违规：' + res.new_events.map(function (e) {
          return (e.violation_types || []).join('、');
        }).join('；') + '（已保存截图）', 'warn');
      }
    }).catch(function (e) {
      if (!S.running) return;
      var msg = e && e.message ? e.message : String(e);
      if (msg !== S.lastErr) { rtToast('实时检测失败：' + msg, 'error'); S.lastErr = msg; }
      setText('rt-detect-status', '异常：' + msg.slice(0, 40));
      S.errStreak += 1;
      // 连续失败 10 次（如模型不可用）自动停止，避免无限报错刷屏
      if (S.errStreak >= 10) {
        rtToast('连续检测失败，已自动停止实时巡检', 'error');
        stopRealtime();
      }
    }).then(function () {
      S.inflight = false;
      pushSample(performance.now() - t0);
      schedule();
    });
  }

  /* ---------------- 开启 / 停止 ---------------- */
  function startRealtime() {
    var err = checkSupport();
    if (err) {
      setText('rt-cam-status', '不可用');
      setText('rt-hint', err);
      rtToast(err, 'error');
      return;
    }
    setText('rt-cam-status', '正在打开…');
    setText('rt-detect-status', '准备中…');
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
    }).then(function (stream) {
      S.stream = stream;
      S.video = $('rt-video');
      S.overlay = $('rt-overlay');
      S.octx = S.overlay ? S.overlay.getContext('2d') : null;
      S.cap = S.cap || document.createElement('canvas');
      S.capCtx = S.cap.getContext('2d');
      if (!S.video) { return; }
      S.video.srcObject = stream;
      var ph = $('rt-placeholder'); if (ph) ph.style.display = 'none';
      return S.video.play().catch(function () { /* 自动播放被拦截不影响抓帧 */ });
    }).then(function () {
      var v = S.video;
      var track = S.stream && S.stream.getVideoTracks()[0];
      var setting = track && track.getSettings ? track.getSettings() : {};
      setText('rt-cam-status', '已开启' +
        (setting.width ? '（' + setting.width + '×' + setting.height + '）' : ''));
      setText('rt-detect-status', '检测中…');
      setText('rt-smooth-status', '正在建立检测状态……');
      setText('rt-hint', '实时巡检进行中：人员中心进入蓝色 ROI 才按区域规则判定；' +
        '画框为单帧原始结果，违规判定用最近 5 帧多数投票后的稳定状态；' +
        '同一人员在同一区域持续违规期间仅记录 1 次；连续多帧确认恢复正常后，' +
        '若再次发生违规，则生成新的违规记录；新出现的人员需连续多帧确认后' +
        '才计为正式人员，期间不会生成违规记录。');
      S.running = true;
      S.t0 = performance.now();
      S.sessionId = 0; S.frames = 0; S.samples = []; S.errStreak = 0; S.lastErr = '';
      $('rt-start').disabled = true;
      $('rt-stop').disabled = false;
      rtToast('摄像头已开启，实时巡检开始', 'success');
      schedule();
    }).catch(function (e) {
      var msg = describeMediaError(e);
      setText('rt-cam-status', '打开失败');
      setText('rt-detect-status', '未开始');
      setText('rt-hint', msg);
      rtToast(msg, 'error');
      if (S.stream) { S.stream.getTracks().forEach(function (t) { t.stop(); }); S.stream = null; }
    });
  }

  function stopRealtime(silent) {
    S.running = false;
    clearTimeout(S.timer); S.timer = null;
    if (S.stream) {
      S.stream.getTracks().forEach(function (t) { t.stop(); });   // 真正释放摄像头
      S.stream = null;
    }
    if (S.video) { try { S.video.pause(); } catch (e) { } S.video.srcObject = null; }
    if (S.overlay && S.octx) { S.octx.clearRect(0, 0, S.overlay.width, S.overlay.height); }
    var ph = $('rt-placeholder'); if (ph) ph.style.display = 'flex';

    if (S.sessionId) {
      var sid = S.sessionId; S.sessionId = 0;
      fetch(window.location.origin + '/api/realtime/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'session_id=' + sid
      }).catch(function () { /* 停止接口失败不影响本地已释放摄像头 */ });
    }
    setText('rt-cam-status', '已停止（摄像头已释放）');
    setText('rt-detect-status', '已停止');
    setText('rt-person-count', '0');
    setText('rt-violation-status', '-');
    setText('rt-fps', '-');
    setText('rt-smooth-status', '-');
    var btnS = $('rt-start'), btnE = $('rt-stop');
    if (btnS) btnS.disabled = false;
    if (btnE) btnE.disabled = true;
    var tb = document.querySelector('#rt-persons tbody'); if (tb) tb.innerHTML = '';
    if (!silent) rtToast('已停止实时巡检，摄像头已释放', 'info');
  }

  /* ---------------- 挂载 ---------------- */
  function init() {
    S.video = $('rt-video');
    S.overlay = $('rt-overlay');
    S.octx = S.overlay ? S.overlay.getContext('2d') : null;
    S.cap = document.createElement('canvas');
    S.capCtx = S.cap.getContext('2d');
    var b1 = $('rt-start'), b2 = $('rt-stop');
    if (b1) b1.onclick = startRealtime;
    if (b2) b2.onclick = function () { stopRealtime(); };

    loadRtLabs();

    // 切到别的页面时自动停止，避免摄像头被后台悄悄占用
    document.querySelectorAll('nav button').forEach(function (b) {
      b.addEventListener('click', function () {
        if (S.running && b.dataset.tab !== 'realtime') stopRealtime(true);
      });
    });
    // 关闭/刷新页面时释放摄像头（浏览器指示灯立即熄灭）
    window.addEventListener('beforeunload', function () {
      if (S.stream) S.stream.getTracks().forEach(function (t) { t.stop(); });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // 供手工调试 / 自动化测试使用
  window.realtime = {
    start: startRealtime, stop: stopRealtime,
    getStream: function () { return S.stream; },   // 仅调试用：验证摄像头是否真正释放
    state: function () {
      return {
        running: S.running, sessionId: S.sessionId, frames: S.frames,
        hasStream: !!S.stream,
        tracks: S.stream ? S.stream.getVideoTracks().map(function (t) {
          return t.readyState;
        }) : []
      };
    }
  };
})();
