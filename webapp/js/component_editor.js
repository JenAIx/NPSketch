/*
 * ComponentEditor — reusable OCS-Plus component reviewer/editor.
 *
 * Renders the layout used by the Review & Label Queue:
 *   left  : the drawing image — switchable Processed ⇄ Original, with an optional crop tool and
 *           an element-highlight overlay — and, below it, the colour-coded
 *           "reference — element locations" canvas
 *   right : an optional scores line and the editable 20×(Presence/Accuracy/Position) grid
 *
 * One DOM-free singleton loads the reference assets (element definitions + the reference image)
 * once and shares them across every instance. Each instance owns its own state/canvases and uses
 * event delegation, so several editors can coexist.
 *
 * Used by:
 *   - webapp/ai_training_data_view.html  (Review & Label Queue + the Training-data preview modal)
 *   - webapp/evaluate.html               (Learn / Label panel)
 *
 * Public API:
 *   const ed = ComponentEditor.create(container, {
 *     showImage, showScores, showComparison, showHeaderTotal,  // booleans
 *     allowCrop,          // show ✂️ Crop on the Original view
 *     onCrop,             // async ({x1,y1,x2,y2}) => void   (host does crop-and-reprocess)
 *     externalTotalEl,    // id/element of an external <span> to keep in sync with the total
 *     onChange,           // (state) => void, fired after every (unlocked) toggle
 *   });
 *   ed.setImage(src)                       // single processed image (no toggle/crop)
 *   ed.setImages({processed, original, imageId})   // enable Processed/Original toggle (+ crop)
 *   ed.setState(components)                // {presence:[20],accuracy:[20],position:[20]}
 *   ed.getState()                          // -> current components (a copy)
 *   ed.setComparison({model, orig})        // model = best-model components, orig = previously-saved
 *   ed.setConfidence(elements)             // live prediction.elements (probs+thresholds) → 🎯 toggle: per-cell margin overlay
 *   ed.setScores(html)                     // raw HTML for the scores line (showScores instances)
 *   ed.setLocked(bool)                     // read-only grid (e.g. a validated image)
 *   ed.total()                             // current Total_Score (0..60)
 *   ed.highlight(e) / ed.refresh()
 */
window.ComponentEditor = (function () {
  const ELS = Array.from({ length: 20 }, (_, i) => i);
  const KEYS = ['presence', 'accuracy', 'position'];
  const KEYS_LABEL = { presence: 'Presence', accuracy: 'Accuracy', position: 'Position' };
  // Confidence of the hard decision (p>=thr), normalized to the room on the DECIDED side of the
  // threshold: 1 = at the extreme prob (0 or 1, fully confident), 0 = exactly at the threshold
  // (max uncertain). This avoids flagging a confident p≈0 as "uncertain" just because the threshold
  // is low (e.g. p=0.00, thr=0.10 → conf=1.0, not 0.10).
  function decisionConf(p, thr) {
    return p >= thr ? (p - thr) / Math.max(1e-6, 1 - thr) : (thr - p) / Math.max(1e-6, thr);
  }
  const UNCERTAIN_FRAC = 0.20;   // decision confidence below this (20%) = "on the fence"
  // element colours — same scheme as the Component Map / Review Queue (e is 0-based)
  const elColor = e => `hsl(${Math.round(e * 360 / 20)}, 75%, 46%)`;
  // short header definitions (OCS-Plus Figure Copy scoring manual → docs/SCORING_CRITERIA.md)
  const ASPECT_HELP = {
    presence: 'Presence (0/1): a recognisable version of this component appears anywhere in the drawing — location-independent.',
    accuracy: 'Accuracy (0/1): drawn with reasonable accuracy — straight lines, clean joins at the template angles (≈90° for the container). Stylus slips & obvious self-corrections are allowed; judges copying ability, not drawing skill.',
    position: 'Position (0/1): positioned correctly relative to its nearest neighbours — dividers must partition in similar proportions; details must match position AND orientation (a flipped diagonal scores 0).',
    el: 'The 20 template components (E01–E20): container edges, dividers, and details (circle, star, cross).',
    sum: 'Σ — this component’s subscore = Presence + Accuracy + Position (0–3). Total_Score = sum over all 20 (0–60).',
  };
  const blank = () => ({ presence: Array(20).fill(0), accuracy: Array(20).fill(0), position: Array(20).fill(0) });
  const clone = c => ({
    presence: (c && c.presence) ? c.presence.slice() : Array(20).fill(0),
    accuracy: (c && c.accuracy) ? c.accuracy.slice() : Array(20).fill(0),
    position: (c && c.position) ? c.position.slice() : Array(20).fill(0),
  });
  const resolve = ref => (typeof ref === 'string') ? document.getElementById(ref) : ref;

  /* ---------------- custom tooltip (native title is unreliable / slow) ---------------- */
  let _tip = null;
  function ensureTip() {
    if (_tip) return;
    _tip = document.createElement('div');
    _tip.style.cssText = 'position:fixed; z-index:100000; max-width:300px; background:#222; color:#fff; ' +
      'padding:8px 11px; border-radius:6px; font-size:12px; line-height:1.4; ' +
      'box-shadow:0 4px 16px rgba(0,0,0,.35); pointer-events:none; display:none;';
    document.body.appendChild(_tip);
    document.addEventListener('mouseover', e => {
      const t = e.target.closest && e.target.closest('[data-tip]');
      if (!t) return;
      _tip.textContent = t.getAttribute('data-tip');
      _tip.style.display = 'block';
      const r = t.getBoundingClientRect();
      const tw = _tip.offsetWidth, th = _tip.offsetHeight;
      let left = r.left, top = r.bottom + 8;
      if (left + tw > window.innerWidth - 8) left = window.innerWidth - tw - 8;
      if (top + th > window.innerHeight - 8) top = r.top - th - 8;   // flip above if no room below
      _tip.style.left = Math.max(8, left) + 'px';
      _tip.style.top = Math.max(8, top) + 'px';
    });
    document.addEventListener('mouseout', e => {
      if (e.target.closest && e.target.closest('[data-tip]')) _tip.style.display = 'none';
    });
  }

  /* ---------------- shared reference assets (loaded once) ---------------- */
  let _elDefs = null, _refImg = null, _refPromise = null;
  function loadRefAssets() {
    if (_refPromise) return _refPromise;
    _refPromise = (async () => {
      try {
        const d = await (await fetch('/api/ai-training/element-definitions?_=' + Date.now())).json();
        _elDefs = {}; (d.elements || []).forEach(el => { _elDefs[el.element - 1] = el.strokes || []; });
      } catch (e) { _elDefs = {}; }
      await new Promise(res => {
        _refImg = new Image();
        _refImg.onload = res; _refImg.onerror = res;
        _refImg.src = '/api/reference-image?_=' + Date.now();
      });
      return { elDefs: _elDefs, refImg: _refImg };
    })();
    return _refPromise;
  }

  // draw one element's strokes into ctx x (extra widens the line, e.g. for highlight)
  function drawStrokes(x, e, extra) {
    const strokes = (_elDefs && _elDefs[e]) || [];
    x.lineCap = 'round'; x.lineJoin = 'round'; x.strokeStyle = x.fillStyle = elColor(e);
    for (const s of strokes) {
      if (s.erase) continue;
      const w = (s.width || 6) + extra, p = s.points || [];
      if (p.length === 1) { x.beginPath(); x.arc(p[0][0], p[0][1], w / 2, 0, 7); x.fill(); continue; }
      x.lineWidth = w; x.beginPath(); x.moveTo(p[0][0], p[0][1]);
      for (let i = 1; i < p.length; i++) x.lineTo(p[i][0], p[i][1]);
      x.stroke();
    }
  }
  // the reference panel: all element locations, faded except the highlighted one
  function drawReference(canvas, hl) {
    if (!canvas) return; const x = canvas.getContext('2d');
    x.clearRect(0, 0, 568, 274); x.fillStyle = '#fff'; x.fillRect(0, 0, 568, 274);
    if (_refImg && _refImg.complete) { x.globalAlpha = 0.18; x.drawImage(_refImg, 0, 0, 568, 274); x.globalAlpha = 1; }
    if (!_elDefs) return;
    for (const e of ELS) {
      x.globalAlpha = (hl == null) ? 0.9 : (e === hl ? 1 : 0.08);
      drawStrokes(x, e, e === hl ? 2 : 0);
    }
    x.globalAlpha = 1;
  }
  // mark the highlighted element ON THE DRAWING (only valid on the 568×274 processed frame)
  function drawImgOverlay(canvas, hl) {
    if (!canvas) return; const x = canvas.getContext('2d');
    x.clearRect(0, 0, 568, 274);
    if (hl == null || !_elDefs) return;
    x.globalAlpha = 0.9; drawStrokes(x, hl, 3); x.globalAlpha = 1;
  }

  /* ---------------- instance ---------------- */
  function create(container, opts) {
    opts = opts || {};
    const root = resolve(container);
    if (!root) throw new Error('ComponentEditor: container not found');
    ensureTip();
    const showImage = opts.showImage !== false;       // default true
    const showReference = opts.showReference !== false; // default true; false → grid-only
    const showScores = !!opts.showScores;
    const showComparison = !!opts.showComparison;
    const showHeaderTotal = !!opts.showHeaderTotal;
    const allowCrop = !!opts.allowCrop;
    const leftFooter = !!opts.leftFooter;   // a host-fillable row under the reference (e.g. action bar)
    const externalTotalEl = opts.externalTotalEl ? resolve(opts.externalTotalEl) : null;
    const footerTag = leftFooter ? '<div class="ce-leftfooter" style="margin-top:12px;"></div>' : '';

    let st = blank(), model = null, orig = null, hl = null, locked = false;
    let conf = null, showConf = false;   // per-cell {p,thr} from a live prediction; toggled display
    let imgSrc = 'processed', imgs = { processed: null, original: null, imageId: null };
    const crop = { active: false, p1: null, p2: null, nW: 0, nH: 0 };

    // ---- build DOM ----
    function refBlock(extraStyle) {
      return `<div class="ce-refblock" style="${extraStyle}">
          <div style="font-size:.78em; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
            Reference — element locations <span style="color:#667eea;">(click an E-cell to highlight)</span>
          </div>
          <canvas class="ce-ref" width="568" height="274" style="width:100%; ${showImage ? '' : 'max-width:420px;'} border:1px solid #ddd; border-radius:8px; background:#fff;"></canvas>
        </div>`;
    }
    const imageCol = `<div class="ce-left" style="flex:1; min-width:300px;">
        <div class="ce-imgtabs" style="display:none; gap:6px; align-items:center; margin-bottom:6px;">
          <button type="button" class="ce-srcbtn" data-src="processed">Processed</button>
          <button type="button" class="ce-srcbtn" data-src="original">Original</button>
          <span class="ce-res" style="margin-left:auto; color:#888; font-size:.8em;"></span>
          <button type="button" class="ce-cropbtn" style="display:none; margin-left:10px;">✂️ Crop</button>
        </div>
        <div class="ce-imgwrap" style="position:relative; line-height:0;">
          <img class="ce-img" style="width:100%; border:1px solid #ddd; border-radius:8px; background:#fff;">
          <canvas class="ce-imgov" width="568" height="274" style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none;"></canvas>
          <canvas class="ce-crop" style="position:absolute; top:0; left:0; display:none; cursor:crosshair;"></canvas>
        </div>
        <div class="ce-cropctl" style="display:none; margin-top:8px; padding:8px 10px; background:#e3f2fd; border-radius:6px; font-size:.88em;">
          <span style="color:#1565c0; font-weight:600;">📍 Click two points to define the crop area.</span>
          <div style="display:flex; gap:8px; margin-top:8px;">
            <button type="button" class="ce-cropapply" disabled>✓ Apply crop &amp; re-optimize</button>
            <button type="button" class="ce-cropcancel">✕ Cancel</button>
          </div>
        </div>
        ${showReference ? refBlock('margin-top:12px;') : ''}
        ${footerTag}
      </div>`;
    const leftCol = showImage ? imageCol
      : (showReference ? `<div class="ce-left" style="flex:0 0 320px; max-width:100%;">${refBlock('')}${footerTag}</div>` : '');
    root.innerHTML =
      `<div class="ce-root" style="display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start;">
         ${leftCol}
         <div class="ce-right" style="flex:1; min-width:340px;">
           ${showScores ? `<div class="ce-scores" style="margin-bottom:10px; font-size:.95em;"></div>` : ''}
           <div class="ce-lockbar" style="display:none; margin-bottom:6px; font-size:.85em; color:#b8860b;">🔒 validated — read-only</div>
           <div class="ce-grid"></div>
           <div class="ce-placeholder" style="display:none;"></div>
         </div>
       </div>`;

    const imgEl = root.querySelector('.ce-img');
    const ovEl = root.querySelector('.ce-imgov');
    const refEl = root.querySelector('.ce-ref');
    const scoresEl = root.querySelector('.ce-scores');
    const gridEl = root.querySelector('.ce-grid');
    const lockBar = root.querySelector('.ce-lockbar');
    const placeholderEl = root.querySelector('.ce-placeholder');
    const leftFooterEl = root.querySelector('.ce-leftfooter');
    const tabsEl = root.querySelector('.ce-imgtabs');
    const cropBtn = root.querySelector('.ce-cropbtn');
    const cropCanvas = root.querySelector('.ce-crop');
    const resEl = root.querySelector('.ce-res');
    const cropCtl = root.querySelector('.ce-cropctl');
    const cropApply = root.querySelector('.ce-cropapply');

    // ---- grid (event-delegated so multiple instances coexist) ----
    function render() {
      let rows = '', contra = 0, chg = 0;
      const cell = (e, k) => {
        const c = st[k][e];
        const m = (showComparison && model) ? model[k][e] : null;
        const o = (showComparison && orig) ? orig[k][e] : null;
        const contradict = m != null && m !== c, changed = o != null && o !== c;
        let badges = '';
        if (showComparison) {
          const mb = m == null ? '' : `<span style="color:${contradict ? '#e67e22' : '#cbcbcb'};font-weight:${contradict ? 700 : 400}">M${m ? '✓' : '✗'}</span>`;
          const sb = changed ? ` <span style="color:#2c7be5;font-weight:700">was${o ? '✓' : '✗'}</span>` : '';
          badges = `<div style="font-size:.6em;line-height:1.1;height:.9em;">${mb}${sb}</div>`;
        }
        // confidence overlay (live, when a prediction's probabilities are loaded + toggled on)
        let tip = '', ring = '';
        if (showConf && conf && conf[k] && conf[k][e]) {
          const cc = conf[k][e], dc = decisionConf(cc.p, cc.thr), unc = dc < UNCERTAIN_FRAC;
          tip = ` data-tip="${KEYS_LABEL[k]} · p=${cc.p.toFixed(2)} · thr=${cc.thr.toFixed(2)} → ${cc.p >= cc.thr ? '✓ present' : '✗ absent'} · Konfidenz ${Math.round(dc * 100)}%${unc ? '  (unsicher)' : ''}"`;
          if (unc) ring = 'box-shadow:0 0 0 2px #f1c40f inset;';
        }
        return `<td data-e="${e}" data-k="${k}"${tip} style="cursor:${locked ? 'default' : 'pointer'};text-align:center;padding:2px;">
          <div style="background:${c ? '#2e7d32' : '#bdbdbd'};color:#fff;border-radius:4px;padding:${showComparison ? '2px 0' : '3px 0'};border:${contradict ? '2px solid #e67e22' : '2px solid transparent'};${ring}${locked ? 'opacity:.85;' : ''}">
            <div style="font-size:.85em;font-weight:700;">${c ? '✓' : '✗'}</div>${badges}
          </div></td>`;
      };
      for (const e of ELS) {
        if (showComparison) for (const k of KEYS) {
          if (model && model[k][e] !== st[k][e]) contra++;
          if (orig && orig[k][e] !== st[k][e]) chg++;
        }
        const sub = st.presence[e] + st.accuracy[e] + st.position[e];
        const elc = `<td data-el="${e}" title="highlight on reference"
          style="cursor:pointer;padding:3px 5px;font-weight:700;color:#fff;text-align:center;background:${elColor(e)};${hl === e ? 'outline:3px solid #111;outline-offset:-1px;' : ''}">E${String(e + 1).padStart(2, '0')}</td>`;
        rows += `<tr>${elc}${cell(e, 'presence')}${cell(e, 'accuracy')}${cell(e, 'position')}<td style="text-align:center;font-weight:700;">${sub}</td></tr>`;
      }
      let head = '';
      if (showHeaderTotal) {
        head =
          `<div style="font-size:1.05em;margin-bottom:4px;">Total_Score: <b>${total()}</b>/60` +
          (showComparison ? `&nbsp;·&nbsp;<span style="color:#e67e22;font-size:.85em;">${contra} differ from model</span>` +
            (orig ? `&nbsp;·&nbsp;<span style="color:#2c7be5;font-size:.85em;">${chg} changed from saved</span>` : '') : '') +
          `</div>`;
        if (showComparison) head +=
          `<div style="font-size:.72em;color:#888;margin-bottom:6px;">green ✓ present / grey ✗ absent · <span style="color:#e67e22;">orange = disagrees with model (M)</span> · <span style="color:#2c7be5;">blue = changed from saved</span> · click E-cell → highlight on reference</div>`;
      }
      // confidence toggle + summary (independent of showHeaderTotal); only when conf data is loaded
      let confBar = '';
      if (conf) {
        let n = 0, mn = null, mnLbl = '';
        for (const e of ELS) for (const k of KEYS) {
          const cc = conf[k] && conf[k][e]; if (!cc) continue;
          const dc = decisionConf(cc.p, cc.thr);
          if (dc < UNCERTAIN_FRAC) n++;
          if (mn == null || dc < mn) { mn = dc; mnLbl = `E${String(e + 1).padStart(2, '0')}-${KEYS_LABEL[k]}`; }
        }
        const badge = n === 0 ? '🟢' : n <= 3 ? '🟡' : '🔴';
        confBar = `<div style="font-size:.8em;margin-bottom:6px;">` +
          `<button type="button" data-ce-conftoggle style="padding:2px 9px;border:1px solid #ccc;border-radius:5px;background:${showConf ? '#fff3cd' : '#f1f3f5'};cursor:pointer;font-weight:600;">🎯 Konfidenz ${showConf ? '▾' : '▸'}</button>` +
          (showConf ? `&nbsp;<span style="color:#555;">${badge} <b>${n}</b>/60 unsicher (Entscheidungs-Konfidenz &lt;${Math.round(UNCERTAIN_FRAC * 100)}%) · min ${mn == null ? '—' : Math.round(mn * 100) + '%'} @ ${mnLbl} · <span style="color:#b8860b;">gelber Ring = Wackelkandidat (hover: p/thr/Konfidenz)</span></span>` : '') +
          `</div>`;
      }
      gridEl.innerHTML = head + confBar +
        `<table style="width:100%;border-collapse:collapse;font-size:.9em;">
           <thead><tr style="background:#f8f9fa;">
             <th data-tip="${ASPECT_HELP.el}" style="cursor:help;">El</th>
             <th data-tip="${ASPECT_HELP.presence}" style="cursor:help;text-decoration:underline dotted;">Presence</th>
             <th data-tip="${ASPECT_HELP.accuracy}" style="cursor:help;text-decoration:underline dotted;">Accuracy</th>
             <th data-tip="${ASPECT_HELP.position}" style="cursor:help;text-decoration:underline dotted;">Position</th>
             <th data-tip="${ASPECT_HELP.sum}" style="cursor:help;">Σ</th>
           </tr></thead>
           <tbody>${rows}</tbody></table>`;
      if (externalTotalEl) externalTotalEl.textContent = total();
    }

    gridEl.addEventListener('click', ev => {
      const ct = ev.target.closest('[data-ce-conftoggle]');
      if (ct) { showConf = !showConf; render(); return; }      // confidence overlay toggle
      const elCell = ev.target.closest('[data-el]');
      if (elCell) { highlight(+elCell.dataset.el); return; }   // highlight always allowed
      if (locked) return;
      const cell = ev.target.closest('[data-e][data-k]');
      if (cell) toggle(+cell.dataset.e, cell.dataset.k);
    });

    // ---- image source toggle + crop ----
    // show the displayed image's natural resolution; updates on Processed/Original switch
    function updateRes() {
      if (!resEl) return;
      resEl.textContent = (imgEl && imgEl.naturalWidth) ? `${imgEl.naturalWidth}×${imgEl.naturalHeight}px` : '';
    }
    function showSource(which) {
      imgSrc = which;
      if (resEl) resEl.textContent = '…';
      if (imgEl) imgEl.src = imgs[which] || '';
      // element overlay only aligns with the 568×274 processed frame
      ovEl.style.display = (which === 'processed') ? 'block' : 'none';
      tabsEl.querySelectorAll('.ce-srcbtn').forEach(b => {
        const on = b.dataset.src === which;
        b.style.background = on ? '#667eea' : '#e9ecef';
        b.style.color = on ? '#fff' : '#444';
      });
      // crop only on the original
      cropBtn.style.display = (allowCrop && which === 'original' && imgs.original) ? 'inline-block' : 'none';
      if (which !== 'original') cancelCrop();
    }
    if (showImage) {
      imgEl.addEventListener('load', updateRes);
      tabsEl.querySelectorAll('.ce-srcbtn').forEach(b => {
        b.style.cssText = 'padding:5px 12px; border:none; border-radius:6px; cursor:pointer; font-size:.85em; font-weight:600;';
        b.addEventListener('click', () => showSource(b.dataset.src));
      });
      [cropBtn, cropApply, root.querySelector('.ce-cropcancel')].forEach(b => {
        b.style.cssText += ';padding:5px 12px; border:none; border-radius:6px; cursor:pointer; font-size:.85em; font-weight:600;';
      });
      cropBtn.style.background = '#17a2b8'; cropBtn.style.color = '#fff';
      cropApply.style.background = '#28a745'; cropApply.style.color = '#fff';
      cropBtn.addEventListener('click', startCrop);
      cropApply.addEventListener('click', applyCrop);
      root.querySelector('.ce-cropcancel').addEventListener('click', cancelCrop);
    }

    function startCrop() {
      if (!imgEl.complete) { imgEl.onload = startCrop; return; }
      crop.active = true; crop.p1 = crop.p2 = null;
      crop.nW = imgEl.naturalWidth; crop.nH = imgEl.naturalHeight;
      cropCanvas.width = imgEl.offsetWidth; cropCanvas.height = imgEl.offsetHeight;
      cropCanvas.style.width = imgEl.offsetWidth + 'px'; cropCanvas.style.height = imgEl.offsetHeight + 'px';
      cropCanvas.style.display = 'block';
      ovEl.style.display = 'none';
      cropCtl.style.display = 'block'; cropBtn.style.display = 'none'; cropApply.disabled = true;
      drawCrop();
    }
    function cropPt(e) {
      const r = cropCanvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }
    if (cropCanvas) {
      cropCanvas.addEventListener('click', e => {
        if (!crop.active) return;
        if (!crop.p1) { crop.p1 = cropPt(e); }
        else if (!crop.p2) { crop.p2 = cropPt(e); cropApply.disabled = false; }
        else { crop.p1 = cropPt(e); crop.p2 = null; cropApply.disabled = true; }
        drawCrop();
      });
      cropCanvas.addEventListener('mousemove', e => {
        if (crop.active && crop.p1 && !crop.p2) drawCrop(cropPt(e));
      });
    }
    function drawCrop(temp) {
      const x = cropCanvas.getContext('2d');
      x.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
      x.fillStyle = 'rgba(0,0,0,.3)'; x.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
      if (!crop.p1) return;
      const p2 = crop.p2 || temp;
      x.fillStyle = '#ff5722'; x.beginPath(); x.arc(crop.p1.x, crop.p1.y, 6, 0, 7); x.fill();
      if (!p2) return;
      const x1 = Math.min(crop.p1.x, p2.x), y1 = Math.min(crop.p1.y, p2.y);
      const w = Math.abs(p2.x - crop.p1.x), h = Math.abs(p2.y - crop.p1.y);
      x.clearRect(x1, y1, w, h);
      x.strokeStyle = '#ff5722'; x.lineWidth = 2; x.setLineDash([5, 5]); x.strokeRect(x1, y1, w, h); x.setLineDash([]);
      x.beginPath(); x.arc(p2.x, p2.y, 6, 0, 7); x.fill();
      const sX = crop.nW / cropCanvas.width, sY = crop.nH / cropCanvas.height;
      x.font = 'bold 14px sans-serif';
      x.fillText(`${Math.round(w * sX)} × ${Math.round(h * sY)} px`, x1 + 5, Math.max(14, y1 - 8));
    }
    function cancelCrop() {
      crop.active = false; crop.p1 = crop.p2 = null;
      if (cropCanvas) cropCanvas.style.display = 'none';
      if (cropCtl) cropCtl.style.display = 'none';
      if (cropBtn && allowCrop && imgSrc === 'original' && imgs.original) cropBtn.style.display = 'inline-block';
    }
    async function applyCrop() {
      if (!crop.p1 || !crop.p2 || !opts.onCrop) return;
      const sX = crop.nW / cropCanvas.width, sY = crop.nH / cropCanvas.height;
      const box = {
        x1: Math.round(Math.min(crop.p1.x, crop.p2.x) * sX),
        y1: Math.round(Math.min(crop.p1.y, crop.p2.y) * sY),
        x2: Math.round(Math.max(crop.p1.x, crop.p2.x) * sX),
        y2: Math.round(Math.max(crop.p1.y, crop.p2.y) * sY),
      };
      cropApply.disabled = true; cropApply.textContent = '⏳ Processing…';
      try {
        await opts.onCrop(box);   // host runs crop-and-reprocess, then should call setImages() again
      } finally {
        cropApply.textContent = '✓ Apply crop & re-optimize';
        cancelCrop();
      }
    }

    // ---- API ----
    function total() { return ELS.reduce((s, e) => s + st.presence[e] + st.accuracy[e] + st.position[e], 0); }
    function setState(comp) { st = clone(comp); render(); }
    function getState() { return clone(st); }
    function setComparison(c) { c = c || {}; model = c.model ? clone(c.model) : null; orig = c.orig ? clone(c.orig) : null; render(); }
    // per-sub-label confidence from a LIVE prediction: elements = prediction.elements
    // (each {element, presence, accuracy, position, thr_presence, thr_accuracy, thr_position}).
    // Stores prob+threshold per cell so render() can show the margin (|p-thr|) on demand.
    function setConfidence(elements) {
      if (!elements || !elements.length) { conf = null; showConf = false; render(); return; }
      const c = { presence: Array(20).fill(null), accuracy: Array(20).fill(null), position: Array(20).fill(null) };
      elements.forEach(el => {
        const e = el.element - 1; if (e < 0 || e > 19) return;
        c.presence[e] = { p: +el.presence, thr: +el.thr_presence };
        c.accuracy[e] = { p: +el.accuracy, thr: +el.thr_accuracy };
        c.position[e] = { p: +el.position, thr: +el.thr_position };
      });
      conf = c; render();
    }
    function setScores(html) { if (scoresEl) scoresEl.innerHTML = html || ''; }
    function setLeftFooter(html) { if (leftFooterEl) leftFooterEl.innerHTML = html || ''; }
    function setLocked(on, hint) {
      locked = !!on;
      if (lockBar) {
        if (hint != null) lockBar.textContent = hint;
        lockBar.style.display = (locked && lockBar.textContent && gridEl.style.display !== 'none') ? 'block' : 'none';
      }
      render();
    }
    // hide the grid (e.g. an image with no component labels) and show a host-supplied prompt instead
    function setGridVisible(visible, html) {
      if (gridEl) gridEl.style.display = visible ? 'block' : 'none';
      if (placeholderEl) {
        placeholderEl.style.display = visible ? 'none' : 'block';
        if (!visible && html != null) placeholderEl.innerHTML = html;
      }
      if (lockBar) lockBar.style.display = (visible && locked) ? 'block' : 'none';
    }
    function setImage(src) {                       // single processed image, no toggle/crop
      imgs = { processed: src, original: null, imageId: null };
      if (tabsEl) tabsEl.style.display = 'none';
      if (imgEl) imgEl.src = src || '';
      if (ovEl) ovEl.style.display = 'block';
      imgSrc = 'processed';
    }
    function setImages(o) {                         // {processed, original, imageId}
      o = o || {};
      imgs = { processed: o.processed || null, original: o.original || null, imageId: o.imageId ?? null };
      cancelCrop();
      if (tabsEl) tabsEl.style.display = imgs.original ? 'flex' : 'none';
      showSource(imgs.processed ? 'processed' : 'original');
    }
    function toggle(e, k) { if (locked) return; st[k][e] = st[k][e] ? 0 : 1; render(); if (opts.onChange) opts.onChange(getState()); }
    function highlight(e) {
      hl = (hl === e ? null : e);
      drawReference(refEl, hl);
      if (imgSrc === 'processed') drawImgOverlay(ovEl, hl);
      render();
    }
    function refresh() { drawReference(refEl, hl); if (imgSrc === 'processed') drawImgOverlay(ovEl, hl); render(); }

    render();
    if (showReference || showImage) loadRefAssets().then(() => { drawReference(refEl, hl); if (imgSrc === 'processed') drawImgOverlay(ovEl, hl); });

    return { setImage, setImages, setState, getState, setComparison, setConfidence, setScores, setLeftFooter, setLocked, setGridVisible, total, highlight, toggle, refresh };
  }

  return { create, loadRefAssets, elColor };
})();
