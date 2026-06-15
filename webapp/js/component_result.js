/*
 * Renders the component-model prediction (60 OCS-Plus sub-labels) as a
 * 20-element x 3-aspect breakdown. Used by evaluate.html and ai_training_data_view.html
 * when predict-single returns training_mode === "components".
 *
 * prediction = {
 *   total_score_hard: int, total_score_soft: float,
 *   elements: [ {element:int, presence:float, accuracy:float, position:float, subscore_hard:int}, ... 20 ]
 * }
 * opts (optional):
 *   isLabels: true -> values are ground-truth 0/1 (show ✓/✗ only, no %), used by the data-view
 *   title:    override the headline caption
 * Returns an HTML string.
 */
function renderComponentBreakdown(prediction, opts) {
  opts = opts || {};
  const isLabels = !!opts.isLabels;
  const els = (prediction && prediction.elements) || [];
  const hard = prediction.total_score_hard;
  const soft = prediction.total_score_soft;
  // Primary number: the calibrated derived score when available, else the hard sum.
  const primary = (prediction.total_score !== undefined && prediction.total_score !== null)
    ? prediction.total_score : hard;
  const calibrated = !!prediction.calibrated;

  // probability -> red->yellow->green background
  const colorFor = (p) => `hsl(${Math.round(p * 120)}, 65%, 45%)`;
  // tick uses the per-label decision threshold when provided (else 0.5)
  const cell = (p, thr) => {
    const t = (typeof thr === 'number') ? thr : 0.5;
    const tick = p >= t ? '✓' : '✗';
    const label = isLabels ? tick : `${tick} ${Math.round(p * 100)}%`;
    return `<td style="padding:4px;text-align:center;">
      <div style="background:${colorFor(p)};color:#fff;border-radius:4px;padding:3px 0;font-size:.8em;font-weight:600;">
        ${label}
      </div></td>`;
  };

  // per-aspect summary (count over the per-label threshold, of 20)
  const cnt = (key) => els.filter(e => e[key] >= (e['thr_' + key] ?? 0.5)).length;
  const summary = `
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin:10px 0;color:#444;font-size:.9em;">
      <span><strong>Presence:</strong> ${cnt('presence')}/20</span>
      <span><strong>Accuracy:</strong> ${cnt('accuracy')}/20</span>
      <span><strong>Position:</strong> ${cnt('position')}/20</span>
    </div>`;

  const rows = els.map(e => `
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:4px 8px;font-weight:600;color:#555;">E${String(e.element).padStart(2,'0')}</td>
      ${cell(e.presence, e.thr_presence)}${cell(e.accuracy, e.thr_accuracy)}${cell(e.position, e.thr_position)}
      <td style="padding:4px;text-align:center;font-weight:700;border-left:2px solid #eee;">${e.subscore_hard}/3</td>
    </tr>`).join('');

  return `
    <div style="text-align:center;margin:10px 0 4px;">
      <div style="font-size:3em;font-weight:bold;color:#667eea;line-height:1;">${primary}<span style="font-size:.4em;color:#999;">/60</span></div>
      <div style="color:#666;font-size:.9em;">Derived Total_Score${calibrated ? ' (calibrated)' : ''}</div>
      <div style="color:#999;font-size:.85em;">hard sum: ${hard} · soft sum: ${soft}</div>
    </div>
    ${summary}
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:.9em;">
        <thead>
          <tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">
            <th style="padding:6px 8px;text-align:left;">Element</th>
            <th style="padding:6px;text-align:center;">Presence</th>
            <th style="padding:6px;text-align:center;">Accuracy</th>
            <th style="padding:6px;text-align:center;">Position</th>
            <th style="padding:6px;text-align:center;border-left:2px solid #dee2e6;">Score</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
        <tfoot>
          <tr style="background:#f8f9fa;border-top:2px solid #dee2e6;font-weight:bold;">
            <td style="padding:6px 8px;">TOTAL</td>
            <td colspan="3" style="padding:6px;text-align:right;color:#666;">Σ sub-scores →</td>
            <td style="padding:6px;text-align:center;border-left:2px solid #dee2e6;">${hard}/60</td>
          </tr>
        </tfoot>
      </table>
    </div>
    <p style="color:#999;font-size:.8em;margin-top:8px;">
      Each cell = model probability that the sub-label is satisfied (✓ at the model's per-label
      threshold${calibrated ? '' : ', else ≥50%'}). Element score = sum of its three hard decisions (0–3).
      ${calibrated ? 'The big number is the calibrated derived Total_Score.' : ''}
      See the Component Map for which element is which.
    </p>`;
}

// expose globally for inline page scripts
window.renderComponentBreakdown = renderComponentBreakdown;
