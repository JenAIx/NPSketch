/*
 * Renders the component-model prediction (60 OCS-Plus sub-labels) as a
 * 20-element x 3-aspect breakdown. Used by draw_testimage.html and upload.html
 * when predict-single returns training_mode === "components".
 *
 * prediction = {
 *   total_score_hard: int, total_score_soft: float,
 *   elements: [ {element:int, presence:float, accuracy:float, position:float, subscore_hard:int}, ... 20 ]
 * }
 * Returns an HTML string.
 */
function renderComponentBreakdown(prediction) {
  const els = (prediction && prediction.elements) || [];
  const hard = prediction.total_score_hard;
  const soft = prediction.total_score_soft;

  // probability -> red->yellow->green background
  const colorFor = (p) => `hsl(${Math.round(p * 120)}, 65%, 45%)`;
  const cell = (p) => {
    const pct = Math.round(p * 100);
    const tick = p >= 0.5 ? '✓' : '✗';
    return `<td style="padding:4px;text-align:center;">
      <div style="background:${colorFor(p)};color:#fff;border-radius:4px;padding:3px 0;font-size:.8em;font-weight:600;">
        ${tick} ${pct}%
      </div></td>`;
  };

  // per-aspect summary (count >= 0.5 of 20)
  const cnt = (key) => els.filter(e => e[key] >= 0.5).length;
  const summary = `
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin:10px 0;color:#444;font-size:.9em;">
      <span><strong>Presence:</strong> ${cnt('presence')}/20</span>
      <span><strong>Accuracy:</strong> ${cnt('accuracy')}/20</span>
      <span><strong>Position:</strong> ${cnt('position')}/20</span>
    </div>`;

  const rows = els.map(e => `
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:4px 8px;font-weight:600;color:#555;">E${String(e.element).padStart(2,'0')}</td>
      ${cell(e.presence)}${cell(e.accuracy)}${cell(e.position)}
      <td style="padding:4px;text-align:center;font-weight:700;border-left:2px solid #eee;">${e.subscore_hard}/3</td>
    </tr>`).join('');

  return `
    <div style="text-align:center;margin:10px 0 4px;">
      <div style="font-size:3em;font-weight:bold;color:#667eea;line-height:1;">${hard}<span style="font-size:.4em;color:#999;">/60</span></div>
      <div style="color:#666;font-size:.9em;">Derived Total_Score (hard sum of 60 sub-labels)</div>
      <div style="color:#999;font-size:.85em;">soft sum: ${soft}</div>
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
      Each cell = model probability that the sub-label is satisfied (✓ at ≥50%). Element score = sum of
      its three hard decisions (0–3). Element names are generic (no OCS-Plus legend available yet).
    </p>`;
}

// expose globally for inline page scripts
window.renderComponentBreakdown = renderComponentBreakdown;
