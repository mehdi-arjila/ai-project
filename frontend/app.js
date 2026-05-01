/**
 * app.js — Main Application Logic
 * Handles: drawing canvas, API communication, UI updates,
 *          training controls, plot modal, and toast notifications.
 */

(function () {
  'use strict';

  const API = 'http://localhost:5000';
  const CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

  // ─── DOM refs ──────────────────────────────────────────────────
  const drawCanvas  = document.getElementById('draw-canvas');
  const ctx         = drawCanvas.getContext('2d');
  const btnClear    = document.getElementById('btn-clear');
  const btnPredict  = document.getElementById('btn-predict');
  const brushSlider = document.getElementById('brush-size');
  const predChar    = document.getElementById('pred-char');
  const predBar     = document.getElementById('pred-bar');
  const predConf    = document.getElementById('pred-conf');
  const top3Grid    = document.getElementById('top3-grid');
  const scoresChart = document.getElementById('scores-chart');
  const correctInput= document.getElementById('correct-label');
  const btnSave     = document.getElementById('btn-save');
  const saveFeedback= document.getElementById('save-feedback');
  const epochInput  = document.getElementById('train-epochs');
  const btnTrain    = document.getElementById('btn-train');
  const trainStatus = document.getElementById('train-status');
  const btnCurves   = document.getElementById('btn-show-curves');
  const btnCM       = document.getElementById('btn-show-cm');
  const statusDot   = document.getElementById('status-dot');
  const statusText  = document.getElementById('status-text');
  const inferOvl    = document.getElementById('inference-overlay');
  const plotModal   = document.getElementById('plot-modal');
  const modalImg    = document.getElementById('modal-img');
  const modalTitle  = document.getElementById('modal-title');
  const modalClose  = document.getElementById('modal-close');
  const modalBd     = document.getElementById('modal-backdrop');
  const btnRotate   = document.getElementById('btn-rotate');
  const btnResetCam = document.getElementById('btn-reset-cam');

  // Layer activation bar elements
  const lsBars = {
    input:        document.getElementById('ls-input'),
    conv1:        document.getElementById('ls-conv1'),
    conv2:        document.getElementById('ls-conv2'),
    dense1:       document.getElementById('ls-dense'),
    output_layer: document.getElementById('ls-output'),
  };

  // ─── Canvas Drawing State ─────────────────────────────────────
  let isDrawing   = false;
  let lastX = 0, lastY = 0;
  let hasContent  = false;
  let brushSize   = 16;
  let trainPollId = null;

  // ─── Canvas Setup ─────────────────────────────────────────────
  function initCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    ctx.strokeStyle = '#fff';
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.lineWidth   = brushSize;
  }

  function getCanvasPos(e) {
    const rect  = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width  / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    const src    = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - rect.left) * scaleX,
      y: (src.clientY - rect.top)  * scaleY,
    };
  }

  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getCanvasPos(e);
    lastX = x; lastY = y;
    drawCanvas.classList.add('drawing');

    // Dot on click
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    hasContent = true;
  }

  function doDraw(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const { x, y } = getCanvasPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = brushSize;
    ctx.stroke();
    lastX = x; lastY = y;
    hasContent = true;

    // Live pixel preview in 3D
    throttledPixelUpdate();
  }

  function endDraw() {
    isDrawing = false;
    drawCanvas.classList.remove('drawing');
  }

  // ─── Throttled pixel update for 3D viz ───────────────────────
  let pixelThrottle = null;
  function throttledPixelUpdate() {
    if (pixelThrottle) return;
    pixelThrottle = setTimeout(() => {
      pixelThrottle = null;
      sendPixelsToViz();
    }, 60);
  }

  function sendPixelsToViz() {
    if (!window.NeuralViz) return;
    const imgData = ctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
    const pixels  = [];
    for (let i = 0; i < imgData.data.length; i += 4 * 4) {
      pixels.push(imgData.data[i] / 255);
    }
    window.NeuralViz.setInputPixels(pixels);
  }

  // ─── Register draw events ─────────────────────────────────────
  drawCanvas.addEventListener('mousedown',  startDraw);
  drawCanvas.addEventListener('mousemove',  doDraw);
  drawCanvas.addEventListener('mouseup',    endDraw);
  drawCanvas.addEventListener('mouseleave', endDraw);
  drawCanvas.addEventListener('touchstart', startDraw, { passive: false });
  drawCanvas.addEventListener('touchmove',  doDraw,    { passive: false });
  drawCanvas.addEventListener('touchend',   endDraw);

  brushSlider.addEventListener('input', () => {
    brushSize    = parseInt(brushSlider.value);
    ctx.lineWidth = brushSize;
  });

  // ─── Clear ────────────────────────────────────────────────────
  btnClear.addEventListener('click', () => {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    hasContent = false;
    resetPrediction();
  });

  function resetPrediction() {
    predChar.textContent = '?';
    predBar.style.width  = '0%';
    predConf.textContent = '—';
    renderTop3([]);
    renderScores([]);
    resetLayerBars();
  }

  // ─── Predict ──────────────────────────────────────────────────
  btnPredict.addEventListener('click', runPredict);

  async function runPredict() {
    if (!hasContent) { showToast('Draw something first!', 'info'); return; }

    setUIBusy(true);
    showInferenceOverlay(true);

    try {
      const imageData = drawCanvas.toDataURL('image/png');
      const response  = await fetch(`${API}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image: imageData }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const result = await response.json();

      // Update UI
      showPrediction(result);

      // Trigger 3D animation
      if (window.NeuralViz) {
        window.NeuralViz.triggerInference(result.all_scores || []);
      }

      // Update layer activation bars
      if (result.activations) updateLayerBars(result.activations);

    } catch (err) {
      showToast(`Prediction failed: ${err.message}`, 'error');
      console.error(err);
    } finally {
      setUIBusy(false);
      showInferenceOverlay(false);
    }
  }

  // ─── Show prediction results ──────────────────────────────────
  function showPrediction(result) {
    // Animate main character
    predChar.classList.remove('pop');
    void predChar.offsetWidth; // reflow
    predChar.textContent = result.predicted || '?';
    predChar.classList.add('pop');

    // Confidence bar
    const pct = Math.round((result.confidence || 0) * 100);
    predBar.style.width  = `${pct}%`;
    predConf.textContent = `${pct}% confidence`;

    // Top 3
    renderTop3(result.top3 || []);

    // All scores
    renderScores(result.all_scores || []);

    // Pre-fill correction label
    if (result.predicted) correctInput.value = result.predicted;
  }

  function renderTop3(top3) {
    if (!top3.length) {
      top3Grid.innerHTML = '<div class="top3-item placeholder">–</div><div class="top3-item placeholder">–</div><div class="top3-item placeholder">–</div>';
      return;
    }
    top3Grid.innerHTML = top3.map((item, i) => `
      <div class="top3-item ${i === 0 ? 'active' : ''}">
        <span class="top3-char">${item.label}</span>
        <span class="top3-conf">${(item.confidence * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  }

  function renderScores(scores) {
    if (!scores.length) {
      scoresChart.innerHTML = '<div class="scores-placeholder">Draw a character and click Predict</div>';
      return;
    }
    const maxScore = Math.max(...scores, 0.001);
    const topIdx   = scores.indexOf(maxScore);

    scoresChart.innerHTML = scores.map((s, i) => `
      <div class="score-row">
        <span class="score-label">${CLASSES[i] || i}</span>
        <div class="score-bar-wrap">
          <div class="score-bar ${i === topIdx ? 'top' : ''}" style="width:${(s / maxScore * 100).toFixed(1)}%"></div>
        </div>
        <span class="score-val">${(s * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  }

  function updateLayerBars(activations) {
    Object.entries(lsBars).forEach(([key, el]) => {
      if (!el) return;
      const act = activations[key];
      if (!act) return;
      const pct = Math.min(100, act.mean * 200);
      el.style.width = `${pct.toFixed(1)}%`;
    });
  }

  function resetLayerBars() {
    Object.values(lsBars).forEach(el => { if (el) el.style.width = '0%'; });
  }

  // ─── Save / Correct ───────────────────────────────────────────
  btnSave.addEventListener('click', async () => {
    const label = correctInput.value.trim().toUpperCase();
    if (!label || label.length !== 1) {
      saveFeedback.textContent = '⚠ Enter a single character label';
      saveFeedback.style.color = 'var(--red)';
      return;
    }
    if (!hasContent) {
      saveFeedback.textContent = '⚠ Draw something first';
      saveFeedback.style.color = 'var(--red)';
      return;
    }
    try {
      const imageData = drawCanvas.toDataURL('image/png');
      const res = await fetch(`${API}/save_drawing`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image: imageData, label }),
      });
      const data = await res.json();
      saveFeedback.textContent = `✓ Saved as "${label}"`;
      saveFeedback.style.color = 'var(--green)';
      showToast(`Drawing saved as "${label}"`, 'success');
    } catch (err) {
      saveFeedback.textContent = `✕ Save failed`;
      saveFeedback.style.color = 'var(--red)';
    }
  });

  // ─── Training ─────────────────────────────────────────────────
  btnTrain.addEventListener('click', async () => {
    const epochs = parseInt(epochInput.value) || 15;
    btnTrain.disabled = true;
    trainStatus.textContent = 'Starting training...';

    try {
      const res = await fetch(`${API}/train`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ epochs, use_user_data: true }),
      });
      const data = await res.json();
      if (res.status === 409) {
        showToast('Training already in progress', 'info');
        btnTrain.disabled = false;
        return;
      }
      showToast('Training started!', 'info');
      trainPollId = setInterval(pollTrainStatus, 3000);
    } catch (err) {
      trainStatus.textContent = `Error: ${err.message}`;
      btnTrain.disabled = false;
    }
  });

  async function pollTrainStatus() {
    try {
      const res  = await fetch(`${API}/train_status`);
      const data = await res.json();
      trainStatus.textContent = data.message || '...';

      if (!data.is_training) {
        clearInterval(trainPollId);
        btnTrain.disabled = false;
        if (data.last_result) {
          const acc = (data.last_result.test_accuracy * 100).toFixed(2);
          showToast(`Training done! Accuracy: ${acc}%`, 'success');
          trainStatus.textContent = `✓ Done — ${acc}% test accuracy`;
        }
      }
    } catch (err) {
      console.warn('Poll error:', err);
    }
  }

  // ─── Plots ────────────────────────────────────────────────────
  btnCurves.addEventListener('click', () => openPlot('training_curves_latest.png', 'Training Accuracy & Loss'));
  btnCM.addEventListener('click',     () => openPlot('confusion_matrix_latest.png', 'Confusion Matrix'));

  function openPlot(filename, title) {
    modalTitle.textContent = title;
    modalImg.src = `${API}/plots/${filename}?t=${Date.now()}`;
    plotModal.classList.remove('hidden');
    modalImg.onerror = () => {
      showToast('No plot available yet. Train the model first.', 'info');
      plotModal.classList.add('hidden');
    };
  }

  modalClose.addEventListener('click', () => plotModal.classList.add('hidden'));
  modalBd.addEventListener('click',    () => plotModal.classList.add('hidden'));

  // ─── Viz Controls ─────────────────────────────────────────────
  btnRotate.addEventListener('click', () => {
    const on = window.NeuralViz?.toggleAutoRotate();
    btnRotate.classList.toggle('active', on);
  });

  btnResetCam.addEventListener('click', () => {
    window.NeuralViz?.resetCamera();
  });

  // ─── UI Helpers ───────────────────────────────────────────────
  function setUIBusy(busy) {
    btnPredict.disabled = busy;
    btnClear.disabled   = busy;
    statusDot.className = 'status-dot ' + (busy ? 'busy' : 'online');
    statusText.textContent = busy ? 'Running inference...' : 'Ready';
  }

  function showInferenceOverlay(show) {
    inferOvl.classList.toggle('hidden', !show);
  }

  // ─── Toast ────────────────────────────────────────────────────
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.animation = 'toast-out .3s ease forwards';
      setTimeout(() => toast.remove(), 350);
    }, 3000);
  }

  // ─── Health Check ─────────────────────────────────────────────
  async function checkHealth() {
    try {
      const res  = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      statusDot.className    = 'status-dot online';
      statusText.textContent = data.model_loaded ? 'Model ready' : 'No model — train first';
      if (!data.model_loaded) {
        showToast('No trained model found. Click "Train" to train the model.', 'info');
      }
    } catch {
      statusDot.className    = 'status-dot offline';
      statusText.textContent = 'Server offline';
      showToast('Cannot connect to server. Is main.py running?', 'error');
    }
  }

  // ─── Keyboard shortcut ────────────────────────────────────────
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter') runPredict();
    if (e.key === 'Escape' || e.key === 'Delete') {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
      hasContent = false;
      resetPrediction();
    }
  });

  // ─── Boot ────────────────────────────────────────────────────
  initCanvas();
  checkHealth();

  // Expose toast globally (used by three_scene if needed)
  window.showToast = showToast;

})();
