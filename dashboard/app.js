/**
 * TMG-GAN Evaluation Console - Frontend Application Logic
 * Academic research interface for multi-model inspection and sample synthesis.
 */

const CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot'];
let allFlows = [];
let currentFlow = null;
let currentGenClassIdx = 5; // Default to Bot
let currentNoiseScale = 1.0;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initStatus();
  initFlowSelector();
  initGenerator();
  initConfusionExplorer();
});

/* ================= UTILITY: CALIBRATED CONFIDENCE FORMATTER ================= */
function formatConfidence(conf) {
  if (typeof conf !== 'number' || isNaN(conf)) return '--%';
  const pct = Math.min(Math.max(conf * 100, 0), 100);
  return `${pct.toFixed(1)}%`;
}

/* ================= 1. TAB NAVIGATION ================= */
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

/* ================= 2. SYSTEM STATUS ================= */
async function initStatus() {
  const statusPillText = document.getElementById('system-status-text');
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'ready') {
      const dev = data.device ? data.device.toUpperCase() : 'CPU';
      statusPillText.textContent = `Inference Device: ${dev} • 9 Models Loaded`;
    }
  } catch (err) {
    statusPillText.textContent = 'Server Online';
  }
}

/* ================= 3. TAB 1: FLOW SELECTOR & CLASSIFIER ================= */
async function initFlowSelector() {
  const classFilter = document.getElementById('flow-class-filter');
  const flowSelect = document.getElementById('flow-select');
  const btnClassify = document.getElementById('btn-run-classify');
  const btnRandom = document.getElementById('btn-pick-random-flow');

  try {
    const res = await fetch('/api/flows');
    const data = await res.json();
    allFlows = data.flows || [];
    renderFlowOptions('all');
  } catch (err) {
    console.error('Failed to load flows:', err);
  }

  classFilter.addEventListener('change', () => {
    renderFlowOptions(classFilter.value);
  });

  flowSelect.addEventListener('change', () => {
    const selectedId = flowSelect.value;
    currentFlow = allFlows.find(f => f.id === selectedId);
    renderFlowDetails(currentFlow);
  });

  btnClassify.addEventListener('click', () => {
    if (currentFlow) {
      runClassification(currentFlow);
    }
  });

  if (btnRandom) {
    btnRandom.addEventListener('click', async () => {
      btnRandom.disabled = true;
      const originalText = btnRandom.innerHTML;
      btnRandom.innerHTML = '<span>Sampling...</span>';
      try {
        const filterVal = classFilter.value === 'challenges' ? 'all' : classFilter.value;
        const res = await fetch(`/api/random_flow?class=${encodeURIComponent(filterVal)}`);
        const randomFlow = await res.json();
        if (randomFlow && !randomFlow.error) {
          if (!allFlows.some(f => f.id === randomFlow.id)) {
            allFlows.unshift(randomFlow);
            const opt = document.createElement('option');
            opt.value = randomFlow.id;
            opt.textContent = `[${randomFlow.class_name}] Held-out Test Flow - Port ${randomFlow.summary_features['Destination Port']}`;
            flowSelect.insertBefore(opt, flowSelect.firstChild);
          }
          flowSelect.value = randomFlow.id;
          currentFlow = randomFlow;
          renderFlowDetails(randomFlow);
          runClassification(randomFlow);
        }
      } catch (e) {
        console.error('Random sampling error:', e);
      } finally {
        btnRandom.innerHTML = originalText;
        btnRandom.disabled = false;
      }
    });
  }
}

function renderFlowOptions(filterClass) {
  const flowSelect = document.getElementById('flow-select');
  flowSelect.innerHTML = '';

  let filtered;
  if (filterClass === 'all') {
    filtered = allFlows;
  } else if (filterClass === 'challenges') {
    filtered = allFlows.filter(f => f.is_challenge);
  } else {
    filtered = allFlows.filter(f => f.class_name === filterClass);
  }

  filtered.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f.id;
    const prefix = f.is_challenge ? '[Discrepancy]' : `[${f.class_name}]`;
    opt.textContent = `${prefix} ${f.id} - Port ${f.summary_features['Destination Port']} (${f.summary_features['Total Fwd Packets']} pkts)`;
    flowSelect.appendChild(opt);
  });

  if (filtered.length > 0) {
    currentFlow = filtered[0];
    renderFlowDetails(currentFlow);
  }
}

function renderFlowDetails(flow) {
  if (!flow) return;

  // Ground truth badge
  const gtBadge = document.getElementById('flow-gt-badge');
  gtBadge.textContent = flow.class_name;
  gtBadge.className = `badge ${flow.class_name === 'Benign' ? 'badge-success' : 'badge-warning'}`;

  // Description
  document.getElementById('flow-desc-text').textContent = flow.description;

  // Key Telemetry
  const telemetryGrid = document.getElementById('telemetry-grid');
  telemetryGrid.innerHTML = '';

  for (const [key, val] of Object.entries(flow.summary_features)) {
    const item = document.createElement('div');
    item.className = 'telemetry-item';
    
    let formattedVal = val;
    if (typeof val === 'number') {
      formattedVal = Number.isInteger(val) ? val.toLocaleString() : val.toFixed(2);
    }

    item.innerHTML = `
      <span class="telemetry-label">${key}</span>
      <span class="telemetry-val">${formattedVal}</span>
    `;
    telemetryGrid.appendChild(item);
  }

  // Pre-fill model predictions if available
  if (flow.model_predictions) {
    updateModelCards(flow.model_predictions, flow.class_name);
  }
}

async function runClassification(flow) {
  const btn = document.getElementById('btn-run-classify');
  const originalHtml = btn.innerHTML;
  btn.innerHTML = '<span>Evaluating Models...</span>';
  btn.disabled = true;

  try {
    const res = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        features: flow.scaled_features,
        is_scaled: true
      })
    });
    const result = await res.json();
    updateModelCards(result.predictions, flow.class_name);
  } catch (err) {
    console.error('Classification error:', err);
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
  }
}

function updateModelCards(predictions, groundTruth) {
  const insightBanner = document.getElementById('insight-banner');
  const verdictBadge = document.getElementById('verdict-status-badge');

  const orig = predictions.original;
  const smote = predictions.smote;
  const tmg = predictions.tmg_gan;

  // Populate cards
  populateCard('orig', orig);
  populateCard('smote', smote);
  populateCard('tmg', tmg);

  // Check for false-positive discrepancy: SMOTE flagged normal traffic as attack, but TMG correctly called Benign
  const isSmoteFalseAlarm = (groundTruth === 'Benign' && smote.predicted_class !== 'Benign' && tmg.predicted_class === 'Benign');

  if (isSmoteFalseAlarm) {
    insightBanner.classList.remove('hidden');
    document.getElementById('insight-title').textContent = 'Decision Boundary Divergence: SMOTE False Positive';
    document.getElementById('insight-desc').textContent = 
      `Ground truth label is Benign. SMOTE overfitted synthetic minority boundaries into normal traffic space, misclassifying this flow as ${smote.predicted_class} (${formatConfidence(smote.confidence)}). ` +
      `TMG-GAN preserved legitimate decision boundaries and correctly classified it as Benign (${formatConfidence(tmg.confidence)}).`;
    
    verdictBadge.textContent = 'SMOTE False Alarm Confirmed';
    verdictBadge.className = 'verdict-badge diverge';
  } else if (smote.predicted_class === tmg.predicted_class && tmg.predicted_class === orig.predicted_class) {
    insightBanner.classList.add('hidden');
    verdictBadge.textContent = 'Classifiers in Agreement';
    verdictBadge.className = 'verdict-badge match';
  } else {
    insightBanner.classList.add('hidden');
    verdictBadge.textContent = 'Classifiers Diverge';
    verdictBadge.className = 'verdict-badge diverge';
  }
}

function populateCard(prefix, pred) {
  if (!pred) return;
  const lbl = document.getElementById(`${prefix}-pred-class`);
  const conf = document.getElementById(`${prefix}-pred-conf`);
  const tag = document.getElementById(`${prefix}-pred-tag`);
  const bars = document.getElementById(`${prefix}-prob-bars`);

  lbl.textContent = pred.predicted_class;
  conf.textContent = `Confidence: ${formatConfidence(pred.confidence)}`;
  
  tag.textContent = pred.is_attack ? 'Malicious Attack' : 'Normal Traffic';
  tag.className = `pred-tag ${pred.is_attack ? 'tag-attack' : 'tag-benign'}`;

  // Render probability distribution bars
  bars.innerHTML = '';
  if (pred.probabilities) {
    for (const [cls, p] of Object.entries(pred.probabilities)) {
      const pct = (p * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-name">${cls}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${pct}%"></div>
        </div>
        <span class="prob-val">${pct}%</span>
      `;
      bars.appendChild(row);
    }
  }
}

/* ================= 4. TAB 2: SYNTHETIC SAMPLE GENERATION ================= */
function initGenerator() {
  const classBtns = document.querySelectorAll('.class-choice-btn');
  const btnGen = document.getElementById('btn-generate-flow');
  const sliderNoise = document.getElementById('adv-noise-scale');
  const noiseValLabel = document.getElementById('adv-noise-val');

  classBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      classBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentGenClassIdx = parseInt(btn.getAttribute('data-class-idx'), 10);
      document.getElementById('gen-metric-target').textContent = `${CLASSES[currentGenClassIdx]} (Class ${currentGenClassIdx})`;
    });
  });

  if (sliderNoise && noiseValLabel) {
    sliderNoise.addEventListener('input', () => {
      currentNoiseScale = parseFloat(sliderNoise.value);
      const labelDesc = currentNoiseScale === 1.0 ? '(Standard Normal)' : (currentNoiseScale > 1.0 ? '(High Dispersion)' : '(Focused)');
      noiseValLabel.textContent = `${currentNoiseScale.toFixed(1)} \u00D7 ${labelDesc}`;
    });
  }

  btnGen.addEventListener('click', () => {
    runLiveGeneration(currentGenClassIdx);
  });

  // Run initial demo generation
  runLiveGeneration(5);
}

async function runLiveGeneration(classIdx) {
  const btn = document.getElementById('btn-generate-flow');
  const originalHtml = btn.innerHTML;
  btn.innerHTML = '<span>Synthesizing Sample...</span>';
  btn.disabled = true;

  try {
    const res = await fetch('/api/synthetic_test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        class_index: classIdx,
        noise_scale: currentNoiseScale
      })
    });
    const sample = await res.json();
    renderGeneratedSample(sample);
  } catch (err) {
    console.error('Generation error:', err);
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
  }
}

function renderGeneratedSample(sample) {
  if (!sample) return;

  const targetName = sample.target_class;
  const dScore = (sample.critic_score !== undefined ? sample.critic_score : sample.discriminator_score).toFixed(4);
  const passed = sample.filter_passed;

  document.getElementById('gen-metric-target').textContent = `${targetName} (Class ${sample.target_index})`;
  document.getElementById('gen-metric-dscore').textContent = dScore >= 0 ? `+${dScore}` : dScore;
  
  const realismElem = document.getElementById('gen-metric-realism');
  if (parseFloat(dScore) >= 0) {
    realismElem.textContent = 'Converged';
    realismElem.className = 'metric-val text-success';
  } else {
    realismElem.textContent = 'Sub-optimal';
    realismElem.className = 'metric-val text-warning';
  }

  document.getElementById('gen-metric-conf').textContent = formatConfidence(sample.filter_confidence);

  const filterBadge = document.getElementById('gen-filter-badge');
  const validationBox = document.getElementById('filter-validation-box');
  const valTitle = document.getElementById('filter-status-title');
  const valDesc = document.getElementById('filter-status-desc');

  if (passed) {
    filterBadge.textContent = 'Algorithm 2: Passed Filter';
    filterBadge.className = 'badge badge-success';
    validationBox.style.background = 'var(--color-success-bg)';
    validationBox.style.borderColor = 'var(--color-success-border)';
    valTitle.textContent = 'Sample Accepted by Algorithm 2 Quality Gate';
    valDesc.textContent = `Tied classifier C(x) verified class fidelity for ${targetName} (Critic score D(x) = ${dScore}). Sample qualifies for augmented training pool.`;
  } else {
    filterBadge.textContent = 'Algorithm 2: Rejected (Resampled)';
    filterBadge.className = 'badge badge-warning';
    validationBox.style.background = 'var(--color-warning-bg)';
    validationBox.style.borderColor = 'var(--color-warning-border)';
    valTitle.textContent = 'Sample Dropped by Algorithm 2 Quality Gate';
    valDesc.textContent = `Candidate flow predicted as ${sample.filter_predicted_class} instead of ${targetName}. Algorithm 2 drops ambiguous boundary samples to protect benign precision.`;
  }

  // Populate 3-Classifier Prediction Cards
  const downstream = sample.downstream_predictions || {};
  
  // 1. Original Model
  const orig = downstream.original;
  if (orig) {
    document.getElementById('adv-orig-pred').textContent = orig.predicted_class;
    document.getElementById('adv-orig-conf').textContent = `Confidence: ${formatConfidence(orig.confidence)}`;
    const origTag = document.getElementById('adv-orig-tag');
    if (orig.predicted_class === 'Benign') {
      origTag.textContent = 'Misclassified as Benign';
      origTag.className = 'adv-tag tag-evaded';
    } else {
      origTag.textContent = `Classified as ${orig.predicted_class}`;
      origTag.className = 'adv-tag tag-detected';
    }
  }

  // 2. SMOTE Model
  const smote = downstream.smote;
  if (smote) {
    document.getElementById('adv-smote-pred').textContent = smote.predicted_class;
    document.getElementById('adv-smote-conf').textContent = `Confidence: ${formatConfidence(smote.confidence)}`;
    const smoteTag = document.getElementById('adv-smote-tag');
    if (smote.predicted_class === 'Benign') {
      smoteTag.textContent = 'Misclassified as Benign';
      smoteTag.className = 'adv-tag tag-evaded';
    } else {
      smoteTag.textContent = `Classified as ${smote.predicted_class}`;
      smoteTag.className = 'adv-tag tag-detected';
    }
  }

  // 3. TMG-GAN Model
  const tmg = downstream.tmg_gan;
  if (tmg) {
    document.getElementById('adv-tmg-pred').textContent = tmg.predicted_class;
    document.getElementById('adv-tmg-conf').textContent = `Confidence: ${formatConfidence(tmg.confidence)}`;
    const tmgTag = document.getElementById('adv-tmg-tag');
    if (tmg.predicted_class === targetName) {
      tmgTag.textContent = `Correctly Classified as ${targetName}`;
      tmgTag.className = 'adv-tag tag-detected';
    } else {
      tmgTag.textContent = `Classified as ${tmg.predicted_class}`;
      tmgTag.className = 'adv-tag tag-evaded';
    }
  }

  // Populate Generated Features Grid
  const featGrid = document.getElementById('gen-features-grid');
  featGrid.innerHTML = '';

  const raw = sample.raw_features || {};
  let count = 0;
  for (const [fName, fVal] of Object.entries(raw)) {
    if (count++ >= 36) break;
    const item = document.createElement('div');
    item.className = 'gen-feat-item';
    item.innerHTML = `
      <span class="gen-feat-name" title="${fName}">${fName}</span>
      <span class="gen-feat-val">${typeof fVal === 'number' ? (fVal >= 1000 ? fVal.toFixed(0) : fVal.toFixed(2)) : fVal}</span>
    `;
    featGrid.appendChild(item);
  }
}

/* ================= 5. TAB 3: CONFUSION MATRIX EXPLORER ================= */
function initConfusionExplorer() {
  const buttons = document.querySelectorAll('.conf-header .btn-secondary');
  const img = document.getElementById('conf-matrix-img');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const matrixKey = btn.getAttribute('data-matrix');
      img.src = `/api/plots/confusion_${matrixKey}.png`;
    });
  });
}
