'use strict';

// ── Data ──────────────────────────────────────────────────────────────────────

const PLANT_EMOJIS = {
  'Manzana': '🍎', 'Arándano': '🫐', 'Cereza': '🍒',
  'Maíz': '🌽', 'Uva': '🍇', 'Naranja': '🍊',
  'Durazno': '🍑', 'Pimiento': '🌶️', 'Papa': '🥔',
  'Frambuesa': '🍓', 'Soya': '🌱', 'Calabaza': '🎃',
  'Fresa': '🍓', 'Tomate': '🍅',
};

const SUPPORTED_PLANTS = [
  { name: 'Manzana',   diseases: 4 },
  { name: 'Arándano',  diseases: 1 },
  { name: 'Cereza',    diseases: 2 },
  { name: 'Maíz',      diseases: 4 },
  { name: 'Uva',       diseases: 4 },
  { name: 'Naranja',   diseases: 1 },
  { name: 'Durazno',   diseases: 2 },
  { name: 'Pimiento',  diseases: 2 },
  { name: 'Papa',      diseases: 3 },
  { name: 'Frambuesa', diseases: 1 },
  { name: 'Soya',      diseases: 1 },
  { name: 'Calabaza',  diseases: 1 },
  { name: 'Fresa',     diseases: 2 },
  { name: 'Tomate',    diseases: 10 },
];

// ── State ─────────────────────────────────────────────────────────────────────

let currentFile = null;

// ── DOM helpers ───────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function show(id)   { $(id) && $(id).classList.remove('d-none'); }
function hide(id)   { $(id) && $(id).classList.add('d-none'); }
function showOnly(id) {
  ['uploadCard','previewCard','loadingCard','resultsSection','errorBox','unknownBox'].forEach(hide);
  show(id);
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  setupDropZone();
  setupFileInput();
  renderPlantsGrid();
  checkModelHealth();
  showOnly('uploadCard');
});

// ── Model health check ────────────────────────────────────────────────────────

async function checkModelHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (!data.model_loaded) show('noModelBanner');
  } catch (_) { /* server may not be ready yet */ }
}

// ── Drop zone ─────────────────────────────────────────────────────────────────

function setupDropZone() {
  const zone = $('dropZone');
  if (!zone) return;

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  zone.addEventListener('click', () => $('fileInput') && $('fileInput').click());
}

function setupFileInput() {
  const input = $('fileInput');
  if (!input) return;
  input.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

// ── File handling ─────────────────────────────────────────────────────────────

function handleFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showError('Formato no válido. Usa JPG, PNG o WebP.');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError('La imagen es demasiado grande. Máximo 16 MB.');
    return;
  }

  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    $('previewImage').src = e.target.result;
    $('fileName').textContent = file.name + ' (' + formatBytes(file.size) + ')';
    showOnly('previewCard');
    show('previewCard');
    hide('uploadCard');
  };
  reader.readAsDataURL(file);
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Analysis ──────────────────────────────────────────────────────────────────

async function analyzeImage() {
  if (!currentFile) return;

  showOnly('loadingCard');
  show('loadingCard');

  const formData = new FormData();
  formData.append('image', currentFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Error desconocido.');
      return;
    }

    if (data.is_unknown) {
      showUnknown(data.message, data.confidence);
      return;
    }

    displayResults(data);
  } catch (_) {
    showError('No se pudo conectar con el servidor. Asegúrate de que la app esté corriendo.');
  }
}

// ── Results ───────────────────────────────────────────────────────────────────

function displayResults(data) {
  const { plant, disease, description, is_healthy: healthy, confidence, top5 } = data;

  // Header color
  const header = $('resultCardHeader');
  header.className = 'card-header ' + (healthy ? 'result-header-healthy' : 'result-header-disease');

  // Icon
  const iconWrap = $('plantIconWrap');
  iconWrap.className = 'plant-icon-wrap ' + (healthy ? 'plant-icon-healthy' : 'plant-icon-disease');
  $('plantIcon').textContent = getEmoji(plant);

  // Badge
  const badge = $('statusBadge');
  badge.textContent = healthy ? '✅ Saludable' : '⚠️ Enfermedad detectada';
  badge.className = 'status-badge ' + (healthy ? 'badge-healthy' : 'badge-disease');

  // Text
  $('plantName').textContent = plant;
  $('diseaseName').textContent = disease;
  const descEl = $('diseaseDescription');
  if (descEl) descEl.textContent = description || '';

  // Confidence
  const pct = (confidence * 100).toFixed(1);
  $('confidenceText').textContent = pct + '%';
  const fill = $('confidenceFill');
  fill.className = 'conf-fill ' + (healthy ? 'conf-fill-healthy' : 'conf-fill-disease');
  fill.style.width = '0%';
  setTimeout(() => { fill.style.width = pct + '%'; }, 80);

  // Top 5
  const container = $('top5Container');
  container.innerHTML = top5.map((pred, i) => {
    const p = (pred.confidence * 100).toFixed(1);
    const cls = pred.is_healthy ? 'pred-bar-fill-healthy' : 'pred-bar-fill-disease';
    return `
      <div class="pred-row">
        <div class="d-flex justify-content-between">
          <span class="pred-label ${i === 0 ? 'top1' : ''}">
            ${i + 1}. ${pred.plant} — ${pred.disease}
          </span>
          <span class="pred-label text-muted">${p}%</span>
        </div>
        <div class="pred-bar-track">
          <div class="${cls}" style="height:100%;width:${p}%;border-radius:4px;transition:width 0.6s ease ${i*0.1}s;"></div>
        </div>
      </div>`;
  }).join('');

  showOnly('resultsSection');
  show('resultsSection');
  $('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetUpload() {
  currentFile = null;
  const input = $('fileInput');
  if (input) input.value = '';
  showOnly('uploadCard');
}

// ── Error / Unknown ───────────────────────────────────────────────────────────

function showError(msg) {
  $('errorMessage').textContent = msg;
  showOnly('errorBox');
  show('errorBox');
  show('uploadCard');
}

function showUnknown(msg, confidence) {
  const pct = confidence != null ? ' (confianza máxima: ' + (confidence * 100).toFixed(1) + '%)' : '';
  $('unknownMessage').textContent = (msg || 'No se reconoció la planta.') + pct;
  showOnly('unknownBox');
  show('unknownBox');
  show('uploadCard');
}

// ── Plants grid ───────────────────────────────────────────────────────────────

function renderPlantsGrid() {
  const grid = $('plantsGrid');
  if (!grid) return;
  grid.innerHTML = SUPPORTED_PLANTS.map(p => `
    <div class="col-6 col-md-4 col-lg-3 mb-2">
      <div class="plant-chip">
        <span>${getEmoji(p.name)}</span>
        <span>${p.name}</span>
        <small class="text-muted">(${p.diseases})</small>
      </div>
    </div>
  `).join('');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getEmoji(plantName) {
  return PLANT_EMOJIS[plantName] || '🌿';
}
