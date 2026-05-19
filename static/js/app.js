'use strict';

// ── Datos ─────────────────────────────────────────────────────────────────────

const PLANT_EMOJIS = {
  'Manzana':   '🍎', 'Arándano':  '🫐', 'Cereza':    '🍒',
  'Maíz':      '🌽', 'Uva':       '🍇', 'Naranja':   '🍊',
  'Durazno':   '🍑', 'Pimiento':  '🌶️', 'Papa':      '🥔',
  'Frambuesa': '🍓', 'Soya':      '🌱', 'Calabaza':  '🎃',
  'Fresa':     '🍓', 'Tomate':    '🍅', 'Banana':    '🍌',
  'Albahaca':  '🌿', 'Frijol':    '🫘', 'Brócoli':   '🥦',
  'Col':       '🥬', 'Zanahoria': '🥕', 'Coliflor':  '🥦',
  'Apio':      '🌿', 'Café':      '☕', 'Pepino':    '🥒',
  'Berenjena': '🍆', 'Ajo':       '🧄', 'Lechuga':   '🥬',
  'Arce':      '🍁', 'Ciruela':   '🫐', 'Arroz':     '🌾',
  'Tabaco':    '🌿', 'Trigo':     '🌾', 'Calabacín': '🥒',
};

// ── Estado ────────────────────────────────────────────────────────────────────

let currentFile   = null;
let currentMode   = 'auto';   // 'auto' | 'plant'
let selectedPlant = null;

// ── Utilidades DOM ────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function show(id)   { $(id) && $(id).classList.remove('d-none'); }
function hide(id)   { $(id) && $(id).classList.add('d-none'); }
function showOnly(id) {
  ['uploadCard','previewCard','loadingCard','resultsSection','errorBox','unknownBox'].forEach(hide);
  show(id);
}

// ── Inicialización ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  setupDropZone();
  setupFileInput();
  checkModelHealth();
  loadPlants();
  showOnly('uploadCard');
});

// ── Verificación de estado del modelo ─────────────────────────────────────────

async function checkModelHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (!data.model_loaded) show('noModelBanner');
  } catch (_) { /* el servidor puede no estar listo aún */ }
}

// ── Selector de modo ──────────────────────────────────────────────────────────

function setMode(mode) {
  currentMode = mode;
  selectedPlant = null;

  $('btnAutoMode').classList.toggle('active', mode === 'auto');
  $('btnPlantMode').classList.toggle('active', mode === 'plant');

  if (mode === 'plant') {
    show('plantSelectorCard');
  } else {
    hide('plantSelectorCard');
  }

  // Limpiar cualquier chip seleccionado activo
  document.querySelectorAll('.plant-chip-pick').forEach(el => el.classList.remove('selected'));
}

// ── Selector de planta ────────────────────────────────────────────────────────

async function loadPlants() {
  try {
    const res  = await fetch('/classes');
    const data = await res.json();
    renderPlantPicker(data.plants);
  } catch (_) { /* servidor no disponible aún — el modo guiado estará vacío hasta la próxima carga */ }
}

function renderPlantPicker(plants) {
  const grid = $('plantPickerGrid');
  if (!grid) return;
  grid.innerHTML = plants.map(p => {
    const emoji = PLANT_EMOJIS[p.name] || '🌿';
    return `<span class="plant-chip-pick" onclick="selectPlant('${p.name.replace(/'/g, "\\'")}', this)">
      ${emoji} ${p.name}
    </span>`;
  }).join('');
}

function selectPlant(name, el) {
  selectedPlant = name;
  document.querySelectorAll('.plant-chip-pick').forEach(c => c.classList.remove('selected'));
  if (el) el.classList.add('selected');
}

// ── Zona de arrastre ──────────────────────────────────────────────────────────

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

// ── Manejo de archivos ────────────────────────────────────────────────────────

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

// ── Análisis ──────────────────────────────────────────────────────────────────

async function analyzeImage() {
  if (!currentFile) return;

  showOnly('loadingCard');
  show('loadingCard');

  const formData = new FormData();
  formData.append('image', currentFile);
  if (currentMode === 'plant' && selectedPlant) {
    formData.append('plant', selectedPlant);
  }

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

// ── Resultados ────────────────────────────────────────────────────────────────

function displayResults(data) {
  const { plant, disease, description, is_healthy: healthy, confidence, top5 } = data;

  // Badge de modo guiado
  const guidedBadgeEl = $('guidedBadge');
  if (guidedBadgeEl) {
    guidedBadgeEl.classList.toggle('d-none', !(currentMode === 'plant' && selectedPlant));
  }

  // Color de la cabecera
  const header = $('resultCardHeader');
  header.className = 'card-header ' + (healthy ? 'result-header-healthy' : 'result-header-disease');

  // Ícono
  const iconWrap = $('plantIconWrap');
  iconWrap.className = 'plant-icon-wrap ' + (healthy ? 'plant-icon-healthy' : 'plant-icon-disease');
  $('plantIcon').textContent = getEmoji(plant);

  // Badge de estado
  const badge = $('statusBadge');
  badge.textContent = healthy ? '✅ Saludable' : '⚠️ Enfermedad detectada';
  badge.className = 'status-badge ' + (healthy ? 'badge-healthy' : 'badge-disease');

  // Texto
  $('plantName').textContent = plant;
  $('diseaseName').textContent = disease;
  const descEl = $('diseaseDescription');
  if (descEl) descEl.textContent = description || '';

  // Confianza
  const pct = (confidence * 100).toFixed(1);
  $('confidenceText').textContent = pct + '%';
  const fill = $('confidenceFill');
  fill.className = 'conf-fill ' + (healthy ? 'conf-fill-healthy' : 'conf-fill-disease');
  fill.style.width = '0%';
  setTimeout(() => { fill.style.width = pct + '%'; }, 80);

  // Top 5 predicciones
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

// ── Restablecer ───────────────────────────────────────────────────────────────

function resetUpload() {
  currentFile = null;
  const input = $('fileInput');
  if (input) input.value = '';
  showOnly('uploadCard');
}

// ── Error / Planta desconocida ────────────────────────────────────────────────

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

// ── Auxiliares ────────────────────────────────────────────────────────────────

function getEmoji(plantName) {
  return PLANT_EMOJIS[plantName] || '🌿';
}
