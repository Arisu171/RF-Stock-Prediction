/* =====================================================================
   App — tiêu thụ luồng SSE và vẽ bảng điều khiển
   =====================================================================
   Không phụ thuộc thư viện ngoài, không tải gì từ mạng. Biểu đồ nến vẽ
   thẳng bằng Canvas 2D — đổi lại vài chục dòng mã, ta được ba thứ: không
   có bước build để hỏng, không phụ thuộc mạng lúc trình bày, và toàn
   quyền kiểm soát cách đánh dấu dự đoán.

   Thứ tự khai báo:

     ①  Trạng thái và hằng số
     ②  Khởi tạo giao diện — nạp danh sách mô hình, gắn sự kiện
     ③  Vòng đời luồng SSE
     ④  Xử lý từng loại sự kiện
     ⑤  Vẽ biểu đồ nến
     ⑥  Vẽ dải kết quả dự đoán
     ⑦  Cập nhật bảng chỉ số
     ⑧  Tab tải dữ liệu
   ===================================================================== */

/* ---------------------------------------------------------------------
   ① Trạng thái và hằng số
   --------------------------------------------------------------------- */
const VISIBLE_BARS = 160;      // số nến hiển thị trong khung nhìn
const OUTCOME_CELLS = 120;     // số ô trên dải kết quả

const COLOR = {
  up: '#2CA02C', down: '#DD4949',
  grid: '#2c3444', axis: '#8b96ab',
  pending: '#3a4356', wick: '#5a6478',
};

const state = {
  source: null,          // EventSource đang mở
  bars: [],              // nến đã nhận
  outcomes: [],          // 'hit' | 'miss' cho từng dự đoán đã chấm
  pending: 0,            // số dự đoán chưa tới hạn kiểm chứng
  steps: 0,
  meta: null,
  uploadToken: null,
  uploadLabel: null,
  frameQueued: false,
};

const element = (id) => document.getElementById(id);

/* ---------------------------------------------------------------------
   ② Khởi tạo giao diện — nạp danh sách rồi gắn sự kiện
   --------------------------------------------------------------------- */
async function initialise() {
  await Promise.all([loadModels(), loadDatasets()]);

  element('start-button').addEventListener('click', startStream);
  element('stop-button').addEventListener('click', stopStream);

  const speed = element('speed-input');
  speed.addEventListener('input', () => {
    element('speed-value').textContent = speed.value === '0' ? 'tối đa' : speed.value;
  });

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  setupUpload();
  window.addEventListener('resize', requestRedraw);
  requestRedraw();
}

async function loadModels() {
  const select = element('model-select');
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error((await response.json()).detail);

    const { models } = await response.json();
    select.innerHTML = '';
    models.forEach((model) => {
      const accuracy = model.metrics?.test_accuracy;
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = accuracy
        ? `${model.label} — test acc ${accuracy.toFixed(4)}`
        : model.label;
      select.appendChild(option);
    });
  } catch (error) {
    select.innerHTML = '<option>Chưa có mô hình nào</option>';
    showVerdict(`Không nạp được danh sách mô hình: ${error.message}`);
  }
}

async function loadDatasets() {
  const select = element('dataset-select');
  const response = await fetch('/api/datasets');
  const { datasets } = await response.json();

  select.innerHTML = '<option value="">— theo mô hình —</option>';
  datasets.forEach((name) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.classList.toggle('is-active', tab.dataset.tab === name);
  });
  element('panel-replay').classList.toggle('is-active', name === 'replay');
  element('panel-upload').classList.toggle('is-active', name === 'upload');
  if (name === 'replay') requestRedraw();
}

/* ---------------------------------------------------------------------
   ③ Vòng đời luồng — mở, đóng, dọn trạng thái cũ
   --------------------------------------------------------------------- */
function startStream() {
  stopStream();
  resetState();

  const parameters = new URLSearchParams({
    model: element('model-select').value,
    speed: element('speed-input').value,
    adaptive: element('adaptive-input').checked ? 'true' : 'false',
  });

  if (state.uploadToken) {
    parameters.set('upload', state.uploadToken);
  } else if (element('dataset-select').value) {
    parameters.set('dataset', element('dataset-select').value);
  }

  const source = new EventSource(`/api/stream?${parameters}`);
  state.source = source;

  source.addEventListener('meta', (event) => handleMeta(JSON.parse(event.data)));
  source.addEventListener('step', (event) => handleStep(JSON.parse(event.data)));
  source.addEventListener('done', (event) => handleDone(JSON.parse(event.data)));
  source.addEventListener('error', (event) => {
    // Sự kiện 'error' do máy chủ gửi có data; lỗi kết nối thì không.
    if (event.data) {
      showVerdict(JSON.parse(event.data).message);
    }
    stopStream();
  });

  element('start-button').disabled = true;
  element('stop-button').disabled = false;
}

function stopStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
  element('start-button').disabled = false;
  element('stop-button').disabled = true;
}

function resetState() {
  state.bars = [];
  state.outcomes = [];
  state.pending = 0;
  state.steps = 0;
  state.meta = null;

  element('meta-strip').classList.add('hidden');
  element('verdict').classList.add('hidden');
  element('current-call').innerHTML = '';
  ['static', 'adaptive', 'baseline'].forEach((line) => updateGauge(line, null));
  ['fact-resolved', 'fact-steps', 'fact-updates'].forEach((id) => {
    element(id).textContent = '0';
  });
  element('fact-trees').textContent = '—';
  requestRedraw();
}

/* ---------------------------------------------------------------------
   ④ Xử lý sự kiện — meta một lần, step mỗi phiên, done ở cuối
   --------------------------------------------------------------------- */
function handleMeta(meta) {
  state.meta = meta;

  element('chart-title').textContent = `Biểu đồ giá — ${meta.label || 'dữ liệu tải lên'}`;
  element('horizon-note').textContent = meta.horizon;
  element('caveat-horizon').textContent = meta.horizon;

  const strip = element('meta-strip');
  strip.innerHTML = `
    <span>Phát lại <b>${meta.replay_rows.toLocaleString('vi-VN')}</b> phiên</span>
    <span>Từ <b>${meta.first_key}</b> đến <b>${meta.last_key}</b></span>
    <span>Đã bỏ qua <b>${meta.skipped_rows.toLocaleString('vi-VN')}</b> phiên
          ${meta.skipped_reason ? `(${meta.skipped_reason})` : ''}</span>
    <span>Tầm nhìn <b>${meta.horizon}</b> phiên</span>
    <span>Ngưỡng <b>${meta.threshold}</b></span>`;
  strip.classList.remove('hidden');
}

function handleStep(step) {
  state.steps += 1;
  state.bars.push({ ...step.bar, key: step.key, prediction: step.prediction });

  if (step.prediction && step.prediction.static) state.pending += 1;
  if (step.resolved) {
    state.pending = Math.max(0, state.pending - 1);
    state.outcomes.push(step.resolved.static_hit ? 'hit' : 'miss');
  }

  showCurrentCall(step);
  ['static', 'adaptive', 'baseline'].forEach((line) => {
    updateGauge(line, step.accuracy[line]);
  });

  element('fact-resolved').textContent = step.resolved_count.toLocaleString('vi-VN');
  element('fact-steps').textContent = state.steps.toLocaleString('vi-VN');
  if (step.adaptation) {
    element('fact-updates').textContent = step.adaptation.num_updates;
    element('fact-trees').textContent = step.adaptation.forest_size;
  }

  requestRedraw();
}

function handleDone(summary) {
  stopStream();

  const accuracy = summary.accuracy || {};
  const modelValue = accuracy.static;
  const baselineValue = accuracy.baseline;
  if (modelValue == null || baselineValue == null) return;

  const gap = modelValue - baselineValue;
  const sign = gap >= 0 ? '+' : '';
  const judgement = gap > 0.02
    ? 'Nhỉnh hơn mốc đoán bừa, nhưng khoảng cách này nằm trong mức nhiễu thống kê.'
    : gap < -0.02
      ? 'Thấp hơn cả mốc đoán bừa trên giai đoạn này.'
      : 'Gần như trùng với mốc đoán bừa.';

  showVerdict(
    `Kết thúc sau ${summary.steps.toLocaleString('vi-VN')} phiên,
     chấm được ${summary.resolved_count.toLocaleString('vi-VN')} dự đoán.
     Mô hình ${(modelValue * 100).toFixed(1)}% so với đoán bừa
     ${(baselineValue * 100).toFixed(1)}% (${sign}${(gap * 100).toFixed(1)} điểm).
     ${judgement}`
  );
}

function showCurrentCall(step) {
  const prediction = step.prediction?.static;
  if (!prediction) {
    element('current-call').innerHTML = `<span>${step.key} — chưa đủ lịch sử</span>`;
    return;
  }

  const isUp = prediction.label === (state.meta?.positive_label ?? 1);
  element('current-call').innerHTML = `
    <span>${step.key} → sau ${state.meta?.horizon ?? '?'} phiên:</span>
    <b class="${isUp ? 'up' : 'down'}">${isUp ? 'TĂNG' : 'GIẢM'}</b>
    <span>(xác suất tăng ${prediction.score.toFixed(4)})</span>`;
}

function showVerdict(message) {
  const box = element('verdict');
  box.textContent = message;
  box.classList.remove('hidden');
}

/* ---------------------------------------------------------------------
   ⑤ Vẽ biểu đồ nến — chỉ vẽ lại tối đa một lần mỗi khung hình
   --------------------------------------------------------------------- */
function requestRedraw() {
  if (state.frameQueued) return;
  state.frameQueued = true;
  requestAnimationFrame(() => {
    state.frameQueued = false;
    drawChart();
    drawOutcomes();
  });
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.height / (canvas.dataset.ratio || 1);

  canvas.width = width * ratio;
  canvas.style.height = `${height}px`;
  canvas.dataset.ratio = ratio;

  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawChart() {
  const canvas = element('price-canvas');
  if (!canvas.clientWidth) return;

  canvas.height = 380;
  const { context, width, height } = prepareCanvas(canvas);
  const padding = { top: 14, right: 62, bottom: 22, left: 10 };

  const visible = state.bars.slice(-VISIBLE_BARS);
  if (visible.length < 2) {
    context.fillStyle = COLOR.axis;
    context.font = '13px "Segoe UI", sans-serif';
    context.fillText('Bấm "Bắt đầu" để phát lại chuỗi giá.', padding.left + 8, height / 2);
    return;
  }

  const lows = visible.map((bar) => bar.low ?? bar.close);
  const highs = visible.map((bar) => bar.high ?? bar.close);
  let minimum = Math.min(...lows);
  let maximum = Math.max(...highs);
  const margin = (maximum - minimum) * 0.08 || 1;
  minimum -= margin;
  maximum += margin;

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const toY = (value) =>
    padding.top + plotHeight * (1 - (value - minimum) / (maximum - minimum));
  const slot = plotWidth / visible.length;

  // Lưới ngang và nhãn giá bên phải
  context.font = '11px "Segoe UI", sans-serif';
  context.textBaseline = 'middle';
  for (let step = 0; step <= 4; step += 1) {
    const value = minimum + ((maximum - minimum) * step) / 4;
    const y = toY(value);

    context.strokeStyle = COLOR.grid;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(padding.left, y + 0.5);
    context.lineTo(width - padding.right, y + 0.5);
    context.stroke();

    context.fillStyle = COLOR.axis;
    context.fillText(formatPrice(value), width - padding.right + 8, y);
  }

  // Nến
  const bodyWidth = Math.max(1.5, Math.min(9, slot * 0.62));
  visible.forEach((bar, index) => {
    const centre = padding.left + slot * (index + 0.5);
    const open = bar.open ?? bar.close;
    const close = bar.close;
    const rising = close >= open;

    context.strokeStyle = COLOR.wick;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(centre, toY(bar.high ?? Math.max(open, close)));
    context.lineTo(centre, toY(bar.low ?? Math.min(open, close)));
    context.stroke();

    const top = toY(Math.max(open, close));
    const bottom = toY(Math.min(open, close));
    context.fillStyle = rising ? COLOR.up : COLOR.down;
    context.fillRect(centre - bodyWidth / 2, top,
                     bodyWidth, Math.max(1.5, bottom - top));
  });

  // Nhãn thời gian ở vài mốc
  context.fillStyle = COLOR.axis;
  context.textBaseline = 'top';
  const tickStep = Math.max(1, Math.floor(visible.length / 6));
  for (let index = 0; index < visible.length; index += tickStep) {
    context.fillText(visible[index].key,
                     padding.left + slot * index, height - padding.bottom + 6);
  }

  // Nến mới nhất — vạch ngang và nhãn nổi bật
  const latest = visible[visible.length - 1];
  const y = toY(latest.close);
  context.strokeStyle = latest.prediction?.static
    ? (latest.prediction.static.label === (state.meta?.positive_label ?? 1)
        ? COLOR.up : COLOR.down)
    : COLOR.axis;
  context.setLineDash([4, 4]);
  context.beginPath();
  context.moveTo(padding.left, y + 0.5);
  context.lineTo(width - padding.right, y + 0.5);
  context.stroke();
  context.setLineDash([]);
}

/* ---------------------------------------------------------------------
   ⑥ Dải kết quả — mỗi ô là một dự đoán đã chấm, ô mờ là đang chờ
   --------------------------------------------------------------------- */
function drawOutcomes() {
  const canvas = element('outcome-canvas');
  if (!canvas.clientWidth) return;

  canvas.height = 46;
  const { context, width, height } = prepareCanvas(canvas);

  const recent = state.outcomes.slice(-OUTCOME_CELLS);
  const cells = [...recent, ...Array(Math.min(state.pending, 12)).fill('pending')];
  if (!cells.length) return;

  const gap = 2;
  const cellWidth = Math.max(2, (width - gap * (cells.length - 1)) / cells.length);

  cells.forEach((outcome, index) => {
    context.fillStyle = outcome === 'hit' ? COLOR.up
                      : outcome === 'miss' ? COLOR.down
                      : COLOR.pending;
    context.fillRect(index * (cellWidth + gap), 8, cellWidth, height - 22);
  });

  const hits = recent.filter((outcome) => outcome === 'hit').length;
  context.fillStyle = COLOR.axis;
  context.font = '11px "Segoe UI", sans-serif';
  context.textBaseline = 'top';
  context.fillText(
    `${recent.length} dự đoán gần nhất — đúng ${hits}, sai ${recent.length - hits}`,
    0, height - 12);
}

/* ---------------------------------------------------------------------
   ⑦ Bảng chỉ số — ba thanh accuracy
   --------------------------------------------------------------------- */
function updateGauge(line, value) {
  const gauge = document.querySelector(`.gauge[data-line="${line}"]`);
  if (!gauge) return;

  const label = gauge.querySelector('b');
  const fill = gauge.querySelector('.gauge-track i');

  if (value == null) {
    label.textContent = '—';
    fill.style.width = '0';
    return;
  }
  label.textContent = `${(value * 100).toFixed(1)}%`;
  fill.style.width = `${Math.min(100, value * 100)}%`;
}

function formatPrice(value) {
  if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString('vi-VN');
  return value.toFixed(2);
}

/* ---------------------------------------------------------------------
   ⑧ Tab tải dữ liệu
   --------------------------------------------------------------------- */
function setupUpload() {
  const dropzone = element('dropzone');
  const input = element('file-input');

  dropzone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files.length) sendFile(input.files[0]);
  });

  ['dragenter', 'dragover'].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add('is-over');
    });
  });
  ['dragleave', 'drop'].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove('is-over');
    });
  });
  dropzone.addEventListener('drop', (event) => {
    if (event.dataTransfer.files.length) sendFile(event.dataTransfer.files[0]);
  });
}

async function sendFile(file) {
  const box = element('upload-result');
  box.classList.remove('hidden', 'is-error');
  box.textContent = `Đang xử lý ${file.name}…`;

  const payload = new FormData();
  payload.append('file', file);

  try {
    const response = await fetch('/api/upload', { method: 'POST', body: payload });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail);

    state.uploadToken = result.token;
    state.uploadLabel = result.label;
    element('dataset-select').value = '';

    box.innerHTML = `<strong>${result.message}</strong>
      Cột nhận được: ${result.columns.join(', ')}.
      Sang tab <em>Phát lại</em> rồi bấm Bắt đầu — luồng sẽ dùng file này.`;
  } catch (error) {
    state.uploadToken = null;
    box.classList.add('is-error');
    box.textContent = error.message;
  }
}

initialise();
