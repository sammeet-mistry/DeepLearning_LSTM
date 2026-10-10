const sampleHistory = [
  33.2, 33.9, 34.5, 34.0, 35.2, 35.9, 36.1, 36.7, 37.0, 37.3,
  38.1, 38.6, 39.0, 39.4, 39.8, 40.1, 39.2, 38.6, 38.0, 37.6,
  36.9, 36.1, 35.7, 35.4, 35.0, 35.6, 36.2, 36.9, 37.5, 38.1
];

const els = {
  status: document.getElementById('serviceStatus'),
  location: document.getElementById('locationSelect'),
  history: document.getElementById('historyInput'),
  predictBtn: document.getElementById('predictBtn'),
  sampleBtn: document.getElementById('sampleBtn'),
  errorBox: document.getElementById('errorBox'),
  warningBox: document.getElementById('warningBox'),
  forecastLocation: document.getElementById('forecastLocation'),
  forecastDate: document.getElementById('forecastDate'),
  predictionValue: document.getElementById('predictionValue'),
  modelId: document.getElementById('modelId'),
  sequenceLength: document.getElementById('sequenceLength'),
  generatedAt: document.getElementById('generatedAt'),
  metricMae: document.getElementById('metricMae'),
  metricRmse: document.getElementById('metricRmse'),
  baselineMae: document.getElementById('baselineMae'),
  baselineRmse: document.getElementById('baselineRmse'),
  chart: document.getElementById('trendChart')
};

function renderHistoryChart(values) {
  const svg = els.chart;
  const width = 760;
  const height = 260;
  const padding = 28;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 32);

  const points = values.map((value, index) => {
    const x = padding + index * ((width - padding * 2) / (Math.max(values.length - 1, 1)));
    const y = height - padding - ((value - min) / (Math.max(max - min, 1) || 1)) * (height - padding * 2);
    return `${x},${y}`;
  }).join(' ');

  const trendPath = `M ${points.replace(/ /g, ' L ')}`;

  svg.innerHTML = `
    <defs>
      <linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#5eead4" stop-opacity="0.35" />
        <stop offset="100%" stop-color="#5eead4" stop-opacity="0.02" />
      </linearGradient>
    </defs>
    <g>
      <polyline points="${points}" fill="none" stroke="#5eead4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>
      <polyline points="${points + ' ' + `${width - padding},${height - padding} ${padding},${height - padding}`}" fill="url(#areaFill)" stroke="none"></polyline>
    </g>
  `;
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.classList.remove('hidden');
}

function clearError() {
  els.errorBox.textContent = '';
  els.errorBox.classList.add('hidden');
}

function showWarning(message) {
  els.warningBox.textContent = message;
  els.warningBox.classList.remove('hidden');
}

function hideWarning() {
  els.warningBox.textContent = '';
  els.warningBox.classList.add('hidden');
}

async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    const statusText = data.model_available ? 'Model ready' : 'Model unavailable';
    els.status.textContent = statusText;
    els.status.style.color = data.model_available ? '#5eead4' : '#fbbf24';
  } catch (error) {
    els.status.textContent = 'API unavailable';
    els.status.style.color = '#f87171';
  }
}

async function loadMetrics() {
  try {
    const response = await fetch('/api/metrics');
    if (!response.ok) {
      return;
    }
    const metrics = await response.json();
    els.metricMae.textContent = Number(metrics.model.mae).toFixed(3) + ' °C';
    els.metricRmse.textContent = Number(metrics.model.rmse).toFixed(3) + ' °C';
    els.baselineMae.textContent = Number(metrics.baseline.mae).toFixed(3) + ' °C';
    els.baselineRmse.textContent = Number(metrics.baseline.rmse).toFixed(3) + ' °C';
  } catch (error) {
    // Ignore metrics when unavailable.
  }
}

function getHistoryValues() {
  const raw = els.history.value
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);

  if (!raw.length) {
    throw new Error('Please enter at least one temperature value.');
  }

  return raw;
}

async function handlePredict() {
  try {
    clearError();
    hideWarning();

    const history = getHistoryValues();
    const payload = {
      location: els.location.value,
      history
    };

    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Forecast request failed.');
    }

    els.forecastLocation.textContent = data.location;
    els.forecastDate.textContent = data.forecast_date;
    els.predictionValue.textContent = `${Number(data.predicted_temperature_c).toFixed(2)} °C`;
    els.modelId.textContent = data.model_id;
    els.sequenceLength.textContent = String(data.input_sequence_length);
    els.generatedAt.textContent = data.generated_at_utc;
    renderHistoryChart(history);

    if (data.warnings && data.warnings.length) {
      showWarning(data.warnings.join(' '));
    }
  } catch (error) {
    showError(error.message || 'Unable to generate forecast.');
  }
}

function populateSample() {
  els.history.value = sampleHistory.join(', ');
  renderHistoryChart(sampleHistory);
}

els.sampleBtn.addEventListener('click', populateSample);
els.predictBtn.addEventListener('click', handlePredict);

populateSample();
loadHealth();
loadMetrics();
