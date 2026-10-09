// Config
const MODEL_PATH = 'models/stacked_ensemble.onnx';
const N_FEATURES = 115;

let session = null;
let currentInput = null;

// Element references
const loadModelBtn = document.getElementById('loadModel');
const modelStatus = document.getElementById('modelStatus');
const csvInput = document.getElementById('csvInput');
const loadSampleBtn = document.getElementById('loadSample');
const loadAttackBtn = document.getElementById('loadAttack');
const inputStatus = document.getElementById('inputStatus');
const predictBtn = document.getElementById('predict');
const resultDiv = document.getElementById('result');
const inferenceTimeSpan = document.getElementById('inferenceTime');
const throughputSpan = document.getElementById('throughput');


// ============================================================
// 1. Load ONNX model
// ============================================================
loadModelBtn.addEventListener('click', async () => {
    try {
        modelStatus.textContent = 'Loading model...';
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'],
        });
        modelStatus.textContent = '✅ Model loaded (Stacked Ensemble)';
        modelStatus.style.color = '#28a745';
        updatePredictButton();
    } catch (err) {
        modelStatus.textContent = '❌ Failed to load model: ' + err.message;
        modelStatus.style.color = '#dc3545';
        console.error(err);
    }
});


// ============================================================
// 2. Input handling
// ============================================================
csvInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        const text = event.target.result;
        const values = text.trim().split('\n')[0].split(',').map(Number);

        if (values.length !== N_FEATURES) {
            inputStatus.textContent = `⚠️ Expected ${N_FEATURES} features, got ${values.length}`;
            inputStatus.style.color = '#dc3545';
            return;
        }

        currentInput = new Float32Array(values);
        inputStatus.textContent = '✅ CSV loaded';
        inputStatus.style.color = '#28a745';
        updatePredictButton();
    };
    reader.readAsText(file);
});

loadSampleBtn.addEventListener('click', () => loadPreset('benign'));
loadAttackBtn.addEventListener('click', () => loadPreset('attack'));


async function loadPreset(type) {
    try {
        const response = await fetch(`samples/${type}.json`);
        const data = await response.json();
        currentInput = new Float32Array(data.features);
        inputStatus.textContent = `✅ ${type.toUpperCase()} sample loaded`;
        inputStatus.style.color = '#28a745';
        updatePredictButton();
    } catch (err) {
        inputStatus.textContent = `⚠️ Preset not found. Upload a CSV instead.`;
        inputStatus.style.color = '#dc3545';
    }
}


function updatePredictButton() {
    predictBtn.disabled = !(session && currentInput);
}


// ============================================================
// 3. Run inference
// ============================================================
predictBtn.addEventListener('click', async () => {
    if (!session || !currentInput) return;

    try {
        const tensor = new ort.Tensor('float32', currentInput, [1, N_FEATURES]);

        const start = performance.now();
        const outputs = await session.run({ input: tensor });
        const elapsed = performance.now() - start;

        console.log('ONNX outputs:', outputs);

        // Model output names: "label" and "probabilities"
        const prediction = Number(outputs['label'].data[0]);
        const probs = outputs['probabilities'].data;
        const confidence = probs[prediction];

        displayResult(prediction, confidence, elapsed);
        inferenceTimeSpan.textContent = `${elapsed.toFixed(2)} ms`;
        throughputSpan.textContent = `${(1000 / elapsed).toFixed(0)} samples/sec`;
    } catch (err) {
        console.error('Inference error:', err);
        resultDiv.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    }
});


function displayResult(prediction, confidence, elapsed) {
    const label = prediction === 0 ? '✅ BENIGN' : '🚨 ATTACK DETECTED';
    const cssClass = prediction === 0 ? 'benign' : 'attack';

    resultDiv.className = cssClass;
    resultDiv.innerHTML = `
        <div class="result-label">${label}</div>
        <div class="result-confidence">Confidence: ${(confidence * 100).toFixed(2)}%</div>
        <div class="result-confidence">Inference: ${elapsed.toFixed(2)} ms</div>
    `;
}






// ============================================================
// Live Stream Simulation
// ============================================================
let streamInterval = null;
let streamSamples = [];
let streamIndex = 0;
let sessionStats = { benign: 0, attack: 0, totalLatency: 0, count: 0 };

const startStreamBtn = document.getElementById('startStream');
const stopStreamBtn  = document.getElementById('stopStream');
const streamLogEl    = document.getElementById('streamLog');
const streamStatsEl  = document.getElementById('streamStats');

console.log('Live stream elements:', {
    startStreamBtn: !!startStreamBtn,
    stopStreamBtn: !!stopStreamBtn,
    streamLogEl: !!streamLogEl,
    streamStatsEl: !!streamStatsEl,
});

async function loadStreamSamples() {
    try {
        const response = await fetch('samples/sample_batch.csv');
        if (!response.ok) throw new Error('HTTP ' + response.status);
        const text = await response.text();
        streamSamples = text.trim().split('\n')
            .map(line => new Float32Array(line.split(',').map(Number)))
            .filter(arr => arr.length === 115);
        console.log('Loaded', streamSamples.length, 'stream samples');
    } catch (err) {
        console.error('Failed to load samples:', err);
        alert('Could not load sample_batch.csv. Check the console.');
    }
}

async function runStreamLoop() {
    if (streamIndex >= streamSamples.length) streamIndex = 0;

    const tensor = new ort.Tensor('float32', streamSamples[streamIndex], [1, 115]);
    const start = performance.now();
    const outputs = await session.run({ input: tensor });
    const elapsed = performance.now() - start;

    const prediction = Number(outputs['label'].data[0]);
    const confidence = outputs['probabilities'].data[prediction];

    sessionStats.count++;
    sessionStats.totalLatency += elapsed;
    if (prediction === 0) sessionStats.benign++;
    else sessionStats.attack++;

    const label = prediction === 0 ? '✅ BENIGN' : '🚨 ATTACK';
    const line = document.createElement('div');
    line.textContent = `Flow ${streamIndex + 1}: ${label} (${(confidence * 100).toFixed(2)}%) — ${elapsed.toFixed(2)} ms`;
    line.style.color = prediction === 0 ? '#28a745' : '#dc3545';
    streamLogEl.prepend(line);

    streamStatsEl.textContent = `Session: ${sessionStats.count} flows | ${sessionStats.benign} benign | ${sessionStats.attack} attacks | Avg latency: ${(sessionStats.totalLatency / sessionStats.count).toFixed(2)} ms`;

    streamIndex++;
}

if (startStreamBtn) {
    startStreamBtn.addEventListener('click', async () => {
        if (!session) {
            alert('Load the ONNX model first.');
            return;
        }

        if (streamSamples.length === 0) {
            await loadStreamSamples();
            if (streamSamples.length === 0) return;
        }

        streamIndex = 0;
        sessionStats = { benign: 0, attack: 0, totalLatency: 0, count: 0 };
        streamLogEl.innerHTML = '';
        startStreamBtn.disabled = true;
        stopStreamBtn.disabled = false;

        // Warm up the WASM engine before the visible stream starts
try {
    const warmupTensor = new ort.Tensor('float32', new Float32Array(115), [1, 115]);
    await session.run({ input: warmupTensor });
    console.log('Warm-up complete');
} catch (err) {
    console.warn('Warm-up failed:', err);
}

        streamInterval = setInterval(async () => {
            try {
                await runStreamLoop();
            } catch (err) {
                console.error('Stream error:', err);
                clearInterval(streamInterval);
                streamInterval = null;
                startStreamBtn.disabled = false;
                stopStreamBtn.disabled = true;
            }
        }, 800);
    });
}

if (stopStreamBtn) {
    stopStreamBtn.addEventListener('click', () => {
        clearInterval(streamInterval);
        streamInterval = null;
        startStreamBtn.disabled = false;
        stopStreamBtn.disabled = true;
    });
}


console.log('Dashboard ready. Load model to begin.');