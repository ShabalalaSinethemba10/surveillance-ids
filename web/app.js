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


console.log('Dashboard ready. Load model to begin.');