import * as ort from 'onnxruntime-web';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
let stream;
let processingFrame = false;
let offscreenCanvas;
let offscreenCtx;
let session;

const modelUrl = 'model/yolov9-c.onnx';
const wasmRoot = 'wasm/';
ort.env.wasm.wasmPaths = wasmRoot;

let lastFrameTime = performance.now();
let fps = 0;

(async () => {
    try {
        session = await ort.InferenceSession.create(modelUrl);
        console.log('Model loaded');
        console.log('Input Names:', session.inputNames);
        console.log('Output Names:', session.outputNames);
    } catch (e) {
        console.error('Error loading the model', e);
    }
})();

function startWebcam() {
    if (stream) {
        stopWebcam();
    }
    startButton.disabled = true;
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            const playPromise = video.play();

            if (playPromise !== undefined) {
                playPromise.then(() => {
                    startButton.disabled = false;
                    console.log('Video playback started');
                    renderVideoFrame();
                    setInterval(processFrame, 100); // Run inference every 100ms
                }).catch(error => {
                    console.error('Error playing the video:', error);
                    startButton.disabled = false;
                });
            }
        })
        .catch(err => {
            console.error('Error accessing the webcam:', err);
            startButton.disabled = false;
        });
}

function stopWebcam() {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null;
    }
}

function renderVideoFrame() {
    if (stream) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        requestAnimationFrame(renderVideoFrame);
    }
}

async function processFrame() {
    if (!session || !stream) {
        return;
    }
    if (!offscreenCanvas) {
        offscreenCanvas = document.createElement('canvas');
        offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 640;
    }

    offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const imageData = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const inputTensor = new ort.Tensor('float32', preprocess(imageData), [1, 3, offscreenCanvas.height, offscreenCanvas.width]);

    console.log('Input Tensor Shape:', inputTensor.dims);

    try {
        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);
        console.log('Inference results:', results);
        postProcess(results);
    } catch (e) {
        console.error('Error during inference', e);
    }
}

function preprocess(imageData) {
    const { data, width, height } = imageData;
    const tensorData = new Float32Array(width * height * 3);
    let offset = 0;

    for (let i = 0; i < data.length; i += 4) {
        tensorData[offset++] = data[i] / 255.0;
        tensorData[offset++] = data[i + 1] / 255.0;
        tensorData[offset++] = data[i + 2] / 255.0;
    }

    return tensorData;
}

function postProcess(results) {
    console.log('Post-processing results:', results);

    const outputNames = Object.keys(results);
    console.log('Output names:', outputNames);

    // Ensure the result arrays exist and have data
    const boxes = results[outputNames[0]]?.data;
    const scores = results[outputNames[1]]?.data;

    if (!boxes || !scores) {
        console.error('Missing expected result data:', {
            boxes,
            scores,
            results
        });
        return;
    }

    console.log("Came to max operation.");

    // Normalize scores to be between 0 and 1
    let maxScore = -Infinity;
    for (let score of scores) {
        if (score > maxScore) {
            maxScore = score;
        }
    }

    console.log("Finished max operation.");

    const normalizedScores = scores.map(score => score / maxScore);

    console.log('Boxes:', boxes); 
    console.log('Scores:', normalizedScores);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Filter out low-confidence detections and draw bounding boxes
    const confidenceThreshold = 0.9;
    const maxDetections = 100; // Limit the number of detections per frame

    let detectionCount = 0;

    for (let i = 0; i < normalizedScores.length; i++) {
        if (detectionCount >= maxDetections) break;

        const score = normalizedScores[i];
        if (score < confidenceThreshold) continue;

        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);

        x1 = Math.max(0, Math.min(canvas.width, x1));
        y1 = Math.max(0, Math.min(canvas.height, y1));
        x2 = Math.max(0, Math.min(canvas.width, x2));
        y2 = Math.max(0, Math.min(canvas.height, y2));

        console.log(`Drawing box: [${x1}, ${y1}, ${x2}, ${y2}] with score: ${score}`);

        // Debugging lines to verify drawing
        console.log('Drawing rectangle with coordinates:', x1, y1, x2, y2);
        console.log('Rectangle width and height:', x2 - x1, y2 - y1);
        console.log('Canvas width and height:', canvas.width, canvas.height);

        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = 'red';
        ctx.fillText(`Score: ${score.toFixed(2)}`, x1, y1 - 5);

        detectionCount++;
    }

    // Calculate and display FPS
    const currentFPS = calculateFPS();
    ctx.fillStyle = 'white';
    ctx.fillRect(10, 10, 100, 20);
    ctx.fillStyle = 'black';
    ctx.fillText(`FPS: ${currentFPS}`, 20, 25);

    console.log('Finished drawing boxes.');
}

function calculateFPS() {
    const now = performance.now();
    const delta = now - lastFrameTime;
    fps = 1000 / delta;
    lastFrameTime = now;
    return fps.toFixed(1);
}

startButton.addEventListener('click', startWebcam);
stopButton.addEventListener('click', stopWebcam);
