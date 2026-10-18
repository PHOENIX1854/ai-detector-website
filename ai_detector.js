// --- DOM Elements ---
const fileInput = document.getElementById('file-upload');
const dragDropArea = document.getElementById('drag-drop-area');
const previewArea = document.getElementById('preview-area');
const previewContent = document.getElementById('preview-content');
const detectButton = document.getElementById('detect-button');
const resultArea = document.getElementById('result-area');
const loadingState = document.getElementById('loading-state');
const resultContent = document.getElementById('result-content');
const backendNote = document.getElementById('backend-note'); // We will hide this now

let selectedFile = null;

// --- Drag and Drop Handlers ---
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dragDropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});
['dragenter', 'dragover'].forEach(eventName => {
    dragDropArea.addEventListener(eventName, () => dragDropArea.classList.add('dragging'), false);
});
['dragleave', 'drop'].forEach(eventName => {
    dragDropArea.addEventListener(eventName, () => dragDropArea.classList.remove('dragging'), false);
});
dragDropArea.addEventListener('drop', handleDrop, false);

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// --- File Input Handler ---
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// --- Core File Logic ---
function handleFile(file) {
    // Reset UI
    resultArea.classList.add('hidden');
    resultContent.classList.add('hidden');
    if (backendNote) backendNote.classList.add('hidden'); // Hide backend note
    loadingState.classList.add('hidden');

    const validImageTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const validVideoTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm'];

    // Use file extensions as a fallback check
    const fileName = file.name.toLowerCase();
    const isImage = validImageTypes.includes(file.type) || ['.jpg', '.jpeg', '.png', '.webp'].some(ext => fileName.endsWith(ext));
    const isVideo = validVideoTypes.includes(file.type) || ['.mp4', '.mov', '.avi', '.mkv', '.webm'].some(ext => fileName.endsWith(ext));

    if (!isImage && !isVideo) {
        console.error('Invalid file type.');
        previewContent.innerHTML = `<p class="p-4 text-red-700">Invalid file type. Please upload a supported image or video.</p>`;
        previewArea.classList.remove('hidden');
        detectButton.disabled = true;
        selectedFile = null;
        return;
    }

    selectedFile = file;
    previewContent.innerHTML = '';

    const fileReader = new FileReader();

    if (isImage) {
        fileReader.onload = () => {
            const img = document.createElement('img');
            img.src = fileReader.result;
            img.className = 'w-full h-auto object-cover rounded-lg';
            previewContent.appendChild(img);
        };
        fileReader.readAsDataURL(file);
    } else if (isVideo) {
        fileReader.onload = () => {
            const video = document.createElement('video');
            video.src = fileReader.result;
            video.className = 'w-full h-auto rounded-lg';
            video.controls = true;
            video.muted = true;
            previewContent.appendChild(video);
        };
        fileReader.readAsDataURL(file);
    }

    previewArea.classList.remove('hidden');
    detectButton.disabled = false;
}

// --- Detect Button Handler ---
detectButton.addEventListener('click', handleDetect);

async function handleDetect() {
    if (!selectedFile) return;

    resultArea.classList.remove('hidden');
    loadingState.classList.remove('hidden');
    resultContent.classList.add('hidden');
    if (backendNote) backendNote.classList.add('hidden');
    detectButton.disabled = true;

    // --- THIS IS THE REAL API CALL ---

    // 1. Create a FormData object to send the file
    const formData = new FormData();
    formData.append('media_file', selectedFile);

    // 2. Send it to your Python backend
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
            // No 'Content-Type' header needed, 'fetch' sets it for FormData
        });

        const data = await response.json();

        if (!response.ok) {
            // If server returns an error (like 400 or 500)
            throw new Error(data.error || `Server error: ${response.statusText}`);
        }

        // 3. Display the *real* result
        displayResult(data);

    } catch (error) {
        console.error('Error during detection:', error);
        displayError(error.message);
    }
    // --- END OF REAL API CALL ---
}

/**
 * Displays the *real* result from the API.
 * @param {object} data - The JSON response from the backend.
 * e.g., { "label": "AI", "confidence": 0.925, "frames_sampled": 120 }
 */
/**
 * Displays the *real* result from the API including the attention heatmap.
 * @param {object} data - The JSON response from the backend.
 */
function displayResult(data) {
    if (data.error) {
        displayError(data.error);
        return;
    }

    const { label, confidence, frames_sampled, keyframe_index, explanation, heatmap_raw, original_image, threshold, tta_enabled } = data;

    const isAI = label.toLowerCase() === 'ai';
    const confidencePercent = (confidence * 100).toFixed(1);

    const cardBg = isAI ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200';
    const textColor = isAI ? 'text-red-900' : 'text-emerald-900';
    const badgeColor = isAI ? 'bg-red-100 text-red-700 border-red-300' : 'bg-emerald-100 text-emerald-700 border-emerald-300';
    const dotColor = isAI ? 'bg-red-500 animate-pulse' : 'bg-emerald-500';
    const barColor = isAI ? 'text-red-600' : 'text-emerald-600';

    let html = `
        <div class="space-y-6">
            <!-- Verdict Card -->
            <div class="${cardBg} border p-6 rounded-xl shadow-sm">
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <div class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${badgeColor}">
                            <span class="w-2 h-2 rounded-full ${dotColor}"></span>
                            ${isAI ? 'AI Generated Media' : 'Real'}
                        </div>
                        <p class="text-3xl font-extrabold mt-2 text-gray-900">${confidencePercent}% <span class="text-base font-normal text-gray-500">Confidence</span></p> 
                        
                        <div class="flex flex-wrap items-center gap-2 mt-2 text-xs text-gray-600 font-medium">
                            <span class="bg-white/90 border border-gray-300 px-2 py-0.5 rounded text-[11px] font-semibold text-gray-700 shadow-2xs">TTA Stabilized</span>
                            <span>•</span>
                            <span class="text-[11px] text-gray-500">Threshold: ${(threshold ? threshold * 100 : 55).toFixed(0)}%</span>
                            ${frames_sampled ? `<span>•</span><span>${frames_sampled} frames${keyframe_index ? ` (Key #${keyframe_index})` : ''}</span>` : ''}
                        </div>
                    </div>
                    <div class="flex-shrink-0 flex items-center justify-center">
                        <div class="relative w-20 h-20">
                            <!-- Circular progress meter -->
                            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                                <circle class="text-gray-200" stroke="currentColor" stroke-width="3.5" fill="transparent" r="15.9155" cx="18" cy="18" />
                                <circle class="${barColor}" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" fill="transparent"
                                    stroke-dasharray="${confidencePercent}, 100" stroke-dashoffset="0" r="15.9155" cx="18" cy="18" />
                            </svg>
                            <div class="absolute inset-0 flex items-center justify-center text-sm font-bold ${textColor}">
                                ${label}
                            </div>
                        </div>
                    </div>
                </div>

                ${explanation ? `
                    <div class="mt-4 pt-3.5 border-t border-gray-200/80 text-xs sm:text-sm text-gray-700 leading-relaxed">
                        ${explanation}
                    </div>
                ` : ''}
            </div>
    `;

    // Visual Attention Heatmap Inspector
    if (heatmap_raw && original_image) {
        html += `
            <div class="bg-gray-50 border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                <div class="flex items-center justify-between">
                    <div>
                        <h4 class="font-bold text-gray-900 flex items-center gap-2 text-sm sm:text-base">
                            <i data-feather="eye" class="h-4 w-4 text-sky-600"></i>
                            Vision Attention Heatmap
                        </h4>
                        <p class="text-xs text-gray-500 mt-0.5">
                            ${frames_sampled ? `Keyframe #${keyframe_index} anomaly heatmap (49 ViT patches)` : 'Visualizes the 49 Vision Transformer patches driving this decision'}
                        </p>
                    </div>
                    <span class="text-xs font-semibold bg-sky-100 text-sky-700 px-2 py-0.5 rounded">ViT-B/32</span>
                </div>

                <!-- Heatmap Canvas Stack -->
                <div class="heatmap-container relative w-full rounded-lg overflow-hidden border border-gray-300 bg-slate-900 flex items-center justify-center">
                    <!-- Base Original Image -->
                    <img id="heatmap-base-img" src="${original_image}" alt="Original" class="w-full h-auto object-contain block select-none">
                    <!-- Heatmap Overlay Layer with Dynamic Opacity -->
                    <img id="heatmap-overlay-layer" src="${heatmap_raw}" alt="Attention Heatmap" 
                         class="absolute inset-0 w-full h-full object-contain pointer-events-none select-none transition-opacity duration-75"
                         style="opacity: 0.5;">
                </div>

                <!-- Opacity Controls -->
                <div class="space-y-2 pt-1">
                    <div class="flex items-center justify-between text-xs font-semibold text-gray-700">
                        <label for="heatmap-opacity-slider" class="flex items-center gap-1.5">
                            <i data-feather="sliders" class="h-3.5 w-3.5 text-gray-500"></i>
                            Heatmap Overlay Intensity:
                        </label>
                        <span id="heatmap-opacity-val" class="font-mono bg-white px-2 py-0.5 rounded border border-gray-300 text-sky-700 font-bold">50%</span>
                    </div>

                    <input id="heatmap-opacity-slider" type="range" min="0" max="100" value="50"
                           class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-sky-600">

                    <!-- Preset Quick Buttons -->
                    <div class="flex justify-between gap-2 pt-1">
                        <button type="button" class="preset-btn flex-1 text-xs py-1.5 px-2 bg-white hover:bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-medium transition shadow-sm" data-opacity="0">
                            Original (0%)
                        </button>
                        <button type="button" class="preset-btn flex-1 text-xs py-1.5 px-2 bg-sky-50 hover:bg-sky-100 border border-sky-300 rounded-md text-sky-700 font-semibold transition shadow-sm" data-opacity="50">
                            Blended (50%)
                        </button>
                        <button type="button" class="preset-btn flex-1 text-xs py-1.5 px-2 bg-white hover:bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-medium transition shadow-sm" data-opacity="100">
                            Heatmap (100%)
                        </button>
                    </div>
                </div>

                <!-- Color Scale Legend -->
                <div class="pt-2 border-t border-gray-200">
                    <div class="flex items-center justify-between text-[11px] text-gray-500 font-medium mb-1">
                        <span>Cool Blue: Neutral / Background</span>
                        <span>Hot Red: Peak Attention / Artifacts</span>
                    </div>
                    <div class="h-2.5 w-full rounded-full shadow-inner" style="background: linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000);"></div>
                </div>
            </div>
        `;
    }

    html += `</div>`;

    resultContent.innerHTML = html;
    resultContent.classList.remove('hidden');
    loadingState.classList.add('hidden');
    detectButton.disabled = false;

    // Attach interactive slider listeners
    const slider = document.getElementById('heatmap-opacity-slider');
    const overlay = document.getElementById('heatmap-overlay-layer');
    const valLabel = document.getElementById('heatmap-opacity-val');

    if (slider && overlay && valLabel) {
        slider.addEventListener('input', (e) => {
            const val = e.target.value;
            overlay.style.opacity = val / 100;
            valLabel.textContent = `${val}%`;
        });

        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const targetOpacity = btn.getAttribute('data-opacity');
                slider.value = targetOpacity;
                overlay.style.opacity = targetOpacity / 100;
                valLabel.textContent = `${targetOpacity}%`;
            });
        });
    }

    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

/**
 * Displays an error message.
 * @param {string} message - The error message.
 */
function displayError(message) {
    resultContent.innerHTML = `
        <div class="bg-red-100 border border-red-200 text-red-800 p-4 rounded-md">
            <h4 class="font-bold">Detection Failed</h4>
            <p class="mt-1 text-sm">${message}</p>
        </div>
    `;
    resultContent.classList.remove('hidden');
    loadingState.classList.add('hidden');
    detectButton.disabled = false;
}

// --- Initialize Icons ---
document.addEventListener('DOMContentLoaded', () => {
    // Make sure feather icons are loaded and replaced
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});

