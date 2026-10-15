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
        const response = await fetch('http://127.0.0.1:5000/api/detect', {
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
function displayResult(data) {
    // Check for an error message in the result
    if (data.error) {
        displayError(data.error);
        return;
    }

    const { label, confidence, frames_sampled } = data;

    const isAI = label.toLowerCase() === 'ai';
    const confidencePercent = (confidence * 100).toFixed(2);

    const bgColor = isAI ? 'bg-red-100' : 'bg-green-100';
    const textColor = isAI ? 'text-red-800' : 'text-green-800';
    const barColor = isAI ? 'bg-red-500' : 'bg-green-500';

    let html = `
        <div class="${bgColor} ${textColor} p-6 rounded-lg shadow-sm">
            <div class="flex flex-col md:flex-row md:items-center md:justify-between">
                <div>
                    <span class="text-sm font-medium uppercase">${label}</span>
                    <p class="text-3xl font-bold mt-1">${confidencePercent}% Confidence</p> 
                </div>
                <div class="mt-4 md:mt-0 md:ml-6 flex-shrink-0">
                    <div class="relative w-24 h-24">
                        <!-- Circular progress bar -->
                        <svg class="w-full h-full" viewBox="0 0 36 36" transform="rotate(-90)">
                            <circle class="text-gray-200"
                                stroke="currentColor" stroke-width="4" fill="transparent"
                                r="15.9155" cx="18" cy="18" />
                            <circle class="${barColor}"
                                stroke="currentColor" stroke-width="4" fill="transparent"
                                stroke-dasharray="${confidencePercent}, 100" 
                                stroke-dashoffset="0"
                                r="15.9155" cx="18" cy="18" />
                        </svg>
                        <div class="absolute inset-0 flex items-center justify-center text-xl font-bold ${textColor}">
                            ${label}
                        </div>
                    </div>
                </div>
            </div>
    `;

    if (frames_sampled) {
        html += `<p class="mt-4 text-sm ${textColor} opacity-80"></p>`;
    }

    html += `</div>`;

    resultContent.innerHTML = html;
    resultContent.classList.remove('hidden');
    loadingState.classList.add('hidden');
    detectButton.disabled = false;
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

