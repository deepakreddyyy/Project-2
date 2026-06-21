document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const dropzoneContent = dropzone.querySelector('.dropzone-content');
    
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const loadingContainer = document.getElementById('loadingContainer');
    const resultsContent = document.getElementById('resultsContent');
    
    const topClass = document.getElementById('topClass');
    const topConfidenceNum = document.getElementById('topConfidenceNum');
    const metaDimensions = document.getElementById('metaDimensions');
    const metaDevice = document.getElementById('metaDevice');
    const cnnCanvas = document.getElementById('cnnCanvas');
    const probabilitiesStack = document.getElementById('probabilitiesStack');
    
    const sampleCards = document.querySelectorAll('.sample-card');
    const sampleSetSelect = document.getElementById('sampleSetSelect');
    
    const API_URL = 'http://127.0.0.1:5000/predict';
    
    // Setup drag and drop listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });
    
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Handle File Selection
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        
        // Remove sample gallery active states
        clearSampleActiveStates();
        
        // Render image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneContent.style.display = 'none';
            previewContainer.style.display = 'block';
            
            // Draw image to the 32x32 canvas
            const img = new Image();
            img.onload = () => {
                drawCnnVision(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Send file to API
        uploadImage(file);
    }
    
    // Draw preprocessed 32x32 image to canvas
    function drawCnnVision(imgElement) {
        const ctx = cnnCanvas.getContext('2d');
        ctx.clearRect(0, 0, 32, 32);
        // Draw image onto 32x32 grid to show pixelation
        ctx.drawImage(imgElement, 0, 0, 32, 32);
    }
    
    // Clear sample cards active border styles
    function clearSampleActiveStates() {
        sampleCards.forEach(card => card.classList.remove('active'));
    }
    
    // Upload image file
    function uploadImage(file) {
        showLoading();
        
        const formData = new FormData();
        formData.append('image', file);
        
        fetch(API_URL, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.message || 'Server error'); });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(err => {
            showError(err.message);
        });
    }
    
    // Update sample cards when set changes
    if (sampleSetSelect) {
        sampleSetSelect.addEventListener('change', () => {
            const setVal = sampleSetSelect.value;
            
            // Update the image source of each sample card
            sampleCards.forEach(card => {
                const sampleClass = card.getAttribute('data-class');
                const cardImg = card.querySelector('img');
                if (cardImg) {
                    cardImg.src = `samples/${sampleClass}_${setVal}.png`;
                }
            });
            
            // If there's an active sample card, update its prediction
            const activeCard = document.querySelector('.sample-card.active');
            if (activeCard) {
                const sampleClass = activeCard.getAttribute('data-class');
                const imgPath = `samples/${sampleClass}_${setVal}.png`;
                imagePreview.src = imgPath;
                
                const img = new Image();
                img.onload = () => {
                    drawCnnVision(img);
                };
                img.src = imgPath;
                
                predictSample(sampleClass, setVal);
            }
        });
    }

    // Click sample gallery cards
    sampleCards.forEach(card => {
        card.addEventListener('click', () => {
            const sampleClass = card.getAttribute('data-class');
            const setVal = sampleSetSelect ? sampleSetSelect.value : 1;
            
            // Set active class
            clearSampleActiveStates();
            card.classList.add('active');
            
            // Set preview image
            const imgPath = `samples/${sampleClass}_${setVal}.png`;
            imagePreview.src = imgPath;
            dropzoneContent.style.display = 'none';
            previewContainer.style.display = 'block';
            
            // Draw image to 32x32 canvas
            const img = new Image();
            img.onload = () => {
                drawCnnVision(img);
            };
            img.src = imgPath;
            
            // Trigger prediction for sample
            predictSample(sampleClass, setVal);
        });
    });
    
    // Send sample class prediction request
    function predictSample(sampleClass, sampleSet = 1) {
        showLoading();
        
        fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                sample_class: sampleClass,
                sample_set: parseInt(sampleSet)
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.message || 'Server error'); });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(err => {
            showError(err.message);
        });
    }
    
    // Remove Button Click
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Stop opening file selector
        resetInput();
    });
    
    function resetInput() {
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropzoneContent.style.display = 'block';
        clearSampleActiveStates();
        resetResults();
    }
    
    // UI state switchers
    function showLoading() {
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'none';
        loadingContainer.style.display = 'flex';
    }
    
    function resetResults() {
        resultsPlaceholder.style.display = 'flex';
        resultsContent.style.display = 'none';
        loadingContainer.style.display = 'none';
    }
    
    function showError(msg) {
        resultsPlaceholder.style.display = 'flex';
        resultsPlaceholder.querySelector('p').innerHTML = `<span style="color: var(--accent-rose); font-weight: 600;">Error:</span> ${msg}<br><br>Please make sure you have run the training script: <code>python train.py</code>.`;
        resultsContent.style.display = 'none';
        loadingContainer.style.display = 'none';
    }
    
    // Display results in the UI
    function displayResults(data) {
        loadingContainer.style.display = 'none';
        resultsContent.style.display = 'flex';
        
        // Update top prediction metrics
        topClass.innerText = data.prediction.toUpperCase();
        topConfidenceNum.innerText = data.confidence;
        
        // Meta details
        metaDimensions.innerText = data.dimensions;
        metaDevice.innerText = data.device;
        
        // Render class probabilities
        probabilitiesStack.innerHTML = '';
        
        data.probabilities.forEach((prob, index) => {
            const isTop = index === 0;
            const rowClass = isTop ? 'prob-row top-match' : 'prob-row';
            
            const row = document.createElement('div');
            row.className = rowClass;
            row.innerHTML = `
                <span class="prob-label">${prob.class}</span>
                <div class="prob-track">
                    <div class="prob-bar" style="width: 0%;"></div>
                </div>
                <span class="prob-val">${prob.confidence}%</span>
            `;
            
            probabilitiesStack.appendChild(row);
            
            // Animate progress bar width after insertion
            setTimeout(() => {
                row.querySelector('.prob-bar').style.width = `${prob.confidence}%`;
            }, 50);
        });
    }
});
