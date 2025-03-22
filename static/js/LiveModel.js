let currentSessionId = null;
let isPaused = false;

function processVideo() {
    const file = document.getElementById('videoInput').files[0];
    if (!file) {
        alert('Please select a video file first!');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.session_id) {
            currentSessionId = data.session_id;
            document.getElementById('videoStream').src = `/video_feed/${data.session_id}`;
            document.getElementById('videoStream').style.display = 'block';
            startStatusUpdates();
            startAnnotationUpdates();
        }
    })
    .catch(error => console.error('Error:', error));
}

function togglePause() {
    if (!currentSessionId) return;
    
    isPaused = !isPaused;
    const action = isPaused ? 'pause' : 'resume';
    
    fetch(`/control/${currentSessionId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: action })
    })
    .then(() => {
        document.getElementById('pauseBtn').textContent = isPaused ? 'Resume' : 'Pause';
    });
}

function startStatusUpdates() {
    setInterval(() => {
        if (!currentSessionId) return;
        
        fetch(`/status/${currentSessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) return;
                
                document.getElementById('weatherStatus').textContent = data.weather;
                document.getElementById('detectionStatus').textContent = 
                    data.detections.length > 0 
                    ? `${data.detections.length} birds detected`
                    : 'No birds detected';
            });
    }, 1000);
}

function startAnnotationUpdates() {
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const video = document.getElementById('videoStream');

    function drawAnnotations() {
        if (!currentSessionId || isPaused) {
            requestAnimationFrame(drawAnnotations);
            return;
        }

        fetch(`/status/${currentSessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) return;

                // Update canvas size
                canvas.width = video.offsetWidth;
                canvas.height = video.offsetHeight;

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                const scaleX = canvas.width / data.video_size.width;
                const scaleY = canvas.height / data.video_size.height;

                data.detections.forEach(det => {
                    const x = det.x * scaleX;
                    const y = det.y * scaleY;
                    const w = det.w * scaleX;
                    const h = det.h * scaleY;

                    ctx.beginPath();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#00FF00';
                    ctx.rect(x, y, w, h);
                    ctx.stroke();
                    
                    ctx.fillStyle = '#00FF00';
                    ctx.font = '14px Arial';
                    ctx.fillText(`${Math.round(det.confidence * 100)}%`, x + 5, y + 15);
                });
            });

        requestAnimationFrame(drawAnnotations);
    }

    drawAnnotations();
}