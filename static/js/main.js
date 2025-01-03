function uploadAudio() {
    const audioFile = document.getElementById('audioFile').files[0];
    const formData = new FormData();
    formData.append('audio', audioFile);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('emoji').textContent = '😕';
            document.getElementById('emotion').textContent = 'Error: ' + data.error;
            document.getElementById('confidence').textContent = '';
        } else {
            document.getElementById('emoji').textContent = data.emoji;
            document.getElementById('emotion').textContent = `Emotion: ${data.emotion}`;
            document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}