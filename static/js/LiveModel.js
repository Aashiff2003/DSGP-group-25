// Function to toggle sidebar
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
  document.getElementById("main-content").classList.toggle("collapsed");
}

// Function to preview the video after selection
function previewVideo(event) {
  const video = document.getElementById('uploadedVideo');
  const file = event.target.files[0];
  
  if (file) {
    const videoURL = URL.createObjectURL(file);
    video.src = videoURL;
    video.style.display = 'block';  // Make sure the video element is visible
  }
}

// Function to upload video and display it
function uploadVideo() {
  const fileInput = document.getElementById('videoUpload');
  const file = fileInput.files[0]; // Get the file from input

  if (!file) {
    alert('Please upload a video first!');
    return;
  }

  const formData = new FormData();
  formData.append('video', file); // Append the video file to FormData

  // Fetch to your server for uploading the video
  fetch('/upload', {
    method: 'POST',
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      if (data.video_url) {
        // Display the uploaded video
        const uploadedVideoElement = document.getElementById('uploadedVideo');
        uploadedVideoElement.src = data.video_url;
        uploadedVideoElement.style.display = 'block'; // Ensure video is visible
      } else {
        alert('Error uploading video');
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
}
