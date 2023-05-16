const fileInput = document.getElementById("images");
const fileLabel = document.getElementById("file-label");

fileInput.addEventListener("change", function() {
  if (this.files && this.files.length > 0) {
    fileLabel.textContent = this.files.length + " files selected";
  } else {
    fileLabel.textContent = "No file chosen";
  }
});
