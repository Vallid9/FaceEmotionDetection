document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const predictButton = document.getElementById("predictButton");
  const resultText = document.getElementById("resultText");

  let model; // Variable to store loaded ONNX model

  async function loadModel() {
    model = await onnx.Model.fromURL("https://raw.githubusercontent.com/Vallid9/FaceEmotionDetection/main/models/Faceemotion_recognition_model.onnx");
  }

  predictButton.addEventListener("click", async () => {
    await loadModel(); // Load model once

    const imageFile = imageInput.files[0];
    const reader = new FileReader();

    reader.onload = async (event) => {
      const imageData = event.target.result;
      const prediction = await predictEmotion(imageData);
      resultText.innerText = `Predicted emotion: ${prediction}`;
    };

    reader.readAsDataURL(imageFile);
  });

  async function predictEmotion(imageData) {
    // Preprocess image according to your model requirements
    const imageTensor = preprocessImage(imageData);

    // Run inference using the loaded model
    const output = await model.run(null, { input: imageTensor });

    // Get and process output
    const predictions = output[0];
    const predictedClass = getPredictedClass(predictions);

    // Return the corresponding emotion name
    return getEmotionName(predictedClass);
  }

  // Your other functions (preprocessImage, getPredictedClass, getEmotionName) here
});
