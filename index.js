document.addEventListener("DOMContentLoaded", async () => {
  await onnx.loadModel(); // Load onnx.js library

  const imageInput = document.getElementById("imageInput");
  const predictButton = document.getElementById("predictButton");
  const resultText = document.getElementById("resultText");

  let model; // Variable to store loaded ONNX model

  async function loadModel() {
    model = await onnx.Model.fromURL(
      "https://raw.githubusercontent.com/Vallid9/FaceEmotionDetection/main/models/Faceemotion_recognition_model.onnx"
    );
  }

  predictButton.addEventListener("click", async () => {
    await loadModel(); // Wait for model loading before continuing

    const imageFile = imageInput.files[0];
    const reader = new FileReader();

    reader.onload = async (event) => {
      const prediction = await predictEmotion(event.target.result);
      resultText.innerText = `Predicted emotion: ${prediction}`;
    };

    reader.readAsDataURL(imageFile);
  });

  async function predictEmotion(imageData) {
    const imageTensor = await convertImageToTensor(imageData);
    const output = await model.run(null, { input: imageTensor });
    const predictedClass = getPredictedClass(output[0]);
    return getEmotionName(predictedClass);
  }

  // Reference code functions
  function convertImageToTensor(image) {
    // ... (refer to provided reference code)
  }

  function argMax(array, dimensions) {
    // ... (refer to provided reference code)
  }

  function processOutput(outputTensor) {
    // ... (refer to provided reference code)
  }

  function getEmotionName(predictedClass) {
    const emotionClasses = ["Fear", "Happy" ,"Sad"]; 
    return emotionClasses[predictedClass];
  }
});
