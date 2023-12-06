const imageInput = document.getElementById("imageInput");
const predictButton = document.getElementById("predictButton");
const resultText = document.getElementById("resultText");

predictButton.addEventListener("click", async () => {
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
  // Load ONNX model
  const model = await onnx.load("Faceemotion_recognition_model.onnx");

  // Create ONNX inference session
  const session = new onnx.InferenceSession(model);

  // Preprocess image
  const imageTensor = preprocessImage(imageData);

  // Run inference
  const output = await session.run(null, { input: imageTensor });

  // Get and process output
  const predictions = output[0];
  const predictedClass = getPredictedClass(predictions);

  // Return predicted emotion
  return getEmotionName(predictedClass);
}
