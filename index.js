<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Emotion Recognition</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
    }
    #imageInput {
      margin-bottom: 10px;
    }
    #output {
      margin-top: 20px;
      border: 1px solid #ccc;
      padding: 10px;
    }
  </style>
</head>
<body>
  <input type="file" id="imageInput" accept="image/*" />
  <button onclick="loadModel().then(() => runInference())">Run Inference</button>

  <div id="output"></div>

  <!-- Include ONNX Runtime JavaScript library -->
  <script src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js"></script>


  <!-- Your custom JavaScript file -->
  <script src="index.js"></script>
</body>
</html>
