async function runExample() {
  
    const x = new Float32Array(9216).fill(0.5); 
    
    let tensorX = new ort.Tensor('float32', x, [1, 3, 48, 48]); // Modify tensor 
    let feeds = { input: tensorX }; // Change 'input' to match your model's input name

    try {
        let session = await ort.InferenceSession.create('EmotionCnnOnnx.onnx');
        let result = await session.run(feeds);
        let outputData = result.output.data;

      
        
        // Example display of results
        displayResults(outputData); // Function to display results (defined in index.html)
    } catch (error) {
        console.error('Error running inference:', error);
        alert('Failed to run inference. Please check the input and try again.');
    }
}

function displayResults(outputData) {
    
    let actualClass = 'happy'; //  actual class label
    let predictedClass = 'sad'; //  predicted class label
    let isCorrect = false; // correctness evaluation logic

    const actualElement = document.getElementById('actual_class');
    const predictedElement = document.getElementById('predicted_class');
    const correctElement = document.getElementById('is_correct');

    actualElement.innerText = `Actual Class: ${actualClass}`;
    predictedElement.innerText = `Predicted Class: ${predictedClass}`;
    correctElement.innerText = `Correct: ${isCorrect}`;
}
