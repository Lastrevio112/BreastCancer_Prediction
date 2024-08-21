document.getElementById('predictionForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    // Create a FormData object to gather the form inputs
    const formData = new FormData(this);

    try {
        // Send the form data to the server via POST
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        // Parse the JSON response
        const result = await response.json();

        // Display the result
        document.getElementById('result').innerText = result.prediction || result.error;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred.';
    }
});