document.getElementById("predict-form").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent page reload

    const formData = new FormData(this);
    const features = {};

    formData.forEach((value, key) => {
        features[key] = value;
    });

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ features })
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.error) {
            document.getElementById("prediction-result").innerHTML = `<p class="error">Error: ${result.error}</p>`;
        } else {
            const insightsHtml = result.insights.map(insight => `
                <div class="alert alert-info">${insight}</div>
            `).join("");

            const predictionHtml = `
                <h2>Prediction: <span class="${result.prediction === 'Diabetic' ? 'text-danger' : 'text-success'}">${result.prediction}</span></h2>
                <h3>Probabilities:</h3>
                <ul>
                    <li><strong>Non-Diabetic:</strong> ${result.probabilities['Non-Diabetic']}</li>
                    <li><strong>Diabetic:</strong> ${result.probabilities['Diabetic']}</li>
                </ul>
                <h3>AI Insights:</h3>
                ${insightsHtml}
            `;

            document.getElementById("prediction-result").innerHTML = predictionHtml;
        }

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("prediction-result").innerHTML = `<p class="error">Failed to get the prediction. Try again.</p>`;
    }
});
