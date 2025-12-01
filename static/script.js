// static/script.js

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("prediction-form");
    const predictionOutput = document.getElementById("prediction-output");
    const filledInputs = document.getElementById("filled-inputs");
    const resultsDiv = document.getElementById("results");

    form.addEventListener("submit", function (event) {
        event.preventDefault();

        // Collect raw values from inputs; they may be empty strings
        const payload = {
            loan_amnt: document.getElementById("loan_amnt").value,
            annual_inc: document.getElementById("annual_inc").value,
            dti: document.getElementById("dti").value,
            int_rate: document.getElementById("int_rate").value,
            installment: document.getElementById("installment").value,
            revol_bal: document.getElementById("revol_bal").value
        };

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then(data => {
            const prob = data.prob_default; // in [0, 1]
            const percent = (prob * 100).toFixed(2);

            predictionOutput.style.display = "block";
            predictionOutput.textContent =
                `Predicted probability of default: ${percent}%`;

            if (data.filled_inputs) {
                filledInputs.style.display = "block";
                filledInputs.textContent =
                    "Features used by the model (blanks replaced by averages):\n" +
                    JSON.stringify(data.filled_inputs, null, 2);
            }

            resultsDiv.style.display = "block";
        })
        .catch(error => {
            console.error("Error during prediction:", error);
            alert("An error occurred while making the prediction.");
        });
    });
});
