document.addEventListener("DOMContentLoaded", function() {
    document.body.style.opacity = "1";
    var navItems = document.querySelectorAll("nav ul li a");
    navItems.forEach(function(item) {
        if (window.location.pathname === item.getAttribute('href')) {
            item.classList.add("active");
        }
    });

    var regressionForm = document.getElementById('regressionForm');
    if (regressionForm) {
        regressionForm.addEventListener('submit', runRegressor);
    }

    var classificationForm = document.getElementById('classificationForm');
    if (classificationForm) {
        classificationForm.addEventListener('submit', runClassifier);
    }
});


document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

function adjustHyperparameters() {
    var classifier = document.getElementById("classifier").value;

    // Reset to default state first
    document.getElementById("svc_params").style.display = "none";
    let noteAll = document.getElementById("note-all");
    let hyperparameterGroups = document.querySelectorAll('.input-group-flex .input-group select');
    
    if (classifier === "svc") {
        document.getElementById("svc_params").style.display = "block";
    } else if (classifier === "all") {
        noteAll.style.display = "block";
        
        // Disable hyperparameter options
        hyperparameterGroups.forEach(group => {
            group.setAttribute('disabled', 'true');
        });
    } else {
        noteAll.style.display = "none";
        
        // Enable hyperparameter options
        hyperparameterGroups.forEach(group => {
            group.removeAttribute('disabled');
        });
    }
}

document.getElementById('regression_algorithm').addEventListener('change', function() {
    // limit to 3 selections
    if (this.selectedOptions.length > 3) {
        alert('You can select up to 3 algorithms only.');
        // remove the latest selection to keep it to 3
        this.options[this.selectedIndex].selected = false;
        return; // exit the event listener after removing the selection
    }

    let featureSelectionDropdown = document.getElementById('feature_selection_method');
    let transformationDropdown = document.getElementById('transformation_method');
    let note = document.getElementById('note');

    // check for lasso or ridge selections
    let selectedValues = Array.from(this.selectedOptions).map(opt => opt.value);
    if (selectedValues.includes('lasso') || selectedValues.includes('ridge')) {
        featureSelectionDropdown.setAttribute('disabled', 'disabled');
        transformationDropdown.setAttribute('disabled', 'disabled');
        note.style.display = 'block';
    } else {
        featureSelectionDropdown.removeAttribute('disabled');
        transformationDropdown.removeAttribute('disabled');
        note.style.display = 'none';
    }
});


async function runRegressor(event) {
    event.preventDefault();

    let resultsSection = document.getElementById("results");
    resultsSection.style.display = "block";
    resultsSection.innerHTML = '<div class="loader"></div><p><em>Some models may take longer to run. Please bear with us!</em></p>';

    let formData = new FormData(event.target);

    let algoSelect = document.getElementById('regression_algorithm');
    if (algoSelect) {
        formData.delete('regression_algorithm');
        Array.from(algoSelect.selectedOptions).forEach(opt => {
            formData.append('regression_algorithm', opt.value);
        });
    }

    let data;
    try {
        let response = await fetch('/run-regressor', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Network response was not ok. Kindly report this issue via our Contact section. We apologize for any inconvenience.");
        }

        data = await response.json();

        // Check if data contains both comparisonTable and bestModelData
        if (!data.comparisonTable || !data.bestModelData) {
            throw new Error("Data is missing from the server response. Kindly report this issue via our Contact section. We apologize for any inconvenience.");
        }

        // Clear previous content
        resultsSection.innerHTML = '<h2>Results</h2>';

        // Display comparison table
        if (data.comparisonTable.length > 0) {
            let tableHTML = '<table id="comparisonTable"><thead><tr><th>Algorithm</th><th>R2</th><th>RMSE</th></tr></thead><tbody>';
            data.comparisonTable.forEach(item => {
                tableHTML += `<tr><td>${item.Algorithm}</td><td>${item.R2}</td><td>${item.RMSE}</td></tr>`;
            });
            tableHTML += '</tbody></table>';
            resultsSection.innerHTML += tableHTML;
        }


        const bestModelR2 = parseFloat(data.bestModelData.R2);
        if (bestModelR2 > 0.99 && bestModelR2 <= 1) {
            window.alert("Warning! The model might be overfitting as it's too perfect on training data.");
        }

        // If 2 or more algorithms are selected, indicate the best algorithm.
        if (algoSelect && algoSelect.selectedOptions.length > 1) {
            resultsSection.innerHTML += `<p><strong>Best Algorithm is:</strong> ${data.bestModelData.Algorithm} due to having highest R2 and lowest RMSE.</p>`;
        }


        // Display the best alpha (only if Ridge or Lasso are selected).
        if (data.bestModelData["Best Alpha"] && ["Ridge", "Lasso"].includes(data.bestModelData.Algorithm)) {
            resultsSection.innerHTML += `<p><strong>Best Alpha:</strong> ${data.bestModelData["Best Alpha"]}</p>`;
        }

         // Add R2 and RMSE explanations
        resultsSection.innerHTML += `
        <p><strong>About R2:</strong></p>
        <ul>
            <li>0.99 - 1: The model might be overfitting as it's too perfect on training data.</li>
            <li>0.9 - 0.99: Represents a model that is highly recommended for deployment.</li>
            <li>0.7 - 0.9: Represents a good model, but further optimization might be helpful.</li>
            <li>Below 0.7: Consider a different model or further feature engineering.</li>
        </ul>

        <p><strong>About RMSE:</strong></p>
        <ul>
            <li>Below 0.05: Low RMSE suggests model predictions are quite accurate.</li>
            <li>0.05 - 0.2: Moderate RMSE suggests model predictions are fairly accurate.</li>
            <li>Above 0.2: High RMSE suggests the model might be making prediction errors.</li>
        </ul>
        `

        // Add a description above the plot
        resultsSection.innerHTML += `<p>The plot below is generated based on the best algorithm selected - ${data.bestModelData.Algorithm}.</p>`;

        // render the scatter plot after updating the results section
        if (data.bestModelData.plotData) {
            renderScatterPlot(data.bestModelData.plotData);
        }

    } catch (error) {
        resultsSection.innerHTML = '<h2>Error</h2>' + error;
    }
}

function getScatterTrace(actual, predicted) {
    return {
        x: actual,
        y: predicted,
        mode: 'markers',
        type: 'scatter',
        name: 'Predicted vs Actual',
        marker: { size: 12 }
    };
}

function renderScatterPlot(plotData) {
    // Basic validations for plotData
    if (!plotData) {
        console.error("No plotData provided");
        return;
    }

    const { actual, predicted } = plotData;

    // Check for validity of the data
    if (!Array.isArray(actual) || !Array.isArray(predicted) || 
        actual.length !== predicted.length || 
        actual.length === 0) {
        console.error("Invalid plotData format or mismatched actual and predicted data lengths.");
        return;
    }

    // Prepare the trace
    const trace = {
        x: actual,
        y: predicted,
        mode: 'markers',
        type: 'scatter',
        name: 'Predicted vs Actual',
        marker: { size: 12 }
    };

    // Layout details
    const layout = {
        xaxis: {
            title: 'Actual Values'
        },
        yaxis: {
            title: 'Predicted Values'
        },
        title: 'Scatter Plot of Predicted vs Actual Values'
    };

    let scatterPlotDiv = document.createElement('div');
    scatterPlotDiv.id = 'scatter-plot';
    document.getElementById("results").appendChild(scatterPlotDiv);

    // Plot the data
    try {
        Plotly.newPlot('scatter-plot', [trace], layout);
    } catch (error) {
        console.error("Failed to render scatter plot:", error);
    }
}

function adjustHyperparameters() {
    var classifierSelect = document.getElementById('classifier');
    var selectedClassifiers = Array.from(classifierSelect.selectedOptions).map(option => option.value);

    // hide SVC and RandomForest parameters by default
    document.getElementById("svc_params").style.display = "none";
    document.getElementById("random_forest_params").style.display = "none";
    document.getElementById("decision_tree_params").style.display = "none";

    // if SVC is one of the selected classifiers, show its parameters
    if (selectedClassifiers.includes("svc")) {
        document.getElementById("svc_params").style.display = "block";
    }

    // if RandomForest is one of the selected classifiers, show its parameters
    if (selectedClassifiers.includes("random_forest")) {
        document.getElementById("random_forest_params").style.display = "block";
    }

    if (selectedClassifiers.includes("decision_tree")) {
        document.getElementById("decision_tree_params").style.display = "block";
    }

    // enforce a maximum of three selections
    if (selectedClassifiers.length > 3) {
        // Deselect the first selected option
        classifierSelect.options[classifierSelect.selectedIndex].selected = false;
        alert('Please select no more than three classifiers.');
        adjustHyperparameters(); // Re-adjust parameters
    }
}

// Add event listener to the classifier select element
document.getElementById('classifier').addEventListener('change', adjustHyperparameters);


function toProperCase(str) {
    const specialCases = { 'xgboost': 'XGBoost', 'catboost': 'CatBoost' };
    return specialCases[str] || str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

async function runClassifier(event) {
    event.preventDefault();

    let resultsSection = document.getElementById("results");
    resultsSection.style.display = "block";

    resultsSection.innerHTML = '<div class="loader"></div><p><em>Some models may take longer to run. Please bear with us!</em></p>';

    let formData = new FormData(event.target);

    try {
        let response = await fetch('/run-classifier', {  
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("There was an error processing your request.");
        }

        let data = await response.json();

        // Clear previous results and remove the loader
        resultsSection.innerHTML = '<h2>Classification Results</h2>';

        // Always show the comparison table
        let tableHTML = '<table><thead><tr><th>Classifier</th><th>Accuracy</th></tr></thead><tbody>';
        data.comparisonTable.forEach((item) => {
            tableHTML += `<tr><td>${toProperCase(item.Classifier)}</td><td>${item.Accuracy.toFixed(3)}</td></tr>`;
        });
        tableHTML += '</tbody></table>';
        resultsSection.innerHTML += tableHTML;

        // Add accuracy explanation
        resultsSection.innerHTML += `
            <p><strong>About Accuracy:</strong></p>
            <ul>
                <li>Above 0.9: Excellent accuracy, model is highly reliable.</li>
                <li>0.8 - 0.9: Good accuracy, model is reliable.</li>
                <li>0.7 - 0.8: Fair accuracy, some improvements needed.</li>
                <li>Below 0.7: Poor accuracy, consider different models or parameters.</li>
            </ul>
        `;

        // Get the best classifier's data
        let bestClassifierData = data.comparisonTable.find(item => item.Classifier === data.bestClassifier);

        // Show confusion matrix and classification report for the best classifier
        resultsSection.innerHTML += `
            <h3>Best Classifier: ${toProperCase(bestClassifierData.Classifier)}</h3>
            <p>Confusion Matrix:</p>
            <img src="/static/img/cm_${bestClassifierData.Classifier}.png" alt="Confusion Matrix Heatmap" style="max-width: 500px;">
            <p>Classification Report:</p>
            <pre>${bestClassifierData.ClassificationReport}</pre>
        `;
    } catch (error) {
        resultsSection.innerHTML = `<h2>Error</h2><p>${error}</p>`;
    }
}


function formatMatrix(matrix) {
    // Convert matrix array to string format for display
    return matrix.map(row => row.join(' ')).join('\n');
}

function makeEditable(id) {
    var element = document.getElementById(id);
    element.contentEditable = true;
    element.focus(); // Focus on the element to start editing

    element.addEventListener('blur', function() {
        saveAnalysisText(id, element.innerText);
        element.contentEditable = false;
    });
}

function saveAnalysisText(id, text) {
    fetch('/update-analysis', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({[id]: text})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Text updated successfully.');
        }
    });
}