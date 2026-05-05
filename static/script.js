document.addEventListener('DOMContentLoaded', async () => {
    const symptomsContainer = document.getElementById('symptoms-container');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const btnText = document.getElementById('btn-text');
    const btnIcon = document.getElementById('btn-icon');

    const diseaseText = document.getElementById('predicted-disease');
    const confidenceText = document.getElementById('prediction-confidence');
    const descText = document.getElementById('prediction-desc');

    let selectedSymptoms = new Set();

    // 1. Fetch available symptoms from API
    try {
        const response = await fetch('/api/symptoms');
        const data = await response.json();
        
        symptomsContainer.innerHTML = '';
        data.symptoms.forEach(symptom => {
            const div = document.createElement('div');
            div.className = 'symptom-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `sym-${symptom}`;
            checkbox.value = symptom;
            
            const label = document.createElement('label');
            label.htmlFor = `sym-${symptom}`;
            label.className = 'symptom-label';
            label.textContent = symptom.replace(/_/g, ' ');

            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    selectedSymptoms.add(symptom);
                } else {
                    selectedSymptoms.delete(symptom);
                }
                predictBtn.disabled = selectedSymptoms.size === 0;
            });

            div.appendChild(checkbox);
            div.appendChild(label);
            symptomsContainer.appendChild(div);
        });
    } catch (error) {
        console.error('Error fetching symptoms:', error);
        symptomsContainer.innerHTML = '<p style="color: #ef4444;">Failed to load symptoms. Please refresh.</p>';
    }

    // 2. Handle Prediction
    predictBtn.addEventListener('click', async () => {
        if (selectedSymptoms.size === 0) return;

        // Show loading state
        loader.style.display = 'block';
        btnText.textContent = 'Analyzing...';
        btnIcon.style.display = 'none';
        predictBtn.disabled = true;
        resultContainer.style.display = 'none';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symptoms: Array.from(selectedSymptoms) })
            });

            const result = await response.json();

            if (result.error) {
                alert(result.error);
            } else {
                // Update UI with result
                diseaseText.textContent = result.disease;
                confidenceText.textContent = `Prediction Confidence: ${result.confidence}`;
                descText.textContent = result.description;

                // Show result with animation
                resultContainer.style.display = 'block';
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Something went wrong. Please try again.');
        } finally {
            loader.style.display = 'none';
            btnText.textContent = 'Predict Diagnosis';
            btnIcon.style.display = 'inline-block';
            predictBtn.disabled = false;
        }
    });
});
