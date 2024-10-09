document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const resultsDiv = document.getElementById('results');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultsDiv.innerHTML = '<p>Loading results...</p>';
        const query = document.getElementById('query').value;
        try {
            const response = await fetch('/', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `query=${encodeURIComponent(query)}`
            });
            const data = await response.json();
            resultsDiv.innerHTML = '';
            const labels = [];
            const scores = [];
            data.results.forEach((result, index) => {
                labels.push(`Doc ${index + 1}`);
                scores.push(result.score);
                const resultContainer = document.createElement('div');
                resultContainer.innerHTML = `
                    <p><strong>Document ${index + 1} (Score: ${result.score.toFixed(4)}):</strong></p>
                    <pre>${result.content}</pre>
                `;
                resultsDiv.appendChild(resultContainer);
            });
            // Draw chart
            const ctx = document.getElementById('chart').getContext('2d');
            if (window.chartInstance) {
                window.chartInstance.destroy();
            }
            window.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Cosine Similarity',
                        data: scores,
                        backgroundColor: 'rgba(0, 123, 255, 0.5)' // Match button color
                    }]
                },
                options: {
                    scales: {
                        y: { beginAtZero: true, max: 1 }
                    }
                }
            });
        } catch (error) {
            resultsDiv.innerHTML = '<p>An error occurred while fetching results.</p>';
            console.error(error);
        }
    });
});
