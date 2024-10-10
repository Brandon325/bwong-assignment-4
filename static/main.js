document.getElementById('search-form').addEventListener('submit', function(e) {
    e.preventDefault();
    let query = document.getElementById('query').value;

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data.documents, data.similarities, data.indices);
        renderChart(data.similarities, data.indices);
    });
});

function displayResults(documents, similarities, indices) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    for (let i = 0; i < documents.length; i++) {
        let docDiv = document.createElement('div');
        let docContent = documents[i];
        docDiv.innerHTML = `
            <p><strong>Document Number:</strong> ${indices[i]}</p>
            <p><strong>Similarity:</strong> ${similarities[i].toFixed(4)}</p>
            <div class="doc-content">${docContent}</div>`;
        resultsDiv.appendChild(docDiv);
    }
}

function renderChart(similarities, indices) {
    let ctx = document.getElementById('chart').getContext('2d');
    if (window.barChart) {
        window.barChart.destroy();
    }
    window.barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: indices.map(i => 'Doc ' + i),
            datasets: [{
                label: 'Cosine Similarity',
                data: similarities,
                backgroundColor: 'rgba(0, 123, 255, 0.5)'
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
