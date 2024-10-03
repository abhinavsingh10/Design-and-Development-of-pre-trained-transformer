document.getElementById('generateForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const prompt = document.getElementById('prompt').value;
    const maxTokens = parseInt(document.getElementById('max_tokens').value, 10);
    const temperature = parseFloat(document.getElementById('temperature').value);

    fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            start_prompt: prompt,
            max_tokens: maxTokens,
            temperature: temperature
        })
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let text = '';

        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    document.getElementById('generatedText').textContent =text;
                    return;
                }
                text += decoder.decode(value, { stream: true });
                document.getElementById('generatedText').textContent = text;
                readStream(); // Continue reading
            });
        }

        readStream();
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('generatedText').textContent = 'Error generating text2______________.';
    });
});


/*slider js*/
var slide = document.getElementById("temperature");
var op = document.getElementById("slider_op");
op.innerHTML = slide.value;
slide.oninput = function(){
    op.innerHTML = slide.value;
}