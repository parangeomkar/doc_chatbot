import numpy as np
import pandas as pd
from sentence_transformers import util
import torch
import requests
from flask import Flask, request, Response, jsonify, render_template_string, stream_with_context

app = Flask(__name__)
model = "llama3:latest"
embedding_model = "all-minilm:latest"
data = pd.read_csv("all-minilm_embeddings.csv")
embeddings = data["embedding"].apply(lambda x: np.array(eval(x)))
sentences = np.array(data["sentence"])
embeddings = torch.tensor(np.array(embeddings.tolist()), dtype=torch.float32).to("cpu")

def generateText(prompt):
    res = requests.post("http://172.210.9.232:8888/api/generate", json={
        'model': model,
        'prompt': prompt,
        "stream": False
    }, stream=False)
    return res

@app.route('/query', methods=['POST'])
def ask():
    data = request.get_json()
    print("Generating embeddings...")
    res = requests.post("http://172.210.9.232:8888/api/embeddings", json={
        'model': embedding_model,
        'prompt': data["query"],
        "options": {
            "temperature": 0.3
        }
    })

    embedding = torch.tensor(np.array((res.json())["embedding"]), dtype=torch.float32).to("cpu")
    dot_scores = util.cos_sim(embedding, embeddings)[0]
    indices = np.argsort(np.array(dot_scores))[-5:]

    context_items = [sentences[i] for i in indices[::-1]]
    context = "- " + "\n- ".join(context_items)
    base_prompt = """Based on the following context items, please answer the query.
    {context}
    \n
    User query: {query}
    Answer:"""
    prompt = base_prompt.format(context=context, query=data["query"])
    
    print("Generating answer...")
    return generateText(prompt).json()

@app.route('/', methods=['GET'])
def get_html():
    # Define your HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample ChatBot</title>
        <script>
            async function ask(){
                const output = document.getElementById('answer');
                output.innerHTML = '';  // Clear previous output

                const query = document.getElementById("query").value;
                const response = await fetch("http://localhost:5000/query",{
                    method: "POST",
                    body: JSON.stringify({
                        query
                    }),
                    headers: {
                        "content-type": "application/json"
                    }
                })
            
                const result = await response.json();
                output.innerHTML = result.response;
            }
        </script>
    </head>
    <body>
        <h1>Ask anything about inventory management and requisitions! (^,^)</h1>
        <div>
            <textarea id="query"></textarea>
            <div id="answer"></div>
            <button onclick="ask()">Submit</button>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run()