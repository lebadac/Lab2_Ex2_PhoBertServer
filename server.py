from flask import Flask, request, jsonify
from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load the pre-trained PhoBERT sentiment classification model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

# Sentiment labels corresponding to the model's output classes
labels = ['negative', 'positive', 'neutral']

# Endpoint to analyze Vietnamese text sentiment using a pre-trained PhoBERT model
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # print(f"Probabilities: {probs.tolist()}")
        pred = torch.argmax(probs, dim=-1).item()
        label = labels[pred]
        return jsonify({'label': label})



#Run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

