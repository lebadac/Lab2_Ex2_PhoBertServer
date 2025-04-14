from flask import Flask, request, jsonify
from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load PhoBERT model
model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

labels = ['negative', 'positive', 'neutral']


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # # In ra các xác suất để kiểm tra (trước khi lấy nhãn)
        # print(f"Probabilities: {probs.tolist()}")

        # Đảm bảo lấy đúng thứ tự: [NEG, POS, NEU]
        pred = torch.argmax(probs, dim=-1).item()

        # Trả lại kết quả với xác suất từng lớp
        label = labels[pred]
        return jsonify({'label': label})




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

