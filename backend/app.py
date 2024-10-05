from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
# Ensure that 'propaganda_model' is the correct path to your model directory
model = BertForSequenceClassification.from_pretrained('propaganda_model')
tokenizer = BertTokenizer.from_pretrained('propaganda_model')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get the input text from the React frontend
        input_text = data.get('text', '')  # Ensure text is retrieved safely

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Check if CUDA is available and use it
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device

        # Pass input through model
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class (0 for no propaganda, 1 for propaganda)
        predicted_class = torch.argmax(logits, dim=1).item()

        # Return the predicted class (1 or 0) to the frontend
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500  # Handle errors gracefully

if __name__ == '__main__':
    app.run(debug=True)
