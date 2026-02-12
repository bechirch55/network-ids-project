from flask import Flask, render_template, request
import pandas as pd
import os
from src import predict as predict_module

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    data = None
    summary = None
    
    if request.method == 'POST':
        print("DEBUG: POST request received") 
        
        if 'file' not in request.files:
            print("DEBUG: No file part in request")
            return "No file part"
            
        file = request.files['file']
        
        if file.filename == '':
            print("DEBUG: No selected file")
            return "No selected file"
            
        if file:
            print(f"DEBUG: Processing file: {file.filename}") 
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            
            original_df = pd.read_csv(filepath)
            print(f"DEBUG: CSV loaded. Rows: {len(original_df)}") 
            try:
                predicted_labels = predict_module.predict_labels(filepath)
                print(f"DEBUG: Prediction successful. First 5 labels: {predicted_labels[:5]}") 
            except Exception as e:
                print(f"DEBUG: Error during prediction: {e}")
                return f"Error during prediction: {e}"

            
            original_df['prediction_label'] = predicted_labels
            data = original_df.head(50).to_dict(orient='records')
            print(f"DEBUG: Data prepared for HTML. Count: {len(data)}") 
            summary = original_df['prediction_label'].value_counts().to_dict()
            print(f"DEBUG: Summary prepared: {summary}") 

    return render_template('index.html', data=data, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)