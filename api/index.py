from flask import Flask, render_template, request
import os
import sys
from collections import Counter
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_extraction import extract_text
from utils.preprocessing import preprocess_text
from utils.nlp_pipeline import run_nlp_pipeline
from utils.feature_extraction import transform_tfidf, bert_similarity
from utils.ml_model import load_models, load_metrics
from utils.scoring import generate_suggestions, calculate_ats_score
from utils.parser import *

# Create Flask app for Vercel
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
)

app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Load models globally
models = None
metrics = None

def load_app_models():
    global models, metrics
    if models is None:
        try:
            print("Loading pre-trained models...")
            models = load_models()
            metrics = load_metrics()
        except Exception as e:
            print(f"Error loading models: {e}")
            from utils.ml_model import train_from_dataset
            print("Training models from dataset...")
            models, metrics = train_from_dataset()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    load_app_models()
    
    files = request.files.getlist('resume')
    job_desc = request.form.get("job_desc", "")

    results = []

    for file in files:
        try:
            # Use BytesIO for serverless compatibility
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            text = extract_text(filepath)
            clean_text = preprocess_text(text)

            nlp_data = run_nlp_pipeline(clean_text)
            skills = nlp_data.get("skills", [])

            # Parsing
            email = extract_email(text)
            phone = extract_phone(text)
            name = extract_name(text)
            education = extract_education(text)
            experience = extract_experience(text)
            sections = detect_sections(text)

            # ML
            X_input = transform_tfidf([clean_text])
            predictions = {k: v.predict(X_input)[0] for k, v in models.items()}
            final_prediction = Counter(predictions.values()).most_common(1)[0][0]

            top_roles = [role for role, _ in Counter(predictions.values()).most_common(3)]

            # Scores
            bert_score = float(bert_similarity(clean_text, job_desc))

            job_keywords = job_desc.lower().split()
            matched = list(set(job_keywords) & set(skills))
            missing = list(set(job_keywords) - set(skills))

            skill_score = (len(matched) / len(set(job_keywords))) * 100 if job_keywords else 0

            score = (
                0.3 * bert_score +
                0.3 * skill_score +
                10 * ("education" in sections) +
                10 * ("experience" in sections) +
                10 * ("projects" in sections)
            )
            score = float(min(score, 100))

            ats_score = float(calculate_ats_score(text, job_desc, skills))
            suggestions = generate_suggestions(nlp_data, clean_text, missing)

            results.append({
                "filename": file.filename,
                "name": name,
                "email": email,
                "phone": phone,
                "education": education,
                "experience": experience,
                "sections": sections,
                "score": score,
                "ats_score": ats_score,
                "skills": skills,
                "missing_skills": missing,
                "final_prediction": final_prediction,
                "top_roles": top_roles,
                "similarity": float(round(bert_score, 2)),
                "suggestions": suggestions
            })
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return render_template("result.html", results=results, metrics=metrics)

# Health check endpoint for Vercel
@app.route('/health')
def health():
    return {"status": "ok"}, 200
