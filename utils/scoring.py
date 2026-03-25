def calculate_ats_score(text, job_desc, skills):
    score = 0

    text_lower = text.lower()

    # Keyword match
    job_keywords = job_desc.lower().split()
    match_count = len(set(job_keywords) & set(skills))

    if job_keywords:
        score += (match_count / len(set(job_keywords))) * 50

    # Sections check
    if "education" in text_lower:
        score += 10
    if "experience" in text_lower:
        score += 10
    if "project" in text_lower:
        score += 10

    # Contact
    if "@" in text_lower:
        score += 10

    return min(score, 100)


def generate_suggestions(nlp_data, text, missing_skills):
    suggestions = []
    text_lower = text.lower()

    skills = nlp_data.get("skills", [])

    if missing_skills:
        suggestions.append("Add missing skills: " + ", ".join(missing_skills))

    if len(skills) < 5:
        suggestions.append("Add more technical skills")

    if "github.com" not in text_lower:
        suggestions.append("Add GitHub profile link")

    if "intern" not in text_lower and "experience" not in text_lower:
        suggestions.append("Add internship or work experience")

    if "project" not in text_lower:
        suggestions.append("Add projects section")

    action_words = ["developed", "built", "designed"]
    if not any(word in text_lower for word in action_words):
        suggestions.append("Use strong action words like Developed, Built, Designed")

    if not suggestions:
        suggestions.append("Your resume looks strong!")

    return suggestions