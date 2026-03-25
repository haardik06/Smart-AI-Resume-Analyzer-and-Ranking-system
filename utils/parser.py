import re

#  Email extraction
def extract_email(text):
    match = re.findall(r'\S+@\S+', text)
    return match[0] if match else "Not Found"


#  Phone extraction
def extract_phone(text):
    match = re.findall(r'\b\d{10}\b', text)
    return match[0] if match else "Not Found"


#  Name extraction (simple heuristic)
def extract_name(text):
    lines = text.split("\n")
    for line in lines[:5]:
        if len(line.split()) <= 3 and line.isalpha():
            return line
    return "Not Found"


#  Education parsing
def extract_education(text):
    keywords = ["b.tech", "m.tech", "bsc", "msc", "degree"]
    return [word for word in keywords if word in text.lower()]


#  Experience parsing
def extract_experience(text):
    if "experience" in text.lower() or "intern" in text.lower():
        return "Experience Present"
    return "No Experience"


#  Section detection
def detect_sections(text):
    sections = []
    for sec in ["education", "experience", "skills", "projects"]:
        if sec in text.lower():
            sections.append(sec)
    return sections