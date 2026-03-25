import spacy

nlp = spacy.load("en_core_web_sm")

skills_db = ["python", "java", "sql", "machine", "learning", "ai"]

def run_nlp_pipeline(text):
    doc = nlp(text)

    tokens = [token.text for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]

    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    skills = list(set([token.text.lower() for token in doc if token.text.lower() in skills_db]))

    chunks = [chunk.text for chunk in doc.noun_chunks]

    return {
        "tokens": tokens,
        "pos": pos_tags,
        "names": names,
        "skills": skills,
        "chunks": chunks
    }