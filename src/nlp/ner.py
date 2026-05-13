import spacy

nlp = spacy.load("en_core_web_sm")


def extract_people(text: str) -> list[str]:
    if not text:
        return []

    doc = nlp(text)

    return [
        ent.text.strip().lower()
        for ent in doc.ents
        if ent.label_ == "PERSON"
    ]


def clean_people(people: list[str]) -> list[str]:
    cleaned = []

    for p in people:
        p = p.strip().lower()

        if len(p) < 5:
            continue
        if len(p.split()) < 2:
            continue

        cleaned.append(p)

    return list(set(cleaned))
