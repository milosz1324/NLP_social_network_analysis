import spacy
from tqdm import tqdm

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


def extract_people_batch(texts: list[str], batch_size: int = 64) -> list[list[str]]:
    """Process all texts through spaCy in batches — much faster than one-by-one."""
    results = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        desc="NER",
        unit="email",
    ):
        people = [
            ent.text.strip().lower()
            for ent in doc.ents
            if ent.label_ == "PERSON"
        ]
        results.append(clean_people(people))
    return results
