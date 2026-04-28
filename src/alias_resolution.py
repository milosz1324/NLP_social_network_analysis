from rapidfuzz import process, fuzz

def resolve_people(names, threshold=92):
    """
    Merges similar / misspelled person names using fuzzy matching.
    """

    canonical = []

    mapping = {}

    for name in sorted(set(names)):

        if not canonical:
            canonical.append(name)
            mapping[name] = name
            continue

        match, score, _ = process.extractOne(
            name,
            canonical,
            scorer=fuzz.ratio
        )

        if score >= threshold:
            mapping[name] = match
        else:
            canonical.append(name)
            mapping[name] = name

    return mapping
