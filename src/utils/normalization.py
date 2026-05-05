import re

def normalize_person(name: str) -> str:
    if not name:
        return ""

    name = name.lower().strip()

    # email → name
    if "@" in name:
        name = name.split("@")[0]

    # usuń kropki i dziwne znaki
    name = name.replace(".", " ")

    # usuń nadmiarowe spacje
    name = re.sub(r"\s+", " ", name)

    return name.strip()
