import json
import random
import re
from typing import List, Dict, Any
numbers = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def spell_number(num_str: str) -> str:
    return ' '.join(numbers[d] for d in num_str)

def generate_credit_card() -> str:
    digits = ''.join(random.choice('0123456789') for _ in range(16))
    return spell_number(digits)

def generate_phone() -> str:
    digits = ''.join(random.choice('0123456789') for _ in range(10))
    return spell_number(digits)

first_names = ['john', 'jane', 'mike', 'sarah', 'david', 'lisa', 'paul', 'anna']
last_names = ['smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis']
providers = ['gmail', 'yahoo', 'hotmail', 'outlook']

def generate_email() -> str:
    first = random.choice(first_names)
    last = random.choice(last_names)
    provider = random.choice(providers)
    return f"{first} dot {last} at {provider} dot com"

def generate_person_name() -> str:
    first = random.choice(first_names)
    last = random.choice(last_names)
    return f"{first} {last}"

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
days = [str(i) for i in range(1, 32)]
years = ['two thousand twenty three', 'two thousand twenty four', 'two thousand twenty five']

def generate_date() -> str:
    month = random.choice(months)
    day = random.choice(days)
    year = random.choice(years)
    return f"{month} {day} {year}"

cities = ['new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville', 'fort worth', 'columbus', 'indianapolis', 'charlotte', 'san francisco', 'seattle', 'denver', 'boston']

def generate_city() -> str:
    return random.choice(cities)

locations = ['central park', 'times square', 'hollywood', 'golden gate bridge', 'statue of liberty', 'grand canyon', 'yellowstone', 'mount rushmore', 'niagara falls', 'disneyland']

def generate_location() -> str:
    return random.choice(locations)

entity_generators = {
    'CREDIT_CARD': generate_credit_card,
    'PHONE': generate_phone,
    'EMAIL': generate_email,
    'PERSON_NAME': generate_person_name,
    'DATE': generate_date,
    'CITY': generate_city,
    'LOCATION': generate_location
}

templates = [
    "my credit card number is {CREDIT_CARD}",
    "call me on {PHONE}",
    "my email is {EMAIL}",
    "i am {PERSON_NAME}",
    "today is {DATE}",
    "i live in {CITY}",
    "the location is {LOCATION}",
    "please contact me at {PHONE} or email {EMAIL}",
    "my name is {PERSON_NAME} and i live in {CITY}",
    "on {DATE} i will be in {LOCATION}",
    "use my credit card {CREDIT_CARD} for payment",
]

def generate_example(entity_types: List[str]) -> Dict[str, Any]:
    # Decide number of entities: mostly 1, sometimes 2 (to expose model to multi-entity contexts)
    num_entities = 1 if random.random() < 0.7 else 2
    selected_types = random.sample(entity_types, num_entities)

    # Choose a template that contains at least one of the selected entity types if possible
    template = None
    random.shuffle(templates)
    for t in templates:
        for st in selected_types:
            if '{' + st + '}' in t:
                template = t
                break
        if template:
            break
    if template is None:
        template = random.choice(templates)

    # Fill any placeholders in template to avoid leftover blanks
    text = template
    for lab in ['CREDIT_CARD', 'PHONE', 'EMAIL', 'PERSON_NAME', 'DATE', 'CITY', 'LOCATION']:
        if '{' + lab + '}' in text:
            text = text.replace('{' + lab + '}', entity_generators[lab](), 1)
    entities = []

    # helper to add filler/noise to simulate STT
    def add_noise(s: str) -> str:
        # occasional filler tokens and hesitations
        fillers = ["", " um", " uh", " please", " thanks"]
        if random.random() < 0.25:
            s = s + random.choice(fillers)
        # randomly remove punctuation (most text has none) and lowercase
        return s.replace(" ,", ",").strip()

    for et in selected_types:
        entity_text = entity_generators[et]()
        # small variations: phone may include country code, credit card grouped or continuous
        if et == 'PHONE' and random.random() < 0.3:
            # sometimes add country code 'nine one' (for 91) or 'one' (US)
            cc = random.choice(['one', 'nine one', ''])
            if cc:
                entity_text = f"{cc} {entity_text}"
        if et == 'CREDIT_CARD' and random.random() < 0.4:
            # group digits into 4-digit groups sometimes
            digits = ''.join(random.choice('0123456789') for _ in range(16))
            groups = [digits[i:i+4] for i in range(0, 16, 4)]
            entity_text = ' '.join(spell_number(g) for g in groups)
        if et == 'EMAIL' and random.random() < 0.3:
            # sometimes omit the 'dot' between provider and com
            entity_text = entity_text.replace(' dot com', ' dot com')

        # If the chosen template already contains this entity we already filled it above,
        # otherwise append it naturally to the sentence (avoid leaving other placeholders)
        if entity_text in text:
            # already filled by placeholder replacement
            pass
        else:
            connector = random.choice([" and ", " , also ", " then ", " "])
            text = text + connector + entity_text

        # compute start/end for this inserted entity (use last occurrence)
        start = text.rfind(entity_text)
        end = start + len(entity_text)
        entities.append({"start": start, "end": end, "label": et})

    # Add occasional filler/noise
    if random.random() < 0.3:
        text = add_noise(text)

    text = re.sub(r'\{[A-Z_]+\}', '', text).strip()
    return {
        "id": f"utt_{random.randint(1000, 9999)}",
        "text": text,
        "entities": entities
    }

def generate_dataset(num_examples: int, entity_types: List[str]) -> List[Dict[str, Any]]:
    data = []
    for _ in range(num_examples):
        data.append(generate_example(entity_types))
    return data

if __name__ == "__main__":
    random.seed(42)
    train_data = generate_dataset(800, ['CREDIT_CARD', 'PHONE', 'EMAIL', 'PERSON_NAME', 'DATE', 'CITY', 'LOCATION'])
    dev_data = generate_dataset(150, ['CREDIT_CARD', 'PHONE', 'EMAIL', 'PERSON_NAME', 'DATE', 'CITY', 'LOCATION'])

    with open('data/train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open('data/dev.jsonl', 'w') as f:
        for item in dev_data:
            f.write(json.dumps(item) + '\n')