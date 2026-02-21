"""
Named Entity Recognition (NER) System
======================================
A rule-based NER system using regex patterns and gazetteers to identify
entities: PERSON, ORGANIZATION, LOCATION, DATE, MONEY.
Includes synthetic text generation, evaluation, and entity visualization.

Author: Data Science Portfolio
"""

import numpy as np
import re
import warnings
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. GAZETTEERS (ENTITY DICTIONARIES)
# ============================================================================

GAZETTEERS = {
    'PERSON': {
        'first_names': [
            'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
            'Joseph', 'Thomas', 'Charles', 'Mary', 'Patricia', 'Jennifer', 'Linda',
            'Barbara', 'Elizabeth', 'Susan', 'Jessica', 'Sarah', 'Karen',
            'Alexander', 'Benjamin', 'Catherine', 'Daniel', 'Eleanor', 'Frederick',
            'George', 'Hannah', 'Isaac', 'Julia', 'Katherine', 'Lawrence',
            'Margaret', 'Nathan', 'Olivia', 'Patrick', 'Rachel', 'Samuel',
            'Victoria', 'Warren',
        ],
        'last_names': [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
            'Davis', 'Rodriguez', 'Martinez', 'Anderson', 'Taylor', 'Thomas',
            'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson', 'White', 'Harris',
            'Clark', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King',
            'Wright', 'Scott', 'Hill', 'Green', 'Adams', 'Baker', 'Nelson',
            'Carter', 'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips',
        ],
        'titles': ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'President', 'Senator',
                   'Governor', 'Director', 'CEO', 'Chairman', 'Secretary'],
    },
    'ORGANIZATION': [
        'Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook', 'Meta', 'Tesla',
        'Netflix', 'IBM', 'Intel', 'Oracle', 'Cisco', 'Adobe', 'Salesforce',
        'United Nations', 'World Health Organization', 'NATO', 'European Union',
        'World Bank', 'International Monetary Fund', 'Red Cross',
        'Harvard University', 'Stanford University', 'MIT',
        'Goldman Sachs', 'JPMorgan Chase', 'Morgan Stanley', 'BlackRock',
        'The New York Times', 'The Washington Post', 'Reuters', 'BBC',
        'NASA', 'SpaceX', 'Boeing', 'Lockheed Martin',
        'Pfizer', 'Johnson & Johnson', 'Moderna',
        'Toyota', 'Volkswagen', 'Samsung', 'Sony',
        'Department of Defense', 'Federal Reserve', 'Supreme Court',
        'Congress', 'Senate', 'House of Representatives',
    ],
    'LOCATION': {
        'cities': [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'San Francisco', 'Seattle', 'Boston', 'Denver', 'Atlanta',
            'London', 'Paris', 'Tokyo', 'Beijing', 'Mumbai',
            'Berlin', 'Sydney', 'Toronto', 'Dubai', 'Singapore',
            'Moscow', 'Seoul', 'Bangkok', 'Istanbul', 'Rome',
            'San Diego', 'Dallas', 'Miami', 'Philadelphia', 'Washington',
        ],
        'states': [
            'California', 'Texas', 'Florida', 'New York', 'Pennsylvania',
            'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan',
            'Virginia', 'Massachusetts', 'Colorado', 'Washington', 'Arizona',
        ],
        'countries': [
            'United States', 'China', 'India', 'Japan', 'Germany',
            'United Kingdom', 'France', 'Brazil', 'Canada', 'Australia',
            'Russia', 'South Korea', 'Mexico', 'Indonesia', 'Italy',
            'Spain', 'Netherlands', 'Saudi Arabia', 'Switzerland', 'Sweden',
        ],
        'landmarks': [
            'Wall Street', 'Silicon Valley', 'Capitol Hill', 'Pentagon',
            'White House', 'Kremlin', 'Buckingham Palace',
        ],
    },
}


# ============================================================================
# 2. SYNTHETIC TEXT GENERATION WITH ENTITY ANNOTATIONS
# ============================================================================

def generate_entity_text(n_documents=50):
    """
    Generate synthetic text documents with known entity positions.
    Returns documents and their ground truth entity annotations.
    """
    templates = [
        "{person} announced that {org} will invest ${money} in a new facility in {location}.",
        "On {date}, {person} of {org} met with officials in {location} to discuss the ${money} deal.",
        "{org} CEO {person} revealed plans to expand operations to {location} by {date}.",
        "The {money} acquisition of {org} by {person}'s company was finalized on {date} in {location}.",
        "According to {person}, {org} reported revenues of ${money} for the quarter ending {date}.",
        "{person} traveled to {location} on {date} for the {org} annual conference.",
        "In {location}, {org} opened a new headquarters worth ${money}, according to {person}.",
        "Dr. {person} published a study funded by {org} on {date} while working in {location}.",
        "The {org} board, led by Chairman {person}, approved a ${money} budget on {date} for {location}.",
        "Senator {person} from {location} proposed a ${money} bill to regulate {org} on {date}.",
        "{person} was appointed as the new director of {org} in {location}, effective {date}.",
        "A ${money} contract between {org} and the city of {location} was signed by {person} on {date}.",
        "Prof. {person} at {org} received a ${money} grant to study climate change in {location}.",
        "{org} spokesperson {person} confirmed that the {location} office will close by {date}, saving ${money}.",
        "On {date}, {person} told reporters in {location} that {org} would cut costs by ${money}.",
    ]

    first_names = GAZETTEERS['PERSON']['first_names']
    last_names = GAZETTEERS['PERSON']['last_names']
    orgs = GAZETTEERS['ORGANIZATION']
    cities = GAZETTEERS['LOCATION']['cities']
    countries = GAZETTEERS['LOCATION']['countries']

    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    documents = []
    annotations = []

    for _ in range(n_documents):
        # Generate entities
        first = np.random.choice(first_names)
        last = np.random.choice(last_names)
        person = f"{first} {last}"
        org = np.random.choice(orgs)
        location = np.random.choice(cities + countries)
        month = np.random.choice(months)
        day = np.random.randint(1, 29)
        year = np.random.randint(2020, 2026)
        date = f"{month} {day}, {year}"
        amount = np.random.choice([
            f"{np.random.randint(1, 999)} million",
            f"{np.random.randint(1, 99)} billion",
            f"{np.random.randint(100, 999)},{np.random.randint(100, 999):03d}",
            f"{np.random.randint(1, 99)}.{np.random.randint(1, 99)} million",
        ])

        # Build 2-4 sentences per document
        n_sentences = np.random.randint(2, 5)
        chosen_templates = np.random.choice(templates, size=n_sentences, replace=False)
        doc_sentences = []
        doc_entities = []

        running_offset = 0
        for template in chosen_templates:
            # Re-generate some entities for variety within the document
            if np.random.random() > 0.5:
                first2 = np.random.choice(first_names)
                last2 = np.random.choice(last_names)
                current_person = f"{first2} {last2}"
            else:
                current_person = person

            if np.random.random() > 0.6:
                current_org = np.random.choice(orgs)
            else:
                current_org = org

            if np.random.random() > 0.6:
                current_location = np.random.choice(cities + countries)
            else:
                current_location = location

            current_month = np.random.choice(months)
            current_day = np.random.randint(1, 29)
            current_year = np.random.randint(2020, 2026)
            current_date = f"{current_month} {current_day}, {current_year}"

            current_amount = np.random.choice([
                f"{np.random.randint(1, 999)} million",
                f"{np.random.randint(1, 99)} billion",
                f"{np.random.randint(100, 999)},{np.random.randint(100, 999):03d}",
            ])

            sentence = template.format(
                person=current_person, org=current_org,
                location=current_location, date=current_date,
                money=current_amount,
            )
            doc_sentences.append(sentence)

            # Find entity positions in sentence
            for entity_text, entity_type in [
                (current_person, 'PERSON'),
                (current_org, 'ORGANIZATION'),
                (current_location, 'LOCATION'),
                (current_date, 'DATE'),
                ('$' + current_amount, 'MONEY'),
            ]:
                start = sentence.find(entity_text)
                if start >= 0:
                    doc_entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'start': running_offset + start,
                        'end': running_offset + start + len(entity_text),
                    })

            running_offset += len(sentence) + 1  # +1 for space

        full_doc = ' '.join(doc_sentences)
        documents.append(full_doc)
        annotations.append(doc_entities)

    print(f"Generated {len(documents)} documents with entity annotations")

    # Count entities by type
    entity_counts = Counter()
    for ann in annotations:
        for ent in ann:
            entity_counts[ent['type']] += 1
    print(f"Entity distribution: {dict(entity_counts)}")

    return documents, annotations


# ============================================================================
# 3. RULE-BASED NER SYSTEM
# ============================================================================

class RuleBasedNER:
    """
    Named Entity Recognition system using regex patterns and gazetteers.

    Entity types supported:
    - PERSON: Names matching gazetteer or title patterns
    - ORGANIZATION: Known organization names and patterns
    - LOCATION: Cities, countries, states, and location patterns
    - DATE: Various date formats
    - MONEY: Currency amounts and patterns
    """

    def __init__(self):
        self.gazetteers = GAZETTEERS
        self._build_patterns()

    def _build_patterns(self):
        """Compile regex patterns for each entity type."""
        self.patterns = {}

        # --- DATE patterns ---
        months = ('January|February|March|April|May|June|July|August|'
                  'September|October|November|December')
        months_short = 'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
        self.patterns['DATE'] = [
            # January 15, 2024
            re.compile(rf'\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b'),
            # Jan 15, 2024
            re.compile(rf'\b({months_short})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b'),
            # 15 January 2024
            re.compile(rf'\b(\d{{1,2}})\s+({months})\s+(\d{{4}})\b'),
            # 2024-01-15 (ISO format)
            re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
            # 01/15/2024 or 01-15-2024
            re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
            # Relative dates
            re.compile(r'\b(today|yesterday|tomorrow)\b', re.IGNORECASE),
            # Weekdays
            re.compile(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'),
        ]

        # --- MONEY patterns ---
        self.patterns['MONEY'] = [
            # $1,234,567 or $1,234,567.89
            re.compile(r'\$[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion)?'),
            # $1.5 million/billion
            re.compile(r'\$\d+(?:\.\d+)?\s+(?:million|billion|trillion)'),
            # 1.5 million dollars
            re.compile(r'\b\d+(?:\.\d+)?\s+(?:million|billion|trillion)\s+dollars\b'),
            # USD 1,234
            re.compile(r'\b(?:USD|EUR|GBP|JPY)\s*[\d,]+(?:\.\d{1,2})?'),
        ]

        # --- PERSON patterns ---
        titles = '|'.join(re.escape(t) for t in self.gazetteers['PERSON']['titles'])
        self.patterns['PERSON_TITLE'] = re.compile(
            rf'\b({titles})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        )

        # Build person name set for gazetteer lookup
        self.person_names = set()
        for first in self.gazetteers['PERSON']['first_names']:
            for last in self.gazetteers['PERSON']['last_names']:
                self.person_names.add(f"{first} {last}")

        # --- ORGANIZATION set ---
        self.org_names = set(self.gazetteers['ORGANIZATION'])

        # Organization suffix patterns
        self.patterns['ORG_SUFFIX'] = re.compile(
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)'
            r'\s+(Inc\.|Corp\.|Ltd\.|LLC|Co\.|Group|Foundation|Institute|Association|Corporation)\b'
        )

        # --- LOCATION sets ---
        self.location_names = set()
        for category in self.gazetteers['LOCATION'].values():
            if isinstance(category, list):
                self.location_names.update(category)
            elif isinstance(category, dict):
                for items in category.values():
                    self.location_names.update(items)

    def recognize(self, text):
        """
        Recognize named entities in the given text.

        Parameters
        ----------
        text : str
            Input text to process.

        Returns
        -------
        list of dict
            Recognized entities with text, type, start, and end positions.
        """
        entities = []
        used_spans = []  # Track recognized spans to avoid overlaps

        def span_overlaps(start, end):
            for s, e in used_spans:
                if start < e and end > s:
                    return True
            return False

        def add_entity(text_match, entity_type, start, end):
            if not span_overlaps(start, end):
                entities.append({
                    'text': text_match,
                    'type': entity_type,
                    'start': start,
                    'end': end,
                })
                used_spans.append((start, end))

        # 1. DATE recognition (highest priority -- dates can look like numbers)
        for pattern in self.patterns['DATE']:
            for match in pattern.finditer(text):
                add_entity(match.group(), 'DATE', match.start(), match.end())

        # 2. MONEY recognition
        for pattern in self.patterns['MONEY']:
            for match in pattern.finditer(text):
                add_entity(match.group(), 'MONEY', match.start(), match.end())

        # 3. ORGANIZATION recognition (before PERSON to handle longer matches first)
        # Gazetteer lookup (longest match first)
        for org in sorted(self.org_names, key=len, reverse=True):
            pattern = re.compile(r'\b' + re.escape(org) + r'\b')
            for match in pattern.finditer(text):
                add_entity(match.group(), 'ORGANIZATION', match.start(), match.end())

        # Organization suffix patterns
        for match in self.patterns['ORG_SUFFIX'].finditer(text):
            add_entity(match.group(), 'ORGANIZATION', match.start(), match.end())

        # 4. PERSON recognition
        # Title-based recognition
        for match in self.patterns['PERSON_TITLE'].finditer(text):
            full_match = match.group()
            # The person name is the part after the title
            name_part = match.group(2)
            add_entity(full_match, 'PERSON', match.start(), match.end())

        # Gazetteer lookup for known full names
        for name in sorted(self.person_names, key=len, reverse=True):
            pattern = re.compile(r'\b' + re.escape(name) + r'\b')
            for match in pattern.finditer(text):
                add_entity(match.group(), 'PERSON', match.start(), match.end())

        # Capitalized word sequences (possible person names) -- heuristic
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b')
        for match in cap_pattern.finditer(text):
            candidate = match.group()
            # Skip if already matched or if it's a common non-name phrase
            if not span_overlaps(match.start(), match.end()):
                words = candidate.split()
                # Check if first word is a known first name
                if words[0] in self.gazetteers['PERSON']['first_names']:
                    add_entity(candidate, 'PERSON', match.start(), match.end())

        # 5. LOCATION recognition
        for loc in sorted(self.location_names, key=len, reverse=True):
            pattern = re.compile(r'\b' + re.escape(loc) + r'\b')
            for match in pattern.finditer(text):
                add_entity(match.group(), 'LOCATION', match.start(), match.end())

        # Sort entities by position
        entities.sort(key=lambda e: e['start'])
        return entities

    def recognize_batch(self, texts):
        """Recognize entities in multiple texts."""
        return [self.recognize(text) for text in texts]


# ============================================================================
# 4. CUSTOM FEATURE EXTRACTION FOR ENTITY DETECTION
# ============================================================================

class EntityFeatureExtractor:
    """
    Extract features for potential entity spans.
    These features can be used with a classifier for ML-based NER.
    """

    def __init__(self):
        self.gazetteers = GAZETTEERS

    def extract_word_features(self, word, context_before='', context_after=''):
        """Extract features for a single word in context."""
        features = {
            'word_lower': word.lower(),
            'is_capitalized': word[0].isupper() if word else False,
            'is_all_caps': word.isupper(),
            'is_all_lower': word.islower(),
            'has_digit': any(c.isdigit() for c in word),
            'is_digit': word.isdigit(),
            'has_hyphen': '-' in word,
            'has_period': '.' in word,
            'word_length': len(word),
            'prefix_2': word[:2].lower() if len(word) >= 2 else word.lower(),
            'prefix_3': word[:3].lower() if len(word) >= 3 else word.lower(),
            'suffix_2': word[-2:].lower() if len(word) >= 2 else word.lower(),
            'suffix_3': word[-3:].lower() if len(word) >= 3 else word.lower(),
            'is_title': word.istitle(),
            'has_dollar': '$' in word,
            'is_first_name': word in self.gazetteers['PERSON']['first_names'],
            'is_last_name': word in self.gazetteers['PERSON']['last_names'],
            'is_title_word': word in self.gazetteers['PERSON']['titles'],
            'prev_word': context_before.split()[-1].lower() if context_before.split() else '',
            'next_word': context_after.split()[0].lower() if context_after.split() else '',
        }
        return features

    def extract_span_features(self, text, start, end):
        """Extract features for a text span."""
        span_text = text[start:end]
        context_before = text[max(0, start - 50):start]
        context_after = text[end:min(len(text), end + 50)]

        features = {
            'span_text': span_text,
            'span_length': len(span_text),
            'n_words': len(span_text.split()),
            'all_capitalized': all(w[0].isupper() for w in span_text.split() if w),
            'has_number': any(c.isdigit() for c in span_text),
            'has_dollar': '$' in span_text,
            'has_comma': ',' in span_text,
            'in_org_gazetteer': span_text in self.gazetteers['ORGANIZATION'],
            'in_location_gazetteer': any(
                span_text in items
                for items in self.gazetteers['LOCATION'].values()
                if isinstance(items, list)
            ),
            'preceded_by_title': any(
                context_before.rstrip().endswith(t)
                for t in self.gazetteers['PERSON']['titles']
            ),
            'context_before': context_before[-30:] if context_before else '',
            'context_after': context_after[:30] if context_after else '',
        }
        return features


# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

def compute_entity_metrics(true_entities, pred_entities, entity_types=None):
    """
    Compute precision, recall, and F1 for entity recognition.

    Uses exact match: both the entity text and type must match.

    Parameters
    ----------
    true_entities : list of list of dict
        Ground truth entities for each document.
    pred_entities : list of list of dict
        Predicted entities for each document.
    entity_types : list of str, optional
        Entity types to evaluate. If None, evaluate all types.

    Returns
    -------
    dict
        Metrics per entity type and overall.
    """
    if entity_types is None:
        entity_types = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'MONEY']

    metrics = {}

    for etype in entity_types:
        tp = 0
        fp = 0
        fn = 0

        for true_ents, pred_ents in zip(true_entities, pred_entities):
            true_set = {(e['text'], e['type']) for e in true_ents if e['type'] == etype}
            pred_set = {(e['text'], e['type']) for e in pred_ents if e['type'] == etype}

            tp += len(true_set & pred_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[etype] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': tp + fn,
        }

    # Overall metrics
    total_tp = sum(m['tp'] for m in metrics.values())
    total_fp = sum(m['fp'] for m in metrics.values())
    total_fn = sum(m['fn'] for m in metrics.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall /
                  (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0.0)

    metrics['OVERALL'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'support': total_tp + total_fn,
    }

    return metrics


def print_evaluation_report(metrics):
    """Print a formatted evaluation report."""
    print(f"\n{'Entity Type':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'Support':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 86)

    for etype in ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'MONEY']:
        if etype in metrics:
            m = metrics[etype]
            print(f"{etype:<16} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['support']:>10d} {m['tp']:>6d} "
                  f"{m['fp']:>6d} {m['fn']:>6d}")

    print("-" * 86)
    m = metrics['OVERALL']
    print(f"{'OVERALL':<16} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {m['support']:>10d} {m['tp']:>6d} "
          f"{m['fp']:>6d} {m['fn']:>6d}")


# ============================================================================
# 6. ENTITY VISUALIZATION
# ============================================================================

ENTITY_COLORS = {
    'PERSON': '#FF6B6B',
    'ORGANIZATION': '#4ECDC4',
    'LOCATION': '#45B7D1',
    'DATE': '#96CEB4',
    'MONEY': '#FFEAA7',
}


def visualize_entities_text(text, entities, max_length=500):
    """
    Create a colored text representation of recognized entities.
    Prints to console with entity type labels.
    """
    if len(text) > max_length:
        text = text[:max_length] + "..."
        entities = [e for e in entities if e['end'] <= max_length]

    # Sort entities by start position
    entities = sorted(entities, key=lambda e: e['start'])

    result_parts = []
    last_end = 0

    for entity in entities:
        # Add text before entity
        if entity['start'] > last_end:
            result_parts.append(text[last_end:entity['start']])

        # Add entity with markup
        entity_text = text[entity['start']:entity['end']]
        result_parts.append(f"[{entity_text}]({entity['type']})")
        last_end = entity['end']

    # Add remaining text
    if last_end < len(text):
        result_parts.append(text[last_end:])

    return ''.join(result_parts)


def plot_entity_distribution(all_entities, save_path='entity_distribution.png'):
    """Plot the distribution of entity types across the corpus."""
    type_counts = Counter()
    for doc_entities in all_entities:
        for entity in doc_entities:
            type_counts[entity['type']] += 1

    entity_types = list(type_counts.keys())
    counts = [type_counts[t] for t in entity_types]
    colors = [ENTITY_COLORS.get(t, '#999999') for t in entity_types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(entity_types, counts, color=colors, edgecolor='gray')
    ax1.set_ylabel('Count')
    ax1.set_title('Entity Type Distribution')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=entity_types, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Entity Type Proportions')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Entity distribution plot saved to {save_path}")


def plot_evaluation_results(metrics, save_path='ner_evaluation.png'):
    """Plot precision, recall, and F1 per entity type."""
    entity_types = [t for t in metrics if t != 'OVERALL']
    n_types = len(entity_types)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grouped bar chart for P, R, F1
    x = np.arange(n_types)
    width = 0.25

    precisions = [metrics[t]['precision'] for t in entity_types]
    recalls = [metrics[t]['recall'] for t in entity_types]
    f1s = [metrics[t]['f1'] for t in entity_types]

    axes[0].bar(x - width, precisions, width, label='Precision', color='steelblue')
    axes[0].bar(x, recalls, width, label='Recall', color='coral')
    axes[0].bar(x + width, f1s, width, label='F1', color='mediumseagreen')

    axes[0].set_xlabel('Entity Type')
    axes[0].set_ylabel('Score')
    axes[0].set_title('NER Evaluation: Precision, Recall, F1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(entity_types, rotation=30, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1.15)

    # Support and error analysis
    tps = [metrics[t]['tp'] for t in entity_types]
    fps = [metrics[t]['fp'] for t in entity_types]
    fns = [metrics[t]['fn'] for t in entity_types]

    axes[1].bar(x - width, tps, width, label='True Positives', color='mediumseagreen')
    axes[1].bar(x, fps, width, label='False Positives', color='coral')
    axes[1].bar(x + width, fns, width, label='False Negatives', color='steelblue')

    axes[1].set_xlabel('Entity Type')
    axes[1].set_ylabel('Count')
    axes[1].set_title('NER Error Analysis')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(entity_types, rotation=30, ha='right')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plot saved to {save_path}")


def plot_entity_spans(text, entities, save_path='entity_spans.png', max_chars=300):
    """
    Create a visual representation of entity spans in text.
    """
    if len(text) > max_chars:
        display_text = text[:max_chars] + '...'
        display_entities = [e for e in entities if e['end'] <= max_chars]
    else:
        display_text = text
        display_entities = entities

    fig, ax = plt.subplots(figsize=(16, 4))

    # Display text
    chars_per_line = 80
    lines = [display_text[i:i + chars_per_line]
             for i in range(0, len(display_text), chars_per_line)]

    y_offset = len(lines) * 0.15 + 0.1
    for line_idx, line in enumerate(lines):
        y = y_offset - line_idx * 0.15
        ax.text(0.02, y, line, fontsize=8, fontfamily='monospace',
                transform=ax.transAxes, verticalalignment='top')

    # Draw entity highlights
    for entity in display_entities:
        line_idx = entity['start'] // chars_per_line
        char_pos = entity['start'] % chars_per_line
        entity_len = entity['end'] - entity['start']

        y = y_offset - line_idx * 0.15 - 0.02
        x_start = 0.02 + char_pos * 0.0095
        x_width = entity_len * 0.0095

        color = ENTITY_COLORS.get(entity['type'], '#999999')
        rect = mpatches.FancyBboxPatch(
            (x_start, y - 0.04), x_width, 0.08,
            boxstyle="round,pad=0.005",
            facecolor=color, alpha=0.3,
            edgecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

    # Legend
    legend_patches = [mpatches.Patch(color=ENTITY_COLORS[t], label=t, alpha=0.5)
                      for t in ENTITY_COLORS]
    ax.legend(handles=legend_patches, loc='lower center', ncol=5,
              fontsize=9, framealpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Named Entity Recognition Visualization', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Entity span visualization saved to {save_path}")


# ============================================================================
# 7. ERROR ANALYSIS
# ============================================================================

def analyze_errors(true_entities, pred_entities, documents, n_examples=5):
    """
    Detailed error analysis of NER predictions.
    """
    print(f"\n{'=' * 60}")
    print("NER ERROR ANALYSIS")
    print(f"{'=' * 60}")

    all_fps = []  # False positives
    all_fns = []  # False negatives

    for doc_idx, (true_ents, pred_ents) in enumerate(zip(true_entities, pred_entities)):
        true_set = {(e['text'], e['type']) for e in true_ents}
        pred_set = {(e['text'], e['type']) for e in pred_ents}

        fps = pred_set - true_set
        fns = true_set - pred_set

        for text, etype in fps:
            all_fps.append({'text': text, 'type': etype, 'doc_idx': doc_idx})
        for text, etype in fns:
            all_fns.append({'text': text, 'type': etype, 'doc_idx': doc_idx})

    # False Positive Analysis
    print(f"\nFalse Positives (incorrectly predicted): {len(all_fps)}")
    fp_by_type = defaultdict(list)
    for fp in all_fps:
        fp_by_type[fp['type']].append(fp)

    for etype, fps in fp_by_type.items():
        print(f"\n  {etype} ({len(fps)} FPs):")
        for fp in fps[:n_examples]:
            snippet = documents[fp['doc_idx']][:80]
            print(f"    - \"{fp['text']}\" in: \"{snippet}...\"")

    # False Negative Analysis
    print(f"\nFalse Negatives (missed entities): {len(all_fns)}")
    fn_by_type = defaultdict(list)
    for fn in all_fns:
        fn_by_type[fn['type']].append(fn)

    for etype, fns in fn_by_type.items():
        print(f"\n  {etype} ({len(fns)} FNs):")
        for fn in fns[:n_examples]:
            snippet = documents[fn['doc_idx']][:80]
            print(f"    - \"{fn['text']}\" in: \"{snippet}...\"")

    # Common error patterns
    print(f"\nMost common false positive entities:")
    fp_texts = Counter((fp['text'], fp['type']) for fp in all_fps)
    for (text, etype), count in fp_texts.most_common(10):
        print(f"  {count:3d}x [{etype}] \"{text}\"")

    print(f"\nMost common missed entities:")
    fn_texts = Counter((fn['text'], fn['type']) for fn in all_fns)
    for (text, etype), count in fn_texts.most_common(10):
        print(f"  {count:3d}x [{etype}] \"{text}\"")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("    NAMED ENTITY RECOGNITION SYSTEM - NLP Portfolio Project")
    print("=" * 70)

    # --- Initialize NER System ---
    print("\n[1] Initializing rule-based NER system...")
    ner = RuleBasedNER()
    print(f"  Gazetteers loaded:")
    print(f"    PERSON: {len(GAZETTEERS['PERSON']['first_names'])} first names, "
          f"{len(GAZETTEERS['PERSON']['last_names'])} last names")
    print(f"    ORGANIZATION: {len(GAZETTEERS['ORGANIZATION'])} entries")
    print(f"    LOCATION: {sum(len(v) for v in GAZETTEERS['LOCATION'].values())} entries")

    # --- Generate Synthetic Data ---
    print("\n[2] Generating synthetic annotated text...")
    documents, true_annotations = generate_entity_text(n_documents=50)

    print("\nSample document:")
    print(f"  \"{documents[0][:200]}...\"")
    print(f"  Entities: {len(true_annotations[0])}")
    for ent in true_annotations[0][:5]:
        print(f"    [{ent['type']:15s}] \"{ent['text']}\"")

    # --- Run NER on All Documents ---
    print("\n[3] Running NER on all documents...")
    predicted_annotations = ner.recognize_batch(documents)

    total_pred = sum(len(p) for p in predicted_annotations)
    total_true = sum(len(t) for t in true_annotations)
    print(f"  True entities:      {total_true}")
    print(f"  Predicted entities: {total_pred}")

    # --- Visualization of Entity Recognition ---
    print("\n[4] Visualizing entity recognition...")
    for i in range(min(3, len(documents))):
        print(f"\n  Document {i + 1}:")
        annotated = visualize_entities_text(documents[i], predicted_annotations[i])
        print(f"  {annotated[:200]}...")

    # --- Evaluation ---
    print("\n[5] Evaluating NER system...")
    metrics = compute_entity_metrics(true_annotations, predicted_annotations)
    print_evaluation_report(metrics)

    # --- Feature Extraction Demo ---
    print("\n[6] Feature extraction demo...")
    feature_extractor = EntityFeatureExtractor()

    sample_text = documents[0]
    sample_entities = predicted_annotations[0]
    if sample_entities:
        ent = sample_entities[0]
        features = feature_extractor.extract_span_features(
            sample_text, ent['start'], ent['end']
        )
        print(f"\n  Features for entity \"{ent['text']}\" ({ent['type']}):")
        for feat_name, feat_val in features.items():
            print(f"    {feat_name:25s}: {feat_val}")

    # --- Test on Custom Sentences ---
    print("\n[7] Testing on custom sentences...")
    test_sentences = [
        "Barack Obama visited Google headquarters in Mountain View on January 15, 2024.",
        "Microsoft CEO Satya Nadella announced a $10 billion investment in Tokyo, Japan.",
        "Dr. Sarah Johnson at Harvard University published her findings on March 3, 2025.",
        "The United Nations held a conference in Geneva with representatives from 50 countries.",
        "Goldman Sachs reported $45.7 billion in revenue for the fiscal year ending December 31, 2024.",
        "SpaceX launched a rocket from Cape Canaveral, Florida on Tuesday.",
        "Prof. Michael Brown received a $2.5 million grant from NASA to study Mars.",
    ]

    for sentence in test_sentences:
        entities = ner.recognize(sentence)
        print(f"\n  Input: \"{sentence}\"")
        if entities:
            for ent in entities:
                print(f"    [{ent['type']:15s}] \"{ent['text']}\"")
        else:
            print("    No entities found.")

    # --- Visualizations ---
    print("\n[8] Generating visualizations...")
    plot_entity_distribution(predicted_annotations)
    plot_evaluation_results(metrics)

    # Entity spans for first document
    if documents:
        plot_entity_spans(documents[0], predicted_annotations[0])

    # --- Error Analysis ---
    print("\n[9] Error analysis...")
    analyze_errors(true_annotations, predicted_annotations, documents)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Documents processed: {len(documents)}")
    print(f"Total true entities: {total_true}")
    print(f"Total predicted entities: {total_pred}")
    print(f"Overall Precision: {metrics['OVERALL']['precision']:.4f}")
    print(f"Overall Recall:    {metrics['OVERALL']['recall']:.4f}")
    print(f"Overall F1:        {metrics['OVERALL']['f1']:.4f}")
    print(f"\nPer-type F1 scores:")
    for etype in ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'MONEY']:
        if etype in metrics:
            print(f"  {etype:15s}: {metrics[etype]['f1']:.4f}")
    print(f"\nGenerated files:")
    print(f"  - entity_distribution.png")
    print(f"  - ner_evaluation.png")
    print(f"  - entity_spans.png")
    print(f"{'=' * 70}")
    print("Named Entity Recognition project complete.")


if __name__ == '__main__':
    main()
