"""
Movie Review Sentiment Analysis
================================
Advanced sentiment analysis for movie reviews with aspect-based analysis,
multiple feature sets, model comparison, error analysis, and interpretability.

Author: Data Science Portfolio
"""

import numpy as np
import pandas as pd
import re
import string
import warnings
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. SENTIMENT LEXICON
# ============================================================================

POSITIVE_WORDS = {
    'excellent', 'amazing', 'wonderful', 'brilliant', 'outstanding', 'superb',
    'fantastic', 'great', 'good', 'love', 'loved', 'enjoy', 'enjoyed',
    'beautiful', 'perfect', 'best', 'masterpiece', 'compelling', 'captivating',
    'impressive', 'remarkable', 'stunning', 'magnificent', 'entertaining',
    'delightful', 'charming', 'touching', 'powerful', 'gripping', 'riveting',
    'engrossing', 'heartwarming', 'hilarious', 'funny', 'witty', 'clever',
    'moving', 'inspiring', 'memorable', 'exceptional', 'phenomenal',
    'breathtaking', 'flawless', 'genius', 'incredible', 'unforgettable',
    'authentic', 'original', 'fresh', 'innovative', 'engaging',
    'recommend', 'recommended', 'favorite', 'gem', 'triumph', 'triumph',
    'stellar', 'terrific', 'splendid', 'admirable', 'absorbing',
}

NEGATIVE_WORDS = {
    'terrible', 'awful', 'horrible', 'bad', 'worst', 'boring', 'dull',
    'disappointing', 'disappointed', 'waste', 'wasted', 'poor', 'weak',
    'mediocre', 'predictable', 'cliche', 'overrated', 'painful', 'annoying',
    'stupid', 'ridiculous', 'pointless', 'forgettable', 'flat', 'lifeless',
    'uninspired', 'unoriginal', 'bland', 'tedious', 'pretentious',
    'confusing', 'confused', 'mess', 'messy', 'disaster', 'atrocious',
    'cringe', 'cringeworthy', 'unwatchable', 'insufferable', 'laughable',
    'pathetic', 'abysmal', 'dreadful', 'appalling', 'dismal', 'lackluster',
    'shallow', 'contrived', 'forced', 'unconvincing', 'wooden', 'stiff',
    'hate', 'hated', 'dislike', 'avoid', 'skip', 'regret',
    'sloppy', 'lazy', 'derivative', 'failed', 'failure', 'flop',
}

NEGATION_WORDS = {
    'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'nor', 'cannot', "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "didn't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", 'hardly', 'barely', 'scarcely',
}

INTENSIFIERS = {
    'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 2.0,
    'truly': 1.5, 'really': 1.5, 'highly': 1.5, 'utterly': 2.0,
    'completely': 1.8, 'totally': 1.8, 'quite': 1.3, 'rather': 1.2,
    'somewhat': 0.8, 'slightly': 0.5, 'barely': 0.3, 'hardly': 0.3,
    'so': 1.5, 'most': 1.5, 'remarkably': 1.8, 'exceptionally': 2.0,
}

# ============================================================================
# 2. SYNTHETIC DATA GENERATION
# ============================================================================

def generate_review_dataset(n_reviews=2000):
    """
    Generate synthetic movie reviews with ratings and aspect-level sentiments.
    Reviews cover aspects: plot, acting, direction, music/soundtrack.
    """

    # Aspect-specific positive phrases
    aspect_positive = {
        'plot': [
            "The plot is brilliantly crafted with unexpected twists.",
            "A compelling storyline that keeps you guessing until the end.",
            "The narrative is engaging and well-paced throughout.",
            "Masterful storytelling with rich character development.",
            "The script is intelligent and thought-provoking.",
            "An original story that feels fresh and innovative.",
            "The plot unfolds beautifully with layers of complexity.",
            "Gripping narrative that never loses momentum.",
            "The screenplay is sharp, witty, and emotionally resonant.",
            "A perfectly structured story with satisfying payoffs.",
        ],
        'acting': [
            "The performances are outstanding across the board.",
            "Incredible acting that brings every character to life.",
            "The lead delivers a career-defining performance.",
            "Stellar ensemble cast with remarkable chemistry.",
            "The actors disappear completely into their roles.",
            "Emotionally powerful performances that move you to tears.",
            "The supporting cast is equally impressive and memorable.",
            "Nuanced and authentic acting throughout the film.",
            "The cast delivers with conviction and emotional depth.",
            "Award-worthy performances from the entire ensemble.",
        ],
        'direction': [
            "Masterful direction that elevates every scene.",
            "The director's vision is bold and perfectly executed.",
            "Stunning cinematography and impeccable visual storytelling.",
            "The director creates an immersive and atmospheric experience.",
            "Brilliant artistic choices that enhance the narrative.",
            "The pacing is perfect under skilled direction.",
            "Visually stunning with creative camera work throughout.",
            "The director brings a unique and refreshing perspective.",
            "Expertly directed with attention to every detail.",
            "The direction transforms a good script into a great film.",
        ],
        'music': [
            "The soundtrack perfectly complements every emotional beat.",
            "A beautiful score that enhances the viewing experience.",
            "The music is hauntingly memorable and emotionally powerful.",
            "Outstanding soundtrack that deserves recognition on its own.",
            "The musical choices are inspired and perfectly timed.",
            "An incredible score that elevates the entire film.",
            "The sound design is immersive and masterfully crafted.",
            "The music creates an unforgettable atmosphere.",
            "Beautiful compositions that linger long after the credits.",
            "The soundtrack is a masterpiece in its own right.",
        ],
    }

    # Aspect-specific negative phrases
    aspect_negative = {
        'plot': [
            "The plot is predictable and full of cliches.",
            "A convoluted storyline that makes no sense.",
            "The narrative is boring and drags on endlessly.",
            "Poorly written script with gaping plot holes.",
            "The story is unoriginal and derivative.",
            "A confusing mess of a plot that goes nowhere.",
            "The screenplay feels lazy and uninspired.",
            "The plot twists are forced and unconvincing.",
            "A weak story that fails to engage on any level.",
            "The narrative structure is a complete disaster.",
        ],
        'acting': [
            "The acting is wooden and completely unconvincing.",
            "Terrible performances that ruin the film.",
            "The lead actor delivers a flat and lifeless performance.",
            "The cast has zero chemistry and it shows.",
            "Cringe-worthy acting that makes you uncomfortable.",
            "The performances are over-the-top and laughable.",
            "Stiff and robotic acting throughout the entire film.",
            "The actors seem bored and disinterested.",
            "Poor casting choices undermine every scene.",
            "The performances range from mediocre to atrocious.",
        ],
        'direction': [
            "The direction is uninspired and pedestrian.",
            "Poor pacing that makes the film feel twice as long.",
            "The director seems lost and unable to find a vision.",
            "Visually bland with no creative ambition.",
            "Choppy editing and poor camera work throughout.",
            "The director fails to bring any coherence to the story.",
            "Sloppy direction that wastes a promising concept.",
            "The film lacks any directorial style or personality.",
            "Poor artistic choices that detract from the experience.",
            "The direction is the weakest aspect of this film.",
        ],
        'music': [
            "The soundtrack is forgettable and generic.",
            "The music feels completely out of place.",
            "A bland and uninspired score that adds nothing.",
            "The sound design is distracting and poorly mixed.",
            "The musical choices are jarring and inappropriate.",
            "The score is repetitive and annoying.",
            "The soundtrack undermines the emotional moments.",
            "No memorable music to speak of in this film.",
            "The music feels like an afterthought.",
            "A disappointing score that fails to enhance the film.",
        ],
    }

    # General review connectors
    connectors = [
        "Overall,", "In summary,", "All in all,", "To sum up,",
        "On the whole,", "In the end,", "Looking back,",
        "That being said,", "Having said that,", "Considering everything,",
    ]

    positive_conclusions = [
        "this is a must-watch film that I highly recommend.",
        "I would definitely watch this again.",
        "this is one of the best films I've seen this year.",
        "a truly remarkable cinematic experience.",
        "I can't recommend this film enough.",
        "this film deserves all the praise it gets.",
        "an absolute gem that shouldn't be missed.",
        "this is filmmaking at its finest.",
    ]

    negative_conclusions = [
        "I would not recommend wasting your time on this.",
        "save your money and skip this one.",
        "this is one of the worst films I've ever seen.",
        "a complete waste of potential and talent.",
        "I regret spending two hours on this disaster.",
        "this film is a total disappointment.",
        "avoid this movie at all costs.",
        "I can't believe this got made.",
    ]

    reviews = []
    labels = []
    ratings = []
    aspect_sentiments = []

    for _ in range(n_reviews):
        # Decide overall sentiment: positive (1) or negative (0)
        is_positive = np.random.random() > 0.45  # Slight positive skew

        aspects = ['plot', 'acting', 'direction', 'music']
        review_parts = []
        doc_aspect_sentiments = {}

        for aspect in aspects:
            if is_positive:
                # Mostly positive, some aspects may be negative
                aspect_positive_prob = np.random.uniform(0.6, 0.95)
            else:
                # Mostly negative, some aspects may be positive
                aspect_positive_prob = np.random.uniform(0.05, 0.4)

            is_aspect_positive = np.random.random() < aspect_positive_prob

            if is_aspect_positive:
                phrase = np.random.choice(aspect_positive[aspect])
                doc_aspect_sentiments[aspect] = 'positive'
            else:
                phrase = np.random.choice(aspect_negative[aspect])
                doc_aspect_sentiments[aspect] = 'negative'

            # Randomly skip some aspects to vary review length
            if np.random.random() > 0.2:
                review_parts.append(phrase)

        # Add conclusion
        connector = np.random.choice(connectors)
        if is_positive:
            conclusion = np.random.choice(positive_conclusions)
        else:
            conclusion = np.random.choice(negative_conclusions)
        review_parts.append(f"{connector} {conclusion}")

        review = ' '.join(review_parts)
        rating = (np.random.uniform(3.5, 5.0) if is_positive
                  else np.random.uniform(1.0, 3.0))
        rating = round(rating, 1)

        reviews.append(review)
        labels.append(1 if is_positive else 0)
        ratings.append(rating)
        aspect_sentiments.append(doc_aspect_sentiments)

    df = pd.DataFrame({
        'review': reviews,
        'sentiment': labels,
        'rating': ratings,
        'sentiment_label': ['positive' if l == 1 else 'negative' for l in labels],
    })

    # Add aspect sentiment columns
    for aspect in ['plot', 'acting', 'direction', 'music']:
        df[f'{aspect}_sentiment'] = [
            asp.get(aspect, 'neutral') for asp in aspect_sentiments
        ]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset generated: {len(df)} reviews")
    print(f"  Positive: {(df['sentiment'] == 1).sum()} ({(df['sentiment'] == 1).mean():.1%})")
    print(f"  Negative: {(df['sentiment'] == 0).sum()} ({(df['sentiment'] == 0).mean():.1%})")
    print(f"  Avg rating: {df['rating'].mean():.2f}")

    return df


# ============================================================================
# 3. TEXT PREPROCESSING
# ============================================================================

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once',
}


def preprocess_review(text):
    """Clean and normalize review text."""
    text = text.lower()
    # Preserve negation contractions
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r"'d", ' would', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


# ============================================================================
# 4. ADVANCED FEATURE EXTRACTION
# ============================================================================

class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract sentiment-specific features from review text.
    Includes lexicon-based scores, POS-like patterns, and stylistic features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            feat = self._extract(text)
            features.append(feat)
        return np.array(features)

    def _extract(self, text):
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) if words else 1

        # Lexicon-based features
        n_positive = sum(1 for w in words if w in POSITIVE_WORDS)
        n_negative = sum(1 for w in words if w in NEGATIVE_WORDS)
        n_negation = sum(1 for w in words if w in NEGATION_WORDS)

        pos_ratio = n_positive / n_words
        neg_ratio = n_negative / n_words
        sentiment_ratio = (n_positive - n_negative) / max(n_words, 1)

        # Intensifier features
        n_intensifiers = sum(1 for w in words if w in INTENSIFIERS)
        total_intensity = sum(INTENSIFIERS.get(w, 0) for w in words)

        # Negation-aware sentiment
        negated_positive = 0
        negated_negative = 0
        for i, word in enumerate(words):
            if word in NEGATION_WORDS and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in POSITIVE_WORDS:
                    negated_positive += 1
                elif next_word in NEGATIVE_WORDS:
                    negated_negative += 1

        # Stylistic features
        n_exclamation = text.count('!')
        n_question = text.count('?')
        n_caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        text_length = len(text)
        sentence_count = max(text.count('.') + text.count('!') + text.count('?'), 1)
        avg_sentence_length = n_words / sentence_count

        # Superlative/comparative patterns (simple heuristic)
        n_superlatives = sum(1 for w in words if w.endswith('est') or w in {'best', 'worst', 'most', 'least'})
        n_comparatives = sum(1 for w in words if w.endswith('er') and len(w) > 3)

        # Aspect keyword counts
        plot_words = {'plot', 'story', 'storyline', 'narrative', 'script', 'screenplay', 'twist', 'ending'}
        acting_words = {'acting', 'performance', 'actor', 'actress', 'cast', 'role', 'character', 'portrayed'}
        direction_words = {'director', 'direction', 'cinematography', 'camera', 'visual', 'pacing', 'editing'}
        music_words = {'music', 'soundtrack', 'score', 'sound', 'song', 'musical', 'composition', 'theme'}

        n_plot_words = sum(1 for w in words if w in plot_words)
        n_acting_words = sum(1 for w in words if w in acting_words)
        n_direction_words = sum(1 for w in words if w in direction_words)
        n_music_words = sum(1 for w in words if w in music_words)

        return [
            n_positive, n_negative, n_negation,
            pos_ratio, neg_ratio, sentiment_ratio,
            n_intensifiers, total_intensity,
            negated_positive, negated_negative,
            n_exclamation, n_question, n_caps_words,
            avg_word_length, text_length, avg_sentence_length,
            n_superlatives, n_comparatives,
            n_plot_words, n_acting_words, n_direction_words, n_music_words,
        ]

    def get_feature_names_out(self):
        return [
            'n_positive', 'n_negative', 'n_negation',
            'pos_ratio', 'neg_ratio', 'sentiment_ratio',
            'n_intensifiers', 'total_intensity',
            'negated_positive', 'negated_negative',
            'n_exclamation', 'n_question', 'n_caps_words',
            'avg_word_length', 'text_length', 'avg_sentence_length',
            'n_superlatives', 'n_comparatives',
            'n_plot_words', 'n_acting_words', 'n_direction_words', 'n_music_words',
        ]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer wrapper for text preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocess_review(text) for text in X]


# ============================================================================
# 5. MODEL BUILDING
# ============================================================================

def build_model_configs():
    """
    Define multiple model configurations with different feature sets.
    """
    configs = {}

    # Config 1: Unigram TF-IDF + Naive Bayes
    configs['NB_Unigram'] = Pipeline([
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
        ('clf', MultinomialNB(alpha=0.1)),
    ])

    # Config 2: Bigram TF-IDF + Naive Bayes
    configs['NB_Bigram'] = Pipeline([
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', MultinomialNB(alpha=0.1)),
    ])

    # Config 3: Complement Naive Bayes (better for imbalanced)
    configs['ComplementNB'] = Pipeline([
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', ComplementNB(alpha=0.5)),
    ])

    # Config 4: SVM with bigrams
    configs['SVM_Bigram'] = Pipeline([
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ('clf', LinearSVC(max_iter=2000, C=1.0, random_state=42)),
    ])

    # Config 5: Logistic Regression with trigrams
    configs['LogReg_Trigram'] = Pipeline([
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3),
                                   sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    # Config 6: Logistic Regression with CountVectorizer
    configs['LogReg_Count'] = Pipeline([
        ('preprocess', TextPreprocessor()),
        ('count', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])

    return configs


# ============================================================================
# 6. ASPECT-BASED SENTIMENT ANALYSIS
# ============================================================================

ASPECT_KEYWORDS = {
    'plot': ['plot', 'story', 'storyline', 'narrative', 'script', 'screenplay',
             'twist', 'ending', 'beginning', 'pacing', 'structured', 'structure',
             'written', 'writing', 'dialogue', 'character development', 'arc',
             'predictable', 'surprising', 'original', 'cliche', 'holes'],
    'acting': ['acting', 'performance', 'performances', 'actor', 'actress',
               'cast', 'ensemble', 'role', 'roles', 'character', 'characters',
               'portrayed', 'delivers', 'convincing', 'wooden', 'stiff',
               'chemistry', 'emotional', 'nuanced', 'over-the-top', 'lead'],
    'direction': ['director', 'direction', 'directed', 'cinematography',
                  'camera', 'visual', 'visually', 'pacing', 'editing',
                  'artistic', 'creative', 'vision', 'shot', 'shots',
                  'framing', 'style', 'atmosphere', 'immersive', 'bland'],
    'music': ['music', 'musical', 'soundtrack', 'score', 'sound', 'sounds',
              'song', 'songs', 'composition', 'theme', 'melody', 'orchestra',
              'instrumental', 'audio', 'mixed', 'design', 'sonic'],
}


def analyze_aspect_sentiment(text, aspect):
    """
    Analyze sentiment for a specific aspect (plot, acting, direction, music).
    Uses keyword matching and surrounding sentiment words.
    """
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text_lower)

    aspect_keywords = ASPECT_KEYWORDS.get(aspect, [])
    relevant_sentences = []

    for sentence in sentences:
        if any(kw in sentence for kw in aspect_keywords):
            relevant_sentences.append(sentence)

    if not relevant_sentences:
        return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0,
                'relevant_text': ''}

    combined_text = ' '.join(relevant_sentences)
    words = combined_text.split()

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    # Check for negation
    negated = False
    for i, w in enumerate(words):
        if w in NEGATION_WORDS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in POSITIVE_WORDS:
                pos_count -= 1
                neg_count += 1
                negated = True
            elif next_word in NEGATIVE_WORDS:
                neg_count -= 1
                pos_count += 1
                negated = True

    total = pos_count + neg_count
    if total == 0:
        sentiment = 'neutral'
        score = 0.0
        confidence = 0.0
    elif pos_count > neg_count:
        sentiment = 'positive'
        score = pos_count / total
        confidence = (pos_count - neg_count) / total
    else:
        sentiment = 'negative'
        score = -neg_count / total
        confidence = (neg_count - pos_count) / total

    return {
        'sentiment': sentiment,
        'score': round(score, 3),
        'confidence': round(confidence, 3),
        'relevant_text': combined_text[:200],
        'positive_words': pos_count,
        'negative_words': neg_count,
        'negated': negated,
    }


def full_aspect_analysis(text):
    """Run aspect-based sentiment analysis for all aspects."""
    results = {}
    for aspect in ['plot', 'acting', 'direction', 'music']:
        results[aspect] = analyze_aspect_sentiment(text, aspect)
    return results


# ============================================================================
# 7. ERROR ANALYSIS AND INTERPRETABILITY
# ============================================================================

def analyze_errors(model_name, pipeline, X_test, y_test, y_pred, top_n=10):
    """
    Detailed error analysis with feature-level explanations.
    """
    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS: {model_name}")
    print(f"{'=' * 60}")

    errors = y_test != y_pred
    n_errors = errors.sum()
    print(f"Total errors: {n_errors} / {len(y_test)} ({n_errors / len(y_test):.2%})")

    # False Positives: negative reviews predicted as positive
    fp_mask = (y_test == 0) & (y_pred == 1)
    print(f"\nFalse Positives (negative predicted as positive): {fp_mask.sum()}")
    if fp_mask.sum() > 0:
        fp_reviews = X_test[fp_mask]
        for i, review in enumerate(fp_reviews[:top_n]):
            print(f"  {i + 1}. \"{review[:100]}...\"")

    # False Negatives: positive reviews predicted as negative
    fn_mask = (y_test == 1) & (y_pred == 0)
    print(f"\nFalse Negatives (positive predicted as negative): {fn_mask.sum()}")
    if fn_mask.sum() > 0:
        fn_reviews = X_test[fn_mask]
        for i, review in enumerate(fn_reviews[:top_n]):
            print(f"  {i + 1}. \"{review[:100]}...\"")

    # Analyze lexicon features of errors
    extractor = SentimentFeatureExtractor()
    error_features = extractor.transform(X_test[errors])
    correct_features = extractor.transform(X_test[~errors])

    feature_names = extractor.get_feature_names_out()
    print(f"\nFeature comparison (errors vs correct):")
    print(f"  {'Feature':<25s} {'Errors':>10s} {'Correct':>10s} {'Diff':>10s}")
    print(f"  {'-' * 55}")
    for i, name in enumerate(feature_names):
        err_mean = error_features[:, i].mean() if len(error_features) > 0 else 0
        cor_mean = correct_features[:, i].mean() if len(correct_features) > 0 else 0
        diff = err_mean - cor_mean
        if abs(diff) > 0.01:
            print(f"  {name:<25s} {err_mean:>10.3f} {cor_mean:>10.3f} {diff:>+10.3f}")


def get_word_importance(pipeline, top_n=20):
    """
    Extract the most important words for positive and negative sentiment.
    Works with models that have coef_ attribute.
    """
    clf = pipeline.named_steps['clf']
    if not hasattr(clf, 'coef_'):
        return None, None

    # Try to get feature names from the vectorizer step
    vectorizer = None
    for step_name, step in pipeline.named_steps.items():
        if hasattr(step, 'get_feature_names_out') and step_name != 'clf':
            vectorizer = step

    if vectorizer is None:
        return None, None

    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_.flatten()

    # Top positive indicators
    pos_indices = np.argsort(coefs)[-top_n:][::-1]
    pos_words = [(feature_names[i], coefs[i]) for i in pos_indices]

    # Top negative indicators
    neg_indices = np.argsort(coefs)[:top_n]
    neg_words = [(feature_names[i], coefs[i]) for i in neg_indices]

    return pos_words, neg_words


def plot_word_importance(pipeline, save_path='word_importance.png', top_n=20):
    """Visualize the most important words for each sentiment class."""
    pos_words, neg_words = get_word_importance(pipeline, top_n=top_n)

    if pos_words is None:
        print("Word importance visualization requires a model with coef_ attribute.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Positive words
    words_p = [w for w, _ in pos_words]
    scores_p = [s for _, s in pos_words]
    y_pos = range(len(words_p))
    ax1.barh(y_pos, scores_p, color='green', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(words_p)
    ax1.set_title(f'Top {top_n} Positive Sentiment Indicators')
    ax1.set_xlabel('Coefficient Weight')
    ax1.invert_yaxis()

    # Negative words
    words_n = [w for w, _ in neg_words]
    scores_n = [abs(s) for _, s in neg_words]
    y_neg = range(len(words_n))
    ax2.barh(y_neg, scores_n, color='red', alpha=0.7)
    ax2.set_yticks(y_neg)
    ax2.set_yticklabels(words_n)
    ax2.set_title(f'Top {top_n} Negative Sentiment Indicators')
    ax2.set_xlabel('|Coefficient Weight|')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word importance plot saved to {save_path}")


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def plot_confusion_matrices(predictions_dict, y_test,
                            save_path='sentiment_confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    n_models = len(predictions_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, y_pred) in enumerate(predictions_dict.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        f1 = f1_score(y_test, y_pred)
        ax.set_title(f'{name}\n(F1={f1:.3f})', fontsize=10)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[0, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_model_comparison(results_dict, save_path='model_comparison.png'):
    """Bar chart comparing models across metrics."""
    df = pd.DataFrame(results_dict).T
    ax = df[['Accuracy', 'Precision', 'Recall', 'F1']].plot(
        kind='bar', figsize=(12, 6), width=0.8
    )
    ax.set_title('Movie Review Sentiment: Model Comparison', fontsize=14)
    ax.set_ylabel('Score')
    ax.set_xlabel('Model Configuration')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison chart saved to {save_path}")


def plot_aspect_analysis(reviews_df, save_path='aspect_analysis.png'):
    """Visualize aspect-based sentiment analysis results."""
    aspects = ['plot', 'acting', 'direction', 'music']

    # Analyze aspects for a sample of reviews
    n_sample = min(200, len(reviews_df))
    sample = reviews_df.head(n_sample)

    aspect_results = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})

    for _, row in sample.iterrows():
        analysis = full_aspect_analysis(row['review'])
        for aspect, result in analysis.items():
            aspect_results[aspect][result['sentiment']] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart
    x = np.arange(len(aspects))
    pos_counts = [aspect_results[a]['positive'] for a in aspects]
    neg_counts = [aspect_results[a]['negative'] for a in aspects]
    neu_counts = [aspect_results[a]['neutral'] for a in aspects]

    ax1.bar(x, pos_counts, label='Positive', color='green', alpha=0.7)
    ax1.bar(x, neg_counts, bottom=pos_counts, label='Negative', color='red', alpha=0.7)
    ax1.bar(x, neu_counts, bottom=[p + n for p, n in zip(pos_counts, neg_counts)],
            label='Neutral', color='gray', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.capitalize() for a in aspects])
    ax1.set_ylabel('Review Count')
    ax1.set_title('Aspect Sentiment Distribution')
    ax1.legend()

    # Aspect sentiment by overall sentiment
    pos_reviews = sample[sample['sentiment'] == 1]
    neg_reviews = sample[sample['sentiment'] == 0]

    pos_aspect_scores = []
    neg_aspect_scores = []

    for aspect in aspects:
        pos_scores = []
        for _, row in pos_reviews.iterrows():
            result = analyze_aspect_sentiment(row['review'], aspect)
            pos_scores.append(result['score'])
        pos_aspect_scores.append(np.mean(pos_scores) if pos_scores else 0)

        neg_scores = []
        for _, row in neg_reviews.iterrows():
            result = analyze_aspect_sentiment(row['review'], aspect)
            neg_scores.append(result['score'])
        neg_aspect_scores.append(np.mean(neg_scores) if neg_scores else 0)

    width = 0.35
    ax2.bar(x - width / 2, pos_aspect_scores, width, label='Positive Reviews',
            color='green', alpha=0.7)
    ax2.bar(x + width / 2, neg_aspect_scores, width, label='Negative Reviews',
            color='red', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.capitalize() for a in aspects])
    ax2.set_ylabel('Average Sentiment Score')
    ax2.set_title('Aspect Scores by Overall Sentiment')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Aspect analysis plot saved to {save_path}")


def plot_rating_distribution(df, save_path='rating_distribution.png'):
    """Plot the distribution of ratings by sentiment class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Rating histogram by sentiment
    pos_ratings = df[df['sentiment'] == 1]['rating']
    neg_ratings = df[df['sentiment'] == 0]['rating']
    ax1.hist(pos_ratings, bins=20, alpha=0.6, label='Positive', color='green')
    ax1.hist(neg_ratings, bins=20, alpha=0.6, label='Negative', color='red')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Rating Distribution by Sentiment')
    ax1.legend()

    # Review length distribution
    df_copy = df.copy()
    df_copy['review_length'] = df_copy['review'].str.len()
    pos_lengths = df_copy[df_copy['sentiment'] == 1]['review_length']
    neg_lengths = df_copy[df_copy['sentiment'] == 0]['review_length']
    ax2.hist(pos_lengths, bins=30, alpha=0.6, label='Positive', color='green')
    ax2.hist(neg_lengths, bins=30, alpha=0.6, label='Negative', color='red')
    ax2.set_xlabel('Review Length (characters)')
    ax2.set_ylabel('Count')
    ax2.set_title('Review Length Distribution by Sentiment')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rating distribution plot saved to {save_path}")


# ============================================================================
# 9. CROSS-VALIDATION
# ============================================================================

def cross_validate_models(configs, X, y, cv=5):
    """Run stratified k-fold cross-validation on all model configurations."""
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION ({cv}-fold Stratified)")
    print(f"{'=' * 60}")

    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, pipeline in configs.items():
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')
        cv_results[name] = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
        }
        print(f"  {name:20s}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return cv_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("    MOVIE REVIEW SENTIMENT ANALYSIS - NLP Portfolio Project")
    print("=" * 70)

    # --- Data Generation ---
    print("\n[1] Generating synthetic movie reviews...")
    df = generate_review_dataset(n_reviews=2000)

    print("\nSample reviews:")
    for _, row in df.head(3).iterrows():
        label = row['sentiment_label'].upper()
        rating = row['rating']
        review = row['review'][:100]
        print(f"  [{label:8s}] (rating: {rating}) {review}...")

    # --- Data exploration ---
    print("\n[2] Data exploration...")
    plot_rating_distribution(df)

    # --- Preprocessing demo ---
    print("\n[3] Text preprocessing:")
    for review in df['review'].head(2):
        processed = preprocess_review(review)
        print(f"  Original : {review[:80]}...")
        print(f"  Processed: {processed[:80]}...")
        print()

    # --- Feature extraction demo ---
    print("[4] Sentiment feature extraction:")
    extractor = SentimentFeatureExtractor()
    sample_features = extractor.transform(df['review'].head(2))
    feature_names = extractor.get_feature_names_out()
    for i, row in enumerate(sample_features):
        print(f"  Review {i + 1} ({df.iloc[i]['sentiment_label']}):")
        for name, val in zip(feature_names, row):
            if abs(val) > 0:
                print(f"    {name:25s}: {val:.3f}")

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'].values, df['sentiment'].values,
        test_size=0.2, random_state=42, stratify=df['sentiment'].values
    )
    print(f"\n[5] Train/Test split: {len(X_train)} train, {len(X_test)} test")

    # --- Model Training and Evaluation ---
    print("\n[6] Training and evaluating models...")
    configs = build_model_configs()
    results = {}
    predictions = {}

    for name, pipeline in configs.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
        }

        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
              f"Recall: {rec:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=['Negative', 'Positive']))

    # --- Visualizations ---
    print("\n[7] Generating visualizations...")
    plot_confusion_matrices(predictions, y_test)
    plot_model_comparison(results)

    # --- Word importance ---
    print("\n[8] Word importance analysis...")
    # Use the best linear model for word importance
    best_linear = None
    best_f1 = 0
    for name, pipeline in configs.items():
        if hasattr(pipeline.named_steps['clf'], 'coef_'):
            if results[name]['F1'] > best_f1:
                best_f1 = results[name]['F1']
                best_linear = name

    if best_linear:
        plot_word_importance(configs[best_linear])
        pos_words, neg_words = get_word_importance(configs[best_linear])

        print(f"\n  Top positive indicators ({best_linear}):")
        for word, score in pos_words[:10]:
            print(f"    {word:25s}: {score:+.4f}")

        print(f"\n  Top negative indicators ({best_linear}):")
        for word, score in neg_words[:10]:
            print(f"    {word:25s}: {score:+.4f}")

    # --- Aspect-Based Sentiment Analysis ---
    print("\n[9] Aspect-based sentiment analysis...")
    plot_aspect_analysis(df)

    # Demo on specific reviews
    demo_reviews = [
        "The plot was incredibly engaging with brilliant twists, but the acting was wooden "
        "and unconvincing. The direction was masterful with stunning visuals, and the "
        "soundtrack perfectly complemented every scene.",

        "A terrible storyline full of plot holes. However, the cast delivered outstanding "
        "performances, especially the lead. The music was forgettable and the direction "
        "felt lazy and uninspired.",
    ]

    for i, review in enumerate(demo_reviews):
        print(f"\n  Demo Review {i + 1}: \"{review[:80]}...\"")
        aspects = full_aspect_analysis(review)
        for aspect, result in aspects.items():
            print(f"    {aspect:12s}: {result['sentiment']:8s} "
                  f"(score: {result['score']:+.3f}, conf: {result['confidence']:.3f})")

    # --- Error Analysis ---
    print("\n[10] Error analysis...")
    best_model = max(results, key=lambda k: results[k]['F1'])
    analyze_errors(best_model, configs[best_model],
                   X_test, y_test, predictions[best_model])

    # --- Cross-Validation ---
    print("\n[11] Cross-validation...")
    cv_results = cross_validate_models(configs, df['review'].values, df['sentiment'].values)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    summary_df = pd.DataFrame(results).T.round(4)
    print(summary_df.to_string())
    print(f"\nBest model: {best_model} (F1={results[best_model]['F1']:.4f})")

    print(f"\nCross-validation results:")
    for name, cv_res in cv_results.items():
        print(f"  {name:20s}: F1 = {cv_res['mean_f1']:.4f} (+/- {cv_res['std_f1']:.4f})")

    print(f"\nGenerated files:")
    print(f"  - rating_distribution.png")
    print(f"  - sentiment_confusion_matrices.png")
    print(f"  - model_comparison.png")
    print(f"  - word_importance.png")
    print(f"  - aspect_analysis.png")
    print(f"{'=' * 70}")
    print("Movie review sentiment analysis project complete.")


if __name__ == '__main__':
    main()
