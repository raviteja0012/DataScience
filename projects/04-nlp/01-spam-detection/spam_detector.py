"""
Spam Detection System
=====================
A complete NLP pipeline for classifying messages as spam or ham (not spam).
Demonstrates text preprocessing, feature engineering, multiple classifiers,
sklearn Pipeline integration, and production-ready prediction.

Author: Data Science Portfolio
"""

import numpy as np
import pandas as pd
import re
import string
import warnings
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================================

def generate_spam_dataset(n_samples=2000):
    """
    Generate a synthetic dataset of spam and ham messages.
    Spam messages mimic common spam patterns: urgency, money offers,
    suspicious links, and excessive punctuation.
    """
    spam_templates = [
        "CONGRATULATIONS! You've won ${amount} in our lottery! Call {phone} NOW to claim!",
        "FREE {product}!!! Click here: http://bit.ly/{code} to claim your prize!!!",
        "URGENT: Your account has been compromised. Verify at http://{domain}.com/verify",
        "You have been selected for a ${amount} gift card! Reply WIN to claim",
        "Make ${amount} per week working from home! No experience needed! Visit {domain}.com",
        "ALERT: {bank} security notice. Update your info at http://{domain}.com/update",
        "Hot singles in your area! Meet them now at http://{domain}.com/dating",
        "Lose {number} pounds in {number} days! Order now: http://{domain}.com/diet",
        "Dear winner, you've been chosen for our ${amount} giveaway. Send details to claim",
        "ACT NOW! Limited time offer: {product} at 90% OFF! Only {number} left!!!",
        "Your {phone_brand} has a virus! Download antivirus NOW: http://{domain}.com/fix",
        "Exclusive deal: Buy {number} get {number} FREE! Use code SAVE{number}",
        "WARNING: Your subscription expires today! Renew at http://{domain}.com/renew",
        "You're our {number}th visitor! Claim your FREE {product} here!!!",
        "BREAKING: Earn ${amount} daily with this one simple trick!!! Click now!!!",
        "Dear customer, your package #{code} is waiting. Pay ${amount} shipping: {domain}.com",
        "FINAL WARNING: Pay ${amount} or face legal action. Call {phone} immediately",
        "Congratulations {name}! You qualify for a ${amount} loan at 0% interest!",
        "FREE trial of {product}! No credit card needed! Sign up: {domain}.com",
        "Your {phone_brand} won a {product}! Confirm at http://{domain}.com/prize",
    ]

    ham_templates = [
        "Hey, are you coming to the meeting at {time} today?",
        "Can you pick up some {food} on your way home?",
        "Thanks for sending the report. I'll review it by {day}.",
        "Happy birthday! Hope you have a wonderful day!",
        "The project deadline has been moved to {day}. Let me know if that works.",
        "Just finished reading {book}. Really enjoyed it!",
        "Running late, traffic is terrible. Be there in {number} minutes.",
        "Did you see the game last night? What a finish!",
        "Can we reschedule our lunch to {time} on {day}?",
        "Your order has been shipped. Expected delivery: {day}.",
        "Thanks for the help yesterday. Really appreciated it.",
        "Movie tonight? I was thinking we could see {movie}.",
        "Don't forget Mom's birthday is on {day}. Should we get flowers?",
        "Great presentation today! The client seemed really impressed.",
        "I'll be working from home tomorrow. Call me if you need anything.",
        "Hey, found that restaurant you mentioned. Want to try it {day}?",
        "The kids' school play is on {day} at {time}. Can you make it?",
        "Just saw your email. I agree with your approach on the {topic} project.",
        "Weather looks great this weekend. Want to go hiking?",
        "Reminder: dentist appointment on {day} at {time}.",
        "Can you send me the slides from today's presentation?",
        "How's the new job going? We should catch up soon.",
        "Picked up your prescription from the pharmacy.",
        "Good luck with the interview tomorrow! You'll do great.",
        "Dinner at our place on {day}? I'm making {food}.",
        "The plumber is coming between {time} and {time} tomorrow.",
        "Saw this article about {topic} and thought of you.",
        "Sorry I missed your call. In a meeting until {time}.",
        "Are you free this weekend? {name} is having a barbecue.",
        "Just wanted to check in. How are you feeling?",
    ]

    products = ['iPhone', 'iPad', 'Samsung Galaxy', 'laptop', 'TV', 'smartwatch', 'AirPods']
    domains = ['secure-verify', 'prize-winner', 'click-now', 'free-gift', 'best-deal']
    banks = ['Chase', 'Wells Fargo', 'Bank of America', 'Citibank']
    phone_brands = ['iPhone', 'Samsung', 'Google Pixel', 'OnePlus']
    foods = ['groceries', 'milk', 'pizza', 'Chinese food', 'tacos', 'pasta']
    books = ['that mystery novel', 'the new bestseller', 'the biography', 'the sci-fi book']
    movies = ['the new Marvel movie', 'that comedy', 'the documentary', 'the thriller']
    topics = ['machine learning', 'marketing', 'design', 'analytics', 'strategy']
    names = ['John', 'Sarah', 'Mike', 'Emily', 'David', 'Lisa', 'Tom', 'Anna']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    times = ['9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm']

    messages = []
    labels = []

    n_spam = n_samples // 3  # ~33% spam is realistic
    n_ham = n_samples - n_spam

    for _ in range(n_spam):
        template = np.random.choice(spam_templates)
        msg = template.format(
            amount=np.random.choice([100, 500, 1000, 5000, 10000, 50000, 1000000]),
            phone=f"{np.random.randint(100,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}",
            product=np.random.choice(products),
            code=''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 6)),
            domain=np.random.choice(domains),
            bank=np.random.choice(banks),
            number=np.random.randint(1, 50),
            phone_brand=np.random.choice(phone_brands),
            name=np.random.choice(names),
        )
        # Add random spam-like modifications
        if np.random.random() > 0.5:
            msg = msg.upper() if np.random.random() > 0.5 else msg
        if np.random.random() > 0.6:
            msg += '!!!' * np.random.randint(1, 4)
        messages.append(msg)
        labels.append(1)

    for _ in range(n_ham):
        template = np.random.choice(ham_templates)
        msg = template.format(
            time=np.random.choice(times),
            food=np.random.choice(foods),
            day=np.random.choice(days),
            book=np.random.choice(books),
            movie=np.random.choice(movies),
            topic=np.random.choice(topics),
            name=np.random.choice(names),
            number=np.random.randint(5, 45),
        )
        messages.append(msg)
        labels.append(0)

    df = pd.DataFrame({'message': messages, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['label_text'] = df['label'].map({0: 'ham', 1: 'spam'})

    print(f"Dataset generated: {len(df)} messages")
    print(f"  Ham:  {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")
    print(f"  Spam: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
    return df


# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================

# Simple stopwords list (avoids NLTK download requirement)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't",
}


def simple_stem(word):
    """A basic suffix-stripping stemmer for demonstration."""
    suffixes = ['ing', 'ly', 'ment', 'ness', 'tion', 'sion', 'ous', 'ive',
                'able', 'ible', 'ful', 'less', 'ed', 'er', 'est', 'es', 's']
    word = word.lower()
    if len(word) <= 4:
        return word
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def preprocess_text(text):
    """
    Full text preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove phone numbers
    5. Remove punctuation
    6. Remove numbers
    7. Remove stopwords
    8. Apply stemming
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' phone ', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [simple_stem(t) for t in tokens]
    text = ' '.join(tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# 3. CUSTOM FEATURE EXTRACTION
# ============================================================================

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract hand-crafted features from raw text messages.
    These features capture stylistic patterns common in spam.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            feat = self._extract_features(text)
            features.append(feat)
        return np.array(features)

    def _extract_features(self, text):
        """Extract numerical features from a single text."""
        length = len(text)
        word_count = len(text.split())
        avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0

        # Character-level features
        n_uppercase = sum(1 for c in text if c.isupper())
        caps_ratio = n_uppercase / max(length, 1)
        n_digits = sum(1 for c in text if c.isdigit())
        digit_ratio = n_digits / max(length, 1)
        n_special = sum(1 for c in text if c in string.punctuation)
        special_ratio = n_special / max(length, 1)
        n_exclamation = text.count('!')
        n_question = text.count('?')
        n_dollar = text.count('$')

        # Pattern features
        has_url = 1 if re.search(r'http|www|\.com|\.ly', text, re.I) else 0
        has_phone = 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0
        has_email = 1 if re.search(r'\S+@\S+', text) else 0

        # Spam keyword indicators
        spam_words = ['free', 'win', 'winner', 'won', 'prize', 'claim', 'urgent',
                      'congratulations', 'offer', 'limited', 'act now', 'click',
                      'subscribe', 'deal', 'discount', 'earn', 'cash', 'money',
                      'credit', 'loan', 'guarantee', 'order', 'buy', 'cheap',
                      'bonus', 'gift', 'selected', 'exclusive', 'alert', 'verify']
        text_lower = text.lower()
        n_spam_words = sum(1 for w in spam_words if w in text_lower)

        # All caps words
        words = text.split()
        n_all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        all_caps_ratio = n_all_caps_words / max(word_count, 1)

        return [
            length, word_count, avg_word_len,
            caps_ratio, digit_ratio, special_ratio,
            n_exclamation, n_question, n_dollar,
            has_url, has_phone, has_email,
            n_spam_words, all_caps_ratio
        ]

    def get_feature_names_out(self):
        return [
            'length', 'word_count', 'avg_word_len',
            'caps_ratio', 'digit_ratio', 'special_ratio',
            'n_exclamation', 'n_question', 'n_dollar',
            'has_url', 'has_phone', 'has_email',
            'n_spam_words', 'all_caps_ratio'
        ]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer wrapper for the text preprocessing function."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocess_text(text) for text in X]


# ============================================================================
# 4. MODEL BUILDING AND EVALUATION
# ============================================================================

def build_pipelines():
    """
    Build multiple classification pipelines with different algorithms.
    Each pipeline includes preprocessing, feature extraction, and a classifier.
    """
    pipelines = {}

    # Pipeline 1: TF-IDF + Naive Bayes
    pipelines['NaiveBayes_TFIDF'] = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB(alpha=0.1)),
    ])

    # Pipeline 2: CountVectorizer + Logistic Regression
    pipelines['LogReg_Count'] = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('count', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    # Pipeline 3: TF-IDF + SVM
    pipelines['SVM_TFIDF'] = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LinearSVC(max_iter=2000, random_state=42)),
    ])

    # Pipeline 4: TF-IDF + Random Forest
    pipelines['RF_TFIDF'] = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    return pipelines


def evaluate_model(name, y_true, y_pred, results_dict):
    """Compute and store classification metrics."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    results_dict[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
    }

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))

    return results_dict


def plot_confusion_matrices(models_predictions, y_test, save_path='confusion_matrices.png'):
    """Plot confusion matrices for all models side by side."""
    n_models = len(models_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, models_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        ax.set_title(f'{name}', fontsize=10)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrices saved to {save_path}")


def plot_model_comparison(results_dict, save_path='model_comparison.png'):
    """Bar chart comparing all models across metrics."""
    df = pd.DataFrame(results_dict).T
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.8)
    ax.set_title('Spam Detection: Model Comparison', fontsize=14)
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison chart saved to {save_path}")


# ============================================================================
# 5. MISCLASSIFICATION ANALYSIS
# ============================================================================

def analyze_misclassifications(model_name, y_true, y_pred, messages, top_n=10):
    """
    Examine messages that were incorrectly classified.
    This helps understand model weaknesses and edge cases.
    """
    print(f"\n{'=' * 60}")
    print(f"MISCLASSIFICATION ANALYSIS: {model_name}")
    print(f"{'=' * 60}")

    mask = y_true != y_pred
    n_errors = mask.sum()
    print(f"Total misclassifications: {n_errors} / {len(y_true)} "
          f"({n_errors / len(y_true):.2%})")

    # False positives: Ham classified as Spam
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_count = fp_mask.sum()
    print(f"\nFalse Positives (Ham -> Spam): {fp_count}")
    if fp_count > 0:
        fp_msgs = messages[fp_mask]
        for i, msg in enumerate(fp_msgs[:top_n]):
            print(f"  {i + 1}. \"{msg[:100]}{'...' if len(msg) > 100 else ''}\"")

    # False negatives: Spam classified as Ham
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_count = fn_mask.sum()
    print(f"\nFalse Negatives (Spam -> Ham): {fn_count}")
    if fn_count > 0:
        fn_msgs = messages[fn_mask]
        for i, msg in enumerate(fn_msgs[:top_n]):
            print(f"  {i + 1}. \"{msg[:100]}{'...' if len(msg) > 100 else ''}\"")


# ============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(pipeline, feature_type='tfidf', top_n=20,
                               save_path='feature_importance.png'):
    """Show which words are most indicative of spam vs ham."""
    clf = pipeline.named_steps['clf']
    vectorizer = pipeline.named_steps.get(feature_type,
                                          pipeline.named_steps.get('count'))

    if vectorizer is None or not hasattr(clf, 'coef_'):
        print("Feature importance analysis requires a linear model with coef_ attribute.")
        return

    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_.flatten() if hasattr(clf.coef_, 'flatten') else clf.coef_

    # Top spam-indicative features
    top_spam_idx = np.argsort(coefs)[-top_n:]
    top_ham_idx = np.argsort(coefs)[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Spam features
    spam_features = [feature_names[i] for i in top_spam_idx]
    spam_scores = [coefs[i] for i in top_spam_idx]
    ax1.barh(spam_features, spam_scores, color='red', alpha=0.7)
    ax1.set_title(f'Top {top_n} Spam Indicators')
    ax1.set_xlabel('Coefficient Weight')

    # Ham features
    ham_features = [feature_names[i] for i in top_ham_idx]
    ham_scores = [coefs[i] for i in top_ham_idx]
    ax2.barh(ham_features, ham_scores, color='green', alpha=0.7)
    ax2.set_title(f'Top {top_n} Ham Indicators')
    ax2.set_xlabel('Coefficient Weight')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance chart saved to {save_path}")

    return spam_features, ham_features


# ============================================================================
# 7. CUSTOM FEATURES PIPELINE
# ============================================================================

def build_combined_pipeline():
    """
    Build a pipeline that combines TF-IDF text features with
    hand-crafted custom features using FeatureUnion.
    """
    text_pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ])

    combined_features = FeatureUnion([
        ('text_features', text_pipeline),
        ('custom_features', CustomFeatureExtractor()),
    ])

    full_pipeline = Pipeline([
        ('features', combined_features),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    return full_pipeline


# ============================================================================
# 8. PRODUCTION PREDICTION FUNCTION
# ============================================================================

class SpamDetector:
    """
    Production-ready spam detection system.
    Wraps the best-performing pipeline and provides a clean API.
    """

    def __init__(self):
        self.pipeline = None
        self.threshold = 0.5
        self.is_fitted = False

    def train(self, messages, labels):
        """Train the spam detector on labeled data."""
        self.pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        self.pipeline.fit(messages, labels)
        self.is_fitted = True
        print("SpamDetector trained successfully.")

    def predict(self, message):
        """
        Predict whether a single message is spam or ham.

        Parameters
        ----------
        message : str
            The text message to classify.

        Returns
        -------
        dict
            Classification result with label, confidence, and features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        if isinstance(message, str):
            messages = [message]
        else:
            messages = list(message)

        predictions = self.pipeline.predict(messages)
        probabilities = self.pipeline.predict_proba(messages)

        results = []
        extractor = CustomFeatureExtractor()
        custom_features = extractor.transform(messages)
        feature_names = extractor.get_feature_names_out()

        for i, msg in enumerate(messages):
            label = 'spam' if predictions[i] == 1 else 'ham'
            confidence = probabilities[i][predictions[i]]
            features = dict(zip(feature_names, custom_features[i]))

            results.append({
                'message': msg[:100] + ('...' if len(msg) > 100 else ''),
                'prediction': label,
                'confidence': round(float(confidence), 4),
                'spam_probability': round(float(probabilities[i][1]), 4),
                'features': features,
            })

        if len(results) == 1:
            return results[0]
        return results

    def predict_batch(self, messages):
        """Predict labels for a batch of messages."""
        return self.predict(messages)


# ============================================================================
# 9. CROSS-VALIDATION
# ============================================================================

def cross_validate_models(pipelines, X, y, cv=5):
    """
    Perform stratified k-fold cross-validation on all pipelines.
    """
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS ({cv}-fold Stratified)")
    print(f"{'=' * 60}")

    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1', n_jobs=-1)
        cv_results[name] = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores,
        }
        print(f"  {name:25s}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return cv_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("        SPAM DETECTION SYSTEM - NLP Portfolio Project")
    print("=" * 70)

    # --- Data Generation ---
    print("\n[1] Generating synthetic dataset...")
    df = generate_spam_dataset(n_samples=2000)

    print("\nSample messages:")
    for _, row in df.head(5).iterrows():
        label = row['label_text'].upper()
        msg = row['message'][:80]
        print(f"  [{label:4s}] {msg}...")

    # --- Preprocessing demo ---
    print("\n[2] Text preprocessing examples:")
    for text in df['message'].head(3):
        processed = preprocess_text(text)
        print(f"  Original : {text[:70]}...")
        print(f"  Processed: {processed[:70]}...")
        print()

    # --- Custom feature extraction demo ---
    print("[3] Custom feature extraction:")
    extractor = CustomFeatureExtractor()
    sample_features = extractor.transform(df['message'].head(3))
    feature_names = extractor.get_feature_names_out()
    for i, row in enumerate(sample_features):
        print(f"  Message {i + 1}:")
        for name, val in zip(feature_names, row):
            print(f"    {name:20s}: {val:.4f}")

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'].values, df['label'].values,
        test_size=0.2, random_state=42, stratify=df['label'].values
    )
    print(f"\n[4] Train/Test split: {len(X_train)} train, {len(X_test)} test")

    # --- Build and evaluate pipelines ---
    print("\n[5] Training and evaluating models...")
    pipelines = build_pipelines()
    results = {}
    predictions = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred
        results = evaluate_model(name, y_test, y_pred, results)

    # --- Combined features pipeline ---
    print("\n[6] Training combined features pipeline (TF-IDF + Custom)...")
    combined_pipeline = build_combined_pipeline()
    combined_pipeline.fit(X_train, y_train)
    y_pred_combined = combined_pipeline.predict(X_test)
    predictions['Combined_LogReg'] = y_pred_combined
    results = evaluate_model('Combined_LogReg', y_test, y_pred_combined, results)

    # --- Visualizations ---
    print("\n[7] Generating visualizations...")
    plot_confusion_matrices(predictions, y_test)
    plot_model_comparison(results)

    # --- Feature importance for Logistic Regression ---
    print("\n[8] Feature importance analysis...")
    analyze_feature_importance(pipelines['LogReg_Count'], feature_type='count')

    # --- Misclassification analysis ---
    print("\n[9] Misclassification analysis...")
    best_model = max(results, key=lambda k: results[k]['F1-Score'])
    analyze_misclassifications(
        best_model, y_test, predictions[best_model], X_test
    )

    # --- Cross-validation ---
    print("\n[10] Cross-validation...")
    cv_results = cross_validate_models(pipelines, df['message'].values, df['label'].values)

    # --- Production prediction demo ---
    print("\n[11] Production prediction demo...")
    detector = SpamDetector()
    detector.train(X_train, y_train)

    test_messages = [
        "Hey, are you free for lunch tomorrow?",
        "CONGRATULATIONS! You've won $10,000! Call NOW to claim your prize!!!",
        "The meeting has been rescheduled to 3pm on Thursday.",
        "FREE iPhone! Click here to claim: http://bit.ly/abc123",
        "Can you pick up the kids from school today?",
        "URGENT: Your bank account needs verification. Visit secure-verify.com NOW",
    ]

    print("\nPrediction Results:")
    print("-" * 80)
    for msg in test_messages:
        result = detector.predict(msg)
        icon = "SPAM" if result['prediction'] == 'spam' else "HAM "
        print(f"  [{icon}] (conf: {result['confidence']:.2f}) {result['message']}")
    print("-" * 80)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    summary_df = pd.DataFrame(results).T.round(4)
    print(summary_df.to_string())
    print(f"\nBest model: {best_model} (F1={results[best_model]['F1-Score']:.4f})")
    print(f"{'=' * 70}")
    print("Spam detection project complete.")


if __name__ == '__main__':
    main()
