"""
Topic Modeling System
=====================
Discovers latent topics in a corpus of documents using LDA and NMF.
Includes synthetic data generation, coherence score optimization,
topic visualization with word clouds, and document-topic assignment.

Author: Data Science Portfolio
"""

import numpy as np
import pandas as pd
import re
import string
import warnings
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. SYNTHETIC CORPUS GENERATION
# ============================================================================

TOPIC_VOCABULARIES = {
    'technology': {
        'keywords': ['software', 'computer', 'technology', 'data', 'algorithm',
                     'artificial', 'intelligence', 'machine', 'learning', 'neural',
                     'network', 'cloud', 'computing', 'digital', 'platform',
                     'startup', 'innovation', 'app', 'smartphone', 'cybersecurity',
                     'programming', 'developer', 'code', 'database', 'server',
                     'internet', 'silicon', 'valley', 'tech', 'robot'],
        'templates': [
            "The new {kw1} {kw2} system leverages advanced {kw3} to improve {kw4} performance across multiple {kw5} platforms.",
            "Researchers at the {kw1} lab have developed a breakthrough in {kw2} {kw3} that could transform how we approach {kw4} and {kw5}.",
            "The latest {kw1} update introduces powerful {kw2} capabilities powered by {kw3} and {kw4} {kw5} integration.",
            "Industry experts predict that {kw1} and {kw2} will dominate the {kw3} landscape, with {kw4} {kw5} leading the way.",
            "A new report highlights the growing importance of {kw1} {kw2} in modern {kw3}, with investments in {kw4} and {kw5} surging.",
        ],
    },
    'sports': {
        'keywords': ['team', 'player', 'game', 'score', 'championship',
                     'season', 'coach', 'tournament', 'athlete', 'training',
                     'victory', 'match', 'league', 'stadium', 'competition',
                     'record', 'medal', 'fitness', 'performance', 'defense',
                     'offense', 'quarterback', 'goal', 'basketball', 'football',
                     'baseball', 'soccer', 'tennis', 'olympic', 'sprint'],
        'templates': [
            "The {kw1} delivered a stunning {kw2} performance in last night's {kw3}, securing their spot in the {kw4} {kw5}.",
            "Star {kw1} broke the {kw2} {kw3} record during the {kw4} {kw5}, thrilling fans worldwide.",
            "The upcoming {kw1} {kw2} promises intense {kw3} action with top {kw4} {kw5} competing for glory.",
            "After a grueling {kw1} of {kw2} and {kw3}, the {kw4} emerged victorious in the {kw5} finals.",
            "The {kw1} {kw2} announced a major trade, bringing in a talented {kw3} to boost their {kw4} {kw5} lineup.",
        ],
    },
    'politics': {
        'keywords': ['government', 'policy', 'election', 'president', 'congress',
                     'legislation', 'democrat', 'republican', 'vote', 'campaign',
                     'senate', 'bill', 'reform', 'budget', 'tax',
                     'healthcare', 'immigration', 'foreign', 'diplomatic', 'administration',
                     'committee', 'bipartisan', 'regulation', 'debate', 'political',
                     'governor', 'mayor', 'district', 'constituent', 'amendment'],
        'templates': [
            "The {kw1} introduced new {kw2} {kw3} aimed at addressing key {kw4} concerns raised by {kw5} leaders.",
            "Ahead of the {kw1}, candidates are focusing on {kw2} {kw3} and {kw4} issues that matter to {kw5}.",
            "The {kw1} {kw2} passed a landmark {kw3} bill that could reshape {kw4} and {kw5} for years to come.",
            "Critics of the {kw1} {kw2} argue that the proposed {kw3} fails to address underlying {kw4} and {kw5} challenges.",
            "In a {kw1} move, the {kw2} reached across the aisle to forge a {kw3} agreement on {kw4} {kw5}.",
        ],
    },
    'science': {
        'keywords': ['research', 'study', 'scientist', 'discovery', 'experiment',
                     'hypothesis', 'laboratory', 'molecule', 'genome', 'climate',
                     'space', 'physics', 'biology', 'chemistry', 'evolution',
                     'species', 'quantum', 'particle', 'telescope', 'fossil',
                     'DNA', 'protein', 'cell', 'vaccine', 'energy',
                     'gravity', 'universe', 'planet', 'ecosystem', 'theory'],
        'templates': [
            "A groundbreaking {kw1} published in Nature reveals new insights into {kw2} {kw3} and its implications for {kw4} {kw5}.",
            "Scientists at the {kw1} have made a remarkable {kw2} about {kw3} that challenges our understanding of {kw4} and {kw5}.",
            "The latest {kw1} on {kw2} {kw3} suggests that {kw4} plays a crucial role in {kw5} development.",
            "New {kw1} data confirms the {kw2} about {kw3} {kw4}, opening new avenues for {kw5} research.",
            "Researchers used advanced {kw1} techniques to study {kw2} {kw3} and discovered unexpected connections to {kw4} {kw5}.",
        ],
    },
    'business': {
        'keywords': ['market', 'company', 'revenue', 'profit', 'investor',
                     'stock', 'CEO', 'startup', 'merger', 'acquisition',
                     'growth', 'quarterly', 'earnings', 'industry', 'consumer',
                     'brand', 'marketing', 'supply', 'chain', 'retail',
                     'economy', 'inflation', 'banking', 'venture', 'capital',
                     'strategy', 'management', 'workforce', 'innovation', 'trade'],
        'templates': [
            "The {kw1} reported strong {kw2} {kw3} this quarter, exceeding {kw4} expectations and boosting {kw5} confidence.",
            "A major {kw1} between two leading {kw2} firms is reshaping the {kw3} landscape and creating new {kw4} {kw5} opportunities.",
            "Analysts predict that {kw1} trends in {kw2} and {kw3} will drive {kw4} {kw5} throughout the coming year.",
            "The {kw1} {kw2} unveiled a new {kw3} {kw4} initiative aimed at capturing emerging {kw5} opportunities.",
            "Rising {kw1} concerns are pushing {kw2} leaders to rethink their {kw3} {kw4} and {kw5} strategies.",
        ],
    },
}


def generate_document_corpus(n_docs_per_topic=100, min_sentences=3, max_sentences=8):
    """
    Generate a synthetic corpus of documents from predefined topic areas.
    Each document is a collection of sentences using topic-specific vocabulary.
    Some documents blend multiple topics to simulate real-world overlap.
    """
    documents = []
    true_topics = []
    topic_names = list(TOPIC_VOCABULARIES.keys())

    for topic_name, topic_data in TOPIC_VOCABULARIES.items():
        keywords = topic_data['keywords']
        templates = topic_data['templates']

        for _ in range(n_docs_per_topic):
            n_sentences = np.random.randint(min_sentences, max_sentences + 1)
            sentences = []

            for _ in range(n_sentences):
                template = np.random.choice(templates)
                kws = np.random.choice(keywords, size=5, replace=False)
                sentence = template.format(kw1=kws[0], kw2=kws[1], kw3=kws[2],
                                           kw4=kws[3], kw5=kws[4])
                sentences.append(sentence)

            # 20% chance of adding a sentence from another topic (cross-topic blending)
            if np.random.random() < 0.2:
                other_topic = np.random.choice([t for t in topic_names if t != topic_name])
                other_data = TOPIC_VOCABULARIES[other_topic]
                other_kws = np.random.choice(other_data['keywords'], size=5, replace=False)
                other_template = np.random.choice(other_data['templates'])
                blended = other_template.format(kw1=other_kws[0], kw2=other_kws[1],
                                                kw3=other_kws[2], kw4=other_kws[3],
                                                kw5=other_kws[4])
                sentences.append(blended)

            doc = ' '.join(sentences)
            documents.append(doc)
            true_topics.append(topic_name)

    df = pd.DataFrame({
        'document': documents,
        'true_topic': true_topics,
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Corpus generated: {len(df)} documents")
    print(f"Topics: {dict(Counter(df['true_topic']))}")
    return df


# ============================================================================
# 2. TEXT PREPROCESSING
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
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now',
    'new', 'also', 'could', 'would', 'may', 'might', 'shall', 'into',
    'its', 'let', 'us', 'like', 'get', 'got', 'much', 'many', 'well',
    'even', 'still', 'way', 'take', 'come', 'make', 'go', 'know', 'see',
    'look', 'find', 'give', 'use', 'tell', 'think', 'say', 'help',
}


def preprocess_document(text):
    """Clean and normalize a single document for topic modeling."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def preprocess_corpus(documents):
    """Apply preprocessing to all documents."""
    return [preprocess_document(doc) for doc in documents]


# ============================================================================
# 3. LDA TOPIC MODELING
# ============================================================================

def train_lda(documents, n_topics=5, max_iter=20, random_state=42):
    """
    Train an LDA model using sklearn's LatentDirichletAllocation.

    Parameters
    ----------
    documents : list of str
        Preprocessed document texts.
    n_topics : int
        Number of topics to discover.
    max_iter : int
        Maximum number of EM iterations.

    Returns
    -------
    lda_model : LatentDirichletAllocation
        Fitted LDA model.
    count_vectorizer : CountVectorizer
        Fitted vectorizer.
    doc_term_matrix : sparse matrix
        Document-term matrix.
    """
    count_vectorizer = CountVectorizer(
        max_df=0.9, min_df=2, max_features=5000
    )
    doc_term_matrix = count_vectorizer.fit_transform(documents)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        random_state=random_state,
        n_jobs=-1,
    )
    lda_model.fit(doc_term_matrix)

    print(f"LDA Model trained with {n_topics} topics")
    print(f"  Log-likelihood: {lda_model.score(doc_term_matrix):.2f}")
    print(f"  Perplexity: {lda_model.perplexity(doc_term_matrix):.2f}")

    return lda_model, count_vectorizer, doc_term_matrix


# ============================================================================
# 4. NMF TOPIC MODELING
# ============================================================================

def train_nmf(documents, n_topics=5, max_iter=200, random_state=42):
    """
    Train an NMF model using sklearn's NMF with TF-IDF features.

    Returns
    -------
    nmf_model : NMF
        Fitted NMF model.
    tfidf_vectorizer : TfidfVectorizer
        Fitted vectorizer.
    tfidf_matrix : sparse matrix
        TF-IDF document-term matrix.
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.9, min_df=2, max_features=5000
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    nmf_model = NMF(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=random_state,
        init='nndsvd',
    )
    nmf_model.fit(tfidf_matrix)

    reconstruction_error = nmf_model.reconstruction_err_
    print(f"NMF Model trained with {n_topics} topics")
    print(f"  Reconstruction error: {reconstruction_error:.4f}")

    return nmf_model, tfidf_vectorizer, tfidf_matrix


# ============================================================================
# 5. TOPIC DISPLAY AND ANALYSIS
# ============================================================================

def display_topics(model, vectorizer, n_top_words=15):
    """Print the top words for each topic discovered by the model."""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_idx, topic_weights in enumerate(model.components_):
        top_word_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        top_scores = [topic_weights[i] for i in top_word_indices]
        topics[topic_idx] = list(zip(top_words, top_scores))

        print(f"\n  Topic {topic_idx + 1}:")
        word_str = ', '.join(top_words)
        print(f"    {word_str}")

    return topics


def get_document_topics(model, matrix):
    """
    Get the topic distribution for each document.

    Returns
    -------
    doc_topics : np.ndarray
        Array of shape (n_docs, n_topics) with topic weights.
    dominant_topics : np.ndarray
        Array of dominant topic index for each document.
    """
    if hasattr(model, 'transform'):
        doc_topics = model.transform(matrix)
    else:
        doc_topics = model.components_.dot(matrix.T).T

    # Normalize rows for NMF (LDA already returns normalized distributions)
    row_sums = doc_topics.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_topics_normalized = doc_topics / row_sums

    dominant_topics = doc_topics_normalized.argmax(axis=1)
    return doc_topics_normalized, dominant_topics


# ============================================================================
# 6. COHERENCE SCORE OPTIMIZATION
# ============================================================================

def compute_coherence_umass(model, vectorizer, doc_term_matrix, n_top_words=10):
    """
    Compute UMass coherence score for a topic model.
    UMass coherence measures word co-occurrence within the corpus.

    C_UMass = (2 / (N * (N-1))) * sum(log((D(w_i, w_j) + 1) / D(w_j)))

    Higher (less negative) values indicate more coherent topics.
    """
    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = (doc_term_matrix > 0).toarray().astype(float)
    n_docs = dense_matrix.shape[0]

    coherence_scores = []

    for topic_weights in model.components_:
        top_word_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
        topic_coherence = 0
        n_pairs = 0

        for i in range(1, len(top_word_indices)):
            for j in range(0, i):
                wi = top_word_indices[i]
                wj = top_word_indices[j]

                # Document frequency of word j
                d_wj = dense_matrix[:, wj].sum()
                # Co-document frequency of words i and j
                d_wi_wj = (dense_matrix[:, wi] * dense_matrix[:, wj]).sum()

                if d_wj > 0:
                    topic_coherence += np.log((d_wi_wj + 1) / d_wj)
                    n_pairs += 1

        if n_pairs > 0:
            topic_coherence /= n_pairs
        coherence_scores.append(topic_coherence)

    return np.mean(coherence_scores)


def optimize_n_topics(documents, method='lda', topic_range=range(2, 12),
                      save_path='coherence_optimization.png'):
    """
    Find the optimal number of topics by evaluating coherence scores
    across a range of topic counts.
    """
    print(f"\nOptimizing number of topics for {method.upper()}...")
    print(f"Testing range: {list(topic_range)}")

    coherence_scores = []
    perplexity_scores = []

    for n_topics in topic_range:
        if method == 'lda':
            model, vectorizer, matrix = train_lda(documents, n_topics=n_topics)
            perplexity_scores.append(model.perplexity(matrix))
        else:
            model, vectorizer, matrix = train_nmf(documents, n_topics=n_topics)

        coherence = compute_coherence_umass(model, vectorizer, matrix)
        coherence_scores.append(coherence)
        print(f"  n_topics={n_topics:2d}: coherence={coherence:.4f}")

    # Plot coherence scores
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Topics')
    ax1.set_ylabel('Coherence Score (UMass)', color=color1)
    ax1.plot(list(topic_range), coherence_scores, 'o-', color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    if perplexity_scores:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Perplexity', color=color2)
        ax2.plot(list(topic_range), perplexity_scores, 's--', color=color2, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)

    best_idx = np.argmax(coherence_scores)
    best_n = list(topic_range)[best_idx]
    ax1.axvline(x=best_n, color='green', linestyle=':', linewidth=2,
                label=f'Best: {best_n} topics')
    ax1.legend()

    plt.title(f'{method.upper()}: Coherence Score vs Number of Topics')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Optimization plot saved to {save_path}")
    print(f"Optimal number of topics: {best_n} (coherence={coherence_scores[best_idx]:.4f})")

    return best_n, coherence_scores


# ============================================================================
# 7. TOPIC VISUALIZATION
# ============================================================================

def plot_topic_word_bars(model, vectorizer, n_top_words=10,
                        save_path='topic_word_bars.png', title_prefix=''):
    """Create horizontal bar plots of top words for each topic."""
    feature_names = vectorizer.get_feature_names_out()
    n_topics = model.components_.shape[0]

    n_cols = min(3, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    if n_topics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for topic_idx, topic_weights in enumerate(model.components_):
        row, col = topic_idx // n_cols, topic_idx % n_cols
        ax = axes[row, col]

        top_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [topic_weights[i] for i in top_indices]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_top_words))
        ax.barh(range(n_top_words), top_scores[::-1], color=colors)
        ax.set_yticks(range(n_top_words))
        ax.set_yticklabels(top_words[::-1])
        ax.set_title(f'{title_prefix}Topic {topic_idx + 1}')
        ax.set_xlabel('Weight')

    # Hide unused axes
    for idx in range(n_topics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Topic word bars saved to {save_path}")


def plot_topic_distribution(doc_topics, true_labels=None,
                            save_path='topic_distribution.png'):
    """
    Visualize the overall topic distribution across the corpus
    and optionally compare with true labels.
    """
    dominant_topics = doc_topics.argmax(axis=1)
    n_topics = doc_topics.shape[1]

    if true_labels is not None:
        unique_labels = sorted(set(true_labels))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Topic distribution
        topic_counts = Counter(dominant_topics)
        topics = range(n_topics)
        counts = [topic_counts.get(t, 0) for t in topics]
        axes[0].bar(topics, counts, color=plt.cm.Set3(np.linspace(0, 1, n_topics)))
        axes[0].set_xlabel('Topic')
        axes[0].set_ylabel('Number of Documents')
        axes[0].set_title('Document Distribution Across Topics')
        axes[0].set_xticks(topics)
        axes[0].set_xticklabels([f'Topic {t + 1}' for t in topics], rotation=45)

        # Topic vs true label heatmap
        confusion = np.zeros((len(unique_labels), n_topics))
        for label, topic in zip(true_labels, dominant_topics):
            label_idx = unique_labels.index(label)
            confusion[label_idx, topic] += 1

        # Normalize rows
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confusion_norm = confusion / row_sums

        sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=[f'Topic {t + 1}' for t in range(n_topics)],
                    yticklabels=unique_labels, ax=axes[1])
        axes[1].set_title('True Label vs Discovered Topic')
        axes[1].set_xlabel('Discovered Topic')
        axes[1].set_ylabel('True Label')
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        topic_counts = Counter(dominant_topics)
        topics = range(n_topics)
        counts = [topic_counts.get(t, 0) for t in topics]
        ax.bar(topics, counts, color=plt.cm.Set3(np.linspace(0, 1, n_topics)))
        ax.set_xlabel('Topic')
        ax.set_ylabel('Number of Documents')
        ax.set_title('Document Distribution Across Topics')
        ax.set_xticks(topics)
        ax.set_xticklabels([f'Topic {t + 1}' for t in topics], rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Topic distribution plot saved to {save_path}")


def plot_word_clouds_simple(model, vectorizer, n_top_words=30,
                            save_path='topic_wordclouds.png'):
    """
    Create a simple word cloud visualization using sized text.
    (Does not require the wordcloud package.)
    """
    feature_names = vectorizer.get_feature_names_out()
    n_topics = model.components_.shape[0]

    n_cols = min(3, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    if n_topics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for topic_idx, topic_weights in enumerate(model.components_):
        row, col = topic_idx // n_cols, topic_idx % n_cols
        ax = axes[row, col]

        top_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = topic_weights[top_indices]

        # Normalize scores to font sizes
        min_score, max_score = top_scores.min(), top_scores.max()
        if max_score > min_score:
            normalized = (top_scores - min_score) / (max_score - min_score)
        else:
            normalized = np.ones_like(top_scores)

        font_sizes = 8 + normalized * 24  # Range: 8 to 32

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Place words in a grid-like pattern
        n_words = len(top_words)
        cols_per_row = 5
        rows_needed = (n_words + cols_per_row - 1) // cols_per_row

        for i, (word, size) in enumerate(zip(top_words, font_sizes)):
            r = i // cols_per_row
            c = i % cols_per_row
            x = (c + 0.5) / cols_per_row
            y = 1 - (r + 0.5) / rows_needed

            color = plt.cm.Dark2(np.random.random())
            ax.text(x, y, word, fontsize=size, ha='center', va='center',
                    color=color, fontweight='bold')

        ax.set_title(f'Topic {topic_idx + 1}', fontsize=14, fontweight='bold')
        ax.axis('off')

    for idx in range(n_topics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word clouds saved to {save_path}")


# ============================================================================
# 8. DOCUMENT-TOPIC ASSIGNMENT
# ============================================================================

def assign_document_topics(df, doc_topics, dominant_topics, n_topics):
    """
    Add topic assignments to the dataframe and analyze results.
    """
    df = df.copy()
    df['dominant_topic'] = dominant_topics
    df['dominant_topic_label'] = [f'Topic {t + 1}' for t in dominant_topics]

    # Add topic probability columns
    for t in range(n_topics):
        df[f'topic_{t + 1}_prob'] = doc_topics[:, t]

    # Topic confidence (max topic probability)
    df['topic_confidence'] = doc_topics.max(axis=1)

    # Analyze alignment with true topics
    print("\nDocument-Topic Assignment Analysis:")
    print("=" * 60)
    print(f"\nTopic assignment distribution:")
    print(df['dominant_topic_label'].value_counts().to_string())

    print(f"\nAverage topic confidence: {df['topic_confidence'].mean():.4f}")
    print(f"Min confidence: {df['topic_confidence'].min():.4f}")
    print(f"Max confidence: {df['topic_confidence'].max():.4f}")

    if 'true_topic' in df.columns:
        print("\nCross-tabulation (True Topic vs Discovered Topic):")
        cross_tab = pd.crosstab(df['true_topic'], df['dominant_topic_label'],
                                margins=True)
        print(cross_tab.to_string())

    return df


def show_representative_documents(df, doc_topics, n_per_topic=3):
    """Show the most representative documents for each topic."""
    n_topics = doc_topics.shape[1]

    print("\nMost Representative Documents per Topic:")
    print("=" * 60)

    for topic_idx in range(n_topics):
        print(f"\n--- Topic {topic_idx + 1} ---")
        topic_probs = doc_topics[:, topic_idx]
        top_doc_indices = topic_probs.argsort()[-n_per_topic:][::-1]

        for rank, idx in enumerate(top_doc_indices, 1):
            prob = topic_probs[idx]
            doc_snippet = df.iloc[idx]['document'][:120]
            true_topic = df.iloc[idx].get('true_topic', 'N/A')
            print(f"  {rank}. [prob={prob:.3f}] (true: {true_topic})")
            print(f"     \"{doc_snippet}...\"")


# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================

def compare_lda_nmf(lda_topics, nmf_topics, lda_coherence, nmf_coherence,
                    save_path='model_comparison.png'):
    """Compare LDA and NMF topic models visually."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coherence comparison
    models = ['LDA', 'NMF']
    scores = [lda_coherence, nmf_coherence]
    colors = ['steelblue', 'coral']
    axes[0].bar(models, scores, color=colors, width=0.5)
    axes[0].set_ylabel('Coherence Score (UMass)')
    axes[0].set_title('Model Coherence Comparison')
    for i, v in enumerate(scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

    # Topic overlap analysis - Jaccard similarity between top words
    n_topics = len(lda_topics)
    overlap_matrix = np.zeros((n_topics, n_topics))

    for i, lda_topic in lda_topics.items():
        lda_words = set(w for w, _ in lda_topic[:10])
        for j, nmf_topic in nmf_topics.items():
            nmf_words = set(w for w, _ in nmf_topic[:10])
            intersection = len(lda_words & nmf_words)
            union = len(lda_words | nmf_words)
            overlap_matrix[i, j] = intersection / union if union > 0 else 0

    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='YlGn',
                xticklabels=[f'NMF T{i + 1}' for i in range(n_topics)],
                yticklabels=[f'LDA T{i + 1}' for i in range(n_topics)],
                ax=axes[1])
    axes[1].set_title('Topic Word Overlap (Jaccard Similarity)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("        TOPIC MODELING SYSTEM - NLP Portfolio Project")
    print("=" * 70)

    # --- Corpus Generation ---
    print("\n[1] Generating synthetic document corpus...")
    df = generate_document_corpus(n_docs_per_topic=100)

    print("\nSample documents:")
    for _, row in df.head(3).iterrows():
        print(f"  [{row['true_topic']:12s}] {row['document'][:100]}...")

    # --- Preprocessing ---
    print("\n[2] Preprocessing corpus...")
    processed_docs = preprocess_corpus(df['document'].values)

    print("\nPreprocessing example:")
    print(f"  Original : {df['document'].iloc[0][:80]}...")
    print(f"  Processed: {processed_docs[0][:80]}...")

    # --- LDA Topic Modeling ---
    print("\n[3] Training LDA model...")
    lda_model, count_vec, count_matrix = train_lda(processed_docs, n_topics=5)

    print("\nLDA Topics:")
    lda_topics = display_topics(lda_model, count_vec)

    # --- NMF Topic Modeling ---
    print("\n[4] Training NMF model...")
    nmf_model, tfidf_vec, tfidf_matrix = train_nmf(processed_docs, n_topics=5)

    print("\nNMF Topics:")
    nmf_topics = display_topics(nmf_model, tfidf_vec)

    # --- Coherence Scores ---
    print("\n[5] Computing coherence scores...")
    lda_coherence = compute_coherence_umass(lda_model, count_vec, count_matrix)
    nmf_coherence = compute_coherence_umass(nmf_model, tfidf_vec, tfidf_matrix)
    print(f"  LDA Coherence (UMass): {lda_coherence:.4f}")
    print(f"  NMF Coherence (UMass): {nmf_coherence:.4f}")

    # --- Topic Number Optimization ---
    print("\n[6] Optimizing number of topics...")
    best_n_lda, lda_scores = optimize_n_topics(
        processed_docs, method='lda', topic_range=range(2, 10),
        save_path='lda_coherence_optimization.png'
    )
    best_n_nmf, nmf_scores = optimize_n_topics(
        processed_docs, method='nmf', topic_range=range(2, 10),
        save_path='nmf_coherence_optimization.png'
    )

    # --- Visualizations ---
    print("\n[7] Generating visualizations...")

    # Word bar charts
    plot_topic_word_bars(lda_model, count_vec, save_path='lda_topic_words.png',
                        title_prefix='LDA ')
    plot_topic_word_bars(nmf_model, tfidf_vec, save_path='nmf_topic_words.png',
                        title_prefix='NMF ')

    # Word clouds
    plot_word_clouds_simple(lda_model, count_vec, save_path='lda_wordclouds.png')
    plot_word_clouds_simple(nmf_model, tfidf_vec, save_path='nmf_wordclouds.png')

    # --- Document-Topic Assignment ---
    print("\n[8] Assigning documents to topics...")
    lda_doc_topics, lda_dominant = get_document_topics(lda_model, count_matrix)
    nmf_doc_topics, nmf_dominant = get_document_topics(nmf_model, tfidf_matrix)

    print("\n--- LDA Document-Topic Assignment ---")
    df_lda = assign_document_topics(df, lda_doc_topics, lda_dominant, n_topics=5)

    print("\n--- NMF Document-Topic Assignment ---")
    df_nmf = assign_document_topics(df, nmf_doc_topics, nmf_dominant, n_topics=5)

    # Topic distribution plots
    plot_topic_distribution(lda_doc_topics, df['true_topic'].values,
                            save_path='lda_topic_distribution.png')
    plot_topic_distribution(nmf_doc_topics, df['true_topic'].values,
                            save_path='nmf_topic_distribution.png')

    # --- Representative Documents ---
    print("\n[9] Finding representative documents...")
    print("\n--- LDA Representative Documents ---")
    show_representative_documents(df, lda_doc_topics, n_per_topic=2)

    print("\n--- NMF Representative Documents ---")
    show_representative_documents(df, nmf_doc_topics, n_per_topic=2)

    # --- Model Comparison ---
    print("\n[10] Comparing LDA vs NMF...")
    compare_lda_nmf(lda_topics, nmf_topics, lda_coherence, nmf_coherence)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Corpus size: {len(df)} documents, {len(TOPIC_VOCABULARIES)} true topics")
    print(f"LDA: {5} topics, coherence = {lda_coherence:.4f}, optimal = {best_n_lda}")
    print(f"NMF: {5} topics, coherence = {nmf_coherence:.4f}, optimal = {best_n_nmf}")
    print(f"\nGenerated files:")
    print(f"  - lda_coherence_optimization.png")
    print(f"  - nmf_coherence_optimization.png")
    print(f"  - lda_topic_words.png / nmf_topic_words.png")
    print(f"  - lda_wordclouds.png / nmf_wordclouds.png")
    print(f"  - lda_topic_distribution.png / nmf_topic_distribution.png")
    print(f"  - model_comparison.png")
    print(f"{'=' * 70}")
    print("Topic modeling project complete.")


if __name__ == '__main__':
    main()
