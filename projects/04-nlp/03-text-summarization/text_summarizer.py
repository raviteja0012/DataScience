"""
Text Summarization System
=========================
Implements multiple extractive summarization approaches:
- TF-IDF based sentence scoring
- TextRank algorithm (graph-based)
- Frequency-based scoring
Includes ROUGE-like evaluation metrics and compression ratio control.

Author: Data Science Portfolio
"""

import numpy as np
import re
import string
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. SAMPLE ARTICLES FOR TESTING
# ============================================================================

SAMPLE_ARTICLES = {
    'ai_technology': """
Artificial intelligence has become one of the most transformative technologies of the 21st century.
Machine learning algorithms are now capable of performing tasks that were once thought to be exclusively
in the domain of human intelligence. From natural language processing to computer vision, AI systems
are achieving remarkable results across a wide range of applications.

Deep learning, a subset of machine learning, has been particularly revolutionary. Neural networks with
many layers can learn hierarchical representations of data, enabling breakthroughs in image recognition,
speech synthesis, and language translation. Companies like Google, Microsoft, and OpenAI have invested
billions of dollars in developing more powerful AI models.

The impact of AI extends beyond technology companies. Healthcare organizations are using machine learning
to diagnose diseases earlier and more accurately. Financial institutions employ AI algorithms for fraud
detection and risk assessment. Manufacturing companies use robotics and computer vision for quality
control and process optimization.

However, the rapid advancement of AI also raises important ethical concerns. Questions about bias in
AI systems, job displacement due to automation, and the potential misuse of AI-generated content are
being actively debated. Researchers and policymakers are working together to establish guidelines and
regulations that ensure AI is developed and deployed responsibly.

The future of AI looks promising but uncertain. Some experts predict that artificial general intelligence,
which would match or exceed human cognitive abilities across all domains, could be achieved within the
next few decades. Others argue that current AI systems, while impressive, are still fundamentally limited
in their understanding and reasoning capabilities. Regardless of the timeline, it is clear that AI will
continue to shape our world in profound ways.
""",

    'climate_change': """
Climate change represents one of the greatest challenges facing humanity in the 21st century. Global
temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times, and the rate
of warming has accelerated in recent decades. Scientists attribute this warming primarily to human
activities, particularly the burning of fossil fuels and deforestation.

The consequences of climate change are already being felt around the world. Rising sea levels threaten
coastal communities and island nations. Extreme weather events, including hurricanes, droughts, and
wildfires, are becoming more frequent and severe. Changes in precipitation patterns are affecting
agriculture and water supplies in many regions.

The scientific community has reached a strong consensus on the need for urgent action. The
Intergovernmental Panel on Climate Change has warned that global emissions must be reduced by
approximately 45 percent by 2030 to limit warming to 1.5 degrees Celsius. Achieving this target
will require rapid transitions in energy systems, transportation, and land use.

Renewable energy sources such as solar and wind power have become increasingly cost-competitive with
fossil fuels. Many countries have set ambitious targets for reducing greenhouse gas emissions, with
some aiming for net-zero emissions by 2050. Electric vehicles are gaining market share, and energy
storage technologies are improving rapidly.

Despite these positive trends, significant obstacles remain. Many developing nations rely heavily on
fossil fuels for economic growth and lack the resources to transition quickly to clean energy.
International cooperation is essential but has proven difficult to sustain. The challenge of climate
change requires a coordinated global response that balances environmental protection with economic
development and social equity.

Individuals can also contribute to climate action through lifestyle choices such as reducing energy
consumption, using public transportation, eating less meat, and supporting sustainable businesses.
While individual actions alone are not sufficient, they can drive demand for cleaner products and
services and send important signals to policymakers and corporations.
""",

    'space_exploration': """
Space exploration has entered a new era characterized by both government-led missions and private
sector innovation. NASA's Artemis program aims to return humans to the Moon and eventually send
astronauts to Mars. Meanwhile, companies like SpaceX and Blue Origin are developing reusable rockets
that could dramatically reduce the cost of accessing space.

The International Space Station has served as a remarkable platform for scientific research and
international cooperation for over two decades. Experiments conducted aboard the station have advanced
our understanding of human physiology in microgravity, materials science, and Earth observation.
However, the station is aging, and plans are underway to develop commercial space stations as successors.

Mars remains the most compelling target for human exploration beyond the Moon. The planet has evidence
of past water activity and may harbor conditions suitable for microbial life. Several robotic missions,
including NASA's Perseverance rover, are currently exploring the Martian surface and collecting samples
for future return to Earth.

Beyond Mars, scientists are studying the outer solar system with increasing interest. Jupiter's moon
Europa and Saturn's moon Enceladus both have subsurface oceans that could potentially support life.
Future missions to these worlds will search for biosignatures and help answer the fundamental question
of whether we are alone in the universe.

The commercialization of space is creating new economic opportunities. Satellite-based services for
communications, navigation, and Earth observation generate hundreds of billions of dollars in revenue
annually. Space tourism is becoming a reality, with several companies offering suborbital flights
to paying customers. The emerging space economy could grow substantially in the coming decades.

Advances in telescope technology are also expanding our understanding of the universe. The James Webb
Space Telescope, launched in late 2021, is providing unprecedented views of distant galaxies, star
formation regions, and exoplanet atmospheres. These observations are helping scientists refine theories
about the origin and evolution of the cosmos.
""",
}


def generate_random_article(n_sentences=20, topic='general'):
    """Generate a random article for testing summarization."""
    sentence_templates = {
        'general': [
            "Research has shown that {subject} plays a significant role in {context}.",
            "Experts in the field argue that {subject} is essential for understanding {context}.",
            "Recent studies indicate that {subject} has a profound impact on {context}.",
            "The relationship between {subject} and {context} has been widely documented.",
            "According to leading scholars, {subject} continues to influence {context} in unexpected ways.",
            "New evidence suggests that {subject} may be more important to {context} than previously thought.",
            "The implications of {subject} for {context} cannot be overstated.",
            "Understanding {subject} is crucial for making progress in {context}.",
            "Many organizations are now investing heavily in {subject} to advance {context}.",
            "The future of {context} depends largely on developments in {subject}.",
        ],
    }
    subjects = ['technology', 'innovation', 'education', 'public policy',
                'scientific research', 'international cooperation', 'economic growth',
                'environmental sustainability', 'digital transformation', 'healthcare']
    contexts = ['modern society', 'global development', 'human progress',
                'economic stability', 'public health', 'environmental protection',
                'social equality', 'technological advancement', 'cultural exchange',
                'scientific discovery']

    templates = sentence_templates['general']
    sentences = []
    for _ in range(n_sentences):
        template = np.random.choice(templates)
        sentence = template.format(
            subject=np.random.choice(subjects),
            context=np.random.choice(contexts),
        )
        sentences.append(sentence)

    return ' '.join(sentences)


# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================

def split_sentences(text):
    """
    Split text into sentences using regex-based rules.
    Handles common abbreviations and edge cases.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on sentence-ending punctuation followed by whitespace and uppercase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Clean up
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def preprocess_for_tfidf(text):
    """Clean text for TF-IDF computation."""
    text = text.lower()
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# 3. TF-IDF BASED SUMMARIZATION
# ============================================================================

def compute_tf(text):
    """Compute term frequency for a document."""
    words = text.lower().split()
    word_count = Counter(words)
    total_words = len(words)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf


def compute_idf(sentences):
    """Compute inverse document frequency across sentences."""
    n_sentences = len(sentences)
    word_doc_count = Counter()

    for sentence in sentences:
        words = set(sentence.lower().split())
        for word in words:
            word_doc_count[word] += 1

    idf = {}
    for word, count in word_doc_count.items():
        idf[word] = np.log((n_sentences + 1) / (count + 1)) + 1  # Smoothed IDF

    return idf


def compute_tfidf_scores(sentences):
    """Compute TF-IDF scores for each word in the corpus of sentences."""
    idf = compute_idf(sentences)
    tfidf_per_sentence = []

    for sentence in sentences:
        tf = compute_tf(sentence)
        tfidf = {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}
        tfidf_per_sentence.append(tfidf)

    return tfidf_per_sentence, idf


def tfidf_summarize(text, compression_ratio=0.3, min_sentences=2):
    """
    Extractive summarization using TF-IDF sentence scoring.

    Each sentence is scored by the average TF-IDF weight of its words.
    Top-scoring sentences are selected while preserving original order.

    Parameters
    ----------
    text : str
        Input text to summarize.
    compression_ratio : float
        Fraction of sentences to include (0 to 1).
    min_sentences : int
        Minimum number of sentences in the summary.

    Returns
    -------
    dict
        Summary text, selected indices, scores, and metadata.
    """
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    if n_sentences <= min_sentences:
        return {
            'summary': text,
            'selected_indices': list(range(n_sentences)),
            'scores': [1.0] * n_sentences,
            'n_original': n_sentences,
            'n_summary': n_sentences,
            'compression_ratio': 1.0,
        }

    # Clean sentences for scoring
    clean_sentences = [preprocess_for_tfidf(s) for s in sentences]

    # Remove common stopwords
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
        'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
        'if', 'then', 'than', 'that', 'this', 'these', 'those', 'it', 'its',
    }

    clean_sentences_filtered = []
    for s in clean_sentences:
        words = [w for w in s.split() if w not in stop_words and len(w) > 2]
        clean_sentences_filtered.append(' '.join(words))

    # Compute TF-IDF
    tfidf_scores, idf = compute_tfidf_scores(clean_sentences_filtered)

    # Score each sentence
    sentence_scores = []
    for i, tfidf in enumerate(tfidf_scores):
        if tfidf:
            score = np.mean(list(tfidf.values()))
            # Bonus for sentence position (first and last sentences often important)
            position_bonus = 0
            if i == 0:
                position_bonus = 0.1
            elif i == n_sentences - 1:
                position_bonus = 0.05
            elif i < n_sentences * 0.2:
                position_bonus = 0.05

            # Penalty for very short sentences
            length_factor = min(len(sentences[i].split()) / 10, 1.0)

            score = score * length_factor + position_bonus
        else:
            score = 0
        sentence_scores.append(score)

    # Select top sentences
    n_select = max(min_sentences, int(n_sentences * compression_ratio))
    n_select = min(n_select, n_sentences)

    ranked_indices = np.argsort(sentence_scores)[::-1][:n_select]
    selected_indices = sorted(ranked_indices.tolist())  # Preserve original order

    summary_sentences = [sentences[i] for i in selected_indices]
    summary = ' '.join(summary_sentences)

    return {
        'summary': summary,
        'selected_indices': selected_indices,
        'scores': sentence_scores,
        'n_original': n_sentences,
        'n_summary': len(summary_sentences),
        'compression_ratio': len(summary_sentences) / n_sentences,
        'original_length': len(text),
        'summary_length': len(summary),
        'length_ratio': len(summary) / max(len(text), 1),
    }


# ============================================================================
# 4. TEXTRANK ALGORITHM
# ============================================================================

def compute_sentence_similarity(s1, s2):
    """
    Compute cosine-like similarity between two sentences
    based on word overlap.
    """
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    # Normalized overlap
    similarity = len(intersection) / (np.log(len(words1)) + np.log(len(words2)) + 1)
    return similarity


def build_similarity_matrix(sentences):
    """
    Build a sentence similarity matrix for TextRank.
    Each entry (i, j) represents the similarity between sentence i and j.
    """
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = compute_sentence_similarity(
                    sentences[i], sentences[j]
                )

    # Normalize: make each row sum to 1 (transition probabilities)
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    similarity_matrix = similarity_matrix / row_sums

    return similarity_matrix


def textrank_scores(similarity_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """
    Compute TextRank scores using the power iteration method.
    This is analogous to PageRank but applied to sentences.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Transition probability matrix.
    damping : float
        Damping factor (default 0.85, same as PageRank).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    scores : np.ndarray
        TextRank score for each sentence.
    """
    n = similarity_matrix.shape[0]
    scores = np.ones(n) / n  # Uniform initialization

    for iteration in range(max_iter):
        prev_scores = scores.copy()
        for i in range(n):
            rank_sum = 0
            for j in range(n):
                if i != j and similarity_matrix[j].sum() > 0:
                    rank_sum += similarity_matrix[j][i] * prev_scores[j]
            scores[i] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        diff = np.abs(scores - prev_scores).sum()
        if diff < tol:
            break

    return scores


def textrank_summarize(text, compression_ratio=0.3, min_sentences=2, damping=0.85):
    """
    Extractive summarization using the TextRank algorithm.

    Builds a graph where sentences are nodes and edges represent
    similarity. Applies PageRank-like scoring to identify important sentences.

    Parameters
    ----------
    text : str
        Input text to summarize.
    compression_ratio : float
        Fraction of sentences to include.
    min_sentences : int
        Minimum number of sentences.
    damping : float
        Damping factor for TextRank.

    Returns
    -------
    dict
        Summary and metadata.
    """
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    if n_sentences <= min_sentences:
        return {
            'summary': text,
            'selected_indices': list(range(n_sentences)),
            'scores': [1.0] * n_sentences,
            'n_original': n_sentences,
            'n_summary': n_sentences,
            'compression_ratio': 1.0,
        }

    # Clean sentences
    clean_sentences = [preprocess_for_tfidf(s) for s in sentences]

    # Build similarity matrix and compute TextRank scores
    sim_matrix = build_similarity_matrix(clean_sentences)
    scores = textrank_scores(sim_matrix, damping=damping)

    # Select top sentences
    n_select = max(min_sentences, int(n_sentences * compression_ratio))
    n_select = min(n_select, n_sentences)

    ranked_indices = np.argsort(scores)[::-1][:n_select]
    selected_indices = sorted(ranked_indices.tolist())

    summary_sentences = [sentences[i] for i in selected_indices]
    summary = ' '.join(summary_sentences)

    return {
        'summary': summary,
        'selected_indices': selected_indices,
        'scores': scores.tolist(),
        'n_original': n_sentences,
        'n_summary': len(summary_sentences),
        'compression_ratio': len(summary_sentences) / n_sentences,
        'original_length': len(text),
        'summary_length': len(summary),
        'length_ratio': len(summary) / max(len(text), 1),
    }


# ============================================================================
# 5. FREQUENCY-BASED SUMMARIZATION
# ============================================================================

def frequency_summarize(text, compression_ratio=0.3, min_sentences=2):
    """
    Simple extractive summarization based on word frequency.

    Sentences containing more frequent (important) words receive
    higher scores. This is the simplest baseline approach.
    """
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    if n_sentences <= min_sentences:
        return {
            'summary': text,
            'selected_indices': list(range(n_sentences)),
            'scores': [1.0] * n_sentences,
            'n_original': n_sentences,
            'n_summary': n_sentences,
            'compression_ratio': 1.0,
        }

    # Count word frequencies across the full text
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
        'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
        'if', 'then', 'than', 'that', 'this', 'these', 'those', 'it', 'its',
    }

    all_words = [w for w in text.lower().split()
                 if w not in stop_words and len(w) > 2]
    word_freq = Counter(all_words)

    # Normalize frequencies
    max_freq = max(word_freq.values()) if word_freq else 1
    word_freq = {w: f / max_freq for w, f in word_freq.items()}

    # Score each sentence
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words = [w for w in sentence.lower().split() if w in word_freq]
        if words:
            score = sum(word_freq[w] for w in words) / len(words)
        else:
            score = 0
        sentence_scores.append(score)

    # Select top sentences
    n_select = max(min_sentences, int(n_sentences * compression_ratio))
    n_select = min(n_select, n_sentences)

    ranked_indices = np.argsort(sentence_scores)[::-1][:n_select]
    selected_indices = sorted(ranked_indices.tolist())

    summary_sentences = [sentences[i] for i in selected_indices]
    summary = ' '.join(summary_sentences)

    return {
        'summary': summary,
        'selected_indices': selected_indices,
        'scores': sentence_scores,
        'n_original': n_sentences,
        'n_summary': len(summary_sentences),
        'compression_ratio': len(summary_sentences) / n_sentences,
        'original_length': len(text),
        'summary_length': len(summary),
        'length_ratio': len(summary) / max(len(text), 1),
    }


# ============================================================================
# 6. ROUGE-LIKE EVALUATION METRICS
# ============================================================================

def compute_rouge_n(hypothesis, reference, n=1):
    """
    Compute ROUGE-N score (unigram or bigram overlap).

    ROUGE-N measures the overlap of n-grams between the hypothesis
    (system summary) and reference (gold summary).

    Returns
    -------
    dict
        Precision, recall, and F1 score.
    """
    def get_ngrams(text, n):
        words = text.lower().split()
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    hyp_ngrams = get_ngrams(hypothesis, n)
    ref_ngrams = get_ngrams(reference, n)

    if not ref_ngrams or not hyp_ngrams:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    hyp_counts = Counter(hyp_ngrams)
    ref_counts = Counter(ref_ngrams)

    # Count matching n-grams
    matches = 0
    for ngram, count in hyp_counts.items():
        matches += min(count, ref_counts.get(ngram, 0))

    precision = matches / len(hyp_ngrams) if hyp_ngrams else 0
    recall = matches / len(ref_ngrams) if ref_ngrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge_l(hypothesis, reference):
    """
    Compute ROUGE-L score based on Longest Common Subsequence.

    ROUGE-L captures sentence-level structure similarity by measuring
    the longest co-occurring sequence of words.

    Returns
    -------
    dict
        Precision, recall, and F1 score.
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    m = len(hyp_words)
    n = len(ref_words)

    if m == 0 or n == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Dynamic programming table for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    precision = lcs_length / m if m > 0 else 0
    recall = lcs_length / n if n > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_summary(hypothesis, reference):
    """
    Comprehensive evaluation of a summary using multiple ROUGE metrics.
    """
    rouge_1 = compute_rouge_n(hypothesis, reference, n=1)
    rouge_2 = compute_rouge_n(hypothesis, reference, n=2)
    rouge_l = compute_rouge_l(hypothesis, reference)

    return {
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l,
    }


# ============================================================================
# 7. COMPARISON OF APPROACHES
# ============================================================================

def compare_summarizers(text, reference_summary=None, compression_ratio=0.3):
    """
    Run all summarization approaches on the same text and compare results.
    """
    methods = {
        'TF-IDF': tfidf_summarize,
        'TextRank': textrank_summarize,
        'Frequency': frequency_summarize,
    }

    results = {}
    for name, method in methods.items():
        result = method(text, compression_ratio=compression_ratio)
        results[name] = result

    # Print comparison
    print(f"\n{'=' * 70}")
    print("SUMMARIZATION COMPARISON")
    print(f"{'=' * 70}")
    print(f"Original text: {len(text)} chars, "
          f"{len(split_sentences(text))} sentences")
    print(f"Target compression: {compression_ratio:.0%}")

    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(f"Summary ({result['n_summary']} sentences, "
              f"{result.get('summary_length', len(result['summary']))} chars, "
              f"compression: {result['compression_ratio']:.1%}):")
        print(f"  {result['summary'][:200]}...")

        if reference_summary:
            rouge_scores = evaluate_summary(result['summary'], reference_summary)
            print(f"  ROUGE-1 F1: {rouge_scores['ROUGE-1']['f1']:.4f}")
            print(f"  ROUGE-2 F1: {rouge_scores['ROUGE-2']['f1']:.4f}")
            print(f"  ROUGE-L F1: {rouge_scores['ROUGE-L']['f1']:.4f}")

    return results


def plot_comparison(results, reference_summary=None,
                    save_path='summarization_comparison.png'):
    """Visualize the comparison of summarization methods."""
    method_names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Compression ratios
    compressions = [results[m]['compression_ratio'] for m in method_names]
    colors = ['steelblue', 'coral', 'mediumseagreen']
    axes[0].bar(method_names, compressions, color=colors)
    axes[0].set_ylabel('Compression Ratio')
    axes[0].set_title('Compression Ratio by Method')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(compressions):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center')

    # 2. Summary lengths
    lengths = [results[m].get('summary_length', len(results[m]['summary']))
               for m in method_names]
    axes[1].bar(method_names, lengths, color=colors)
    axes[1].set_ylabel('Characters')
    axes[1].set_title('Summary Length by Method')
    for i, v in enumerate(lengths):
        axes[1].text(i, v + 5, str(v), ha='center')

    # 3. ROUGE scores (if reference available)
    if reference_summary:
        rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        x = np.arange(len(rouge_metrics))
        width = 0.25

        for i, method in enumerate(method_names):
            rouge = evaluate_summary(results[method]['summary'], reference_summary)
            f1_scores = [rouge[m]['f1'] for m in rouge_metrics]
            axes[2].bar(x + i * width, f1_scores, width, label=method,
                        color=colors[i])

        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('ROUGE Scores')
        axes[2].set_xticks(x + width)
        axes[2].set_xticklabels(rouge_metrics)
        axes[2].legend()
        axes[2].set_ylim(0, 1)
    else:
        axes[2].text(0.5, 0.5, 'No reference summary\navailable for ROUGE',
                     ha='center', va='center', fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title('ROUGE Scores (N/A)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


def plot_sentence_scores(text, save_path='sentence_scores.png'):
    """
    Visualize sentence importance scores from all methods.
    """
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    # Get scores from each method
    tfidf_result = tfidf_summarize(text, compression_ratio=1.0)
    textrank_result = textrank_summarize(text, compression_ratio=1.0)
    freq_result = frequency_summarize(text, compression_ratio=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    sentence_labels = [f'S{i + 1}' for i in range(n_sentences)]
    x = range(n_sentences)

    # Normalize scores to [0, 1] for comparison
    def normalize(scores):
        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            return (scores - min_s) / (max_s - min_s)
        return np.ones_like(scores) * 0.5

    methods_data = [
        ('TF-IDF Scores', normalize(tfidf_result['scores']),
         tfidf_summarize(text, compression_ratio=0.3)['selected_indices'], 'steelblue'),
        ('TextRank Scores', normalize(textrank_result['scores']),
         textrank_summarize(text, compression_ratio=0.3)['selected_indices'], 'coral'),
        ('Frequency Scores', normalize(freq_result['scores']),
         frequency_summarize(text, compression_ratio=0.3)['selected_indices'], 'mediumseagreen'),
    ]

    for ax, (title, scores, selected, color) in zip(axes, methods_data):
        bar_colors = [color if i in selected else 'lightgray' for i in range(n_sentences)]
        ax.bar(x, scores, color=bar_colors, edgecolor='gray', linewidth=0.5)
        ax.set_ylabel('Normalized Score')
        ax.set_title(title)
        ax.set_ylim(0, 1.1)

    axes[-1].set_xlabel('Sentence Index')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(sentence_labels, rotation=45, fontsize=8)

    plt.suptitle('Sentence Importance Scores by Method\n(Colored = selected for summary)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sentence scores plot saved to {save_path}")


# ============================================================================
# 8. COMPRESSION RATIO ANALYSIS
# ============================================================================

def analyze_compression_ratios(text, ratios=None, method='tfidf',
                               save_path='compression_analysis.png'):
    """
    Analyze how summary quality changes with different compression ratios.
    """
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    summarize_fn = {
        'tfidf': tfidf_summarize,
        'textrank': textrank_summarize,
        'frequency': frequency_summarize,
    }[method]

    # Use a constructed reference (top sentences by all methods combined)
    all_scores = np.zeros(len(split_sentences(text)))
    for fn in [tfidf_summarize, textrank_summarize, frequency_summarize]:
        result = fn(text, compression_ratio=1.0)
        scores = np.array(result['scores'])
        scores = (scores - scores.min()) / (max(scores.max() - scores.min(), 1e-10))
        all_scores += scores
    top_indices = np.argsort(all_scores)[::-1][:max(3, len(all_scores) // 3)]
    reference_sentences = [split_sentences(text)[i] for i in sorted(top_indices)]
    reference = ' '.join(reference_sentences)

    results_data = []
    for ratio in ratios:
        result = summarize_fn(text, compression_ratio=ratio)
        rouge = evaluate_summary(result['summary'], reference)
        results_data.append({
            'ratio': ratio,
            'n_sentences': result['n_summary'],
            'length': result.get('summary_length', len(result['summary'])),
            'rouge1_f1': rouge['ROUGE-1']['f1'],
            'rouge2_f1': rouge['ROUGE-2']['f1'],
            'rougeL_f1': rouge['ROUGE-L']['f1'],
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ratios_list = [d['ratio'] for d in results_data]

    # ROUGE scores vs compression ratio
    ax1.plot(ratios_list, [d['rouge1_f1'] for d in results_data], 'o-',
             label='ROUGE-1', color='steelblue')
    ax1.plot(ratios_list, [d['rouge2_f1'] for d in results_data], 's-',
             label='ROUGE-2', color='coral')
    ax1.plot(ratios_list, [d['rougeL_f1'] for d in results_data], '^-',
             label='ROUGE-L', color='mediumseagreen')
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'{method.upper()}: ROUGE vs Compression Ratio')
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Summary length vs compression ratio
    ax2.plot(ratios_list, [d['n_sentences'] for d in results_data], 'o-',
             color='purple')
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('Number of Sentences')
    ax2.set_title('Summary Size vs Compression Ratio')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Compression analysis saved to {save_path}")

    return results_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("        TEXT SUMMARIZATION SYSTEM - NLP Portfolio Project")
    print("=" * 70)

    # --- Process each sample article ---
    for article_name, article_text in SAMPLE_ARTICLES.items():
        print(f"\n\n{'#' * 70}")
        print(f"ARTICLE: {article_name.upper()}")
        print(f"{'#' * 70}")

        sentences = split_sentences(article_text)
        print(f"\nOriginal: {len(article_text)} characters, {len(sentences)} sentences")
        print(f"First 200 chars: {article_text.strip()[:200]}...")

        # --- TF-IDF Summarization ---
        print(f"\n[1] TF-IDF Based Summarization:")
        tfidf_result = tfidf_summarize(article_text, compression_ratio=0.3)
        print(f"    Selected {tfidf_result['n_summary']}/{tfidf_result['n_original']} sentences")
        print(f"    Compression: {tfidf_result['compression_ratio']:.1%}")
        print(f"    Summary: {tfidf_result['summary'][:200]}...")

        # --- TextRank Summarization ---
        print(f"\n[2] TextRank Summarization:")
        textrank_result = textrank_summarize(article_text, compression_ratio=0.3)
        print(f"    Selected {textrank_result['n_summary']}/{textrank_result['n_original']} sentences")
        print(f"    Compression: {textrank_result['compression_ratio']:.1%}")
        print(f"    Summary: {textrank_result['summary'][:200]}...")

        # --- Frequency-based Summarization ---
        print(f"\n[3] Frequency-based Summarization:")
        freq_result = frequency_summarize(article_text, compression_ratio=0.3)
        print(f"    Selected {freq_result['n_summary']}/{freq_result['n_original']} sentences")
        print(f"    Compression: {freq_result['compression_ratio']:.1%}")
        print(f"    Summary: {freq_result['summary'][:200]}...")

        # --- Cross-evaluation between methods ---
        print(f"\n[4] Cross-evaluation (using TF-IDF summary as reference):")
        reference = tfidf_result['summary']
        for name, result in [('TextRank', textrank_result), ('Frequency', freq_result)]:
            rouge = evaluate_summary(result['summary'], reference)
            print(f"    {name} vs TF-IDF:")
            print(f"      ROUGE-1: P={rouge['ROUGE-1']['precision']:.3f} "
                  f"R={rouge['ROUGE-1']['recall']:.3f} F1={rouge['ROUGE-1']['f1']:.3f}")
            print(f"      ROUGE-2: P={rouge['ROUGE-2']['precision']:.3f} "
                  f"R={rouge['ROUGE-2']['recall']:.3f} F1={rouge['ROUGE-2']['f1']:.3f}")
            print(f"      ROUGE-L: P={rouge['ROUGE-L']['precision']:.3f} "
                  f"R={rouge['ROUGE-L']['recall']:.3f} F1={rouge['ROUGE-L']['f1']:.3f}")

    # --- Detailed comparison on first article ---
    print(f"\n\n{'#' * 70}")
    print("DETAILED COMPARISON ON AI TECHNOLOGY ARTICLE")
    print(f"{'#' * 70}")

    article = SAMPLE_ARTICLES['ai_technology']

    # Compare all methods
    results = compare_summarizers(article, compression_ratio=0.3)

    # Visualizations
    print("\n[5] Generating visualizations...")
    plot_comparison(results, save_path='summarization_comparison.png')
    plot_sentence_scores(article, save_path='sentence_scores.png')

    # Compression ratio analysis
    print("\n[6] Analyzing compression ratios...")
    for method in ['tfidf', 'textrank', 'frequency']:
        analyze_compression_ratios(
            article, method=method,
            save_path=f'compression_analysis_{method}.png'
        )

    # --- Different compression ratios demo ---
    print(f"\n\n{'#' * 70}")
    print("COMPRESSION RATIO DEMO")
    print(f"{'#' * 70}")

    for ratio in [0.2, 0.3, 0.5]:
        result = tfidf_summarize(article, compression_ratio=ratio)
        print(f"\nCompression ratio: {ratio:.0%}")
        print(f"  Sentences: {result['n_summary']}/{result['n_original']}")
        print(f"  Length: {result.get('summary_length', len(result['summary']))} / "
              f"{result.get('original_length', len(article))} chars")
        print(f"  Summary: {result['summary'][:150]}...")

    # --- Random article test ---
    print(f"\n\n{'#' * 70}")
    print("RANDOM ARTICLE TEST")
    print(f"{'#' * 70}")

    random_article = generate_random_article(n_sentences=15)
    print(f"\nGenerated article ({len(random_article)} chars):")
    print(f"  {random_article[:200]}...")

    random_results = compare_summarizers(random_article, compression_ratio=0.3)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Processed {len(SAMPLE_ARTICLES)} sample articles + 1 random article")
    print(f"Methods compared: TF-IDF, TextRank, Frequency-based")
    print(f"Evaluation metrics: ROUGE-1, ROUGE-2, ROUGE-L")
    print(f"\nGenerated files:")
    print(f"  - summarization_comparison.png")
    print(f"  - sentence_scores.png")
    print(f"  - compression_analysis_tfidf.png")
    print(f"  - compression_analysis_textrank.png")
    print(f"  - compression_analysis_frequency.png")
    print(f"{'=' * 70}")
    print("Text summarization project complete.")


if __name__ == '__main__':
    main()
