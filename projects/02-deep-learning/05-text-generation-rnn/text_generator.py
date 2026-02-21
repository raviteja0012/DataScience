"""
Character-Level Text Generation with LSTM
==========================================

A production-quality pipeline for generating Shakespeare-style text
using character-level language modeling with stacked LSTMs.

Architecture:
    - Character embedding layer
    - Two stacked LSTM layers with dropout
    - Dense output with softmax over character vocabulary
    - Temperature-based sampling for diversity control
    - Teacher forcing during training

Dataset:
    Synthetic Shakespeare-style text corpus generated programmatically.

Author: Data Science Portfolio
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Central configuration for the text generation pipeline."""

    # Sequence
    SEQ_LENGTH = 100
    STEP_SIZE = 3  # Sliding window step

    # Model
    EMBEDDING_DIM = 64
    LSTM_UNITS_1 = 256
    LSTM_UNITS_2 = 128
    DROPOUT_RATE = 0.2

    # Training
    BATCH_SIZE = 128
    EPOCHS = 50
    INITIAL_LR = 1e-3

    # Generation
    TEMPERATURES = [0.2, 0.5, 0.8, 1.0, 1.2]
    GENERATION_LENGTH = 500
    SEED_LENGTH = 40

    # Paths
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "text_generator_model.keras"
    HISTORY_PLOT = OUTPUT_DIR / "training_loss.png"
    SAMPLES_PATH = OUTPUT_DIR / "generated_samples.txt"
    CORPUS_PATH = OUTPUT_DIR / "training_corpus.txt"

    RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Corpus Generation
# ---------------------------------------------------------------------------
def generate_shakespeare_corpus():
    """Generate a synthetic Shakespeare-style text corpus.

    Creates a substantial body of text (~60,000 characters) combining
    sonnets, dramatic dialogue, soliloquies, and prose passages in
    Early Modern English style.

    Returns:
        String containing the full corpus text.
    """
    sonnets = [
        """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance, or nature's changing course, untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.""",

        """When I do count the clock that tells the time,
And see the brave day sunk in hideous night;
When I behold the violet past prime,
And sable curls all silver'd o'er with white;
When lofty trees I see barren of leaves
Which erst from heat did canopy the herd,
And summer's green all girded up in sheaves
Borne on the bier with white and bristly beard,
Then of thy beauty do I question make,
That thou among the wastes of time must go,
Since sweets and beauties do themselves forsake
And die as fast as they see others grow;
And nothing 'gainst Time's scythe can make defence
Save breed, to brave him when he takes thee hence.""",

        """Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove.
O no, it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wand'ring bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come;
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me prov'd,
I never writ, nor no man ever lov'd.""",

        """My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.
I have seen roses damask'd, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet well I know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground.
And yet, by heaven, I think my love as rare
As any she belied with false compare.""",

        """That time of year thou mayst in me behold
When yellow leaves, or none, or few, do hang
Upon those boughs which shake against the cold,
Bare ruin'd choirs, where late the sweet birds sang.
In me thou see'st the twilight of such day
As after sunset fadeth in the west,
Which by and by black night doth take away,
Death's second self, that seals up all in rest.
In me thou see'st the glowing of such fire
That on the ashes of his youth doth lie,
As the death-bed whereon it must expire,
Consum'd with that which it was nourish'd by.
This thou perceiv'st, which makes thy love more strong,
To love that well which thou must leave ere long.""",
    ]

    dialogues = [
        """HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.""",

        """MACBETH: Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more: it is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.""",

        """ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.
It is my lady, O, it is my love!
O, that she knew she were!""",

        """JULIET: O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
What's in a name? that which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title.""",

        """PROSPERO: Our revels now are ended. These our actors,
As I foretold you, were all spirits and
Are melted into air, into thin air:
And, like the baseless fabric of this vision,
The cloud-capp'd towers, the gorgeous palaces,
The solemn temples, the great globe itself,
Ye all which it inherit, shall dissolve
And, like this insubstantial pageant faded,
Leave not a rack behind. We are such stuff
As dreams are made on, and our little life
Is rounded with a sleep.""",

        """PORTIA: The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath: it is twice blest;
It blesseth him that gives and him that takes:
'Tis mightiest in the mightiest: it becomes
The throned monarch better than his crown;
His sceptre shows the force of temporal power,
The attribute to awe and majesty,
Wherein doth sit the dread and fear of kings;
But mercy is above this sceptred sway;
It is enthroned in the hearts of kings,
It is an attribute to God himself.""",

        """KING HENRY: Once more unto the breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a man
As modest stillness and humility:
But when the blast of war blows in our ears,
Then imitate the action of the tiger;
Stiffen the sinews, summon up the blood,
Disguise fair nature with hard-favour'd rage;
Then lend the eye a terrible aspect.""",

        """OBERON: I know a bank where the wild thyme blows,
Where oxlips and the nodding violet grows,
Quite over-canopied with luscious woodbine,
With sweet musk-roses and with eglantine:
There sleeps Titania sometime of the night,
Lull'd in these flowers with dances and delight;
And there the snake throws her enamell'd skin,
Weed wide enough to wrap a fairy in.""",

        """PUCK: If we shadows have offended,
Think but this, and all is mended,
That you have but slumber'd here
While these visions did appear.
And this weak and idle theme,
No more yielding but a dream,
Gentles, do not reprehend:
If you pardon, we will mend.""",

        """BRUTUS: There is a tide in the affairs of men,
Which, taken at the flood, leads on to fortune;
Omitted, all the voyage of their life
Is bound in shallows and in miseries.
On such a full sea are we now afloat,
And we must take the current when it serves,
Or lose our ventures.""",
    ]

    prose_passages = [
        """The world is a stage, and all the men and women merely players.
They have their exits and their entrances, and one man in his time
plays many parts. His acts being seven ages. At first, the infant,
mewling and puking in the nurse's arms. Then the whining schoolboy,
with his satchel and shining morning face, creeping like snail
unwillingly to school. And then the lover, sighing like furnace,
with a woeful ballad made to his mistress' eyebrow.""",

        """There is nothing either good or bad, but thinking makes it so.
The fault, dear Brutus, is not in our stars, but in ourselves,
that we are underlings. What a piece of work is man, how noble in
reason, how infinite in faculty, in form and moving how express and
admirable, in action how like an angel, in apprehension how like a
god: the beauty of the world, the paragon of animals.""",

        """If music be the food of love, play on, give me excess of it;
that surfeiting, the appetite may sicken, and so die. Love looks not
with the eyes, but with the mind, and therefore is winged Cupid
painted blind. The course of true love never did run smooth. Doubt
thou the stars are fire, doubt that the sun doth move, doubt truth
to be a liar, but never doubt my love.""",

        """All that glisters is not gold; often have you heard that told.
Many a man his life hath sold but my outside to behold. Gilded
tombs do worms enfold. Had you been as wise as bold, young in limbs,
in judgment old, your answer had not been inscroll'd. Fare you well;
your suit is cold. So be gone: you are a fool; I will ever be your
fool. Better a witty fool than a foolish wit.""",

        """We know what we are, but know not what we may be. There are more
things in heaven and earth, Horatio, than are dreamt of in your
philosophy. Though this be madness, yet there is method in it. Brevity
is the soul of wit. To thine own self be true, and it must follow,
as the night the day, thou canst not then be false to any man. Give
every man thy ear, but few thy voice; take each man's censure, but
reserve thy judgment.""",

        """O, wonder! How many goodly creatures are there here! How beauteous
mankind is! O brave new world, that has such people in it! The evil
that men do lives after them; the good is oft interred with their
bones. Cowards die many times before their deaths; the valiant never
taste of death but once. Of all the wonders that I yet have heard,
it seems to me most strange that men should fear, seeing that death,
a necessary end, will come when it will come.""",

        """Friends, Romans, countrymen, lend me your ears; I come to bury
Caesar, not to praise him. The evil that men do lives after them;
the good is oft interred with their bones. So let it be with Caesar.
The noble Brutus hath told you Caesar was ambitious: if it were so,
it was a grievous fault, and grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest, I come to speak in
Caesar's funeral. He was my friend, faithful and just to me.""",

        """How poor are they that have not patience! What wound did ever heal
but by degrees? Thou know'st we work by wit, and not by witchcraft;
and wit depends on dilatory time. The robbed that smiles, steals
something from the thief; he robs himself that spends a bootless
grief. Our bodies are our gardens, to the which our wills are
gardeners; so that if we will plant nettles, or sow lettuce, the
power and corrigible authority of this lies in our wills.""",
    ]

    # Combine all sections with separators
    corpus_parts = []

    # Repeat and interleave to build volume
    for i in range(4):
        for sonnet in sonnets:
            corpus_parts.append(sonnet)
        for dialogue in dialogues:
            corpus_parts.append(dialogue)
        for passage in prose_passages:
            corpus_parts.append(passage)

    corpus = "\n\n".join(corpus_parts)
    return corpus


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def prepare_sequences(text):
    """Convert text into training sequences and targets.

    Creates overlapping sequences of characters for next-character prediction.
    Each input is a sequence of SEQ_LENGTH characters; the target is the
    following character.

    Args:
        text: Full corpus string.

    Returns:
        Tuple of (sequences, targets, char_to_idx, idx_to_char, chars)
        sequences: integer array of shape (N, SEQ_LENGTH)
        targets: integer array of shape (N,)
    """
    print("=" * 60)
    print("Preparing Character Sequences")
    print("=" * 60)

    # Build character vocabulary
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"  Corpus length     : {len(text):,} characters")
    print(f"  Unique characters : {len(chars)}")
    print(f"  Sequence length   : {Config.SEQ_LENGTH}")
    print(f"  Step size         : {Config.STEP_SIZE}")

    # Create sequences with sliding window
    sequences = []
    targets = []

    for i in range(0, len(text) - Config.SEQ_LENGTH, Config.STEP_SIZE):
        seq = text[i : i + Config.SEQ_LENGTH]
        target = text[i + Config.SEQ_LENGTH]
        sequences.append([char_to_idx[ch] for ch in seq])
        targets.append(char_to_idx[target])

    sequences = np.array(sequences, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    print(f"  Training sequences: {len(sequences):,}")
    print()

    return sequences, targets, char_to_idx, idx_to_char, chars


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_text_generator_model(vocab_size):
    """Build a stacked LSTM model for character-level text generation.

    Architecture:
        Embedding(vocab_size, 64)
        -> LSTM(256, return_sequences=True) -> Dropout(0.2)
        -> LSTM(128) -> Dropout(0.2)
        -> Dense(vocab_size, softmax)

    Teacher forcing is implicit: at each training step, the model
    receives the true previous characters as input.

    Args:
        vocab_size: Number of unique characters in vocabulary.

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="TextGenerator_LSTM")

    model.add(layers.Embedding(
        input_dim=vocab_size,
        output_dim=Config.EMBEDDING_DIM,
        input_length=Config.SEQ_LENGTH,
        name="char_embedding",
    ))

    model.add(layers.LSTM(
        Config.LSTM_UNITS_1,
        return_sequences=True,
        name="lstm_1",
    ))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name="drop_1"))

    model.add(layers.LSTM(
        Config.LSTM_UNITS_2,
        return_sequences=False,
        name="lstm_2",
    ))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name="drop_2"))

    model.add(layers.Dense(vocab_size, activation="softmax", name="output"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Sampling / Generation
# ---------------------------------------------------------------------------
def sample_with_temperature(predictions, temperature=1.0):
    """Sample a character index from the model's output distribution.

    Lower temperature -> more conservative (picks likely characters).
    Higher temperature -> more creative (more uniform distribution).

    Args:
        predictions: Probability distribution over characters, shape (vocab_size,).
        temperature: Sampling temperature (0 < temperature).

    Returns:
        Sampled character index (integer).
    """
    predictions = np.asarray(predictions).astype("float64")
    # Apply temperature scaling in log space
    log_preds = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(log_preds)
    probas = exp_preds / np.sum(exp_preds)
    # Sample from the distribution
    return np.random.choice(len(probas), p=probas)


def generate_text(model, seed_text, char_to_idx, idx_to_char, length, temperature):
    """Generate text starting from a seed string.

    Args:
        model: Trained Keras model.
        seed_text: Starting text (at least SEQ_LENGTH characters).
        char_to_idx: Character to index mapping.
        idx_to_char: Index to character mapping.
        length: Number of characters to generate.
        temperature: Sampling temperature.

    Returns:
        Generated text string (seed + generated).
    """
    generated = list(seed_text)
    current_seq = [char_to_idx.get(ch, 0) for ch in seed_text[-Config.SEQ_LENGTH:]]

    for _ in range(length):
        x = np.array([current_seq])
        predictions = model.predict(x, verbose=0)[0]
        next_idx = sample_with_temperature(predictions, temperature)
        next_char = idx_to_char[next_idx]

        generated.append(next_char)
        current_seq = current_seq[1:] + [next_idx]

    return "".join(generated)


# ---------------------------------------------------------------------------
# Custom Callback for Text Sampling During Training
# ---------------------------------------------------------------------------
class TextSamplerCallback(callbacks.Callback):
    """Generate sample text at the end of selected epochs.

    This provides a qualitative view of how the model's generation
    quality improves during training.
    """

    def __init__(self, seed_text, char_to_idx, idx_to_char, sample_every=10):
        super().__init__()
        self.seed_text = seed_text
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.sample_every = sample_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.sample_every == 0:
            print(f"\n  --- Sample at epoch {epoch + 1} (temperature=0.5) ---")
            text = generate_text(
                self.model, self.seed_text,
                self.char_to_idx, self.idx_to_char,
                length=200, temperature=0.5,
            )
            # Show only the generated part
            generated_part = text[len(self.seed_text):]
            print(f"  {generated_part[:200]}")
            print("  ---\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, sequences, targets, seed_text, char_to_idx, idx_to_char):
    """Train the text generation model.

    Uses teacher forcing (standard for sequence models in Keras):
    the model receives the true previous characters as input at each
    timestep during training.

    Args:
        model: Compiled Keras model.
        sequences: Input sequences, shape (N, SEQ_LENGTH).
        targets: Target characters, shape (N,).
        seed_text: Seed text for sample generation callback.
        char_to_idx: Character to index mapping.
        idx_to_char: Index to character mapping.

    Returns:
        Keras History object.
    """
    print("=" * 60)
    print("Training Model (with Teacher Forcing)")
    print("=" * 60)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    val_size = int(len(sequences) * 0.1)
    x_train = sequences[val_size:]
    y_train = targets[val_size:]
    x_val = sequences[:val_size]
    y_val = targets[:val_size]

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        TextSamplerCallback(
            seed_text=seed_text,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            sample_every=10,
        ),
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=cb_list,
        verbose=1,
    )

    print(f"\n  Training complete. Model saved to: {Config.MODEL_PATH}\n")
    return history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_training_loss(history):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].set_title("Character Prediction Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.HISTORY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss plot saved to: {Config.HISTORY_PLOT}")


# ---------------------------------------------------------------------------
# Text Generation & Evaluation
# ---------------------------------------------------------------------------
def generate_samples_at_temperatures(model, seed_text, char_to_idx, idx_to_char):
    """Generate and display text samples at various temperatures.

    Args:
        model: Trained model.
        seed_text: Starting text string.
        char_to_idx: Character to index mapping.
        idx_to_char: Index to character mapping.
    """
    print("=" * 60)
    print("Generated Text Samples at Different Temperatures")
    print("=" * 60)

    all_samples = []

    for temp in Config.TEMPERATURES:
        print(f"\n  Temperature = {temp}")
        print("  " + "-" * 56)

        generated = generate_text(
            model, seed_text, char_to_idx, idx_to_char,
            length=Config.GENERATION_LENGTH,
            temperature=temp,
        )

        # Show only the generated portion
        new_text = generated[len(seed_text):]
        print(f"  SEED: \"{seed_text[-50:]}\"")
        print(f"  GENERATED:")

        # Format output with wrapping
        line_width = 70
        for j in range(0, len(new_text), line_width):
            print(f"    {new_text[j:j+line_width]}")

        all_samples.append((temp, new_text))

    # Save all samples to file
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.SAMPLES_PATH, "w") as f:
        f.write("Character-Level Text Generation - Generated Samples\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Seed text: \"{seed_text}\"\n")
        f.write(f"Generation length: {Config.GENERATION_LENGTH} characters\n\n")

        for temp, text in all_samples:
            f.write(f"\n{'='*60}\n")
            f.write(f"Temperature = {temp}\n")
            f.write(f"{'='*60}\n\n")
            f.write(text)
            f.write("\n")

    print(f"\n  All samples saved to: {Config.SAMPLES_PATH}\n")


def analyze_generation_diversity(model, seed_text, char_to_idx, idx_to_char):
    """Analyze how temperature affects generation diversity.

    Generates multiple samples at each temperature and measures
    character-level diversity metrics.
    """
    print("=" * 60)
    print("Temperature Diversity Analysis")
    print("=" * 60)

    results = []
    for temp in Config.TEMPERATURES:
        chars_generated = []
        for _ in range(5):
            text = generate_text(
                model, seed_text, char_to_idx, idx_to_char,
                length=200, temperature=temp,
            )
            new_text = text[len(seed_text):]
            chars_generated.extend(list(new_text))

        unique_chars = len(set(chars_generated))
        total_chars = len(chars_generated)
        diversity = unique_chars / len(char_to_idx)

        results.append({
            "temperature": temp,
            "unique_chars": unique_chars,
            "total_chars": total_chars,
            "diversity_ratio": diversity,
        })

        print(f"  Temp {temp:.1f}: {unique_chars} unique chars / "
              f"{len(char_to_idx)} vocabulary = {diversity:.2%} diversity")

    print()
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Execute the full text generation pipeline."""
    print("\n" + "=" * 60)
    print("  Character-Level Text Generation with LSTM")
    print("=" * 60 + "\n")

    tf.random.set_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate corpus
    print("=" * 60)
    print("Generating Shakespeare-style Corpus")
    print("=" * 60)
    corpus = generate_shakespeare_corpus()
    print(f"  Corpus length: {len(corpus):,} characters")

    # Save corpus
    with open(Config.CORPUS_PATH, "w") as f:
        f.write(corpus)
    print(f"  Corpus saved to: {Config.CORPUS_PATH}\n")

    # 2. Prepare sequences
    sequences, targets, char_to_idx, idx_to_char, chars = prepare_sequences(corpus)

    # 3. Build model
    vocab_size = len(chars)
    model = build_text_generator_model(vocab_size)

    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    model.summary()
    print(f"\n  Total parameters : {model.count_params():,}")
    print(f"  Vocabulary size  : {vocab_size}")
    print()

    # 4. Select seed text for generation
    seed_start = len(corpus) // 2
    seed_text = corpus[seed_start : seed_start + Config.SEQ_LENGTH]
    print(f"  Seed text: \"{seed_text[:60]}...\"\n")

    # 5. Train
    history = train_model(
        model, sequences, targets,
        seed_text, char_to_idx, idx_to_char,
    )

    # 6. Plot loss
    plot_training_loss(history)

    # 7. Generate samples at different temperatures
    generate_samples_at_temperatures(model, seed_text, char_to_idx, idx_to_char)

    # 8. Diversity analysis
    analyze_generation_diversity(model, seed_text, char_to_idx, idx_to_char)

    # 9. Save model
    model.save(Config.MODEL_PATH)
    print(f"  Final model saved to: {Config.MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
