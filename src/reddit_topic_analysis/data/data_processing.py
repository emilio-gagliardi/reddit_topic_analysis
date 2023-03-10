import os
import string

import better_profanity
import dotenv
from loguru import logger
import praw
import prawcore
import pandas as pd
import numpy
from typing import List, Dict, Tuple, Any, TypeVar
import re
import spacy
from spacy.lang.en.lemmatizer import Lemmatizer
from spacy.tokenizer import Tokenizer
from spacy.tokens.span import Span
import textatistic
import sklearn.feature_extraction.text
import scipy.sparse
from reddit_topic_analysis.main import dicts_to_json
# enable CUDA support
spacy.prefer_gpu()
# import textacy


PROJECT_NAME = "reddit_topic_analysis"

# NLPTokenizer = TypeVar('NLPTokenizer', bound=spacy.tokenizer.Tokenizer)
# NLPLemmatizer = TypeVar('NLPLemmatizer', bound=spacy.lang.en.lemmatizer.Lemmatizer)
# NLPSentences = TypeVar('NLPSentences', bound=spacy.tokens.span.Span)


def get_project_dir(cwd: str, base_dir: str) -> str:
    """
    Returns the topmost directory of the project directory structure,
    given the current working directory.
    """
    # Get the current working directory
    current_dir = cwd
    current_file = os.path.basename(__file__)
    # Loop until we find the topmost directory
    while True:
        # Check if we've reached the root directory
        print(current_dir)
        if current_dir == os.path.abspath(os.sep):
            raise ValueError("Could not find project directory")

        # Check if the current directory has the expected structure
        if (os.path.basename(current_dir) == base_dir and
                os.path.basename(os.path.dirname(current_dir)) in ("src", "tests")):
            return os.path.dirname(os.path.dirname(current_dir))

        # Move up to the parent directory
        current_dir = os.path.dirname(current_dir)


def load_environment_variables(env_file_path: str) -> None:
    if os.path.isfile(env_file_path):
        # Load the environment variables from the .env file
        dotenv.load_dotenv(dotenv_path=env_file_path)
        logger.info("Loading environment variables.")
    else:
        logger.warning("File not found. Could not load environment variables.")


def model_setup(language_model_path: str):
    rta_nlp = spacy.load("en_core_web_sm")
    print(language_model_path)
    save_spacy_language_model(rta_nlp, language_model_path)


def acceptable_chars() -> str:

    return string.ascii_letters + string.digits + '_!@#$%^&*()'


def save_spacy_language_model(model: spacy.language.Language, model_name: str) -> None:
    """write the spacy language model to disk.
    Args:
        model: The spacy language model.
        model_name: The full path of where to save the model.
        """
    try:
        model.to_disk(model_name)
        logger.info(f"Saved spacy language model to {model_name}")
    except OSError as e:
        logger.error(f"Could not save spacy language model to {model_name}")
        logger.error(e)


def load_spacy_language_model(model_name: str) -> spacy.language.Language:
    """Load the spacy language model from disk.
    Args:
        model_name: The full path of where the model was saved to disk.
    Returns:
        spacy.language.Language: The spacy language model.
        """
    try:
        model = spacy.load(model_name)
        logger.info(f"Loaded spacy language model from {model_name}")
        return model
    except OSError as e:
        logger.error(f"Could not load spacy language model from {model_name}")
        logger.error(e)
        return None


def set_reddit_credentials(client_id: str, client_secret: str) -> None:
    """Set the Reddit credentials environment variables."""
    if not isinstance(client_id, str) or not isinstance(client_secret, str):
        raise TypeError("Client ID and secret must be strings.")
    if not client_id.strip() or not client_secret.strip():
        raise ValueError("Client ID and secret cannot be empty.")
    if not all(c in acceptable_chars() for c in client_id) or not all(c in acceptable_chars() for c in client_secret):
        raise ValueError("Invalid client_id or client_secret")
    os.environ['REDDIT_CLIENT_ID'] = client_id
    os.environ['REDDIT_CLIENT_SECRET'] = client_secret


def get_reddit_credentials() -> Tuple[str, str]:
    """Get the Reddit credentials from environment variables.

    Returns:
        Tuple[str, str]: The Reddit client ID and client secret.
           """
    reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
    reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    if reddit_client_id is None or reddit_client_secret is None:
        raise KeyError("No environmental variables set for Reddit credentials. Load or set them first.")
    return reddit_client_id, reddit_client_secret


def connect_to_reddit_with_oauth(client_id: str, client_secret: str, redirect_uri: str = 'http://localhost:8000',
                                 user_agent: str = 'bhg_topic_analysis_trends') -> praw.Reddit:
    """Connect to Reddit API using OAuth2.
    Args:
        client_id (str): The Reddit client ID.
        client_secret (str): The Reddit client secret.
        redirect_uri (str): The redirect URI. defaults to http://localhost:8000
        user_agent (str): The user agent. 'bhg_topic_analysis_trends'
    Returns:
        praw.Reddit: The Reddit connection.
    """
    try:
        reddit = praw.Reddit(client_id=client_id,
                             client_secret=client_secret,
                             redirect_uri=redirect_uri,
                             user_agent=user_agent)
    except prawcore.exceptions.RequestException:
        raise Exception("Could not connect to Reddit API. Please check your network connection and API credentials.")

    return reddit


def get_one_subreddit(reddit_conn: praw.Reddit, name: str = 'Intune') -> praw.models.Subreddit:
    if not isinstance(reddit_conn, praw.Reddit):
        logger.debug(f"failed to connect to subreddit: {name}")
        raise TypeError('reddit_conn must be an instance of praw.Reddit')

    subreddit = reddit_conn.subreddit(name)

    return subreddit


def extract_submission_info(subreddit_conn: praw.models.Subreddit,
                            submission_type: str,
                            limit: int = 10) -> List[Dict[str, Any]]:
    """Extracts useful submission information from a subreddit.
    Args:
        subreddit_conn (praw.models.Subreddit): The subreddit connection.
        submission_type (str): The type of submission to extract.
        limit (int): The number of submissions to extract. Defaults to 10.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the submission information.
    """
    if not isinstance(subreddit_conn, praw.models.Subreddit):
        logger.debug(f"Invalid subreddit object")
        raise TypeError('subreddit_conn must be an instance of praw.models.Subreddit')

    allowed_types = ["hot", "top", "new", "controversial", "gilded", "rising"]

    if submission_type not in allowed_types:
        raise ValueError(f"Submission type must be one of: {allowed_types}")

    submission_subset_list = []

    for submission in getattr(subreddit_conn, submission_type)(limit=limit):

        if isinstance(submission, praw.models.Submission):
            submission_subset = {"id": submission.id,
                                 "type": submission_type,
                                 "title": submission.title,
                                 "body": submission.selftext,
                                 "created": submission.created,
                                 "ups": submission.ups,
                                 "downs": submission.downs,
                                 "view_count": submission.view_count,
                                 "likes": submission.likes,
                                 "url": submission.url,
                                 "score": submission.score}
            # replace_more is an attribute that exposes all comments in a submission.
            # otherwise only the top level comments are exposed.
            submission.comments.replace_more(limit=None)
            submission_subset["comments"] = {}

            for i, comment in enumerate(submission.comments.list()):
                try:
                    encoded_text = comment.body.encode("utf-8")
                    submission_subset["comments"][i] = encoded_text
                except UnicodeEncodeError:
                    logger.warning("Could not encode comment body.")
                    cleaned_text = ''
                    for char in comment.body:
                        if ord(char) < 128:  # Check if character is ASCII
                            cleaned_text += char
                    submission_subset["comments"][i] = cleaned_text.encode('utf-8')
            submission_subset_list.append(submission_subset)
        else:
            logger.error(f"Submission {submission.id} is not of type praw.models.Submission")

    return submission_subset_list


def count_words(model, text: str) -> int:
    """Count the number of words in a text after the text has been cleaned for punctuation.
    Args:
        model: The tokenizer to use.
        text (str): The text to count the words in.
    Returns:
        int: The number of tokens in the text excluding punctuation and stopwords.
    """
    doc = model(text)
    word_tokens = [token.text for token in doc if token.is_alpha]
    return len(word_tokens)


def count_hashtags(text: str) -> int:
    """Count the number of twitter style hashtags in a text.
    for example, #hashtag
    Args:
        text (str): The text to count the hashtags in.
    Returns:
        int: The number of hashtags in the text.
    """
    words = text.split()
    hashtags = [word for word in words if word.startswith("#")]
    return len(hashtags)


def count_mentions(text: str) -> int:
    """Count the number of twitter style mentions in a text.
    for example, @mention
    Args:
        text (str): The text to count the hashtags in.
    Returns:
        int: The number of mentions in the text.
    """
    words = text.split()
    mentions = [word for word in words if word.startswith("@")]

    return len(mentions)


def compute_readability(text: str) -> Dict[str, float]:
    """Compute the readability scores for a text.
    Textatistic is a Python library that computes readability scores for a text.
    1. Flesch reading ease, 2. Gunning fog index
    Args:
        text (str): The text to compute the readability scores for.
    Returns:
        Dict[str, float]: A dictionary containing the readability scores.
    """
    readability_scores = textatistic.Textatistic(text).scores
    return readability_scores


def get_sentences(model, text: str) -> List[Any]:
    """
    get the sentences from a text. Using SpaCy's default sentence tokenizer.
    SpaCy returns a generator of spacy.Spans, so we convert it to a list.
    Args:
        sentence_tokenizer: a model that takes text and returns a set of sentences.
        text: the text to get the sentences from.
    sentence_tokenizer: a model that takes text and returns a set of sentences.
    Returns:
        List[Any]: a list of sentences.
    """
    doc = model(text)
    sentences = list(doc.sents)
    return sentences


def get_all_tokens(model, text) -> List[str]:
    """
    get all tokens from a text including punctuation and stopwords.
    Args:
        tokenizer: a class that takes a text and returns a set of tokens.
        text:

    Returns:

    """
    doc = model(text)
    tokens = [token.text for token in doc]
    return tokens


def get_lemmas(model,  text: str, keep_punctuation=False, keep_stop_words=False, stop_words=None) -> List[str]:
    """Get lemmas for a text exclude punctuation and stopwords by default.
    Currently, using SpaCy's default lemmatizer pipe .:. function syntax
    follows SpaCy's implementation.
    Args:
        stop_words:
        keep_stop_words:
        keep_punctuation:
        lemmatizer: The SpaCy lemmatizer currently.
        text (str): The text to get the lemmas from.
    Returns:
        List[str]: A list of lemmas.
    """
    if stop_words is None:
        keep_stop_words = True
    try:
        doc = model(text)
    except ValueError:
        logger.error(f"ValueError: {text}")
        return []
    if keep_punctuation and keep_stop_words:
        print("both True")
        lemmas = [token.lemma_ for token in doc]
    elif keep_punctuation:
        print("keep punctuation True")
        lemmas = [token.lemma_ for token in doc if token.lemma_ not in stop_words]
    elif keep_stop_words:
        print("keep stop True")
        lemmas = [token.lemma_ for token in doc if token.is_alpha]
    else:
        print("both False")
        lemmas = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]

    return lemmas


def get_all_pos(model, text: str) -> List[Tuple[str, str]]:
    """Get all parts of speech for a text.
    Currently, using SpaCy's default parsing pipe .:. function syntax
    follows SpaCy's usage.
    Args:
        parser: The SpaCy default parser in spacy.Language
        text (str): The text to get the lemmas from.
    Returns:
        List[Tuple[str, str]]: A list of tuples (word, pos).
    """
    doc = model(text)
    pos = [(token.text, token.pos_) for token in doc]
    return pos


def remove_stopwords(tokens: List, stopwords: List) -> List:
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def preprocess_for_vectorization(model, text: str, **kwargs) -> str:
    # Create Doc object
    doc = model(text)
    stopwords = kwargs.get("stopwords", None)
    # Generate lemmas
    lemmas = get_lemmas(doc)
    if stopwords:
        lemmas = remove_stopwords(lemmas, stopwords)

    return ' '.join(lemmas)


# Returns number of proper nouns
def count_proper_nouns(model, text: str) -> int:
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of proper nouns
    return pos.count("PROPN")


# Returns number of other nouns
def count_nouns(model, text: str) -> int:
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of other nouns
    return pos.count("NOUN")


def find_persons(model, text: str) -> List[str]:
    # Create Doc object
    doc = model(text)
    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    # Return persons
    return persons


def find_nouns(model, text: str) -> List[str]:
    # Create Doc object
    doc = model(text)
    # Identify the nouns
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    # Return nouns
    return nouns


def find_proper_nouns(model, text: str) -> List[str]:
    # Create Doc object
    doc = model(text)
    # Identify the nouns
    nouns = [token.text for token in doc if token.pos_ == 'PROPN']
    # Return nouns
    return nouns


def find_verbs(model, text: str) -> List[str]:
    # Create Doc object
    doc = model(text)
    # Identify the verbs
    verbs = [token.text for token in doc if token.pos_ == 'VERB']
    # Return verbs
    return verbs


def create_bow_matrix(text: str) -> scipy.sparse.csr_matrix:
    """Returns a sparse matrix of word counts"""
    # Create CountVectorizer object
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    # Generate matrix of word vectors
    bow_matrix = vectorizer.fit_transform(text)
    return bow_matrix


def create_tfidf_matrix(text: str) -> scipy.sparse.csr_matrix:
    """Returns a sparse matrix of TF-IDF scores"""
    # Create TfidfVectorizer object
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    # Generate matrix of word vectors
    tfidf_matrix = vectorizer.fit_transform(text)
    return tfidf_matrix


def create_similarity_matrix(text, type="linear"):
    """Returns a dense matrix of pairwise similarity scores"""
    similarity_matrix = scipy.sparse.csr_matrix((0, 0), dtype=int)
    if type == "linear":
        similarity_matrix = sklearn.metrics.pairwise.linear_kernel(text)
    elif type == "cosine":
        similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(text)
    return similarity_matrix


def map_sparse_matrix_to_labels(labels: List, matrix: scipy.sparse.csr_matrix) -> pd.Series:
    """Each row in the sparse matrix represents a document.
    Labels are the document names or titles or custom label
    """
    mappings = pd.Series(numpy.arange(len(matrix.toarray())), index=labels)
    return mappings


def get_most_similar_documents(similarity_matrix, mappings, document, n=5):
    # Get index of document
    idx = mappings[document]
    # Get pairwise similarity scores
    pairwise_similarities = similarity_matrix[idx]
    # Sort the similarity scores
    most_similar = pairwise_similarities.argsort()[::-1]
    # Get top n most similar documents
    most_similar = most_similar[1:n + 1]
    # Get the names of the most similar documents
    most_similar = [mappings[i] for i in most_similar]
    return most_similar


def save_sparse_matrix(matrix: scipy.sparse.csr_matrix, filename: str) -> None:
    if isinstance(matrix, scipy.sparse.csr_matrix):
        try:
            scipy.sparse.save_npz(filename, matrix)
        except OSError as e:
            print(f"Error: {e}")
    else:
        raise ValueError("Matrix must be of type scipy.sparse.csr_matrix")


def load_sparse_matrix(filename: str) -> scipy.sparse.csr_matrix:
    matrix = scipy.sparse.csr_matrix((0, 0), dtype=int)
    try:
        matrix = scipy.sparse.load_npz(filename)
    except OSError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    return matrix


def clean_url(text: str) -> str:
    """Clean the URL from the submission body using regex
    :param text: Text to clean
    :returns: Cleaned text"""
    cleaned_text = re.sub(r"http\S+", "", text)

    return cleaned_text


def clean_non_alphanumeric(text: str) -> str:
    cleaned_text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
    return cleaned_text


def clean_lowercase(text: str) -> str:
    cleaned_text = text.lower()
    return cleaned_text


def clean_profanity(text: list, hide_asterisk=False) -> list:
    """Remove profanity from the text
    :param hide_asterisk: If True, replace asterisks with empty string
    :param text: List of tokens
    :returns: List of tokens"""
    better_profanity.profanity.load_censor_words()

    filtered_text = list(map(better_profanity.profanity.censor, text))
    if hide_asterisk:
        filtered_text = map(lambda x: re.sub(r"\*", "", x), filtered_text)

    return filtered_text


def remove_words_by_length(text: list, length=2) -> list:
    """Remove tokens that are less than 2 characters long
    :param length: 
    :param text: List of tokens
    :returns: List of tokens"""
    filtered_text = [w for w in text if len(w) > length]
    return filtered_text


def clean_filler_words(text: list) -> list:
    """Remove overly used words that don't add value
    :param text: List of tokens
    :returns: List of tokens"""
    empty_words = ["like", "get", "thanks", "also", "still", "though"]
    filtered_text = [w for w in text if w not in empty_words]
    return filtered_text


def convert_to_string(text: list) -> str:
    """Convert a list of tokens to a string
    :param text: List of tokens
    :returns: String"""
    text = ' '.join(text)
    return text


def get_subreddit_submissions(subreddit: str, submission_type: str, limit: int = 10) -> List[Dict]:
    """Get submissions from a subreddit
    :param subreddit: string of subreddit name
    :param submission_type: Type of submission "hot", "top", "new", "controversial", "gilded", "rising"
    :param limit: Number of submissions to get
    :returns: List of submissions"""
    reddit_conn = connect_to_reddit_with_oauth()
    subreddit_conn = get_one_subreddit(reddit_conn, subreddit)
    submission_subset_list = extract_submission_info(subreddit_conn, submission_type, limit)

    return submission_subset_list


def main():
    # print("data_processing.py main()")
    # print(os.getcwd())
    # PROJECT_DIR = get_project_dir(os.getcwd(), PROJECT_NAME)
    # DATA_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "data")
    # MODEL_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "model")
    # LOGS_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "logs")
    # CONFIG_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, ".config")
    #
    # logger.debug(f"project dir is {PROJECT_DIR}")
    # logger.debug(f"data dir is {DATA_DIR}")
    # logger.debug(f"logs dir is {LOGS_DIR}")
    # logger.debug(f"config dir is {CONFIG_DIR}")
    #
    # env_variables_path = os.path.join(CONFIG_DIR, 'environment.env')
    # load_environment_variables(env_variables_path)
    # (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, or CRITICAL)
    #
    # os.environ["LOGURU_LEVEL"] = "DEBUG"
    # log_dev_file = os.path.join(LOGS_DIR, 'development.log')
    # logger.add(open(log_dev_file, "a"),
    #            format="{time:YYYY-MM-DD at HH:mm:ss} {module}::{function} [{level}] {message}",
    #            level="DEBUG")
    #
    # model_path = os.path.join(MODEL_DIR, "rta_nlp")
    # if not os.path.isdir(model_path):
    #     logger.debug(f"Model not found at {model_path}. Creating new model.")
    #     model_setup(model_path)
    # nlp = load_spacy_language_model(model_path)
    # spacy_stopwords = nlp.Defaults.stop_words
    # creds = get_reddit_credentials()
    # reddit_conn = connect_to_reddit_with_oauth(creds[0], creds[1])
    # subreddit_conn = get_one_subreddit(reddit_conn, "Intune")
    # submissions = extract_submission_info(subreddit_conn, "hot", 3)
    # json_payload = dicts_to_json(submissions)
    # sample = "Pandas is an industry standard for analyzing data in Python. With a few keystrokes, you can load, filter, restructure, and visualize gigabytes of heterogeneous information."
    # expected_words = ['Pandas', 'is', 'an', 'industry', 'standard', 'for', 'analyzing', 'data', 'in', 'Python', '.', 'With', 'a', 'few', 'keystrokes', ',', 'you', 'can', 'load', ',', 'filter', ',', 'restructure', ',', 'and', 'visualize', 'gigabytes', 'of', 'heterogeneous', 'information', '.']
    # expected_lemmas = ['panda', 'industry', 'standard', 'analyze', 'datum', 'Python', 'keystroke', 'load', 'filter', 'restructure', 'visualize', 'gigabyte', 'heterogeneous', 'information']
    # expected_sentences = get_sentences(nlp, sample)
    # expected_pos = [('Pandas', 'NOUN'), ('is', 'AUX'), ('an', 'DET'), ('industry', 'NOUN'), ('standard', 'NOUN'), ('for', 'ADP'), ('analyzing', 'VERB'), ('data', 'NOUN'), ('in', 'ADP'), ('Python', 'PROPN'), ('.', 'PUNCT'), ('With', 'ADP'), ('a', 'DET'), ('few', 'ADJ'), ('keystrokes', 'NOUN'), (',', 'PUNCT'), ('you', 'PRON'), ('can', 'AUX'), ('load', 'VERB'), (',', 'PUNCT'), ('filter', 'NOUN'), (',', 'PUNCT'), ('restructure', 'NOUN'), (',', 'PUNCT'), ('and', 'CCONJ'), ('visualize', 'VERB'), ('gigabytes', 'NOUN'), ('of', 'ADP'), ('heterogeneous', 'ADJ'), ('information', 'NOUN'), ('.', 'PUNCT')]
    # doc_no_tokens = nlp("")
    # doc_sample_tokens = nlp(sample)


    # doc_sents = [sent for sent in doc_sample_tokens.sents]
    # all_tokens = [token.text for token in doc_sample_tokens]
    # word_tokens = [token.text for token in doc_sample_tokens if token.is_alpha]
    # filtered_tokens = [token for token in word_tokens if token not in spacy_stopwords]
    # lemma_tokens = [token.lemma_ for token in doc_sample_tokens if token.is_alpha and token.lemma_ not in spacy_stopwords]
    # func_lemmas = get_lemmas(nlp, sample, stop_words=spacy_stopwords)
    # func_pos = get_all_pos(nlp, sample)
    # print(all_tokens[0], type(all_tokens[0]))
    # print(lemma_tokens[0], type(lemma_tokens[0]))
    # print(word_tokens, str(len(word_tokens)))
    # print(filtered_tokens, str(len(filtered_tokens)))
    #
    # print(type(expected_sentences[0]))
    # print(get_all_tokens(nlp, sample), str(len(get_all_tokens(nlp, sample))))
    # print(lemma_tokens, str(len(lemma_tokens)))
    # print(func_lemmas, str(len(func_lemmas)))
    # print(func_pos, str(len(func_pos)))
    return True


if __name__ == "__main__":
    main()
