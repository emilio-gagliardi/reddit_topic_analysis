import os
import better_profanity
import dotenv
from loguru import logger
import praw
import pandas as pd
import numpy
from typing import Optional, List, Dict, Tuple, Any
import re
import spacy
from spacy.lang.en import STOP_WORDS as spacy_stopwords
from spacy.tokens import Span
import textatistic
import sklearn.feature_extraction.text
import scipy.sparse

# import textacy
# enable CUDA support
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")


def hello():
    print("hello from data_processing.py")


def get_project_dir(cwd: str, base_dir: str) -> None:
    """
    Returns the topmost directory of the project directory structure,
    given the current working directory.
    """
    # Get the current working directory
    current_dir = cwd

    # Loop until we find the topmost directory
    while True:
        # Check if we've reached the root directory
        if current_dir == os.path.abspath(os.sep):
            raise ValueError("Could not find project directory")

        # Check if the current directory has the expected structure
        if os.path.basename(current_dir) == base_dir and \
                os.path.basename(os.path.dirname(current_dir)) == "src":
            return os.path.dirname(os.path.dirname(current_dir))

        # Move up to the parent directory
        current_dir = os.path.dirname(current_dir)


def load_environment_variables(env_file: str) -> None:
    if os.path.isfile(env_file):
        # Load the environment variables from the .env file
        dotenv.load_dotenv(dotenv_path=env_file)
        logger.info("Loading environment variables.")
    else:
        logger.warning("File not found. Could not load environment variables.")


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
    os.environ['REDDIT_CLIENT_ID'] = client_id
    os.environ['REDDIT_CLIENT_SECRET'] = client_secret


def get_reddit_credentials() -> Tuple[str, str]:
    """Get the Reddit credentials from environment variables.

    Returns:
        Tuple[str, str]: The Reddit client ID and client secret.
           """
    return os.environ['REDDIT_CLIENT_ID'], os.environ['REDDIT_CLIENT_SECRET']


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
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         redirect_uri=redirect_uri,
                         user_agent=user_agent)
    return reddit


def get_one_subreddit(reddit_conn: praw.Reddit, name: str = 'Intune') -> praw.models.Subreddit:
    subreddit = reddit_conn.subreddit(name)
    logger.success(f"Connected to subreddit: {subreddit.display_name}")

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


def count_words(text: str) -> int:
    doc = nlp(text)
    return len(doc)


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


def get_sentences(text: str) -> List[Span]:
    doc = nlp(text)
    sentences = list(doc.sents)
    return sentences


def get_all_tokens(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def get_all_lemmas(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas


def get_all_pos(text):
    doc = nlp(text)
    pos = [(token.text, token.pos_) for token in doc]
    return pos


def remove_stopwords(tokens, stopwords):
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def preprocess_for_vectorization(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = get_all_lemmas(doc)
    # Remove stopwords
    a_lemmas = remove_stopwords(lemmas, spacy_stopwords)

    return ' '.join(a_lemmas)


# Returns number of proper nouns
def count_proper_nouns(text):
    # Create doc object
    doc = nlp(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of proper nouns
    return pos.count("PROPN")


# Returns number of other nouns
def count_nouns(text):
    # Create doc object
    doc = nlp(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of other nouns
    return pos.count("NOUN")


def find_persons(text):
    # Create Doc object
    doc = nlp(text)
    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    # Return persons
    return persons


def find_nouns(text):
    # Create Doc object
    doc = nlp(text)
    # Identify the nouns
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    # Return nouns
    return nouns


def find_proper_nouns(text):
    # Create Doc object
    doc = nlp(text)
    # Identify the nouns
    nouns = [token.text for token in doc if token.pos_ == 'PROPN']
    # Return nouns
    return nouns


def find_verbs(text):
    # Create Doc object
    doc = nlp(text)
    # Identify the verbs
    verbs = [token.text for token in doc if token.pos_ == 'VERB']
    # Return verbs
    return verbs


def create_bow_matrix(text):
    """Returns a sparse matrix of word counts"""
    # Create CountVectorizer object
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    # Generate matrix of word vectors
    bow_matrix = vectorizer.fit_transform(text)
    return bow_matrix


def create_tfidf_matrix(text):
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


def map_sparse_matrix_to_labels(labels, matrix: scipy.sparse.csr_matrix):
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


def save_sparse_matrix(matrix, filename):
    if isinstance(matrix, scipy.sparse.csr_matrix):
        try:
            scipy.sparse.save_npz(filename, matrix)
        except OSError as e:
            print(f"Error: {e}")
    else:
        raise ValueError("Matrix must be of type scipy.sparse.csr_matrix")


def load_sparse_matrix(filename):
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


def get_subreddit_submissions(subreddit, submission_type, limit):
    """Get submissions from a subreddit
    :param subreddit: string of subreddit name
    :param submission_type: Type of submission "hot", "top", "new", "controversial", "gilded", "rising"
    :param limit: Number of submissions to get
    :returns: List of submissions"""
    reddit_conn = connect_to_reddit_with_oauth()
    subreddit_conn = get_one_subreddit(reddit_conn, subreddit)
    submission_subset_list = extract_submission_info(subreddit_conn, submission_type, limit)

    return submission_subset_list


def setup():
    print("Setting up project environment variables. data_processing.py setup()")
    PROJECT_NAME = "reddit_topic_analysis"
    PROJECT_DIR = get_project_dir(os.getcwd(), PROJECT_NAME)
    DATA_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "data")
    LOGS_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "logs")
    CONFIG_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, ".config")

    logger.debug(f"project dir is {PROJECT_DIR}")
    logger.debug(f"data dir is {DATA_DIR}")
    logger.debug(f"logs dir is {LOGS_DIR}")
    logger.debug(f"config dir is {CONFIG_DIR}")

    env_variables_path = os.path.join(CONFIG_DIR, 'environment.env')
    load_environment_variables(env_variables_path)
    # (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, or CRITICAL)
    LOGURU_LEVEL = "INFO"
    os.environ["LOGURU_LEVEL"] = "DEBUG"
    log_dev_file = os.path.join(LOGS_DIR, 'development.log')
    logger.add(open(log_dev_file, "a"),
               format="{time:YYYY-MM-DD at HH:mm:ss} {module}::{function} [{level}] {message}",
               level="DEBUG")


def main():
    print("data_processing.py main()")
    # json_payload = reddit_topic_analysis.main.dicts_to_json(submission_subset_list)
    # logger.debug(json_payload)
    # save to MongoDB
    # save_to_mongo_one(json_payload, "reddit_submissions")

    # get raw submissions from MongoDB and preprocess for NLP
    # results = load_from_mongo("reddit_submissions", None, 150)
    # text_cols = ["title", "body", "comments"]
    # raw_text = results[text_cols].copy()

    # preprocess the text from reddit submissions
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_url)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_url)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_non_alphanumeric)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_non_alphanumeric)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_lowercase)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_lowercase)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_tokenization)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_tokenization)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_stopwords)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_stopwords)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_profanity)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_profanity)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(remove_words_by_length)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(remove_words_by_length)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(clean_filler_words)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(clean_filler_words)
    #
    # raw_text.loc[:, "title"] = raw_text["title"].apply(convert_to_string)
    # raw_text.loc[:, "body"] = raw_text["body"].apply(convert_to_string)
    # apply same preprocessing to comments which are nested dicts
    # for i, row in raw_text.iterrows():
    # print(row[1]["comments"])
    # comments_dict = row["comments"]
    # modified_dict = {key: convert_to_string(
    #     remove_words_by_length(
    #         clean_filler_words(
    #             clean_profanity(
    #                 clean_stopwords(
    #                     clean_tokenization(
    #                         clean_lowercase(
    #                             clean_non_alphanumeric(
    #                                 clean_url(value))))))))) for key, value in comments_dict.items()}
    # raw_text.at[i, "comments"] = modified_dict
    # list_of_dicts = raw_text.to_dict(orient="records")
    # logger.debug(list_of_dicts)
    # json_payload = dicts_to_json(list_of_dicts)
    # save_to_mongo_one(json_payload, "cleaned_submissions")

    # get cleaned submissions from MongoDB
    # dataset = load_from_mongo("cleaned_submissions", None, 150)

    # split data into titles and bodies and comments
    # titles = dataset["title"]
    # bodies = dataset["body"]
    # text_cols = ["title", "body", "comments"]
    # clean_tokenized_words = [titles, bodies]
    # clean_tokenized_comments_dump = dataset["comments"]
    # for item in range(2):
    #     compute value_counts() for each item in the list
    # plt.figure(figsize=(20, 10))
    # pd.Series(' '.join([i for i in clean_tokenized_words[item]]).split()).value_counts(normalize=True).head(20).plot(kind="bar")
    # plt.title(f"Top 20 words in {text_cols[item]}")
    # plt.show()
    #
    # clean_tokenized_comments_list = []
    # comments are stored in a dictionary, so we need to iterate through the dictionary

    # for i, item in clean_tokenized_comments_dump.iteritems():
    #     comments_dict = item
    #     for key, value in comments_dict.items():
    #         tokenized_list = value.split()
    #         clean_tokenized_comments_list.extend(tokenized_list)

    # logger.debug(clean_tokenized_comments_list)
    # plt.figure(figsize=(20, 10))
    # pd.Series(clean_tokenized_comments_list).value_counts().head(50).plot(kind="bar")
    # plt.title("Top 50 words in comments")
    # plt.show()

    # need to compute metrics for each submission
    # need to store metrics of each submission somewhere
    # need to compute daily trends of topics
    # need to get daily trends and return as some datastructure
    # need to detect when a word in submission is not in vocabulary
    # need to add new words to vocabulary
    # need to regenerate word vectors when new words are added to vocabulary

    return True


if __name__ == "__main__":
    main()
