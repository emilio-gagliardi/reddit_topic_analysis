import spacy
from spacy.lang.en import STOP_WORDS as spacy_stopwords
import textatistic
import sklearn.feature_extraction.text
import pandas as pd
import scipy.sparse
# import textacy
# enable CUDA support
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")


def get_sample_data(path) -> list:
    with open(path, "r") as file:
        text = file.read().replace("\n\n", " ")
        chapters = text.split("CHAPTER ")[1:]
        chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
        return chapters


def count_words(text):
    doc = nlp(text)
    return len(doc)


def count_hashtags(text):
    words = text.split()
    hashtags = [word for word in words if word.startswith("#")]

    return len(hashtags)


def count_mentions(text):
    words = text.split()
    mentions = [word for word in words if word.startswith("@")]

    return len(mentions)


def compute_readability(text):
    readability_scores = textatistic.Textatistic(text).scores
    return readability_scores


def get_sentences(text):
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


def remove_stopwords(lemmas, stopwords):
    filtered_lemmas = [lemma for lemma in lemmas if lemma not in stopwords]
    return filtered_lemmas


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


def main():
    # load small model vs. 'md' vs 'lg'
    chapters = get_sample_data("../data/alice.txt")
    # print(chapters[0])
    doc = nlp(chapters[2])
    sentences = list(doc.sents)
    sentence = sentences[2]
    # print(sentence)
    # get named entities
    doc_ents = list(doc.ents)
    sentence_ents = list(sentence.ents)
    # the numeric representation of the named entity in the vocabulary
    # print(sentence_ents[0].label)
    # the named entity type
    # print(sentence_ents[0].label_)
    # the text of the named entity
    # print(sentence_ents[0].text)
    nouns = []
    for token in sentence:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
    print(nouns)
    # noun chunks are groupings of nouns e.g, the white rabbit, the queen of hearts
    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        if "daisy-chain" in chunk.text:
            print(chunk.text)
    people = []
    # get all the people from the document
    for ent in doc_ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)

#     extract verbs
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": "ADV"}, {"POS": "VERB"}]
    matcher.add("AdverbVerbPattern", [pattern])
    matches = matcher(doc)
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        print(matched_span.text)


if __name__ == "__main__":
    main()
