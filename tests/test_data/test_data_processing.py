import string

import reddit_topic_analysis.data.data_processing as sut
import os
import pytest
import tempfile
import random
import praw
from unittest.mock import MagicMock
from unittest import TestCase, mock
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab


def acceptable_chars() -> str:
    return string.ascii_letters + string.digits + '_!@#$%^&*()'


def test_load_environment_variables():
    # Create a temporary file with fake environment variables
    fake_env_vars = {'KEY1': 'VALUE1', 'KEY2': 'VALUE2'}
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for key, value in fake_env_vars.items():
            f.write(f'{key}={value}\n')
        env_file_path = f.name

    # Load the environment variables from the temporary file
    sut.load_environment_variables(env_file_path)

    # Check if the environment variables are set correctly
    assert os.environ['KEY1'] == 'VALUE1'
    assert os.environ['KEY2'] == 'VALUE2'

    # Clean up the temporary file
    os.remove(env_file_path)


@pytest.fixture
def reddit_client_id():
    # Generate a random client ID with acceptable characters
    return ''.join(random.choices(acceptable_chars(), k=20))


@pytest.fixture
def reddit_client_secret():
    # Generate a random client secret with acceptable characters
    return ''.join(random.choices(acceptable_chars(), k=32))


def test_set_reddit_credentials(reddit_client_id, reddit_client_secret):
    # Test that the function sets the environment variables correctly
    sut.set_reddit_credentials(reddit_client_id, reddit_client_secret)
    assert os.environ['REDDIT_CLIENT_ID'] == reddit_client_id
    assert os.environ['REDDIT_CLIENT_SECRET'] == reddit_client_secret


def test_set_reddit_credentials_raises_exception_if_none():
    # Test that the function raises an exception if client_id or client_secret is None
    with pytest.raises(Exception):
        sut.set_reddit_credentials(None, 'client_secret')
    with pytest.raises(Exception):
        sut.set_reddit_credentials('client_id', None)


def test_set_reddit_credentials_raises_exception_if_not_string():
    # Test that the function raises an exception if client_id or client_secret is not a string
    with pytest.raises(Exception):
        sut.set_reddit_credentials(123, 'client_secret')
    with pytest.raises(Exception):
        sut.set_reddit_credentials('client_id', 123)


def test_set_reddit_credentials_does_not_modify_other_env_vars(reddit_client_id, reddit_client_secret):
    # Test that the function does not modify any other environment variables
    os.environ['OTHER_VAR'] = 'other_value'
    sut.set_reddit_credentials(reddit_client_id, reddit_client_secret)
    assert os.environ['OTHER_VAR'] == 'other_value'


def test_connect_to_reddit_with_oauth():
    pass


def test_get_one_subreddit():
    subreddit = MagicMock(spec=praw.models.Subreddit)
    subreddit.display_name = "Intune"
    reddit = MagicMock(spec=praw.Reddit)
    reddit.subreddit = MagicMock(return_value=subreddit)
    result = sut.get_one_subreddit(reddit)
    assert result == subreddit


def test_extract_submission_info():
    pass


class TestCountWords(TestCase):
    def test_count_words_len_only(self):
        tokenizer_mock = MagicMock()
        # Configure the mock to return a mock Doc object
        doc_mock = MagicMock()
        doc_mock.__len__.return_value = 5
        tokenizer_mock.return_value = doc_mock

        # Call the function with the mock tokenizer
        result = sut.count_words(tokenizer_mock, "This is a test sentence.")

        # Check that the result is correct
        self.assertEqual(result, 5)

    def test_count_words_correct_tokens(self):
        # Define the input text and expected number of tokens
        input_text = "This is a test sentence."
        expected_num_tokens = 5

        # Create a mock tokenizer and mock doc object
        tokenizer_mock = mock.MagicMock(spec=sut.NLPTokenizer)
        vocab_mock = mock.MagicMock(spec=spacy.vocab.Vocab)
        doc_mock = Doc(vocab_mock, words=input_text.split())
        tokenizer_mock.return_value = doc_mock

        # call the function being tested
        result = sut.count_words(tokenizer_mock, input_text)

        # assert the result
        self.assertEqual(result, expected_num_tokens)


class TestCountHashtags(TestCase):

    def test_empty_string(self):
        text = ""
        self.assertEqual(sut.count_hashtags(text), 0)

    def test_no_hashtags(self):
        text = "This is a test string without any hashtags."
        self.assertEqual(sut.count_hashtags(text), 0)

    def test_correct_number_of_hashtags(self):
        text = "This is a #test string #with a few #hashtags."
        self.assertEqual(sut.count_hashtags(text), 3)
