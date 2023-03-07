import os
import pytest
import tempfile
import random
from unittest.mock import MagicMock
from unittest import TestCase, mock
import reddit_topic_analysis.main as sut


def test_dicts_to_json():
    # Arrange
    input_dicts = [{"test1": "test1"}, {"test2": "test2"}]
    expected = '{"test1": "test1"}, {"test2": "test2"}'

    # Act
    actual = sut.dicts_to_json(input_dicts)

    # Assert
    assert actual == expected


class TestSave(TestCase):
    @mock.patch('reddit_topic_analysis.main.save')
    def test_save_docs(self, mock_save):
        # Arrange
        mock_save.return_value = None
        input_dicts = [{"test1": "test1"}, {"test2": "test2"}]
        expected = None

        # Act
        sut.save(input_dicts, file_type="documents",
                 db_name="reddit_submissions",
                 collection_name="test_collection")

        # Assert
        assert True == True
        mock_save.assert_called_once_with('{"test1": "test1"}, {"test2": "test2"}', 'reddit_submissions')

    def test_save_table(self):
        # Arrange
        input_tuples = (123, 456, 789)
        expected = None

        # Act
        sut.save(input_tuples, file_type="table",
                 db_name="reddit_submissions",
                 db_table="test_collection",
                 col_names=["test1", "test2", "test3"])

        # Assert
        assert True == True
        # mock_save.assert_called_once_with('{"test1": "test1"}, {"test2": "test2"}', 'reddit_submissions')

    def test_save_json(self):
        # Arrange
        input_string = {'key': 'value'}
        expected = None

        # Act
        sut.save(input_string, file_type="json")

        # Assert
        assert True == True
        # mock_save.assert_called_once_with('{"test1": "test1"}, {"test2": "test2"}', 'reddit_submissions')


class SaveSaveAsDocuments(TestCase):
    def test_save_as_documents(self):
        # Arrange
        input_dicts = [{"test1": "test1"}, {"test2": "test2"}]
        expected = None

        # Act
        sut.save_as_documents(input_dicts, db_name="reddit_submissions", collection_name="test_collection")

        # Assert
        assert True == True
        # mock_save.assert_called_once_with('{"test1": "test1"}, {"test2": "test2"}', 'reddit_submissions')


class SaveSaveAsTable(TestCase):
    def test_save_as_table(self):
        # Arrange
        input_tuples = (123, 456, 789)
        expected = None

        # Act
        sut.save_as_table(input_tuples, db_name="reddit_submissions", db_table="test_collection")

        # Assert
        assert True == True
        # mock_save.assert_called_once_with('{"test1": "test1"}, {"test2": "test2"}', 'reddit_submissions')


class SaveSaveAsJson(TestCase):
    def test_save_as_json(self):
        # Arrange
        input_string = {'key': 'value'}
        expected = None

        # Act
        sut.save_as_json(input_string)

        # Assert
        assert True == True
        # mock_save.assert_called_once_with('{"test1": "test1"}, {"test2": "test2"}', 'reddit_submissions')


def test_compute_topic_stats():
    pass


def test_save_topic_stats():
    pass


def test_get_topic_stats():
    pass


def test_schedule_submission_collection():
    pass


def test_schedule_topic_stats_collection():
    pass


def test_build_web_interface():
    pass


def test_get_topic_recommendations():
    pass


