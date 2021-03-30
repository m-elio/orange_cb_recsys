import shutil
import os

from whoosh.fields import Schema, TEXT, KEYWORD
from whoosh.index import create_in, open_dir
from whoosh.formats import Frequency
from whoosh.analysis import SimpleAnalyzer
from whoosh.query import Term

from orange_cb_recsys.content_analyzer.memory_interfaces.memory_interfaces import TextInterface
import math


class IndexInterface(TextInterface):
    """
    Abstract class that takes care of serializing and deserializing text in an indexed structure
    using the Whoosh library. The attribute schema_defined is used to determine if a schema is already
    defined in the index. By doing so dynamic schema creation is made possible and the schema will be based
    on the first element that will be added to the index

    Args:
        directory (str): Path of the directory where the content will be serialized
    """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.__doc = None
        self.__writer = None
        self.__doc_index = 0
        self.__schema_defined = False

    def __str__(self):
        return "IndexInterface"

    def init_writing(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            ix = create_in(self.directory, Schema())
        else:
            ix = open_dir(self.directory)
        if len(ix.schema.names()) != 0:
            self.__schema_defined = True
        self.__writer = ix.writer()

    def new_content(self):
        """
        The new content is a document that will be indexed. In this case the document is a dictionary with
        the name of the field as key and the data inside the field as value
        """
        self.__doc = {}

    def new_field(self, field_name: str, field_data):
        """
        Add a new field. If the schema is not yet defined the writer will add the field_name inside the schema

        Args:
            field_name (str): Name of the new field
            field_data: Data to put into the field
        """
        if not self.__schema_defined:
            self.__writer.add_field(field_name, KEYWORD(stored=True, vector=Frequency()))
        self.__doc[field_name] = field_data

    def new_searching_field(self, field_name, field_data):
        """
        Add a new searching field. It will be used by the search engine recommender.
        If the schema is not yet defined the writer will add the field_name inside the schema

        Args:
            field_name (str): Name of the new field
            field_data: Data to put into the field
        """
        if not self.__schema_defined:
            self.__writer.add_field(field_name, TEXT(stored=True, analyzer=SimpleAnalyzer()))
        self.__doc[field_name] = field_data

    def serialize_content(self) -> int:
        """
        Serialize the content. If the schema is not yet defined the writer will commit the additions made
        to it previously and change the boolean attribute scema_defined to true
        """
        if not self.__schema_defined:
            self.__writer.commit()
            self.__writer = open_dir(self.directory).writer()
            self.__schema_defined = True
        self.__writer.add_document(**self.__doc)
        del self.__doc
        self.__doc_index += 1
        return self.__doc_index - 1

    def stop_writing(self):
        """
        Stop the index writer and commit the operations
        """
        self.__writer.commit()

    def get_tf_idf(self, field_name: str, content_id: str):
        """
        Calculates the tf-idf for the words contained in the field of the content whose id
        is content_id.
        The tf-idf computation formula is: tf-idf = (1 + log10(tf)) * log10(idf)

        Args:
            field_name (str): Name of the field containing the words for which calculate the tf-idf
            content_id (str): Id of the content that contains the specified field

        Returns:
             words_bag (Dict <str, float>): Dictionary whose keys are the words contained in the field,
                and the corresponding values are the tf-idf values
        """
        ix = open_dir(self.directory)
        words_bag = {}
        with ix.searcher() as searcher:
            query = Term("content_id", content_id)
            doc_num = searcher.search(query).docnum(0)
            list_with_freq = [term_with_freq for term_with_freq
                              in searcher.vector(doc_num, field_name).items_as("frequency")]
            for term, freq in list_with_freq:
                tf = 1 + math.log10(freq)
                idf = math.log10(searcher.doc_count()/searcher.doc_frequency(field_name, term))
                words_bag[term] = tf*idf
        return words_bag

    def delete_index(self):
        shutil.rmtree(self.directory, ignore_errors=True)
