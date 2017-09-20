"""Document class, read document, clean document, get terms."""
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter



class Document(object):

    def __init__(self, path):
        self.path = path
        self._name = path.split('/')[-1]
        self._term = None

    def read(self):
        """Get terms within documents."""
        try:
            with open(self.path, 'r') as f:
                self._term = f.read()
                return self
        except EnvironmentError:
            raise IOError("File not found")

    def lower(self):
        """Terms to lower case."""
        self._term = self._term.lower()
        return self

    def del_punc(self):
        """Remove punc."""
        self._term = self._term.translate(
        	None,
        	string.punctuation
        	)
        return self

    def del_space_stop(self):
        """Remove spaces, stopwords."""
        cached = stopwords.words("english")
        self._term = ' '.join([word for word in self._term.split() if word not in cached])
        return self

    @property
    def terms(self):
        """Finish process"""
        self.read().lower().del_punc().del_space_stop()
        return self._term

    @property 
    def name(self):
        """doc name"""
        return self._name
