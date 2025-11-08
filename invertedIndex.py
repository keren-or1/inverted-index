"""
InvertedIndex class for building an inverted index from AP collection documents.

This module implements an efficient inverted index data structure that:
- Maps each term to a list of documents containing it
- Uses internal document IDs (integers) for efficient processing
- Maintains mapping between internal IDs and original document IDs
- Supports document frequency lookups
"""

import os
import re
import zipfile
from collections import defaultdict
from xml.etree import ElementTree as ET


class InvertedIndex:
    """
    An inverted index for efficient document retrieval.

    Structure:
    - Dictionary mapping terms to postings lists
    - Each posting list contains internal document IDs
    - Maintains bidirectional mapping between internal and original doc IDs
    """

    def __init__(self):
        """Initialize the inverted index."""
        self.index = defaultdict(set)  # term -> set of internal doc IDs
        self.doc_id_map = {}  # internal_id -> original_doc_id
        self.reverse_doc_id_map = {}  # original_doc_id -> internal_id
        self.next_internal_id = 0
        self.punctuation_pattern = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')

    def _clean_text(self, text):
        """
        Remove specified punctuation marks from text.
        Text is already lowercased by the AP collection.

        Args:
            text: Raw text from document

        Returns:
            Cleaned text with punctuation removed
        """
        return self.punctuation_pattern.sub('', text)

    def _tokenize(self, text):
        """
        Tokenize text into words by splitting on whitespace.

        Args:
            text: Cleaned text from document

        Returns:
            List of tokens
        """
        return text.split()

    def _get_internal_id(self, original_doc_id):
        """
        Get or create internal ID for a document.

        Args:
            original_doc_id: Original document ID from AP collection

        Returns:
            Internal document ID
        """
        if original_doc_id not in self.reverse_doc_id_map:
            internal_id = self.next_internal_id
            self.next_internal_id += 1
            self.doc_id_map[internal_id] = original_doc_id
            self.reverse_doc_id_map[original_doc_id] = internal_id
            return internal_id
        return self.reverse_doc_id_map[original_doc_id]

    def _parse_document(self, doc_element):
        """
        Parse a single document from trectext format.

        Args:
            doc_element: XML element representing a document

        Returns:
            Tuple of (doc_id, text) or (None, None) if parsing fails
        """
        try:
            # Get document ID
            docno_elem = doc_element.find('DOCNO')
            if docno_elem is None or docno_elem.text is None:
                return None, None
            original_doc_id = docno_elem.text.strip()

            # Get document text from all TEXT tags
            text_parts = []
            for text_elem in doc_element.findall('TEXT'):
                if text_elem.text:
                    text_parts.append(text_elem.text)

            if not text_parts:
                return original_doc_id, ""

            text = ' '.join(text_parts)
            return original_doc_id, text
        except Exception as e:
            print(f"Error parsing document: {e}")
            return None, None

    def build_index_from_zip(self, zip_file_path):
        """
        Build inverted index from a single zip file containing AP documents.

        Args:
            zip_file_path: Path to the zip file containing AP documents
        """
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Process each file in the zip
                for file_info in zip_ref.filelist:
                    if not file_info.filename.endswith('.zip'):
                        with zip_ref.open(file_info) as f:
                            xml_content = f.read().decode('utf-8', errors='ignore')
                            self._process_xml_content(xml_content)
        except Exception as e:
            print(f"Error processing zip file {zip_file_path}: {e}")

    def build_index_from_directory(self, data_dir_path):
        """
        Build inverted index from all zip files in a directory.

        Args:
            data_dir_path: Path to directory containing AP_Coll_Parsed_*.zip files
        """
        if not os.path.exists(data_dir_path):
            print(f"Data directory not found: {data_dir_path}")
            return

        # Find all zip files
        zip_files = sorted([f for f in os.listdir(data_dir_path)
                           if f.startswith('AP_Coll_Parsed_') and f.endswith('.zip')])

        print(f"Found {len(zip_files)} zip files")

        for i, zip_file in enumerate(zip_files, 1):
            zip_path = os.path.join(data_dir_path, zip_file)
            print(f"Processing {i}/{len(zip_files)}: {zip_file}")
            self.build_index_from_zip(zip_path)

    def _process_xml_content(self, xml_content):
        """
        Process XML content and extract documents.

        Args:
            xml_content: Raw XML string containing documents
        """
        try:
            # Parse XML with error handling for malformed documents
            root = ET.fromstring(xml_content)

            # Find all DOC elements
            for doc_elem in root.findall('.//DOC'):
                original_doc_id, text = self._parse_document(doc_elem)
                if original_doc_id and text:
                    self.add_document(original_doc_id, text)
        except ET.ParseError:
            # Try to extract documents manually if XML parsing fails
            self._extract_documents_manual(xml_content)

    def _extract_documents_manual(self, xml_content):
        """
        Manually extract documents from XML when parsing fails.

        Args:
            xml_content: Raw XML string containing documents
        """
        # Split by <DOC> tags and process each document
        doc_pattern = r'<DOC>(.*?)</DOC>'
        matches = re.finditer(doc_pattern, xml_content, re.DOTALL)

        for match in matches:
            doc_str = match.group(1)

            # Extract DOCNO
            docno_match = re.search(r'<DOCNO>\s*(.*?)\s*</DOCNO>', doc_str)
            if not docno_match:
                continue
            original_doc_id = docno_match.group(1).strip()

            # Extract TEXT content
            text_matches = re.findall(r'<TEXT>(.*?)</TEXT>', doc_str, re.DOTALL)
            if not text_matches:
                self.add_document(original_doc_id, "")
                continue

            text = ' '.join(text_matches)
            self.add_document(original_doc_id, text)

    def add_document(self, original_doc_id, text):
        """
        Add a document to the inverted index.

        Args:
            original_doc_id: Original document ID from AP collection
            text: Document text to index
        """
        # Get internal ID for this document
        internal_id = self._get_internal_id(original_doc_id)

        # Clean and tokenize text
        cleaned_text = self._clean_text(text)
        tokens = self._tokenize(cleaned_text)

        # Add each unique term to the index
        for token in tokens:
            if token:  # Skip empty tokens
                self.index[token].add(internal_id)

    def get_postings(self, term):
        """
        Get postings list for a term (sorted list of internal doc IDs).

        Args:
            term: The search term

        Returns:
            Sorted list of internal document IDs containing the term
        """
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def get_postings_with_original_ids(self, term):
        """
        Get postings list with original document IDs.

        Args:
            term: The search term

        Returns:
            List of original document IDs containing the term
        """
        postings = self.get_postings(term)
        return [self.doc_id_map[internal_id] for internal_id in postings]

    def get_document_frequency(self, term):
        """
        Get the document frequency of a term.

        Args:
            term: The search term

        Returns:
            Number of documents containing the term
        """
        return len(self.index.get(term, set()))

    def get_all_terms(self):
        """
        Get all terms in the index.

        Returns:
            List of all terms
        """
        return list(self.index.keys())

    def get_term_statistics(self):
        """
        Get document frequency statistics for all terms.

        Returns:
            Dictionary mapping terms to their document frequencies
        """
        return {term: len(postings) for term, postings in self.index.items()}

    def get_vocabulary_size(self):
        """Get the number of unique terms in the index."""
        return len(self.index)

    def get_collection_size(self):
        """Get the total number of documents indexed."""
        return self.next_internal_id

    def get_original_doc_id(self, internal_id):
        """
        Convert internal document ID to original document ID.

        Args:
            internal_id: Internal document ID

        Returns:
            Original document ID
        """
        return self.doc_id_map.get(internal_id)

    def save_index(self, file_path):
        """
        Save the index to a file (optional, for persistence).

        Args:
            file_path: Path where to save the index
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump({
                'index': dict(self.index),
                'doc_id_map': self.doc_id_map,
                'reverse_doc_id_map': self.reverse_doc_id_map,
                'next_internal_id': self.next_internal_id
            }, f)

    def load_index(self, file_path):
        """
        Load the index from a file.

        Args:
            file_path: Path to the saved index file
        """
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.index = defaultdict(set, {k: set(v) for k, v in data['index'].items()})
            self.doc_id_map = data['doc_id_map']
            self.reverse_doc_id_map = data['reverse_doc_id_map']
            self.next_internal_id = data['next_internal_id']
