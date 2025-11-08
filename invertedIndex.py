"""Inverted index implementation for efficient document retrieval.

This module provides an inverted index data structure that:
- Maps each term to a sorted list of documents containing it
- Uses internal document IDs (integers) for efficient processing
- Maintains bidirectional mapping between internal and original IDs
- Supports document frequency lookups and vocabulary analysis
- Maintains sorted postings lists for efficient merge-based operations
"""

import os
import re
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


class InvertedIndex:
    """An inverted index for efficient document retrieval from AP collection.

    Attributes:
        index: Dictionary mapping terms to sorted lists of internal document IDs
        doc_id_map: Mapping from internal IDs to original document IDs
        reverse_doc_id_map: Mapping from original IDs to internal IDs
        next_internal_id: Counter for assigning sequential internal IDs
    """

    def __init__(self) -> None:
        """Initialize an empty inverted index."""
        self.index: Dict[str, List[int]] = defaultdict(list)
        self.doc_id_map: Dict[int, str] = {}
        self.reverse_doc_id_map: Dict[str, int] = {}
        self.next_internal_id: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words by splitting on whitespace.

        Args:
            text: Text from document (already preprocessed by AP collection).

        Returns:
            List of tokens.
        """
        return text.split()

    def _get_internal_id(self, original_doc_id: str) -> int:
        """Get or create an internal ID for a document.

        Args:
            original_doc_id: Original document ID from AP collection.

        Returns:
            Internal document ID.
        """
        if original_doc_id not in self.reverse_doc_id_map:
            internal_id = self.next_internal_id
            self.next_internal_id += 1
            self.doc_id_map[internal_id] = original_doc_id
            self.reverse_doc_id_map[original_doc_id] = internal_id
            return internal_id
        return self.reverse_doc_id_map[original_doc_id]

    def build_index_from_zip(self, zip_file_path: str) -> None:
        """Build inverted index from a single zip file.

        Args:
            zip_file_path: Path to the zip file containing AP documents.
        """
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                for file_info in zip_ref.filelist:
                    if not file_info.filename.endswith(".zip"):
                        with zip_ref.open(file_info) as f:
                            xml_content = f.read().decode("utf-8", errors="ignore")
                            doc_count = self._process_xml_content(xml_content)
                            print(f"Extracted {doc_count} documents from file {file_info.filename}")
        except Exception as e:
            print(f"Error processing zip file {zip_file_path}: {e}")

    def build_index_from_directory(self, data_dir_path: str) -> None:
        """Build inverted index from all zip files in a directory.

        Args:
            data_dir_path: Path to directory containing AP_Coll_Parsed_*.zip files.
        """
        if not os.path.exists(data_dir_path):
            print(f"Data directory not found: {data_dir_path}")
            return

        zip_files = sorted(
            [
                f
                for f in os.listdir(data_dir_path)
                if f.startswith("AP_Coll_Parsed_") and f.endswith(".zip")
            ]
        )

        print(f"Found {len(zip_files)} zip files")

        for i, zip_file in enumerate(zip_files, 1):
            zip_path = os.path.join(data_dir_path, zip_file)
            print(f"Processing {i}/{len(zip_files)}: {zip_file}")
            self.build_index_from_zip(zip_path)

    def _process_xml_content(self, xml_content: str) -> int:
        """Process XML content and extract documents.

        Args:
            xml_content: Raw XML string containing documents.
        Returns:
            Number of documents.
        """
        doc_pattern = r"<DOC>(.*?)</DOC>"
        matches = re.finditer(doc_pattern, xml_content, re.DOTALL)

        counter = 0
        for match in matches:
            doc_str = match.group(1)

            docno_match = re.search(r"<DOCNO>\s*(.*?)\s*</DOCNO>", doc_str)
            if not docno_match:
                continue
            original_doc_id = docno_match.group(1).strip()

            text_matches = re.findall(r"<TEXT>(.*?)</TEXT>", doc_str, re.DOTALL)
            if not text_matches:
                self.add_document(original_doc_id, "")
                continue

            text = " ".join(text_matches)
            self.add_document(original_doc_id, text)
            counter += 1
        return counter

    def add_document(self, original_doc_id: str, text: str) -> None:
        """Add a document to the inverted index.

        Args:
            original_doc_id: Original document ID from AP collection.
            text: Document text to index (already preprocessed by AP collection).
        """
        internal_id = self._get_internal_id(original_doc_id)

        tokens = self._tokenize(text)

        for token in tokens:
            if token:
                postings = self.index[token]
                # Append new internal_id (always in ascending order since IDs are sequential)
                postings.append(internal_id)

    def get_postings(self, term: str) -> List[int]:
        """Get postings list for a term (sorted list of internal doc IDs).

        Args:
            term: The search term.

        Returns:
            Sorted list of internal document IDs containing the term.
        """
        if term in self.index:
            return self.index[term]
        return []

    def get_postings_with_original_ids(self, term: str) -> List[str]:
        """Get postings list with original document IDs.

        Args:
            term: The search term.

        Returns:
            List of original document IDs containing the term.
        """
        postings = self.get_postings(term)
        return [self.doc_id_map[internal_id] for internal_id in postings]

    def get_document_frequency(self, term: str) -> int:
        """Get the document frequency of a term.

        Args:
            term: The search term.

        Returns:
            Number of documents containing the term.
        """
        return len(self.index.get(term, set()))

    def get_all_terms(self) -> List[str]:
        """Get all terms in the index.

        Returns:
            List of all terms.
        """
        return list(self.index.keys())

    def get_term_statistics(self) -> Dict[str, int]:
        """Get document frequency statistics for all terms.

        Returns:
            Dictionary mapping terms to their document frequencies.
        """
        return {term: len(postings) for term, postings in self.index.items()}

    def get_vocabulary_size(self) -> int:
        """Get the number of unique terms in the index.

        Returns:
            Vocabulary size.
        """
        return len(self.index)

    def get_collection_size(self) -> int:
        """Get the total number of documents indexed.

        Returns:
            Total number of documents.
        """
        return self.next_internal_id

    def get_original_doc_id(self, internal_id: int) -> Optional[str]:
        """Convert internal document ID to original document ID.

        Args:
            internal_id: Internal document ID.

        Returns:
            Original document ID or None if not found.
        """
        return self.doc_id_map.get(internal_id)
