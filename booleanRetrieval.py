"""Boolean query processor for inverted index.

This module implements Boolean query processing using Reverse Polish Notation (RPN).
It supports AND, OR, and NOT operators with efficient merge-based set operations.
"""

from typing import List

from invertedIndex import InvertedIndex


class BooleanRetrieval:
    """Processes Boolean queries against an inverted index.

    Uses Reverse Polish Notation for query evaluation and maintains
    sorted postings lists for efficient set operations.

    Attributes:
        index: The inverted index to query against.
    """

    def __init__(self, inverted_index: InvertedIndex) -> None:
        """Initialize Boolean retrieval engine with an inverted index.

        Args:
            inverted_index: An InvertedIndex object.
        """
        self.index = inverted_index

    def process_query(self, query_string: str) -> List[int]:
        """Process a Boolean query in Reverse Polish Notation.

        Query format: term1 term2 AND term3 OR NOT
        Operators: AND, OR, NOT
        NOT is treated as "AND NOT" - it negates the following term.

        Args:
            query_string: Query string in RPN/mixed format.

        Returns:
            Sorted list of internal document IDs matching the query.

        Raises:
            ValueError: If query is malformed or invalid.
        """
        tokens = query_string.split()
        stack: List[List[int]] = []

        i = 0
        while i < len(tokens):
            token = tokens[i].strip()

            if not token:
                i += 1
                continue

            if token == "AND":
                if len(stack) < 2:
                    raise ValueError("Invalid query: insufficient operands for AND")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._merge_and(operand1, operand2)
                stack.append(result)

            elif token == "OR":
                if len(stack) < 2:
                    raise ValueError("Invalid query: insufficient operands for OR")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._merge_or(operand1, operand2)
                stack.append(result)

            elif token == "NOT":
                if i + 1 < len(tokens) and tokens[i + 1] not in ["AND", "OR"]:
                    i += 1
                    next_token = tokens[i].strip()

                    postings = self.index.get_postings(next_token)
                    negated = self._merge_not(postings)

                    if len(stack) > 0:
                        operand1 = stack.pop()
                        result = self._merge_and(operand1, negated)
                        stack.append(result)
                    else:
                        stack.append(negated)
                else:
                    if len(stack) < 1:
                        raise ValueError("Invalid query: NOT without operand")
                    operand = stack.pop()
                    negated = self._merge_not(operand)

                    if len(stack) > 0:
                        operand1 = stack.pop()
                        result = self._merge_and(operand1, negated)
                        stack.append(result)
                    else:
                        stack.append(negated)

            else:
                # It's a term, get its postings list
                postings = self.index.get_postings(token)
                stack.append(postings)

            i += 1

        if len(stack) != 1:
            raise ValueError(f"Invalid query: malformed expression (stack size: {len(stack)})")

        return stack[0]

    def _merge_and(self, postings1: List[int], postings2: List[int]) -> List[int]:
        """Compute intersection of two sorted postings lists.

        Uses merge algorithm on sorted lists for O(n + m) complexity.

        Args:
            postings1: First sorted postings list.
            postings2: Second sorted postings list.

        Returns:
            Sorted list of documents in both postings lists.
        """
        result: List[int] = []
        i = j = 0

        while i < len(postings1) and j < len(postings2):
            if postings1[i] == postings2[j]:
                result.append(postings1[i])
                i += 1
                j += 1
            elif postings1[i] < postings2[j]:
                i += 1
            else:
                j += 1

        return result

    def _merge_or(self, postings1: List[int], postings2: List[int]) -> List[int]:
        """Compute union of two sorted postings lists.

        Uses merge algorithm on sorted lists for O(n + m) complexity.

        Args:
            postings1: First sorted postings list.
            postings2: Second sorted postings list.

        Returns:
            Sorted list of all documents in either postings list.
        """
        result: List[int] = []
        i = j = 0

        while i < len(postings1) and j < len(postings2):
            if postings1[i] == postings2[j]:
                result.append(postings1[i])
                i += 1
                j += 1
            elif postings1[i] < postings2[j]:
                result.append(postings1[i])
                i += 1
            else:
                result.append(postings2[j])
                j += 1

        result.extend(postings1[i:])
        result.extend(postings2[j:])

        return result

    def _merge_not(self, postings: List[int]) -> List[int]:
        """Compute complement of a postings list.

        Returns documents NOT containing the term.

        Args:
            postings: Sorted postings list.

        Returns:
            Sorted list of all other documents in collection.
        """
        all_docs = set(range(self.index.get_collection_size()))
        result = sorted(all_docs - set(postings))
        return result

    def retrieve(self, query_string: str) -> List[str]:
        """Retrieve documents matching a Boolean query.

        Args:
            query_string: Query string in RPN format.

        Returns:
            List of original document IDs matching the query.
        """
        try:
            internal_ids = self.process_query(query_string)
            original_ids = [
                self.index.get_original_doc_id(internal_id) for internal_id in internal_ids
            ]
            return [doc_id for doc_id in original_ids if doc_id is not None]
        except ValueError as e:
            print(f"Query error: {e}")
            return []

    def retrieve_raw(self, query_string: str) -> List[int]:
        """Retrieve internal document IDs matching a Boolean query.

        Args:
            query_string: Query string in RPN format.

        Returns:
            List of internal document IDs matching the query.
        """
        try:
            return self.process_query(query_string)
        except ValueError as e:
            print(f"Query error: {e}")
            return []
