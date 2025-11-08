"""
BooleanRetrieval class for processing Boolean queries using an inverted index.

This module implements Boolean query processing using Reverse Polish Notation (RPN).
It supports AND, OR, and NOT operators and efficiently processes sorted postings lists.
"""


class BooleanRetrieval:
    """
    Processes Boolean queries against an inverted index.

    The class uses Reverse Polish Notation for query evaluation and maintains
    sorted postings lists for efficient set operations.
    """

    def __init__(self, inverted_index):
        """
        Initialize Boolean retrieval engine with an inverted index.

        Args:
            inverted_index: An InvertedIndex object
        """
        self.index = inverted_index

    def process_query(self, query_string):
        """
        Process a Boolean query in Reverse Polish Notation.

        Query format: term1 term2 AND term3 OR NOT
        Operators: AND, OR, NOT
        NOT is treated as "AND NOT" - it negates the next term and ANDs with previous result.

        Args:
            query_string: Query string in RPN/mixed format

        Returns:
            List of internal document IDs matching the query (sorted)
        """
        tokens = query_string.split()
        stack = []

        i = 0
        while i < len(tokens):
            token = tokens[i].strip()

            if not token:
                i += 1
                continue

            if token == 'AND':
                # Pop two operands and perform intersection
                if len(stack) < 2:
                    raise ValueError("Invalid query: insufficient operands for AND")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._merge_and(operand1, operand2)
                stack.append(result)

            elif token == 'OR':
                # Pop two operands and perform union
                if len(stack) < 2:
                    raise ValueError("Invalid query: insufficient operands for OR")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._merge_or(operand1, operand2)
                stack.append(result)

            elif token == 'NOT':
                # NOT is treated as "AND NOT"
                # Check if there's a next term or if NOT is at the end
                if i + 1 < len(tokens) and tokens[i + 1] not in ['AND', 'OR']:
                    # NOT followed by a term: "X NOT term" -> get "term" and negate it
                    i += 1
                    next_token = tokens[i].strip()

                    postings = self.index.get_postings(next_token)
                    negated = self._merge_not(postings)

                    # If there's something on the stack, AND with the negated result
                    if len(stack) > 0:
                        operand1 = stack.pop()
                        result = self._merge_and(operand1, negated)
                        stack.append(result)
                    else:
                        stack.append(negated)
                else:
                    # NOT at end or followed by operator: negate the top of stack
                    if len(stack) < 1:
                        raise ValueError("Invalid query: NOT without operand")
                    operand = stack.pop()
                    negated = self._merge_not(operand)

                    # If there was something before this, AND them
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

    def _merge_and(self, postings1, postings2):
        """
        Compute intersection of two sorted postings lists.

        Uses merge algorithm on sorted lists for O(n + m) complexity.

        Args:
            postings1: First sorted postings list
            postings2: Second sorted postings list

        Returns:
            Sorted list of documents in both postings lists
        """
        result = []
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

    def _merge_or(self, postings1, postings2):
        """
        Compute union of two sorted postings lists.

        Uses merge algorithm on sorted lists for O(n + m) complexity.

        Args:
            postings1: First sorted postings list
            postings2: Second sorted postings list

        Returns:
            Sorted list of all documents in either postings list
        """
        result = []
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

        # Add remaining elements
        result.extend(postings1[i:])
        result.extend(postings2[j:])

        return result

    def _merge_not(self, postings):
        """
        Compute complement of a postings list (documents NOT containing the term).

        Args:
            postings: Sorted postings list

        Returns:
            Sorted list of all other documents in collection
        """
        # Get all possible document IDs
        all_docs = set(range(self.index.get_collection_size()))
        # Remove documents in postings
        result = sorted(all_docs - set(postings))
        return result

    def retrieve(self, query_string):
        """
        Retrieve documents matching a Boolean query.

        Args:
            query_string: Query string in RPN format

        Returns:
            List of original document IDs matching the query
        """
        try:
            internal_ids = self.process_query(query_string)
            # Convert internal IDs to original document IDs
            original_ids = [self.index.get_original_doc_id(internal_id)
                          for internal_id in internal_ids]
            return original_ids
        except ValueError as e:
            print(f"Query error: {e}")
            return []

    def retrieve_raw(self, query_string):
        """
        Retrieve internal document IDs matching a Boolean query.

        Args:
            query_string: Query string in RPN format

        Returns:
            List of internal document IDs matching the query
        """
        try:
            return self.process_query(query_string)
        except ValueError as e:
            print(f"Query error: {e}")
            return []
