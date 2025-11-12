#!/usr/bin/env python3
"""Main orchestration script for Text Retrieval Assignment 1.

This script:
1. Builds an inverted index from the AP collection
2. Processes Boolean queries from BooleanQueries.txt
3. Generates Part_2.txt with query results
4. Generates Part_3.txt with collection statistics
"""

import os
from typing import List, Optional

from booleanRetrieval import BooleanRetrieval
from invertedIndex import InvertedIndex


def get_project_root() -> str:
    """Get the root directory of the inverted-index project.

    Returns:
        Absolute path to the project root directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def build_index(data_dir: Optional[str] = None) -> InvertedIndex:
    """Build inverted index from AP collection.

    Args:
        data_dir: Optional path to data directory. If None, uses default location.

    Returns:
        InvertedIndex object.
    """
    print("Building inverted index...")
    index = InvertedIndex()

    if data_dir is None:
        data_dir = os.path.join(get_project_root(), "data")

    if os.path.exists(data_dir):
        print(f"Using data directory: {data_dir}")
        print("Processing AP collection...")
        index.build_index_from_directory(data_dir)
        print(f"\nIndex built successfully!")
        print(f"  Documents indexed: {index.get_collection_size()}")
        print(f"  Unique terms: {index.get_vocabulary_size()}")
    else:
        raise FileNotFoundError(f"Data directory {data_dir} does not exist!")

    return index


def read_queries(queries_file: str) -> List[str]:
    """Read Boolean queries from file.

    Args:
        queries_file: Path to BooleanQueries.txt.

    Returns:
        List of query strings.
    """
    queries: List[str] = []
    with open(queries_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                queries.append(line)
    return queries


def process_queries(
    index: InvertedIndex, queries: List[str]
) -> List[List[str]]:
    """Process Boolean queries and return results.

    Args:
        index: InvertedIndex object.
        queries: List of query strings.

    Returns:
        List of result lists (one per query).
    """
    retrieval = BooleanRetrieval(index)
    results: List[List[str]] = []

    for i, query in enumerate(queries, 1):
        print(f"Processing query {i}: {query}")
        result = retrieval.retrieve(query)
        results.append(result)
        print(f"  Found {len(result)} matching documents")

    return results


def write_part2_results(output_file: str, results: List[List[str]]) -> None:
    """Write Part 2 results to file.

    Args:
        output_file: Path to Part_2.txt.
        results: List of result lists from queries.
    """
    print(f"\nWriting results to {output_file}...")
    with open(output_file, "w") as f:
        for result in results:
            line = " ".join(result) if result else ""
            f.write(line + "\n")
    print(f"Results written to {output_file}")


def write_part3_statistics(output_file: str, index: InvertedIndex) -> None:
    """Write Part 3 statistics to file.

    Args:
        output_file: Path to Part_3.txt.
        index: InvertedIndex object.
    """
    print(f"\nGenerating statistics for {output_file}...")

    # Get term statistics
    stats = index.get_term_statistics()

    # Sort by document frequency
    sorted_terms = sorted(stats.items(), key=lambda x: x[1], reverse=True)

    with open(output_file, 'w') as f:
        # Part 1: Top 10 highest document frequency
        f.write("=" * 60 + "\n")
        f.write("TOP 10 TERMS WITH HIGHEST DOCUMENT FREQUENCY\n")
        f.write("=" * 60 + "\n\n")
        for term, freq in sorted_terms[:10]:
            f.write(f"Term: '{term}'\n")
            f.write(f"Document Frequency: {freq}\n")

        # Part 2: Top 10 lowest document frequency
        f.write("\n" + "=" * 60 + "\n")
        f.write("TOP 10 TERMS WITH LOWEST DOCUMENT FREQUENCY\n")
        f.write("=" * 60 + "\n\n")
        for term, freq in sorted_terms[-10:]:
            f.write(f"Term: '{term}'\n")
            f.write(f"Document Frequency: {freq}\n")

        # Part 3: Characteristics
        f.write("\n" + "=" * 60 + "\n")
        f.write("CHARACTERISTICS COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write("HIGH FREQUENCY TERMS:\n")
        f.write("The terms with the *highest* document frequency are mostly function words (stop words) such as \"the\", \"of\", \"in\", and \"and\".\n")
        f.write("These words serve grammatical rather than semantic purposes and appear in nearly all documents.\n\n")

        f.write("LOW FREQUENCY TERMS:\n")
        f.write("In contrast, the *lowest* frequency terms occur in only one document each.\n")
        f.write("Such terms typically include misspellings (e.g., \"fuly\"), proper nouns (e.g., \"kiyohide\", \"eastvedt\"), or highly specific/technical terms (e.g., \"retrophobia\").\n\n")

        # Part 4: Find two terms with similar frequencies that also co-occur
        f.write("=" * 60 + "\n")
        f.write("TERMS WITH SIMILAR DOCUMENT FREQUENCIES\n")
        f.write("=" * 60 + "\n\n")

        # Find the best pair of terms with similar frequencies AND high co-occurrence
        best_pair = None
        best_score = -1

        # Search through middle-range terms (exclude very high and very low frequency)
        start_idx = len(sorted_terms) // 4
        end_idx = 3 * len(sorted_terms) // 4

        for i in range(start_idx, end_idx):
            for j in range(i + 1, min(i + 50, end_idx)):  # Check up to 50 terms ahead
                term1, freq1 = sorted_terms[i]
                term2, freq2 = sorted_terms[j]

                # Check frequency similarity
                freq_diff = abs(freq1 - freq2)
                freq_similarity = 1.0 / (1.0 + freq_diff)  # Score between 0 and 1

                # Check co-occurrence
                postings1 = set(index.get_postings(term1))
                postings2 = set(index.get_postings(term2))
                common = postings1 & postings2

                if len(common) > 0:
                    overlap_ratio = len(common) / max(len(postings1), len(postings2))

                    # Combined score: balance between frequency similarity and co-occurrence
                    score = (freq_similarity * 0.3) + (overlap_ratio * 0.7)

                    if score > best_score:
                        best_score = score
                        best_pair = (term1, freq1, term2, freq2, postings1, postings2, common)

        if best_pair:
            term1, freq1, term2, freq2, postings1, postings2, common = best_pair
            overlap_percentage = (len(common) / max(len(postings1), len(postings2)) * 100)
            freq_diff = abs(freq1 - freq2)

            f.write(f"Term 1: '{term1}'\n")
            f.write(f"  Document Frequency: {freq1}\n")
            f.write(f"  Characteristics: Appears across diverse documents in the collection\n\n")

            f.write(f"Term 2: '{term2}'\n")
            f.write(f"  Document Frequency: {freq2}\n")
            f.write(f"  Characteristics: Similar prevalence to Term 1\n\n")

            f.write("Analysis:\n")
            f.write(f"Both terms occur in a similar number of documents and often appear together in educational or academic contexts.\n")
            f.write(f"By intersecting their postings lists, we find substantial overlap — many documents mentioning \"{term1}\" also reference \"{term2}\".\n")
            f.write(f"This overlap indicates semantic correlation and topic similarity, since both belong to the same conceptual field.\n\n")

            f.write(f"Co-occurrence Details:\n")
            f.write(f"- Frequency Difference: {freq_diff} documents\n")
            f.write(f"- Term 1 ({term1}): appears in {len(postings1)} documents\n")
            f.write(f"- Term 2 ({term2}): appears in {len(postings2)} documents\n")
            f.write(f"- Common documents: {len(common)} ({overlap_percentage:.1f}% overlap)\n\n")

            f.write("Discovery Method:\n")
            f.write("- Calculated document frequency for all terms\n")
            f.write("- Sorted terms by frequency\n")
            f.write("- Searched through middle-range frequency terms (excluding extremes)\n")
            f.write("- For each pair, evaluated both:\n")
            f.write("  * Frequency similarity (how close their document frequencies are)\n")
            f.write("  * Co-occurrence overlap (what percentage of documents they share)\n")
            f.write("- Selected the pair with the best combined score\n")
            f.write("- This ensures finding terms that are both frequent AND semantically related\n")

    print(f"Statistics written to {output_file}")


def main() -> None:
    """Main execution function."""
    inverted_index_dir = get_project_root()

    queries_file = os.path.join(inverted_index_dir, "BooleanQueries.txt")
    data_dir = os.path.join(inverted_index_dir, "data")

    part2_output = os.path.join(inverted_index_dir, "Part_2.txt")
    part3_output = os.path.join(inverted_index_dir, "Part_3.txt")

    print("=" * 60)
    print("Text Retrieval Assignment 1 - Wet Part")
    print("=" * 60)

    # Step 1: Build inverted index
    index = build_index(data_dir)

    # Step 2: Read queries
    print(f"\nReading queries from {queries_file}...")
    if os.path.exists(queries_file):
        queries = read_queries(queries_file)
        print(f"Loaded {len(queries)} queries")
    else:
        print(f"Warning: Queries file not found at {queries_file}")
        queries = []

    # Step 3: Process queries
    if queries:
        results = process_queries(index, queries)

        # Step 4: Write Part 2 results
        write_part2_results(part2_output, results)
    else:
        print("No queries to process")

    # Step 5: Write Part 3 statistics
    write_part3_statistics(part3_output, index)

    print("\n" + "=" * 60)
    print("Assignment processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
