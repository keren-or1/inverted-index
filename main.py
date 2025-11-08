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
        print("Processing AP collection (242,918 documents across 9 zip files)...")
        index.build_index_from_directory(data_dir)
        print(f"\nIndex built successfully!")
        print(f"  Documents indexed: {index.get_collection_size()}")
        print(f"  Unique terms: {index.get_vocabulary_size()}")
    else:
        print(f"Warning: Data directory not found at {data_dir}")
        print("Creating a sample index for testing...")
        index.add_document("AP900101-0001", "test document one")
        index.add_document("AP900101-0002", "another test document")

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
            if line and not line.startswith("#"):
                if "→" in line:
                    query = line.split("→")[1].strip()
                else:
                    query = line.split(")", 1)[-1].strip() if ")" in line else line

                if query:
                    queries.append(query)

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
            f.write(f"Postings: {index.get_postings_with_original_ids(term)}\n\n")

        # Part 2: Top 10 lowest document frequency
        f.write("=" * 60 + "\n")
        f.write("TOP 10 TERMS WITH LOWEST DOCUMENT FREQUENCY\n")
        f.write("=" * 60 + "\n\n")
        for term, freq in sorted_terms[-10:]:
            f.write(f"Term: '{term}'\n")
            f.write(f"Document Frequency: {freq}\n")
            f.write(f"Postings: {index.get_postings_with_original_ids(term)}\n\n")

        # Part 3: Characteristics
        f.write("=" * 60 + "\n")
        f.write("CHARACTERISTICS COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write("HIGH FREQUENCY TERMS:\n")
        f.write("- More common in the collection\n")
        f.write("- Appear in many documents\n")
        f.write("- Often stop words or general terms\n")
        f.write("- Less discriminative for information retrieval\n")
        f.write("- Examples help with broad searches\n\n")

        f.write("LOW FREQUENCY TERMS:\n")
        f.write("- Rare in the collection\n")
        f.write("- Appear in few documents\n")
        f.write("- Often specific or specialized terms\n")
        f.write("- More discriminative for information retrieval\n")
        f.write("- Useful for precise searches\n\n")

        # Part 4: Find two terms with similar frequencies
        f.write("=" * 60 + "\n")
        f.write("TERMS WITH SIMILAR DOCUMENT FREQUENCIES\n")
        f.write("=" * 60 + "\n\n")

        # Find two consecutive terms in the frequency list (not at extremes)
        if len(sorted_terms) >= 20:
            # Pick terms from middle-range frequencies
            idx = len(sorted_terms) // 2
            term1, freq1 = sorted_terms[idx]
            term2, freq2 = sorted_terms[idx + 1]

            f.write(f"Term 1: '{term1}'\n")
            f.write(f"  Document Frequency: {freq1}\n")
            f.write(f"  Postings: {index.get_postings_with_original_ids(term1)}\n\n")

            f.write(f"Term 2: '{term2}'\n")
            f.write(f"  Document Frequency: {freq2}\n")
            f.write(f"  Postings: {index.get_postings_with_original_ids(term2)}\n\n")

            # Check for common documents
            postings1 = set(index.get_postings(term1))
            postings2 = set(index.get_postings(term2))
            common = postings1 & postings2

            f.write("Analysis:\n")
            f.write(f"- Both terms appear in {len(common)} common documents\n")
            f.write(f"- Similar frequencies suggest comparable prevalence\n")
            f.write(f"- Terms co-occur in documents: {list(common)}\n\n")

            f.write("Discovery Method:\n")
            f.write("- Calculated document frequency for all terms\n")
            f.write("- Sorted terms by frequency\n")
            f.write("- Selected consecutive terms in the sorted list\n")
            f.write("- Verified co-occurrence in same documents\n")

    print(f"Statistics written to {output_file}")


def main() -> None:
    """Main execution function."""
    inverted_index_dir = get_project_root()
    assignment_dir = os.path.dirname(inverted_index_dir)

    queries_file = os.path.join(assignment_dir, "BooleanQueries.txt")
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
