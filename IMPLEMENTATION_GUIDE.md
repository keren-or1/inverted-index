# Text Retrieval Assignment 1 - Implementation Guide

## Overview

This project implements an **Inverted Index** data structure and **Boolean Retrieval** system for text search and document retrieval from the AP collection.

## Project Structure

```
inverted-index/
├── invertedIndex.py          # Core inverted index implementation
├── booleanRetrieval.py       # Boolean query processing engine
├── main.py                   # Main script to build index and process queries
├── Part_2.txt               # Query results (output)
├── Part_3.txt               # Collection statistics (output)
└── IMPLEMENTATION_GUIDE.md  # This file
```

## Component Descriptions

### 1. InvertedIndex Class (`invertedIndex.py`)

**Purpose**: Build and maintain an inverted index from the AP document collection.

**Key Features**:
- Maps each term to a postings list of internal document IDs
- Maintains bidirectional mapping between internal and original document IDs
- Supports efficient term frequency queries
- Handles XML parsing in trectext format
- Automatic text cleaning (punctuation removal, lowercasing)

**Main Methods**:
- `build_index_from_directory(data_dir)` - Build index from all AP zip files
- `add_document(original_doc_id, text)` - Add a document to the index
- `get_postings(term)` - Get internal IDs of documents containing term
- `get_document_frequency(term)` - Get number of documents with term
- `get_postings_with_original_ids(term)` - Get original document IDs

**Index Structure**:
```python
index = {
    'term1': {internal_id1, internal_id2, ...},
    'term2': {internal_id3, internal_id4, ...},
    ...
}
```

### 2. BooleanRetrieval Class (`booleanRetrieval.py`)

**Purpose**: Process Boolean queries against the inverted index.

**Query Format**: Reverse Polish Notation (RPN) with special handling for NOT
- Supports: AND, OR, NOT operators
- NOT is treated as "AND NOT" (negates the following term)
- Uses efficient merge-based set operations

**Key Features**:
- Linear-time merge operations on sorted postings lists
- No set data structures (as per requirements)
- Proper handling of complex Boolean expressions

**Main Methods**:
- `process_query(query_string)` - Process RPN query, return internal IDs
- `retrieve(query_string)` - Process query, return original document IDs
- `_merge_and(list1, list2)` - Efficient intersection
- `_merge_or(list1, list2)` - Efficient union
- `_merge_not(list)` - Complement operation

**Example Queries**:
```
iran israel AND              → Documents with both "iran" AND "israel"
southwest airlines OR africa NOT  → (southwest OR airlines) AND (NOT africa)
winner                       → Documents containing "winner"
death cancer OR us NOT       → (death OR cancer) AND (NOT us)
space station NOT moon AND   → space AND (NOT moon) AND station
```

### 3. Main Script (`main.py`)

**Purpose**: Orchestrate index building, query processing, and result generation.

**Workflow**:
1. Build inverted index from AP collection (or sample data if files not found)
2. Read Boolean queries from `BooleanQueries.txt`
3. Process each query using BooleanRetrieval
4. Generate output files:
   - `Part_2.txt`: Query results (document IDs per query)
   - `Part_3.txt`: Collection statistics

## Running the Implementation

### Prerequisites
```bash
# No external dependencies required for core functionality
# Optional: for PDF generation
pip install reportlab
```

### Build Index and Process Queries
```bash
cd inverted-index
python3 main.py
```

### Output Files

**Part_2.txt**: Contains one line per query with matching document IDs
```
AP880219-0002 AP880314-0254 AP880404-0200 ...
AP880503-0228 AP880221-0077 ...
...
```

**Part_3.txt**: Contains:
- Top 10 terms by highest document frequency
- Top 10 terms by lowest document frequency
- Characteristic comparison of high/low frequency terms
- Two terms with similar frequencies that co-occur

## Algorithm Complexity

### Inverted Index Construction
- **Time**: O(N) where N = total tokens in collection
- **Space**: O(V + P) where V = vocabulary size, P = total postings

### Boolean Query Processing
- **AND**: O(n + m) where n, m are postings list lengths
- **OR**: O(n + m)
- **NOT**: O(D) where D = total documents
- Complex queries: O(D × number of terms)

## Key Design Decisions

1. **Internal Document IDs**: Use sequential integers for efficiency in merge operations
2. **Sorted Postings Lists**: Enable O(n+m) merge-based set operations
3. **No Set Data Structures**: Meet assignment requirement by using merge algorithms
4. **Field-Agnostic Tokenization**: All tokens treated equally (simple but extensible)
5. **Error Handling**: Graceful degradation with sample data when AP collection unavailable

## Testing

### With Sample Data
When AP collection is not available, the script creates sample documents:
```
Document 1: "test document one"
Document 2: "another test document"
```

Results validate:
- Index creation functionality
- Query processing logic
- Output file generation

### With Full AP Collection
Place unzipped files in `../data/` directory:
```
../data/
├── AP_Coll_Parsed_1.zip
├── AP_Coll_Parsed_2.zip
└── ... (up to AP_Coll_Parsed_9.zip)
```

## Extensibility

The implementation can be extended to support:

1. **Positional Indexing**: Store positions within documents
2. **Phrase Queries**: "words in order" searches
3. **Range Queries**: Numeric field searches
4. **Relevance Ranking**: TF-IDF, BM25 scoring
5. **Field-Specific Search**: Index different document fields separately
6. **Compression**: Compress postings lists for space efficiency
7. **Persistence**: Save/load index to disk using pickle

## Debugging

### Check Index Building
```python
from invertedIndex import InvertedIndex
index = InvertedIndex()
index.add_document("doc1", "test document")
print(f"Vocabulary size: {index.get_vocabulary_size()}")
print(f"Documents: {index.get_collection_size()}")
```

### Test Query Processing
```python
from booleanRetrieval import BooleanRetrieval
retrieval = BooleanRetrieval(index)
results = retrieval.retrieve("test")
print(f"Found {len(results)} documents")
```

## Performance Notes

- Small collections (< 1M docs): All data fits in memory
- Large collections (> 100M docs): Consider:
  - Incremental indexing
  - Posting list compression
  - External sorting
  - Distributed processing

## References

The implementation follows standard information retrieval practices as covered in the course:
- Van Rijsbergen et al. "Information Retrieval" chapters on indexing
- Introduction to Information Retrieval (Manning, Raghavan, Schütze)
- Boolean retrieval fundamentals
