"""
Hybrid retrieval system that combines vector search with BM25 lexical search &
numerical query-adaptive proximity scoring.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from rank_bm25 import BM25Okapi

from log import logger


class DynamicHybridRetriever:
    """
    Implements a hybrid search approach with dynamic, query-adaptive weights
    and improved sorting for printing document attributes.
    """

    def __init__(self, vector_db, llm=None):
        """
        Initialize the hybrid retriever.

        Args:
            vector_db: Vector database manager instance
            llm: Optional LLM for query analysis
        """
        self.vector_db = vector_db
        self.llm = llm
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.initialized = False

        # Try to load spaCy if available
        try:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for improved query analysis")
            except Exception as e:  # Specify the exception type
                logger.warning(
                    f"Could not load spaCy model, will install and download: {e}"
                )
                try:
                    # Try to download the model
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Downloaded and loaded spaCy model")
                except Exception as e:
                    logger.warning(
                        f"Failed to download spaCy model, "
                        f"falling back to basic tokenization: {e}"
                    )
                    self.nlp = None
        except ImportError:
            logger.warning("spaCy not available, falling back to basic tokenization")
            self.nlp = None

        # Define attribute metadata for scoring
        self.field_metadata = {
            "Ink Coverage": {
                "aliases": [
                    "ink",
                    "coverage",
                    "color",
                    "density",
                    "heavy",
                    "medium",
                    "light",
                    "painted",
                ],
                "values": ["heavy", "medium", "light", "painted"],
                "analysis_weight": 0.0,  # Dynamically set based on query
                # "default_importance": 0.8,  # Base importance if mentioned
                "default_importance": 0.0,  # Base importance if mentioned
            },
            "Media Coating": {
                "aliases": [
                    "coating",
                    "coated",
                    "uncoated",
                    "treatment",
                    "surface treatment",
                ],
                "values": ["coated", "uncoated"],
                "analysis_weight": 0.0,
                # "default_importance": 0.7,
                "default_importance": 0.0,
            },
            "Media Finish": {
                "aliases": [
                    "finish",
                    "texture",
                    "surface",
                    "silk",
                    "matte",
                    "gloss",
                    "glossy",
                    "satin",
                ],
                "values": ["silk", "matte", "gloss", "glossy", "satin"],
                "analysis_weight": 0.0,
                # "default_importance": 0.6,
                "default_importance": 0.0,
            },
            "Media Weight GSM": {
                "aliases": [
                    "gsm",
                    "weight",
                    "grammage",
                    "thickness",
                    "g/m2",
                    "paper weight",
                ],
                "numeric": True,
                "analysis_weight": 0.0,
                # "default_importance": 0.75,  # Higher default for GSM
                "default_importance": 0.0,
            },
            "Press Model": {
                "aliases": ["press", "model", "printer", "machine", "device"],
                "analysis_weight": 0.0,
                # "default_importance": 0.5,
                "default_importance": 0.0,
            },
            "location": {
                "aliases": ["location", "place", "site", "facility", "where"],
                "analysis_weight": 0.0,
                # "default_importance": 0.4,
                "default_importance": 0.0,
            },
        }

    def initialize_bm25(self, documents: List[Dict[str, Any]]):
        """
        Initialize the BM25 index with documents.

        Args:
            documents: List of document dictionaries
        """
        logger.info("Initializing BM25 index")
        self.documents = documents
        self.doc_ids = [doc.get("event_id", str(i)) for i, doc in enumerate(documents)]

        # Create document texts for BM25
        doc_texts = []
        for doc in documents:
            # Combine all metadata fields into a single text
            text_parts = []
            for key, value in doc.items():
                if value and isinstance(value, (str, int, float)):
                    text_parts.append(f"{key}: {value}")

            doc_text = " ".join(text_parts)
            doc_texts.append(doc_text)

        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(text) for text in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.initialized = True
        logger.info(f"BM25 index initialized with {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 processing.

        Args:
            text: The text to tokenize

        Returns:
            List of tokens
        """
        if self.nlp:
            # Use spaCy for better tokenization
            doc = self.nlp(text.lower())
            # Filter out stopwords and punctuation
            return [
                token.text for token in doc if not token.is_stop and not token.is_punct
            ]
        else:
            # Fallback to basic tokenization
            text = re.sub(r"[^\w\s]", " ", text.lower())
            return [token for token in text.split() if token and len(token) > 2]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to range [0, 1].

        Args:
            scores: List of raw scores

        Returns:
            List of normalized scores
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # If all scores are the same, return equal normalized scores
        if max_score == min_score:
            return [1.0] * len(scores)

        # Otherwise normalize to [0, 1] range
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _extract_numeric_values(self, query: str) -> Dict[str, float]:
        """
        Extract numerical values from the query with improved pattern matching.

        Args:
            query: The user query

        Returns:
            Dictionary mapping field names to numeric values
        """
        numeric_values = {}

        # Extract GSM value
        gsm_patterns = [
            r"(\d+)\s*gsm",  # 220 gsm
            r"(\d+)\s*g/m2",  # 220 g/m2
            r"(\d+)\s*gram",  # 220 gram
            r"(\d+)\s*g\.?\/m2",  # Various g/m2 formats
            r"weight\s+(\d+)",  # weight 220
            r"(\d+)\s*weight",  # 220 weight
        ]

        for pattern in gsm_patterns:
            match = re.search(pattern, query.lower())
            if match:
                numeric_values["Media Weight GSM"] = float(match.group(1))
                break

        # Look for "closest to X" patterns
        closest_match = re.search(r"closest\s+to\s+(\d+)", query.lower())
        if closest_match:
            value = float(closest_match.group(1))

            # If we already found a GSM value elsewhere, prefer that one
            if "Media Weight GSM" not in numeric_values:
                # Assume it's GSM if not specified
                numeric_values["Media Weight GSM"] = value

            # Also add a flag to indicate this is a proximity search
            numeric_values["_proximity_search"] = True

        # Look for "near X" patterns
        near_match = re.search(r"near\s+(\d+)", query.lower())
        if near_match and "Media Weight GSM" not in numeric_values:
            numeric_values["Media Weight GSM"] = float(near_match.group(1))
            numeric_values["_proximity_search"] = True

        return numeric_values

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to determine important fields and relationships.

        Args:
            query: The user query

        Returns:
            Dictionary of query analysis results
        """
        # Reset field weights
        for field in self.field_metadata:
            self.field_metadata[field]["analysis_weight"] = 0.0

        # Extract numerical queries
        numerical_queries = self._extract_numeric_values(query)

        # Flag for proximity search
        proximity_search = numerical_queries.get("_proximity_search", False)
        if "_proximity_search" in numerical_queries:
            del numerical_queries["_proximity_search"]

        # GSM is mentioned in the query
        if "Media Weight GSM" in numerical_queries:
            # Mark that GSM is important in this query
            self.field_metadata["Media Weight GSM"][
                "analysis_weight"
            ] = self.field_metadata["Media Weight GSM"]["default_importance"]

            # If this is a proximity search, give extra weight to GSM
            if proximity_search:
                self.field_metadata["Media Weight GSM"]["analysis_weight"] += 0.15

        # Parse query with NLP if available
        tokens = []
        query_lower = query.lower()

        if self.nlp:
            doc = self.nlp(query_lower)
            tokens = [
                token for token in doc if not token.is_stop and not token.is_punct
            ]

            # Check field mentions in the query
            for field, metadata in self.field_metadata.items():
                # Check for any aliases of this field
                for alias in metadata.get("aliases", []):
                    if alias in query_lower:
                        # Field is mentioned - give it its default importance
                        metadata["analysis_weight"] = metadata["default_importance"]
                        break

                # For non-numeric fields, check for value mentions
                if not metadata.get("numeric", False):
                    for value in metadata.get("values", []):
                        if value in query_lower:
                            # Value is mentioned - boost importance further
                            metadata["analysis_weight"] = min(
                                1.0, metadata["analysis_weight"] + 0.2
                            )
                            break
        else:
            # Simple tokenization fallback
            # words = query_lower.split()

            # Check field mentions
            for field, metadata in self.field_metadata.items():
                # Check for any aliases of this field
                for alias in metadata.get("aliases", []):
                    if alias in query_lower:
                        # Field is mentioned - give it its default importance
                        metadata["analysis_weight"] = metadata["default_importance"]
                        break

                # For non-numeric fields, check for value mentions
                if not metadata.get("numeric", False):
                    for value in metadata.get("values", []):
                        if value in query_lower:
                            # Value is mentioned - boost importance further
                            metadata["analysis_weight"] = min(
                                1.0, metadata["analysis_weight"] + 0.2
                            )
                            break

        # Determine order of mention to prioritize fields
        field_order = []
        if self.nlp:
            # Use positional analysis to determine mention order
            field_positions = {}
            for token in tokens:
                for field, metadata in self.field_metadata.items():
                    # Check if token is related to this field
                    if token.text in metadata.get(
                        "aliases", []
                    ) or token.text in metadata.get("values", []):
                        field_positions[field] = min(
                            field_positions.get(field, token.i), token.i
                        )

            # Sort fields by position of first mention
            field_order = sorted(
                field_positions.keys(), key=lambda f: field_positions[f]
            )
        else:
            # Simple approach - just look for the first mention of each field alias or value
            field_positions = {}
            for field, metadata in self.field_metadata.items():
                field_position = float("inf")

                # Check aliases
                for alias in metadata.get("aliases", []):
                    alias_pos = query_lower.find(alias)
                    if alias_pos != -1:
                        field_position = min(field_position, alias_pos)

                # Check values
                for value in metadata.get("values", []):
                    value_pos = query_lower.find(value)
                    if value_pos != -1:
                        field_position = min(field_position, value_pos)

                if field_position != float("inf"):
                    field_positions[field] = field_position

            # Sort fields by position of first mention
            field_order = sorted(
                field_positions.keys(), key=lambda f: field_positions[f]
            )

        # Boost weights based on mention order (earlier mentions get higher weight)
        order_boost = 0.2
        for i, field in enumerate(field_order):
            # Earlier mentions get higher boost (diminishing)
            boost = order_boost * (1.0 - (i / max(1, len(field_order))))
            self.field_metadata[field]["analysis_weight"] += boost

        # Detect specific attributes in the query for direct matching
        field_values = {}

        # Ink Coverage
        if "heavy" in query_lower:
            field_values["Ink Coverage"] = "Heavy"
        elif "medium" in query_lower:
            field_values["Ink Coverage"] = "Medium"
        elif "light" in query_lower:
            field_values["Ink Coverage"] = "Light"

        # Media Coating
        if "coated" in query_lower:
            field_values["Media Coating"] = "Coated"
        elif "uncoated" in query_lower:
            field_values["Media Coating"] = "Uncoated"

        # Media Finish
        if "silk" in query_lower:
            field_values["Media Finish"] = "Silk"
        elif "matte" in query_lower:
            field_values["Media Finish"] = "Matte"
        elif "gloss" in query_lower or "glossy" in query_lower:
            field_values["Media Finish"] = "Glossy"

        # Make sure vector similarity always has at least some weight
        vector_weight = 0.15
        bm25_weight = 0.10

        # Calculate proportional weights for field components
        field_weights = {
            field: metadata["analysis_weight"]
            for field, metadata in self.field_metadata.items()
        }
        total_field_weight = sum(field_weights.values())

        # If no fields were prioritized, use default distribution
        if total_field_weight == 0:
            # Default weights if no fields are mentioned
            normalized_weights = {"vector": 0.7, "bm25": 0.3, "fields": {}}
        else:
            # Normalize field weights to distribute the non-vector portion
            available_weight = 1.0 - vector_weight - bm25_weight

            if total_field_weight > 0:
                for field in field_weights:
                    field_weights[field] = (
                        field_weights[field] / total_field_weight
                    ) * available_weight

            # Combine vector and field weights
            normalized_weights = {
                "vector": vector_weight,
                "bm25": bm25_weight,
                "fields": field_weights,
            }

        return {
            "numerical_queries": numerical_queries,
            "field_values": field_values,
            "weights": normalized_weights,
            "prioritized_fields": field_order,
            "proximity_search": proximity_search,
        }

    def _calculate_field_score(
        self, doc: Dict[str, Any], field: str, target_value: Any
    ) -> float:
        """
        Calculate score for a specific field with improved GSM proximity calculation.

        Args:
            doc: Document metadata
            field: Field name
            target_value: Target value to match

        Returns:
            Field score (0-1)
        """
        if field not in doc or doc[field] is None:
            return 0.0

        # Special handling for Media Weight GSM
        if (
            field == "Media Weight GSM"
            and self.field_metadata.get(field, {}).get("numeric", False)
            and isinstance(target_value, (int, float))
        ):
            try:
                doc_value = float(doc[field])
                diff = abs(doc_value - target_value)

                # Simpler, more direct calculation based purely on absolute difference
                # The closer the value, the higher the score
                if diff == 0:  # Exact match
                    return 1.0
                else:
                    # Max difference considered is 200, beyond that all scores are the minimum
                    max_diff = 200
                    # Simple linear scaling, higher for closer values
                    proximity = max(0.1, 1.0 - (diff / max_diff))
                    return proximity
            except (ValueError, TypeError):
                return 0.0

        # Handle other numeric fields
        elif self.field_metadata.get(field, {}).get("numeric", False) and isinstance(
            target_value, (int, float)
        ):
            try:
                doc_value = float(doc[field])
                diff = abs(doc_value - target_value)

                # Regular proximity calculation
                if diff == 0:  # Exact match
                    return 1.0
                elif diff <= 20:  # Very close
                    return 0.9 - (diff / 200)
                elif diff <= 50:  # Moderately close
                    return 0.8 - (diff / 250)
                elif diff <= 100:  # Somewhat close
                    return 0.6 - (diff / 500)
                else:  # Far away
                    return max(0.1, 1.0 - (diff / 200))
            except (ValueError, TypeError):
                return 0.0

        # Handle categorical fields
        if isinstance(target_value, str) and isinstance(doc[field], str):
            if doc[field].lower() == target_value.lower():
                return 1.0  # Exact match

            # Check for partial matches
            doc_val_lower = doc[field].lower()
            target_val_lower = target_value.lower()

            # Check if one contains the other
            if target_val_lower in doc_val_lower or doc_val_lower in target_val_lower:
                return 0.7  # Partial match

            # Check for word overlap
            doc_words = set(doc_val_lower.split())
            target_words = set(target_val_lower.split())
            overlap = len(doc_words.intersection(target_words))

            if overlap > 0:
                return 0.5 * (overlap / len(target_words))  # Word overlap

        return 0.0  # No match

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        where_clause: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
        """
        Retrieve documents using dynamic hybrid search with query-adaptive weighting.

        Args:
            query: Text query
            query_embedding: Vector embedding of the query
            where_clause: Optional filtering criteria
            n_results: Maximum number of results to return

        Returns:
            Tuple of (documents, scores, explanations)
        """
        if not self.initialized:
            logger.warning("BM25 index not initialized, using vector search only")
            # Fallback to vector search only
            results = self.vector_db.query_by_embedding(
                query_embedding, where_clause, n_results
            )
            return (
                results["metadatas"][0] if results["metadatas"] else [],
                results["distances"][0] if results["distances"] else [],
                ["Vector similarity only"]
                * len(results["metadatas"][0] if results["metadatas"] else []),
            )

        logger.debug(f"Performing dynamic hybrid search for: '{query}'")

        # Analyze query to determine field weights and important values
        query_analysis = self._analyze_query(query)
        numerical_queries = query_analysis["numerical_queries"]
        field_values = query_analysis["field_values"]
        weights = query_analysis["weights"]
        prioritized_fields = query_analysis["prioritized_fields"]
        # proximity_search = query_analysis["proximity_search"]

        # Log the analysis results
        logger.debug(
            f"Query analysis: numerical_queries={numerical_queries}, field_values={field_values}"
        )
        logger.debug(f"Dynamic weights: {weights}")
        logger.debug(f"Prioritized fields: {prioritized_fields}")

        # Get vector search results
        vector_results = self.vector_db.query_by_embedding(
            query_embedding,
            where_clause,
            max(n_results * 2, 34),  # Get more results for reranking
        )

        vector_docs = (
            vector_results["metadatas"][0] if vector_results["metadatas"] else []
        )
        vector_distances = (
            vector_results["distances"][0] if vector_results["distances"] else []
        )

        if not vector_docs:
            logger.warning("No vector search results found")
            return [], [], []

        # Convert vector distances to similarity scores (higher is better)
        max_distance = max(vector_distances) if vector_distances else 1.0
        vector_scores = [
            1.0 - (distance / max_distance) for distance in vector_distances
        ]
        normalized_vector_scores = self._normalize_scores(vector_scores)

        logger.debug("===== Documents sorted by VECTOR score =====")
        # Create a list of (doc, score) pairs
        vector_doc_scores = list(zip(vector_docs, normalized_vector_scores))
        # Sort by vector score (descending)
        vector_sorted = sorted(vector_doc_scores, key=lambda x: x[1], reverse=True)
        # Log the top 10 documents
        for i, (doc, score) in enumerate(vector_sorted[:10]):
            doc_id = doc.get("event_id", "unknown")
            ink = doc.get("Ink Coverage", "N/A")
            coating = doc.get("Media Coating", "N/A")
            gsm = doc.get("Media Weight GSM", "N/A")
            logger.debug(
                f"Vector Rank {i + 1}: Doc {doc_id} - Score: {score:.4f} - Ink: {ink}, Coating: {coating}, GSM: {gsm}"
            )

        # Get BM25 scores
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            # If query tokens are empty, use vector search only
            logger.warning("Empty tokenized query, using vector search only")
            results = [
                (doc, score, "Vector similarity only")
                for doc, score in zip(
                    vector_docs[:n_results], normalized_vector_scores[:n_results]
                )
            ]
            docs, scores, explanations = zip(*results) if results else ([], [], [])
            return list(docs), list(scores), list(explanations)

        # Get BM25 scores for documents returned by vector search
        bm25_scores = []
        field_scores = {field: [] for field in self.field_metadata}

        for doc in vector_docs:
            # BM25 score
            doc_id = doc.get("event_id", "")
            if doc_id in self.doc_ids:
                idx = self.doc_ids.index(doc_id)
                score = self.bm25.get_scores(tokenized_query)[idx]
                bm25_scores.append(score)
            else:
                bm25_scores.append(0.0)

            # Calculate field-specific scores
            for field, metadata in self.field_metadata.items():
                field_score = 0.0

                # If we have a numerical target for this field
                if field in numerical_queries:
                    field_score = self._calculate_field_score(
                        doc, field, numerical_queries[field]
                    )
                # If we have a categorical target for this field
                elif field in field_values:
                    field_score = self._calculate_field_score(
                        doc, field, field_values[field]
                    )
                # Otherwise use general field importance
                elif metadata["analysis_weight"] > 0:
                    # Check if field exists in document
                    if field in doc and doc[field]:
                        # Look for exact/partial matches with known values
                        for value in metadata.get("values", []):
                            if value in query.lower():
                                field_score = self._calculate_field_score(
                                    doc, field, value
                                )
                                if field_score > 0:
                                    break

                field_scores[field].append(field_score)

        # Normalize scores
        normalized_bm25_scores = self._normalize_scores(bm25_scores)

        logger.debug("===== Documents sorted by BM25 score =====")
        # Create a list of (doc, score) pairs
        bm25_doc_scores = list(zip(vector_docs, normalized_bm25_scores))
        # Sort by BM25 score (descending)
        bm25_sorted = sorted(bm25_doc_scores, key=lambda x: x[1], reverse=True)
        # Log the top 10 documents
        for i, (doc, score) in enumerate(bm25_sorted):
            doc_id = doc.get("event_id", "unknown")
            ink = doc.get("Ink Coverage", "N/A")
            coating = doc.get("Media Coating", "N/A")
            gsm = doc.get("Media Weight GSM", "N/A")
            logger.debug(
                f"BM25 Rank {i + 1}: Doc {doc_id} - Score: {score:.4f} - Ink: {ink}, Coating: {coating}, GSM: {gsm}"
            )
        normalized_field_scores = {
            field: self._normalize_scores(scores)
            for field, scores in field_scores.items()
        }

        # Calculate combined scores based on dynamic query weights
        combined_scores = []
        explanations = []

        for i, doc in enumerate(vector_docs):
            # Vector component
            score_components = {
                "vector": normalized_vector_scores[i] * weights["vector"],
                "bm25": normalized_bm25_scores[i] * weights["bm25"],
                "fields": {},
            }

            # Field components
            for field, field_weight in weights["fields"].items():
                if i < len(normalized_field_scores.get(field, [])):
                    field_score = normalized_field_scores[field][i] * field_weight
                    score_components["fields"][field] = field_score

            # Calculate final score
            final_score = (
                score_components["vector"]
                + score_components["bm25"]
                + sum(score_components["fields"].values())
            )
            combined_scores.append(final_score)

            # Generate explanation
            explanation_parts = []

            # Add vector component
            vector_contribution = normalized_vector_scores[i] * weights["vector"]
            if vector_contribution > 0.01:
                explanation_parts.append(
                    f"Vector similarity: {normalized_vector_scores[i]:.2f} × {weights['vector']:.2f}"
                )

            # Add BM25 component
            bm25_contribution = normalized_bm25_scores[i] * weights["bm25"]
            if bm25_contribution > 0.01:
                explanation_parts.append(
                    f"Keyword matching: {normalized_bm25_scores[i]:.2f} × {weights['bm25']:.2f}"
                )

            # Add field components
            for field in sorted(
                weights["fields"].keys(),
                key=lambda f: weights["fields"][f]
                * normalized_field_scores.get(f, [0])[i]
                if i < len(normalized_field_scores.get(f, []))
                else 0,
                reverse=True,
            ):
                if (
                    i < len(normalized_field_scores.get(field, []))
                    and normalized_field_scores[field][i] > 0
                ):
                    field_contribution = (
                        normalized_field_scores[field][i] * weights["fields"][field]
                    )

                    if (
                        field_contribution > 0.01
                    ):  # Only include significant contributions
                        field_display = field.replace(
                            "Media ", ""
                        )  # Shorter display name
                        field_value = doc.get(field, "unknown")

                        if field in numerical_queries:
                            target = numerical_queries[field]
                            explanation_parts.append(
                                f"{field_display}: {field_value} "
                                f"(target: {target}) "
                                f"scored {normalized_field_scores[field][i]:.2f} "
                                f"× {weights['fields'][field]:.2f}"
                            )
                        elif field in field_values:
                            target = field_values[field]
                            explanation_parts.append(
                                f"{field_display}: {field_value} "
                                f"matched '{target}' "
                                f"scored {normalized_field_scores[field][i]:.2f} "
                                f"× {weights['fields'][field]:.2f}"
                            )
                        else:
                            explanation_parts.append(
                                f"{field_display}: {field_value} "
                                f"scored {normalized_field_scores[field][i]:.2f} "
                                f"× {weights['fields'][field]:.2f}"
                            )

            explanation = "Matched based on: " + "; ".join(explanation_parts)
            explanations.append(explanation)

        # Check if this is a numerical proximity query without hardcoding phrases
        if numerical_queries and any(
            field
            for field in numerical_queries
            if self.field_metadata.get(field, {}).get("numeric", False)
            and self.field_metadata.get(field, {}).get("analysis_weight", 0) > 0.2
        ):
            # Get the documents that match all categorical criteria
            matching_docs = []
            for i, doc in enumerate(vector_docs):
                # Check categorical matches
                matches_criteria = True
                for field, value in field_values.items():
                    if field in doc and doc[field] is not None:
                        if str(doc[field]).lower() != str(value).lower():
                            matches_criteria = False
                            break

                if matches_criteria:
                    # Calculate proximity scores for all numerical fields
                    proximity_distances = {}
                    for field, target in numerical_queries.items():
                        if self.field_metadata.get(field, {}).get("numeric", False):
                            if field in doc and doc[field] is not None:
                                try:
                                    doc_value = float(doc[field])
                                    proximity_distances[field] = abs(doc_value - target)
                                except (ValueError, TypeError):
                                    proximity_distances[field] = float("inf")
                            else:
                                proximity_distances[field] = float("inf")

                    # Use the distance from the highest-weighted numeric field
                    best_field = None
                    best_weight = 0
                    for field in proximity_distances:
                        field_weight = weights["fields"].get(field, 0)
                        if field_weight > best_weight:
                            best_weight = field_weight
                            best_field = field

                    if best_field:
                        proximity_distance = proximity_distances[best_field]
                        matching_docs.append((doc, explanations[i], proximity_distance))

            # If we have matched documents, sort by proximity distance
            if matching_docs:
                # Sort by proximity (ascending - closer is better)
                matching_docs.sort(key=lambda x: x[2])

                # Return the sorted documents
                sorted_docs = [item[0] for item in matching_docs]
                sorted_explanations = [item[1] for item in matching_docs]

                # Generate appropriate scores (1.0 for perfect match, descending from there)
                max_distance = (
                    max(item[2] for item in matching_docs) if matching_docs else 1.0
                )
                if max_distance == 0:  # Avoid division by zero
                    max_distance = 1.0
                sorted_scores = [
                    1.0 - (item[2] / (max_distance * 2)) for item in matching_docs
                ]

                return (
                    sorted_docs[:n_results],
                    sorted_scores[:n_results],
                    sorted_explanations[:n_results],
                )

        # Create result tuples
        results = list(zip(vector_docs, combined_scores, explanations))

        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top n results
        results = results[:n_results]
        docs, scores, explanations = zip(*results) if results else ([], [], [])

        logger.debug(f"Dynamic hybrid search returned {len(docs)} documents")
        return list(docs), list(scores), list(explanations)
