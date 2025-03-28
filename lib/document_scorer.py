"""
Document scoring for the RAG Chatbot.
"""

import re
from typing import Dict, List, Tuple, Any

from config import MAX_DISPLAY_RESULTS
from log import logger


class DocumentScorer:
    """
    Scores and ranks documents based on relevance to query.

    This class implements algorithms for determining how well
    documents match user queries and provides explanations.
    """

    def score_document(self, doc: Dict[str, Any], query: str) -> Tuple[int, str]:
        """
        Score a document based on how well it matches the query.

        Args:
            doc: Document metadata dictionary
            query: User's search query

        Returns:
            Tuple of (score, explanation) where score is an integer
            and explanation is a string describing the match
        """
        score = 0
        explanation = []

        # Extract key terms from query
        query_lower = query.lower()

        # Check for ink coverage
        if "ink" in query_lower:
            if "heavy" in query_lower and doc.get("Ink Coverage") == "Heavy":
                score += 10
                explanation.append("Matches requested heavy ink coverage")
            elif "medium" in query_lower and doc.get("Ink Coverage") == "Medium":
                score += 10
                explanation.append("Matches requested medium ink coverage")
            elif "light" in query_lower and doc.get("Ink Coverage") == "Light":
                score += 10
                explanation.append("Matches requested light ink coverage")

        # Check for media coating
        if "coat" in query_lower:
            if "coated" in query_lower and "coated" in str(doc.get("Media Coating", "")).lower():
                score += 10
                explanation.append("Matches requested coated media")
            elif "uncoated" in query_lower and "uncoated" in str(doc.get("Media Coating", "")).lower():
                score += 10
                explanation.append("Matches requested uncoated media")

        # Check for media finish
        if "silk" in query_lower and "silk" in str(doc.get("Media Finish", "")).lower():
            score += 10
            explanation.append("Matches requested silk finish")
        elif "matte" in query_lower and "matte" in str(doc.get("Media Finish", "")).lower():
            score += 10
            explanation.append("Matches requested matte finish")
        elif "gloss" in query_lower and "gloss" in str(doc.get("Media Finish", "")).lower():
            score += 10
            explanation.append("Matches requested gloss finish")

        # Check for GSM weight
        gsm_matches = re.findall(r'(\d+)\s*gsm', query_lower)
        if gsm_matches and "Media Weight GSM" in doc:
            requested_gsm = int(gsm_matches[0])
            doc_gsm = int(doc["Media Weight GSM"]) if doc["Media Weight GSM"] else 0
            # Score based on how close the document GSM is to the requested GSM
            gsm_diff = abs(requested_gsm - doc_gsm)
            if gsm_diff == 0:
                score += 15
                explanation.append(f"Exact match for requested {requested_gsm} GSM")
            elif gsm_diff <= 20:
                score += 10
                explanation.append(f"Close match for GSM: {doc_gsm} vs requested {requested_gsm}")
            elif gsm_diff <= 50:
                score += 5
                explanation.append(f"Approximate GSM match: {doc_gsm} vs requested {requested_gsm}")

        # Check for location if mentioned
        if "location" in doc and doc["location"]:
            location = str(doc["location"]).lower()
            if location in query_lower:
                score += 10
                explanation.append(f"Matches requested location: {doc['location']}")

        # Check for press model if mentioned
        if "Press Model" in doc and doc["Press Model"]:
            press_model = str(doc["Press Model"]).lower()
            # Check for partial matches in query
            for word in press_model.split():
                if len(word) > 3 and word in query_lower:  # Only check significant words
                    score += 10
                    explanation.append(f"Matches requested press model: {doc['Press Model']}")
                    break

        return score, ". ".join(explanation) if explanation else "Matched based on general relevance."

    def rank_documents(
            self,
            documents: List[Dict[str, Any]],
            query: str,
            max_results: int = MAX_DISPLAY_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Rank documents based on their relevance to the query.

        Args:
            documents: List of document metadata dictionaries
            query: User's search query
            max_results: Maximum number of results to return

        Returns:
            List of ranked document dictionaries with explanations
        """
        # Apply scoring to each document
        scored_docs = []
        for doc in documents:
            score, explanation = self.score_document(doc, query)
            doc_copy = doc.copy()
            doc_copy["score"] = score
            doc_copy["Reason for Selection"] = explanation
            scored_docs.append(doc_copy)

        # Sort by score in descending order
        ranked_results = sorted(scored_docs, key=lambda x: x.get("score", 0), reverse=True)

        # Limit to top results
        top_ranked = ranked_results[:max_results] if len(ranked_results) >= max_results else ranked_results

        # Clean up score field before returning
        for doc in top_ranked:
            if "score" in doc:
                del doc["score"]

        return top_ranked