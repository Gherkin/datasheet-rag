"""Data models for the concept graph layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Concept(BaseModel):
    """A concept extracted from document chunks.

    Concepts represent ideas, specifications, or topics that appear across
    one or more chunks and potentially across documents. Examples:
      - "thermal resistance junction-to-ambient"
      - "ESD protection HBM rating"
      - "I2C bus timing specifications"
      - "dropout voltage vs load current"
    """

    id: str = Field(description="Unique concept ID")
    name: str = Field(description="Short human-readable concept label")
    description: str = Field(
        default="",
        description="LLM-generated description of what this concept covers",
    )
    canonical_form: str = Field(
        default="",
        description="Normalised form for deduplication (lowercase, stemmed)",
    )

    # Which chunks reference this concept
    chunk_ids: list[str] = Field(default_factory=list)

    # Which documents this concept appears in
    doc_ids: list[str] = Field(default_factory=list)

    # Related concepts (concept_id → similarity score)
    related_concepts: dict[str, float] = Field(default_factory=dict)

    # Embedding of the concept itself
    embedding: list[float] | None = None


class ConceptLink(BaseModel):
    """A link between a chunk and a concept, with a relevance score."""

    chunk_id: str
    concept_id: str
    relevance: float = Field(ge=0.0, le=1.0, description="How central this concept is to the chunk")
    context: str = Field(
        default="",
        description="Short explanation of how the concept manifests in this chunk",
    )


class ConceptGraph(BaseModel):
    """The full concept graph — concepts + their links to chunks.

    Supports two query patterns:
    1. concept_id → chunks  (find all chunks about a concept)
    2. chunk_id → concepts  (find all concepts in a chunk)
    3. concept_id → related concepts → their chunks  (lateral navigation)
    """

    concepts: dict[str, Concept] = Field(default_factory=dict)
    links: list[ConceptLink] = Field(default_factory=list)

    # Indexes built on demand
    _by_chunk: dict[str, list[ConceptLink]] | None = None
    _by_concept: dict[str, list[ConceptLink]] | None = None

    def _build_indexes(self) -> None:
        self._by_chunk = {}
        self._by_concept = {}
        for link in self.links:
            self._by_chunk.setdefault(link.chunk_id, []).append(link)
            self._by_concept.setdefault(link.concept_id, []).append(link)

    def concepts_for_chunk(self, chunk_id: str) -> list[ConceptLink]:
        if self._by_chunk is None:
            self._build_indexes()
        assert self._by_chunk is not None
        return self._by_chunk.get(chunk_id, [])

    def chunks_for_concept(self, concept_id: str) -> list[ConceptLink]:
        if self._by_concept is None:
            self._build_indexes()
        assert self._by_concept is not None
        return self._by_concept.get(concept_id, [])

    def related_chunks(self, chunk_id: str, concept_id: str) -> list[ConceptLink]:
        """Find other chunks that share a specific concept with the given chunk."""
        links = self.chunks_for_concept(concept_id)
        return [link for link in links if link.chunk_id != chunk_id]

    def lateral_search(
        self, chunk_id: str, top_k: int = 5
    ) -> list[tuple[str, list[ConceptLink]]]:
        """From a chunk, find related concepts, then find chunks for each.

        Returns list of (concept_id, [links to other chunks]) sorted by
        concept relevance to the source chunk.
        """
        source_concepts = self.concepts_for_chunk(chunk_id)
        source_concepts.sort(key=lambda c: -c.relevance)

        results: list[tuple[str, list[ConceptLink]]] = []
        seen_chunks: set[str] = {chunk_id}

        for sc in source_concepts[:top_k]:
            related = [
                link for link in self.chunks_for_concept(sc.concept_id)
                if link.chunk_id not in seen_chunks
            ]
            if related:
                results.append((sc.concept_id, related))
                seen_chunks.update(link.chunk_id for link in related)

        return results
