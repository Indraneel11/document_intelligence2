import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentGraphBuilder:
    """
    Builds a deterministic document structure graph from:

    1. multimodal_enriched_document.json
    2. multimodal_chunks.json

    The graph represents document structure and provenance.

    No LLM/API calls are used here.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        enriched_document_path: Path,
        chunks_path: Path,
    ):
        self.enriched_document_path = Path(enriched_document_path)
        self.chunks_path = Path(chunks_path)

        self.document: Dict[str, Any] = {}
        self.chunks_document: Dict[str, Any] = {}

        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

        self.node_ids = set()
        self.element_lookup: Dict[str, Dict[str, Any]] = {}
        self.page_lookup: Dict[int, Dict[str, Any]] = {}
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}

        self.section_counter = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_inputs(self):
        """Load the two JSON inputs."""

        if not self.enriched_document_path.exists():
            raise FileNotFoundError(
                f"Enriched document not found: "
                f"{self.enriched_document_path}"
            )

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: "
                f"{self.chunks_path}"
            )

        with self.enriched_document_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            self.document = json.load(f)

        with self.chunks_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            self.chunks_document = json.load(f)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def node_id(node_type: str, identifier: Any) -> str:
        """Create globally unique node IDs."""

        return f"{node_type}:{identifier}"

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Add a node while preventing duplicate IDs."""

        if node_id in self.node_ids:
            raise ValueError(
                f"Duplicate graph node ID detected: {node_id}"
            )

        node = {
            "id": node_id,
            "type": node_type,
            "properties": properties or {},
        }

        self.nodes.append(node)
        self.node_ids.add(node_id)

    def add_edge(
        self,
        source: str,
        target: str,
        relationship: str,
        weight: float = 1.0,
        provenance: str = "deterministic",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a directed graph relationship."""

        if source not in self.node_ids:
            raise ValueError(
                f"Edge source node does not exist: {source}"
            )

        if target not in self.node_ids:
            raise ValueError(
                f"Edge target node does not exist: {target}"
            )

        edge = {
            "source": source,
            "target": target,
            "relationship": relationship,
            "weight": weight,
            "provenance": provenance,
        }

        if metadata:
            edge["metadata"] = metadata

        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Document structure
    # ------------------------------------------------------------------

    def build_document_node(self):
        """Create the root document node."""

        document_metadata = self.document.get(
            "document",
            {}
        )

        source = document_metadata.get(
            "source",
            "unknown"
        )

        total_pages = document_metadata.get(
            "total_pages",
            len(self.document.get("pages", []))
        )

        document_node_id = self.node_id(
            "document",
            "root"
        )

        self.add_node(
            document_node_id,
            "document",
            {
                "source": source,
                "total_pages": total_pages,
            },
        )

    def build_page_nodes(self):
        """Create page nodes and document → page relationships."""

        document_node_id = self.node_id(
            "document",
            "root"
        )

        pages = self.document.get("pages", [])

        for page in pages:
            page_number = page.get("page_number")

            if page_number is None:
                raise ValueError(
                    "Encountered page without page_number."
                )

            if page_number in self.page_lookup:
                raise ValueError(
                    f"Duplicate page number: {page_number}"
                )

            page_node_id = self.node_id(
                "page",
                page_number
            )

            elements = page.get("elements", [])

            self.add_node(
                page_node_id,
                "page",
                {
                    "page_number": page_number,
                    "element_count": len(elements),
                },
            )

            self.page_lookup[page_number] = page

            self.add_edge(
                document_node_id,
                page_node_id,
                "contains",
                provenance="document_structure",
            )

    # ------------------------------------------------------------------
    # Element structure
    # ------------------------------------------------------------------

    def build_element_nodes(self):
        """
        Create an element node for every element in the enriched
        document.

        Layout elements are intentionally retained here even though
        they are excluded from semantic chunking.
        """

        for page in self.document.get("pages", []):
            page_number = page.get("page_number")

            page_node_id = self.node_id(
                "page",
                page_number
            )

            elements = page.get("elements", [])

            previous_element_id = None

            for element in elements:
                element_id = element.get("element_id")

                if not element_id:
                    raise ValueError(
                        f"Element on page {page_number} "
                        f"is missing element_id."
                    )

                if element_id in self.element_lookup:
                    raise ValueError(
                        f"Duplicate element_id detected: "
                        f"{element_id}"
                    )

                element_node_id = self.node_id(
                    "element",
                    element_id
                )

                modality = element.get(
                    "modality",
                    "unknown"
                )

                element_type = element.get(
                    "type",
                    "unknown"
                )

                properties = {
                    "element_id": element_id,
                    "page_number": page_number,
                    "type": element_type,
                    "modality": modality,
                    "bbox": element.get("bbox"),
                }

                metadata = element.get(
                    "metadata",
                    {}
                )

                if isinstance(metadata, dict):
                    properties["metadata"] = metadata

                self.add_node(
                    element_node_id,
                    "element",
                    properties,
                )

                self.element_lookup[element_id] = element

                # Page → element
                self.add_edge(
                    page_node_id,
                    element_node_id,
                    "contains",
                    provenance="document_structure",
                )

                # Element ordering
                if previous_element_id is not None:
                    previous_node_id = self.node_id(
                        "element",
                        previous_element_id
                    )

                    self.add_edge(
                        previous_node_id,
                        element_node_id,
                        "follows",
                        provenance="mineru_document_order",
                    )

                previous_element_id = element_id

    # ------------------------------------------------------------------
    # Chunk structure
    # ------------------------------------------------------------------

    def build_chunk_nodes(self):
        """
        Create chunk nodes from multimodal_chunks.json.

        Chunk elements reference canonical element nodes rather than
        creating duplicate element payloads.
        """

        chunks = self.chunks_document.get(
            "chunks",
            []
        )

        previous_chunk_id = None

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Chunk without chunk_id detected."
                )

            if chunk_id in self.chunk_lookup:
                raise ValueError(
                    f"Duplicate chunk_id detected: {chunk_id}"
                )

            chunk_node_id = self.node_id(
                "chunk",
                chunk_id
            )

            element_ids = chunk.get(
                "element_ids",
                []
            )

            modalities = chunk.get(
                "modalities",
                []
            )

            properties = {
                "chunk_id": chunk_id,
                "size": chunk.get("size"),
                "modalities": modalities,
                "has_cross_modal": chunk.get(
                    "has_cross_modal",
                    False
                ),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "oversized": chunk.get(
                    "oversized",
                    False
                ),
                "element_count": len(element_ids),
            }

            self.add_node(
                chunk_node_id,
                "chunk",
                properties,
            )

            self.chunk_lookup[chunk_id] = chunk

            # ----------------------------------------------------------
            # Chunk → element
            # ----------------------------------------------------------

            for element_id in element_ids:

                if element_id not in self.element_lookup:
                    raise ValueError(
                        f"Chunk {chunk_id} references unknown "
                        f"element {element_id}"
                    )

                element_node_id = self.node_id(
                    "element",
                    element_id
                )

                self.add_edge(
                    chunk_node_id,
                    element_node_id,
                    "contains",
                    provenance="chunk_membership",
                )

            # ----------------------------------------------------------
            # Chunk → page
            # ----------------------------------------------------------

            page_numbers = self._get_chunk_pages(
                element_ids
            )

            for page_number in page_numbers:

                page_node_id = self.node_id(
                    "page",
                    page_number
                )

                self.add_edge(
                    chunk_node_id,
                    page_node_id,
                    "spans_page",
                    provenance="chunk_membership",
                )

            # ----------------------------------------------------------
            # Chunk ordering
            # ----------------------------------------------------------

            if previous_chunk_id is not None:

                previous_chunk_node_id = self.node_id(
                    "chunk",
                    previous_chunk_id
                )

                self.add_edge(
                    previous_chunk_node_id,
                    chunk_node_id,
                    "follows",
                    provenance="chunk_order",
                )

            previous_chunk_id = chunk_id

    def _get_chunk_pages(
        self,
        element_ids: List[str]
    ) -> List[int]:
        """Return unique pages occupied by a chunk."""

        pages = set()

        for element_id in element_ids:

            element = self.element_lookup.get(
                element_id
            )

            if element is None:
                continue

            page_number = element.get(
                "page_number"
            )

            if page_number is not None:
                pages.add(page_number)

        return sorted(pages)

    # ------------------------------------------------------------------
    # Section hierarchy
    # ------------------------------------------------------------------

    def build_section_hierarchy(self):
        """
        Build deterministic section hierarchy using title metadata.

        MinerU title levels are treated as structural evidence, not
        as guaranteed semantic truth.

        Example:

            Title level 1
                ↓
            Title level 2
                ↓
            Title level 3

        Each section remains connected to the heading that defines it.
        """

        section_stack = []

        for page in self.document.get("pages", []):

            elements = page.get(
                "elements",
                []
            )

            for element in elements:

                element_type = element.get(
                    "type"
                )

                if element_type != "title":
                    continue

                metadata = element.get(
                    "metadata",
                    {}
                )

                level = metadata.get(
                    "level"
                )

                if not isinstance(level, int):
                    continue

                if level <= 0:
                    continue

                # Remove sections at the same or deeper level.
                while section_stack and section_stack[-1][
                    "level"
                ] >= level:
                    section_stack.pop()

                self.section_counter += 1

                section_id = (
                    f"section_{self.section_counter:04d}"
                )

                section_node_id = self.node_id(
                    "section",
                    section_id
                )

                element_id = element.get(
                    "element_id"
                )

                title_text = element.get(
                    "text"
                )

                self.add_node(
                    section_node_id,
                    "section",
                    {
                        "section_id": section_id,
                        "level": level,
                        "heading_element_id": element_id,
                        "title": title_text,
                    },
                )

                # Heading → section
                element_node_id = self.node_id(
                    "element",
                    element_id
                )

                self.add_edge(
                    element_node_id,
                    section_node_id,
                    "defines_section",
                    provenance="mineru_title_level",
                    metadata={
                        "level": level
                    },
                )

                # Parent section → child section
                if section_stack:

                    parent_section_id = section_stack[-1][
                        "section_id"
                    ]

                    parent_node_id = self.node_id(
                        "section",
                        parent_section_id
                    )

                    self.add_edge(
                        parent_node_id,
                        section_node_id,
                        "contains",
                        provenance="mineru_title_level",
                    )

                section_stack.append(
                    {
                        "section_id": section_id,
                        "level": level,
                    }
                )

    # ------------------------------------------------------------------
    # Element → section assignment
    # ------------------------------------------------------------------

    def assign_elements_to_sections(self):
        """
        Assign elements to the currently active section.

        This follows document order and the title hierarchy created
        above.
        """

        section_stack = []

        # Map heading element ID → section ID.
        heading_to_section = {}

        for node in self.nodes:

            if node["type"] != "section":
                continue

            properties = node.get(
                "properties",
                {}
            )

            heading_element_id = properties.get(
                "heading_element_id"
            )

            section_id = properties.get(
                "section_id"
            )

            if heading_element_id and section_id:
                heading_to_section[
                    heading_element_id
                ] = {
                    "section_id": section_id,
                    "level": properties.get(
                        "level"
                    ),
                }

        for page in self.document.get("pages", []):

            for element in page.get("elements", []):

                element_id = element.get(
                    "element_id"
                )

                element_type = element.get(
                    "type"
                )

                metadata = element.get(
                    "metadata",
                    {}
                )

                # A title changes the active section.
                if element_type == "title":

                    section_info = heading_to_section.get(
                        element_id
                    )

                    if section_info:

                        level = section_info["level"]

                        while (
                            section_stack
                            and section_stack[-1]["level"]
                            >= level
                        ):
                            section_stack.pop()

                        section_stack.append(
                            section_info
                        )

                    # The title itself is connected through
                    # defines_section, so no contains edge to
                    # itself is created.
                    continue

                if not section_stack:
                    continue

                current_section = section_stack[-1]

                section_node_id = self.node_id(
                    "section",
                    current_section["section_id"]
                )

                element_node_id = self.node_id(
                    "element",
                    element_id
                )

                self.add_edge(
                    section_node_id,
                    element_node_id,
                    "contains",
                    provenance="section_assignment",
                )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        """Run structural graph validation."""

        errors = []

        # --------------------------------------------------------------
        # Node uniqueness
        # --------------------------------------------------------------

        node_ids = [
            node["id"]
            for node in self.nodes
        ]

        if len(node_ids) != len(set(node_ids)):
            errors.append(
                "Duplicate node IDs exist."
            )

        # --------------------------------------------------------------
        # Edge references
        # --------------------------------------------------------------

        for edge in self.edges:

            if edge["source"] not in self.node_ids:
                errors.append(
                    f"Missing edge source: "
                    f"{edge['source']}"
                )

            if edge["target"] not in self.node_ids:
                errors.append(
                    f"Missing edge target: "
                    f"{edge['target']}"
                )

        # --------------------------------------------------------------
        # Element coverage
        # --------------------------------------------------------------

        expected_elements = len(
            self.element_lookup
        )

        graph_element_nodes = sum(
            1
            for node in self.nodes
            if node["type"] == "element"
        )

        if expected_elements != graph_element_nodes:
            errors.append(
                "Element node count does not match "
                "canonical document element count."
            )

        # --------------------------------------------------------------
        # Chunk references
        # --------------------------------------------------------------

        for chunk_id, chunk in self.chunk_lookup.items():

            for element_id in chunk.get(
                "element_ids",
                []
            ):

                if element_id not in self.element_lookup:
                    errors.append(
                        f"Chunk {chunk_id} references "
                        f"missing element {element_id}"
                    )

        if errors:
            raise ValueError(
                "Graph validation failed:\n- "
                + "\n- ".join(errors)
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """Build and return the complete graph."""

        self.load_inputs()

        self.build_document_node()
        self.build_page_nodes()
        self.build_element_nodes()
        self.build_chunk_nodes()
        self.build_section_hierarchy()
        self.assign_elements_to_sections()

        self.validate()

        return self.to_dict()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _node_counts(self):
        counts = defaultdict(int)

        for node in self.nodes:
            counts[node["type"]] += 1

        return dict(sorted(counts.items()))

    def _edge_counts(self):
        counts = defaultdict(int)

        for edge in self.edges:
            counts[edge["relationship"]] += 1

        return dict(sorted(counts.items()))

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable graph representation."""

        document_metadata = self.document.get(
            "document",
            {}
        )

        return {
            "graph": {
                "schema_version": self.SCHEMA_VERSION,
                "strategy": (
                    "Deterministic document structure graph"
                ),
                "source_document": document_metadata.get(
                    "source"
                ),
                "source_enriched_document": str(
                    self.enriched_document_path
                ),
                "source_chunks": str(
                    self.chunks_path
                ),
                "node_counts": self._node_counts(),
                "edge_counts": self._edge_counts(),
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
            },
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def save(self, output_path: Path):
        """Build and save graph to JSON."""

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        graph = self.build()

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                graph,
                f,
                indent=2,
                ensure_ascii=False
            )

        return graph