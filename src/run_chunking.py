import json
from collections import Counter
from pathlib import Path

from chunking.optimized_chunker import OptimizedChunker
from chunking.semantic_filter import is_semantically_chunkable


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "multimodal_enriched_document.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "multimodal_chunks.json"
)


# ============================================================
# Configuration
# ============================================================

MAX_CHUNK_SIZE = 512
OVERLAP = 50
MAX_OVERLAP_ELEMENTS = 3

VALID_MODALITIES = {
    "text",
    "layout",
    "table",
    "visual",
    "unknown",
}


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document():
    """
    Load the Phase 3.2 enriched multimodal document.

    Phase 4 intentionally does not fall back to the Phase 3.1
    document because doing so could silently remove visual
    enrichment from the RAG pipeline.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nPhase 4 requires the Phase 3.2 enriched document:\n"
            f"{INPUT_FILE}\n\n"
            "Run Phase 3.2 multimodal enrichment first."
        )

    print(
        "Loading enriched multimodal document:"
    )

    print(INPUT_FILE)

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_document(
    document,
):
    """
    Validate the Phase 3.2 document before chunking.

    This protects Phase 4 from consuming an incomplete or
    structurally invalid multimodal document.
    """

    print()
    print("=" * 80)
    print("PHASE 4 INPUT VALIDATION")
    print("=" * 80)

    pages = document.get(
        "pages"
    )

    if not isinstance(
        pages,
        list,
    ):

        raise ValueError(
            "Invalid document: 'pages' must be a list."
        )

    total_elements = 0
    visual_elements = 0
    visual_analysis_count = 0

    modality_counts = {}

    errors = []

    for page_index, page in enumerate(
        pages,
        start=1,
    ):

        if not isinstance(
            page,
            dict,
        ):

            errors.append(
                f"Page {page_index}: page must be an object."
            )

            continue

        elements = page.get(
            "elements"
        )

        if not isinstance(
            elements,
            list,
        ):

            errors.append(
                f"Page {page_index}: "
                "'elements' must be a list."
            )

            continue

        for element_index, element in enumerate(
            elements,
            start=1,
        ):

            total_elements += 1

            if not isinstance(
                element,
                dict,
            ):

                errors.append(
                    f"Page {page_index}, element "
                    f"{element_index}: element must be an object."
                )

                continue

            modality = element.get(
                "modality"
            )

            if modality not in VALID_MODALITIES:

                errors.append(
                    f"Element "
                    f"{page_index}:{element_index} "
                    f"has invalid modality: "
                    f"{modality!r}"
                )

            modality_counts[
                modality
            ] = (
                modality_counts.get(
                    modality,
                    0,
                )
                + 1
            )

            # ----------------------------------------------------
            # Visual validation
            # ----------------------------------------------------

            if modality == "visual":

                visual_elements += 1

                if element.get(
                    "visual_analysis"
                ):

                    visual_analysis_count += 1

    # --------------------------------------------------------
    # Print validation statistics
    # --------------------------------------------------------

    print(
        f"Pages             : {len(pages)}"
    )

    print(
        f"Total elements    : {total_elements}"
    )

    print()
    print(
        "Modality counts:"
    )

    for modality, count in sorted(
        modality_counts.items(),
        key=lambda item: str(item[0]),
    ):

        print(
            f"  {str(modality):<12} {count}"
        )

    print()
    print(
        f"Visual elements   : {visual_elements}"
    )

    print(
        f"Visuals enriched  : "
        f"{visual_analysis_count}"
    )

    # --------------------------------------------------------
    # Ensure all visual elements are enriched.
    # --------------------------------------------------------

    if (
        visual_elements > 0
        and visual_analysis_count != visual_elements
    ):

        errors.append(
            f"Only {visual_analysis_count} of "
            f"{visual_elements} visual elements "
            "contain visual_analysis."
        )

    # --------------------------------------------------------
    # Final validation result.
    # --------------------------------------------------------

    if errors:

        print()
        print(
            "Validation: FAILED"
        )

        print()

        for error in errors:

            print(
                f"  ERROR: {error}"
            )

        raise ValueError(
            "Phase 4 input validation failed."
        )

    print()
    print(
        "Validation: PASSED"
    )

    print(
        "=" * 80
    )


# ============================================================
# SEMANTIC ELEMENT FILTERING
# ============================================================

def flatten_semantic_elements(
    document,
):
    """
    Flatten the document while applying the semantic chunking
    eligibility policy.

    The canonical document is never modified.

    Layout elements such as:
        - page headers
        - page footers
        - page numbers

    remain in the canonical document but are excluded from
    retrieval-oriented semantic chunks.
    """

    semantic_elements = []

    total_elements = 0
    excluded_elements = 0

    excluded_by_modality = {}

    for page in document["pages"]:

        for element in page["elements"]:

            total_elements += 1

            if not is_semantically_chunkable(
                element
            ):

                excluded_elements += 1

                modality = element.get(
                    "modality",
                    "unknown",
                )

                excluded_by_modality[
                    modality
                ] = (
                    excluded_by_modality.get(
                        modality,
                        0,
                    )
                    + 1
                )

                continue

            semantic_elements.append(
                element
            )

    return (
        semantic_elements,
        total_elements,
        excluded_elements,
        excluded_by_modality,
    )


# ============================================================
# CHUNK VALIDATION
# ============================================================

def validate_chunks(
    chunks,
    semantic_elements,
):
    """
    Validate the generated chunks.

    Checks:
        - chunks are not empty
        - chunk IDs are unique
        - no layout elements enter chunks
        - all semantic elements are represented
        - all visual elements are represented
        - modality metadata is correct
        - cross-modal metadata is correct
        - page metadata is correct
        - overlap does not exceed configured limits
    """

    print()
    print("=" * 80)
    print("PHASE 4 OUTPUT VALIDATION")
    print("=" * 80)

    errors = []

    if not chunks:

        errors.append(
            "No chunks were generated."
        )

    # --------------------------------------------------------
    # Validate chunk IDs and contents.
    # --------------------------------------------------------

    chunk_ids = set()

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_id = chunk.get(
            "chunk_id"
        )

        if not chunk_id:

            errors.append(
                f"Chunk {index} has no chunk_id."
            )

        elif chunk_id in chunk_ids:

            errors.append(
                f"Duplicate chunk_id: {chunk_id}"
            )

        else:

            chunk_ids.add(
                chunk_id
            )

        elements = chunk.get(
            "elements",
            []
        )

        if not elements:

            errors.append(
                f"{chunk_id}: empty chunk."
            )

        element_ids = chunk.get(
            "element_ids",
            []
        )

        if len(elements) != len(
            element_ids
        ):

            errors.append(
                f"{chunk_id}: element_ids count "
                "does not match elements count."
            )

        # ----------------------------------------------------
        # Check duplicate element IDs inside one chunk.
        # ----------------------------------------------------

        if len(element_ids) != len(
            set(element_ids)
        ):

            errors.append(
                f"{chunk_id}: duplicate element IDs "
                "inside the same chunk."
            )

        # ----------------------------------------------------
        # Validate elements inside each chunk.
        # ----------------------------------------------------

        for element in elements:

            if not isinstance(
                element,
                dict,
            ):

                errors.append(
                    f"{chunk_id}: contains a non-object element."
                )

                continue

            # ------------------------------------------------
            # Layout must never enter semantic chunks.
            # ------------------------------------------------

            if element.get(
                "modality"
            ) == "layout":

                errors.append(
                    f"{chunk_id}: layout element "
                    f"{element.get('element_id')} "
                    "was included."
                )

    # --------------------------------------------------------
    # Collect input element IDs.
    # --------------------------------------------------------

    input_ids = {
        element.get("element_id")
        for element in semantic_elements
        if element.get("element_id") is not None
    }

    # --------------------------------------------------------
    # Count element occurrences across chunks.
    #
    # Elements can legitimately occur more than once because
    # of overlap.
    # --------------------------------------------------------

    element_occurrences = Counter()

    for chunk in chunks:

        for element_id in chunk.get(
            "element_ids",
            [],
        ):

            element_occurrences[
                element_id
            ] += 1

    chunk_element_ids = set(
        element_occurrences.keys()
    )

    # --------------------------------------------------------
    # Every semantic element must appear at least once.
    # --------------------------------------------------------

    missing_ids = (
        input_ids
        - chunk_element_ids
    )

    if missing_ids:

        errors.append(
            "Semantic elements missing from chunks: "
            f"{sorted(missing_ids)}"
        )

    # --------------------------------------------------------
    # Validate overlap occurrence limit.
    #
    # With sequential bounded overlap, an element should not
    # normally occur in more than two adjacent chunks.
    #
    # We therefore treat >2 occurrences as suspicious rather
    # than immediately failing arbitrary future chunking logic.
    # --------------------------------------------------------

    excessive_overlap = {
        element_id: count
        for element_id, count
        in element_occurrences.items()
        if count > 2
    }

    if excessive_overlap:

        errors.append(
            "Some elements occur in more than two chunks, "
            "indicating excessive overlap: "
            f"{excessive_overlap}"
        )

    # --------------------------------------------------------
    # Verify visual elements survived.
    # --------------------------------------------------------

    input_visual_ids = {
        element.get("element_id")
        for element in semantic_elements
        if (
            element.get("modality")
            == "visual"
            and element.get("element_id") is not None
        )
    }

    chunk_visual_ids = set()

    for chunk in chunks:

        for element in chunk.get(
            "elements",
            [],
        ):

            if (
                element.get("modality")
                == "visual"
            ):

                element_id = element.get(
                    "element_id"
                )

                if element_id is not None:

                    chunk_visual_ids.add(
                        element_id
                    )

    missing_visuals = (
        input_visual_ids
        - chunk_visual_ids
    )

    if missing_visuals:

        errors.append(
            "Visual elements missing from chunks: "
            f"{sorted(missing_visuals)}"
        )

    # --------------------------------------------------------
    # Validate chunk metadata.
    # --------------------------------------------------------

    for chunk in chunks:

        chunk_id = chunk.get(
            "chunk_id"
        )

        elements = chunk.get(
            "elements",
            []
        )

        expected_modalities = [
            element.get(
                "modality",
                "unknown",
            )
            for element in elements
        ]

        actual_modalities = chunk.get(
            "modalities",
            []
        )

        if actual_modalities != expected_modalities:

            errors.append(
                f"{chunk_id}: modality metadata mismatch."
            )

        expected_cross_modal = (
            len(set(expected_modalities)) > 1
        )

        if chunk.get(
            "has_cross_modal"
        ) != expected_cross_modal:

            errors.append(
                f"{chunk_id}: has_cross_modal metadata mismatch."
            )

        # ----------------------------------------------------
        # Validate page range.
        # ----------------------------------------------------

        page_numbers = [
            element.get(
                "page_number"
            )
            for element in elements
            if isinstance(
                element.get(
                    "page_number"
                ),
                int,
            )
            and not isinstance(
                element.get(
                    "page_number"
                ),
                bool,
            )
        ]

        if page_numbers:

            expected_page_start = min(
                page_numbers
            )

            expected_page_end = max(
                page_numbers
            )

            if chunk.get(
                "page_start"
            ) != expected_page_start:

                errors.append(
                    f"{chunk_id}: page_start mismatch."
                )

            if chunk.get(
                "page_end"
            ) != expected_page_end:

                errors.append(
                    f"{chunk_id}: page_end mismatch."
                )

    # --------------------------------------------------------
    # Final validation result.
    # --------------------------------------------------------

    if errors:

        print(
            "Validation: FAILED"
        )

        print()

        for error in errors:

            print(
                f"  ERROR: {error}"
            )

        raise ValueError(
            "Phase 4 output validation failed."
        )

    print(
        f"Chunks validated : {len(chunks)}"
    )

    print(
        "Semantic coverage : 100%"
    )

    print(
        "Visual coverage   : 100%"
    )

    print(
        "Validation        : PASSED"
    )

    print(
        "=" * 80
    )

    return element_occurrences


# ============================================================
# OVERLAP STATISTICS
# ============================================================

def calculate_overlap_statistics(
    element_occurrences,
):
    """
    Calculate statistics describing element overlap.

    These statistics are useful for detecting excessive
    duplication without confusing legitimate overlap with
    missing coverage.
    """

    if not element_occurrences:

        return {
            "overlapped_elements": 0,
            "total_repeated_element_occurrences": 0,
            "max_element_occurrences": 0,
        }

    overlapped_elements = sum(
        1
        for count in element_occurrences.values()
        if count > 1
    )

    repeated_occurrences = sum(
        count - 1
        for count in element_occurrences.values()
        if count > 1
    )

    max_occurrences = max(
        element_occurrences.values()
    )

    return {
        "overlapped_elements": (
            overlapped_elements
        ),

        "total_repeated_element_occurrences": (
            repeated_occurrences
        ),

        "max_element_occurrences": (
            max_occurrences
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MULTIMODAL CHUNKING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load Phase 3.2 document.
    # --------------------------------------------------------

    document = load_document()

    # --------------------------------------------------------
    # Validate Phase 3.2 output.
    # --------------------------------------------------------

    validate_input_document(
        document
    )

    # --------------------------------------------------------
    # Semantic filtering.
    # --------------------------------------------------------

    (
        elements,
        total_elements,
        excluded_elements,
        excluded_by_modality,
    ) = flatten_semantic_elements(
        document
    )

    print()
    print(
        f"Total document elements : "
        f"{total_elements}"
    )

    print(
        f"Semantic input elements : "
        f"{len(elements)}"
    )

    print(
        f"Excluded elements       : "
        f"{excluded_elements}"
    )

    if excluded_by_modality:

        print()
        print(
            "Excluded by modality:"
        )

        for modality, count in sorted(
            excluded_by_modality.items()
        ):

            print(
                f"  {modality:<12} {count}"
            )

    # --------------------------------------------------------
    # RAG-Anything-inspired configuration.
    # --------------------------------------------------------

    chunker = OptimizedChunker(
        max_chunk_size=MAX_CHUNK_SIZE,
        overlap=OVERLAP,
        max_overlap_elements=MAX_OVERLAP_ELEMENTS,
    )

    # --------------------------------------------------------
    # Generate chunks.
    # --------------------------------------------------------

    chunks = chunker.chunk_document(
        elements
    )

    print()
    print(
        f"Generated chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Add stable chunk IDs.
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk["chunk_id"] = (
            f"chunk_{index:05d}"
        )

    # --------------------------------------------------------
    # Validate chunks.
    # --------------------------------------------------------

    element_occurrences = validate_chunks(
        chunks,
        elements,
    )

    # --------------------------------------------------------
    # Calculate overlap statistics.
    # --------------------------------------------------------

    overlap_statistics = (
        calculate_overlap_statistics(
            element_occurrences
        )
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    cross_modal_chunks = sum(
        1
        for chunk in chunks
        if chunk["has_cross_modal"]
    )

    oversized_chunks = sum(
        1
        for chunk in chunks
        if chunk.get("oversized", False)
    )

    modality_counts = {}

    for chunk in chunks:

        for modality in set(
            chunk["modalities"]
        ):

            modality_counts[
                modality
            ] = (
                modality_counts.get(
                    modality,
                    0,
                )
                + 1
            )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "document": {
            "source": document.get(
                "document",
                {},
            ).get(
                "source"
            ),

            "total_pages": document.get(
                "document",
                {},
            ).get(
                "total_pages"
            ),
        },

        "chunking": {
            "strategy": (
                "RAG-Anything-guide "
                "OptimizedChunker"
            ),

            "max_chunk_size": (
                MAX_CHUNK_SIZE
            ),

            "overlap": (
                OVERLAP
            ),

            "max_overlap_elements": (
                MAX_OVERLAP_ELEMENTS
            ),

            "total_document_elements": (
                total_elements
            ),

            "excluded_elements": (
                excluded_elements
            ),

            "excluded_by_modality": (
                excluded_by_modality
            ),

            "total_input_elements": (
                len(elements)
            ),

            "total_chunks": (
                len(chunks)
            ),

            "cross_modal_chunks": (
                cross_modal_chunks
            ),

            "oversized_chunks": (
                oversized_chunks
            ),

            "overlap_statistics": (
                overlap_statistics
            ),

            "chunk_modality_counts": (
                modality_counts
            ),
        },

        "chunks": chunks,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CHUNKING SUMMARY")
    print("=" * 80)

    print(
        f"Document elements  : "
        f"{total_elements}"
    )

    print(
        f"Semantic elements  : "
        f"{len(elements)}"
    )

    print(
        f"Excluded elements  : "
        f"{excluded_elements}"
    )

    print(
        f"Output chunks      : "
        f"{len(chunks)}"
    )

    print(
        f"Cross-modal chunks : "
        f"{cross_modal_chunks}"
    )

    print(
        f"Oversized chunks   : "
        f"{oversized_chunks}"
    )

    print()

    print(
        "Overlap statistics:"
    )

    print(
        f"  Overlapped elements       : "
        f"{overlap_statistics['overlapped_elements']}"
    )

    print(
        f"  Repeated occurrences     : "
        f"{overlap_statistics['total_repeated_element_occurrences']}"
    )

    print(
        f"  Max element occurrences  : "
        f"{overlap_statistics['max_element_occurrences']}"
    )

    print()

    print(
        "Excluded elements:"
    )

    if excluded_by_modality:

        for modality, count in sorted(
            excluded_by_modality.items()
        ):

            print(
                f"  {modality:<12} {count}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Chunk modalities:"
    )

    for modality, count in sorted(
        modality_counts.items()
    ):

        print(
            f"  {modality:<12} {count}"
        )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()