from typing import Any, Dict, List, Tuple


class OptimizedChunker:
    """
    Structure-aware multimodal chunker inspired by the
    RAG-Anything implementation guide.

    Core strategy:
        - sequential element processing
        - modality-aware size estimation
        - maximum chunk budget
        - bounded element-level overlap
        - preservation of multimodal element boundaries
        - provenance metadata

    Default configuration follows the guide:
        max_chunk_size = 512
        overlap = 50

    Additional production safeguard:
        max_overlap_elements = 3

    Important:
        The size value is an estimated token-like budget.
        It is not an exact tokenizer measurement.
    """

    VISUAL_MODALITIES = {
        "visual",
        "image",
        "chart",
        "equation",
    }

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        max_overlap_elements: int = 3,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= max_chunk_size:
            raise ValueError(
                "overlap must be smaller than max_chunk_size."
            )

        if max_overlap_elements <= 0:
            raise ValueError(
                "max_overlap_elements must be greater than 0."
            )

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.max_overlap_elements = max_overlap_elements

    # ============================================================
    # MAIN CHUNKING
    # ============================================================

    def chunk_document(
        self,
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Chunk a sequential list of multimodal elements.

        Elements are never reordered.

        Multimodal elements are treated as atomic units and are
        never split internally.

        Returns:
            List of chunk dictionaries.
        """

        if not isinstance(elements, list):
            raise TypeError(
                "elements must be a list."
            )

        chunks: List[Dict[str, Any]] = []

        current_chunk: List[
            Dict[str, Any]
        ] = []

        current_size = 0

        for element in elements:

            if not isinstance(element, dict):
                continue

            element_size = self.estimate_size(
                element
            )

            # ----------------------------------------------------
            # If adding this element exceeds the budget,
            # finalize the current chunk first.
            # ----------------------------------------------------

            if (
                current_chunk
                and current_size + element_size
                > self.max_chunk_size
            ):

                previous_chunk = current_chunk

                chunks.append(
                    self.finalize_chunk(
                        previous_chunk
                    )
                )

                # ------------------------------------------------
                # Create bounded element-level overlap.
                # ------------------------------------------------

                current_chunk = (
                    self.create_overlap_chunk(
                        previous_chunk
                    )
                )

                current_size = sum(
                    self.estimate_size(item)
                    for item in current_chunk
                )

                # ------------------------------------------------
                # Extremely large elements can make even the
                # overlap + new element exceed the budget.
                #
                # In that situation, remove the overlap rather
                # than creating an unnecessarily oversized chunk.
                # ------------------------------------------------

                if (
                    current_chunk
                    and current_size + element_size
                    > self.max_chunk_size
                    and element_size
                    <= self.max_chunk_size
                ):
                    current_chunk = []
                    current_size = 0

            # ----------------------------------------------------
            # Add current element.
            #
            # An individual element may itself be larger than
            # max_chunk_size. It remains atomic rather than being
            # incorrectly split.
            # ----------------------------------------------------

            current_chunk.append(
                element
            )

            current_size += element_size

        # --------------------------------------------------------
        # Final chunk.
        # --------------------------------------------------------

        if current_chunk:

            chunks.append(
                self.finalize_chunk(
                    current_chunk
                )
            )

        return chunks

    # ============================================================
    # SIZE ESTIMATION
    # ============================================================

    def estimate_size(
        self,
        element: Dict[str, Any],
    ) -> int:
        """
        Estimate token-like size for a multimodal element.

        Strategy follows the RAG-Anything guide:

            text:
                approximately 1 token / 4 characters

            table:
                rows * columns * 2

            visual/equation:
                fixed cost of 100

            other:
                fixed cost of 50

        These values are intentionally heuristic.
        """

        if not isinstance(element, dict):
            return 1

        modality = element.get(
            "modality"
        )

        # --------------------------------------------------------
        # Text
        # --------------------------------------------------------

        if modality == "text":

            content = self._get_text_content(
                element
            )

            if not content:
                return 1

            return max(
                1,
                len(content) // 4,
            )

        # --------------------------------------------------------
        # Table
        # --------------------------------------------------------

        if modality == "table":

            rows, columns = (
                self._estimate_table_dimensions(
                    element
                )
            )

            return max(
                1,
                rows * columns * 2,
            )

        # --------------------------------------------------------
        # Visual / image / chart / equation
        # --------------------------------------------------------

        if modality in self.VISUAL_MODALITIES:

            return 100

        # --------------------------------------------------------
        # Everything else
        # --------------------------------------------------------

        return 50

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================

    @staticmethod
    def _get_text_content(
        element: Dict[str, Any],
    ) -> str:
        """
        Extract textual content from supported schemas.
        """

        text = element.get(
            "text"
        )

        if isinstance(
            text,
            str,
        ):
            return text.strip()

        content = element.get(
            "content"
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        return ""

    # ============================================================
    # TABLE SIZE ESTIMATION
    # ============================================================

    def _estimate_table_dimensions(
        self,
        element: Dict[str, Any],
    ) -> Tuple[int, int]:
        """
        Estimate table dimensions.

        Priority:
            1. Explicit element dimensions
            2. Metadata dimensions
            3. Structured content dimensions
            4. Row data
            5. HTML markup
            6. Conservative fallback
        """

        # --------------------------------------------------------
        # 1. Explicit dimensions
        # --------------------------------------------------------

        rows = element.get(
            "rows"
        )

        columns = element.get(
            "columns"
        )

        if self._valid_dimensions(
            rows,
            columns,
        ):
            return rows, columns

        # --------------------------------------------------------
        # 2. Metadata dimensions
        # --------------------------------------------------------

        metadata = element.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):

            rows = metadata.get(
                "rows"
            )

            columns = metadata.get(
                "columns"
            )

            if self._valid_dimensions(
                rows,
                columns,
            ):
                return rows, columns

        # --------------------------------------------------------
        # 3. Structured content dimensions
        # --------------------------------------------------------

        content = element.get(
            "content"
        )

        if isinstance(
            content,
            dict,
        ):

            content_rows = content.get(
                "rows"
            )

            content_columns = content.get(
                "columns"
            )

            if self._valid_dimensions(
                content_rows,
                content_columns,
            ):
                return (
                    content_rows,
                    content_columns,
                )

            # ----------------------------------------------------
            # 4. Row data
            # ----------------------------------------------------

            row_data = content.get(
                "data"
            )

            if isinstance(
                row_data,
                list,
            ) and row_data:

                inferred_rows = len(
                    row_data
                )

                inferred_columns = 1

                for row in row_data:

                    if isinstance(
                        row,
                        list,
                    ):

                        inferred_columns = max(
                            inferred_columns,
                            len(row),
                        )

                    elif isinstance(
                        row,
                        dict,
                    ):

                        inferred_columns = max(
                            inferred_columns,
                            len(row),
                        )

                return (
                    max(1, inferred_rows),
                    max(1, inferred_columns),
                )

        # --------------------------------------------------------
        # 5. HTML fallback
        # --------------------------------------------------------

        html = element.get(
            "text"
        )

        if not isinstance(
            html,
            str,
        ):

            html = element.get(
                "content"
            )

        if isinstance(
            html,
            str,
        ):

            lower = html.lower()

            row_count = lower.count(
                "<tr"
            )

            if row_count > 0:

                first_row_start = lower.find(
                    "<tr"
                )

                first_row_end = lower.find(
                    "</tr>",
                    first_row_start,
                )

                if (
                    first_row_start >= 0
                    and first_row_end >= 0
                ):

                    first_row = lower[
                        first_row_start:first_row_end
                    ]

                    column_count = (
                        first_row.count(
                            "<td"
                        )
                        + first_row.count(
                            "<th"
                        )
                    )

                    if column_count > 0:

                        return (
                            row_count,
                            column_count,
                        )

        # --------------------------------------------------------
        # 6. Conservative fallback
        # --------------------------------------------------------

        return 1, 1

    # ============================================================
    # TABLE VALIDATION
    # ============================================================

    @staticmethod
    def _valid_dimensions(
        rows: Any,
        columns: Any,
    ) -> bool:
        """
        Validate inferred table dimensions.
        """

        return (
            isinstance(rows, int)
            and isinstance(columns, int)
            and not isinstance(rows, bool)
            and not isinstance(columns, bool)
            and rows > 0
            and columns > 0
        )

    # ============================================================
    # OVERLAP
    # ============================================================

    def create_overlap_chunk(
        self,
        previous_chunk: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Create a bounded element-level overlap from the end
        of the previous chunk.

        Two limits are enforced:

            1. Token-like overlap budget <= self.overlap
            2. Number of repeated elements <=
               self.max_overlap_elements

        This prevents documents containing many tiny elements
        from producing excessive chunk duplication.

        Elements are never split.
        """

        overlap_elements: List[
            Dict[str, Any]
        ] = []

        overlap_size = 0

        for element in reversed(
            previous_chunk
        ):

            # ----------------------------------------------------
            # Stop once the maximum number of repeated elements
            # has been reached.
            # ----------------------------------------------------

            if (
                len(overlap_elements)
                >= self.max_overlap_elements
            ):
                break

            element_size = self.estimate_size(
                element
            )

            # ----------------------------------------------------
            # Preserve the configured overlap budget.
            # ----------------------------------------------------

            if (
                overlap_size + element_size
                <= self.overlap
            ):

                overlap_elements.insert(
                    0,
                    element,
                )

                overlap_size += (
                    element_size
                )

            else:

                break

        return overlap_elements

    # ============================================================
    # FINALIZE CHUNK
    # ============================================================

    def finalize_chunk(
        self,
        chunk_elements: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Convert elements into a retrieval-ready chunk.

        Original elements are preserved for:
            - provenance
            - graph construction
            - multimodal retrieval
            - later context expansion
        """

        if not chunk_elements:
            raise ValueError(
                "Cannot finalize an empty chunk."
            )

        modalities = [
            element.get(
                "modality",
                "unknown",
            )
            for element in chunk_elements
        ]

        unique_modalities = set(
            modalities
        )

        page_numbers = [
            element.get(
                "page_number"
            )
            for element in chunk_elements
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

        element_ids = [
            element.get(
                "element_id"
            )
            for element in chunk_elements
            if element.get(
                "element_id"
            ) is not None
        ]

        size = sum(
            self.estimate_size(
                element
            )
            for element in chunk_elements
        )

        oversized = (
            size > self.max_chunk_size
        )

        return {
            "elements": chunk_elements,

            "element_ids": element_ids,

            "modalities": modalities,

            "size": size,

            "has_cross_modal": (
                len(unique_modalities) > 1
            ),

            "page_start": (
                min(page_numbers)
                if page_numbers
                else None
            ),

            "page_end": (
                max(page_numbers)
                if page_numbers
                else None
            ),

            "oversized": oversized,
        }