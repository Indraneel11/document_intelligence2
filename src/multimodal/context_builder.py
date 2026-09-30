from typing import List, Dict, Any, Optional


class ContextBuilder:
    """
    RAG-Anything-style context builder.

    Context is built around the target element using page boundaries.

    Default behavior:
        target page ± context_window

    The context can include:
        - text elements
        - headers/footers
        - captions and footnotes from multimodal elements

    This follows the page-based ContextExtractor approach used by
    RAG-Anything rather than introducing a separate relevance model.
    """

    TEXT_TYPES = {
        "paragraph",
        "title",
        "page_footnote",
    }

    HEADER_TYPES = {
        "page_header",
        "page_footer",
    }

    MULTIMODAL_TYPES = {
        "image",
        "chart",
        "table",
    }

    def __init__(
        self,
        context_window: int = 1,
        context_mode: str = "page",
        max_context_tokens: int = 2000,
        include_headers: bool = True,
        include_captions: bool = True,
        context_filter_content_types: Optional[List[str]] = None,
    ):
        self.context_window = context_window
        self.context_mode = context_mode
        self.max_context_tokens = max_context_tokens
        self.include_headers = include_headers
        self.include_captions = include_captions

        if context_filter_content_types is None:
            self.context_filter_content_types = ["text"]
        else:
            self.context_filter_content_types = (
                context_filter_content_types
            )

        if self.context_mode not in {"page"}:
            raise ValueError(
                f"Unsupported context mode: {self.context_mode}. "
                "Currently supported: 'page'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(
        self,
        pages: List[Dict[str, Any]],
        target_page_index: int,
        target_element_index: int,
    ) -> Dict[str, Any]:
        """
        Build context around a target multimodal element.

        For page mode:

            [target page - window]
                    ↓
               target page
                    ↓
            [target page + window]

        Page boundaries are always respected.
        """

        if not pages:
            return {
                "context_mode": self.context_mode,
                "context_window": self.context_window,
                "target_page": None,
                "target_element_id": None,
                "pages_included": [],
                "context": "",
            }

        if not (
            0 <= target_page_index < len(pages)
        ):
            raise IndexError(
                f"Invalid target_page_index: {target_page_index}"
            )

        target_page = pages[target_page_index]

        elements = target_page.get("elements", [])

        if not (
            0 <= target_element_index < len(elements)
        ):
            raise IndexError(
                f"Invalid target_element_index: "
                f"{target_element_index}"
            )

        target_element = elements[target_element_index]

        # --------------------------------------------------------------
        # Determine pages included in the context window.
        # --------------------------------------------------------------

        start_page = max(
            0,
            target_page_index - self.context_window,
        )

        end_page = min(
            len(pages),
            target_page_index + self.context_window + 1,
        )

        selected_pages = pages[
            start_page:end_page
        ]

        # --------------------------------------------------------------
        # Build context.
        # --------------------------------------------------------------

        context_parts = []

        for page in selected_pages:

            page_number = page.get(
                "page_number",
                start_page + 1,
            )

            page_content = self._extract_page_content(
                page
            )

            if not page_content:
                continue

            context_parts.append(
                f"[Page {page_number}]\n"
                + "\n".join(page_content)
            )

        context = "\n\n".join(context_parts)

        # --------------------------------------------------------------
        # Apply context token limit.
        # --------------------------------------------------------------

        context = self._truncate_context(
            context
        )

        return {
            "context_mode": self.context_mode,
            "context_window": self.context_window,
            "target_page": target_page.get(
                "page_number",
                target_page_index + 1,
            ),
            "target_element_id": target_element.get(
                "element_id"
            ),
            "pages_included": [
                page.get(
                    "page_number",
                    index + 1,
                )
                for index, page in enumerate(
                    selected_pages,
                    start=start_page,
                )
            ],
            "context": context,
        }

    # ------------------------------------------------------------------
    # Page content extraction
    # ------------------------------------------------------------------

    def _extract_page_content(
        self,
        page: Dict[str, Any],
    ) -> List[str]:
        """
        Extract context-bearing content from one page.

        Text is controlled by context_filter_content_types.

        Captions/footnotes from multimodal elements can optionally
        be included separately.
        """

        content = []

        for element in page.get("elements", []):

            element_type = element.get("type")

            # ----------------------------------------------------------
            # Text content
            # ----------------------------------------------------------

            if (
                "text"
                in self.context_filter_content_types
                and element_type in self.TEXT_TYPES
            ):
                text = element.get("text")

                if text and text.strip():
                    content.append(
                        text.strip()
                    )

                continue

            # ----------------------------------------------------------
            # Headers and footers
            # ----------------------------------------------------------

            if (
                self.include_headers
                and element_type in self.HEADER_TYPES
            ):
                text = element.get("text")

                if text and text.strip():
                    content.append(
                        text.strip()
                    )

                continue

            # ----------------------------------------------------------
            # Captions and footnotes
            # ----------------------------------------------------------

            if (
                self.include_captions
                and element_type in self.MULTIMODAL_TYPES
            ):
                caption = element.get("caption")
                footnote = element.get("footnote")

                if caption and caption.strip():
                    content.append(
                        f"[Caption] {caption.strip()}"
                    )

                if footnote and footnote.strip():
                    content.append(
                        f"[Footnote] {footnote.strip()}"
                    )

        return content

    # ------------------------------------------------------------------
    # Context truncation
    # ------------------------------------------------------------------

    def _truncate_context(
        self,
        context: str,
    ) -> str:
        """
        Temporary token-limit implementation.

        Approximation:
            1 token ≈ 4 characters

        Later this can be replaced with the same tokenizer strategy
        used by the selected LLM/provider.
        """

        if not context:
            return ""

        max_characters = (
            self.max_context_tokens * 4
        )

        if len(context) <= max_characters:
            return context

        truncated = context[
            :max_characters
        ]

        # Prefer ending at a sentence boundary.
        last_period = truncated.rfind(".")

        if (
            last_period >=
            int(max_characters * 0.7)
        ):
            truncated = (
                truncated[:last_period + 1]
            )

        return truncated.strip()