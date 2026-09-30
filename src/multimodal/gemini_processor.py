import json
import mimetypes
import os
import re
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class GeminiDailyQuotaExceeded(Exception):
    """Raised when Gemini's daily project quota has been exhausted."""


class GeminiTemporaryRateLimit(Exception):
    """Raised when Gemini temporarily rate-limits a request."""


class GeminiVLMProcessor:
    def __init__(
        self,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in the environment."
            )

        self.client = genai.Client(api_key=api_key)

        self.model = model or os.getenv(
            "GEMINI_VLM_MODEL",
            "gemini-3.8-flash",
        )

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ============================================================
    # ERROR CLASSIFICATION
    # ============================================================

    def _extract_retry_delay(
        self,
        error: Exception,
    ) -> Optional[float]:
        """
        Extract a server-provided retry delay from Gemini error text.

        Examples:
            retryDelay: '50s'
            retryDelay: '4s'
        """

        error_text = str(error)

        match = re.search(
            r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
            error_text,
            re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

        return None

    def _is_daily_quota_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Detect errors indicating that Gemini's DAILY project quota
        has been exhausted.

        Important:
        Per-minute/RPM quota errors must NOT be classified as daily
        quota errors because those should be retried.
        """

        error_text = str(error).lower()

        daily_quota_patterns = [
            "generate_requests_per_day",
            "generaterequestsperday",
            "requestsperdayperprojectpermodel",
            "perdayperprojectpermodel",
            "daily quota",
            "quota exceeded for today",
            "daily limit",
        ]

        return any(
            pattern in error_text
            for pattern in daily_quota_patterns
        )

    def _is_rate_limit_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Detect HTTP 429 / temporary rate-limit errors.
        """

        error_text = str(error).lower()

        return (
            "429" in error_text
            or "resource_exhausted" in error_text
            or "rate limit" in error_text
            or "ratelimit" in error_text
        )

    def _is_temporary_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Detect temporary server/network errors that are safe to retry.
        """

        error_text = str(error).lower()

        return (
            "503" in error_text
            or "service unavailable" in error_text
            or "unavailable" in error_text
            or "timeout" in error_text
            or "timed out" in error_text
        )

    # ============================================================
    # RESPONSE VALIDATION
    # ============================================================

    def _validate_analysis(
        self,
        analysis: Dict[str, Any],
    ) -> None:
        """
        Validate Gemini's structured visual-analysis response.
        """

        required_fields = {
            "visual_description": str,
            "key_information": list,
            "trends_and_relationships": list,
            "main_takeaway": str,
            "important_entities": list,
            "confidence": str,
        }

        for field, expected_type in required_fields.items():

            if field not in analysis:
                raise ValueError(
                    f"Missing required field: {field}"
                )

            if not isinstance(
                analysis[field],
                expected_type,
            ):
                raise ValueError(
                    f"Invalid type for '{field}'. "
                    f"Expected {expected_type.__name__}."
                )

        if analysis["confidence"] not in {
            "High",
            "Medium",
            "Low",
        }:
            raise ValueError(
                "confidence must be one of: High, Medium, Low"
            )

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        caption: Optional[str],
        footnote: Optional[str],
        context: str,
    ) -> str:

        return f"""
You are analyzing a visual element extracted from a complex PDF.

The image/chart itself is the PRIMARY SOURCE OF TRUTH.

Use the surrounding document context only as supporting information.

Do not invent:
- values
- labels
- trends
- entities
- relationships
- conclusions

Only report information that can reasonably be supported by the
visual and the supplied document context.

Analyze the visual carefully.

Return ONLY valid JSON.

Do not use:
- markdown
- code fences
- explanations outside JSON
- additional fields

Required JSON schema:

{{
  "visual_description": "Detailed description of what is visibly present.",
  "key_information": [
    "Important visible facts, values, labels, categories, or observations."
  ],
  "trends_and_relationships": [
    "Important relationships, comparisons, patterns, or trends."
  ],
  "main_takeaway": "The most important conclusion from the visual.",
  "important_entities": [
    "Important entities, categories, organizations, concepts, or labels."
  ],
  "confidence": "High"
}}

Caption:
{caption or "None"}

Footnote:
{footnote or "None"}

Surrounding document context:
{context or "None"}
"""

    # ============================================================
    # VISUAL ANALYSIS
    # ============================================================

    def analyze_visual(
        self,
        image_path: str,
        caption: Optional[str] = None,
        footnote: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:

        # --------------------------------------------------------
        # Validate image path
        # --------------------------------------------------------

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Visual file not found: {image_path}"
            )

        # --------------------------------------------------------
        # Read image
        # --------------------------------------------------------

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)

        if not mime_type:
            mime_type = "image/png"

        # --------------------------------------------------------
        # Build prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            caption=caption,
            footnote=footnote,
            context=context,
        )

        last_error = None

        # --------------------------------------------------------
        # Retry loop
        # --------------------------------------------------------

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            print(
                f"Gemini request "
                f"{attempt}/{self.max_retries}..."
            )

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                # ------------------------------------------------
                # Empty response
                # ------------------------------------------------

                if not response.text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                # ------------------------------------------------
                # Parse JSON
                # ------------------------------------------------

                analysis = json.loads(
                    response.text
                )

                # ------------------------------------------------
                # Validate schema
                # ------------------------------------------------

                self._validate_analysis(
                    analysis
                )

                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                return {
                    "model": self.model,
                    "analysis": analysis,
                }

            except Exception as exc:

                last_error = exc

                print(
                    f"Gemini request failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                # =================================================
                # DAILY QUOTA
                # =================================================

                if self._is_daily_quota_error(exc):

                    print(
                        "\n"
                        "==================================================\n"
                        "GEMINI DAILY QUOTA EXHAUSTED\n"
                        "==================================================\n"
                        "Stopping immediately.\n"
                        "The enrichment runner will save its checkpoint.\n"
                        "Do not retry this request.\n"
                        "==================================================\n"
                    )

                    raise GeminiDailyQuotaExceeded(
                        "Gemini daily quota has been exhausted."
                    ) from exc

                # =================================================
                # TEMPORARY RATE LIMIT / 429
                # =================================================

                if self._is_rate_limit_error(exc):

                    retry_delay = (
                        self._extract_retry_delay(exc)
                    )

                    if retry_delay is None:
                        retry_delay = (
                            self.retry_delay * attempt
                        )

                    print(
                        f"Temporary Gemini rate limit. "
                        f"Waiting {retry_delay:.1f} seconds..."
                    )

                    if attempt < self.max_retries:

                        time.sleep(
                            retry_delay
                        )

                        continue

                    break

                # =================================================
                # TEMPORARY SERVER / NETWORK ERROR
                # =================================================

                if self._is_temporary_error(exc):

                    retry_delay = (
                        self.retry_delay * attempt
                    )

                    print(
                        f"Temporary Gemini error. "
                        f"Retrying in {retry_delay:.1f} seconds..."
                    )

                    if attempt < self.max_retries:

                        time.sleep(
                            retry_delay
                        )

                        continue

                    break

                # =================================================
                # NON-RETRYABLE ERROR
                # =================================================

                print(
                    "Non-retryable Gemini error."
                )

                break

        # --------------------------------------------------------
        # All attempts failed
        # --------------------------------------------------------

        raise RuntimeError(
            f"Gemini visual analysis failed after "
            f"{self.max_retries} attempts."
        ) from last_error