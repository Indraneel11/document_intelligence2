import os
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_TOKEN = os.getenv("MINERU_API_TOKEN")

if not API_TOKEN:
    raise RuntimeError(
        "MINERU_API_TOKEN is missing from .env"
    )

BASE_URL = "https://mineru.net"

# Network settings
API_TIMEOUT = 60
UPLOAD_CONNECT_TIMEOUT = 30
UPLOAD_READ_TIMEOUT = 600
DOWNLOAD_TIMEOUT = 300

# Retry settings
MAX_UPLOAD_RETRIES = 3
RETRY_DELAY = 5


# ============================================================
# HTTP helpers
# ============================================================

def get_headers():
    """
    Headers required for MinerU API requests.

    These headers are used for requests to mineru.net.
    They are NOT used for the presigned OSS upload.
    """
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }


# ============================================================
# Step 1: Create upload task
# ============================================================

def create_upload_task(pdf_path: Path):
    """
    Ask MinerU for a temporary presigned upload URL.
    """

    url = f"{BASE_URL}/api/v4/file-urls/batch"

    payload = {
        "files": [
            {
                "name": pdf_path.name
            }
        ],
        "model_version": "vlm"
    }

    print("Creating MinerU task...")

    response = requests.post(
        url,
        headers=get_headers(),
        json=payload,
        timeout=API_TIMEOUT,
    )

    response.raise_for_status()

    result = response.json()

    if result.get("code") != 0:
        raise RuntimeError(
            f"MinerU error while creating upload task: {result}"
        )

    data = result["data"]

    batch_id = data["batch_id"]
    upload_url = data["file_urls"][0]

    return batch_id, upload_url


# ============================================================
# Step 2: Upload PDF
# ============================================================

def upload_pdf(pdf_path: Path, upload_url: str):
    """
    Upload the PDF to the temporary MinerU OSS URL.

    IMPORTANT:
    MinerU's presigned upload must not receive an
    Authorization header or a custom Content-Type header.
    """

    file_size = pdf_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    upload_host = requests.utils.urlparse(
        upload_url
    ).netloc

    print(
        f"Uploading PDF: {pdf_path.name} "
        f"({file_size_mb:.2f} MB)"
    )

    print(
        f"Upload host: {upload_host}"
    )

    print(
        f"File size: {file_size} bytes"
    )

    for attempt in range(
        1,
        MAX_UPLOAD_RETRIES + 1
    ):

        print(
            f"Upload attempt "
            f"{attempt}/{MAX_UPLOAD_RETRIES}..."
        )

        try:

            # Open the file for every attempt so that
            # every retry starts from byte zero.
            with open(
                pdf_path,
                "rb"
            ) as file:

                response = requests.put(
                    upload_url,
                    data=file,

                    # Explicitly tell OSS the exact
                    # size of the request body.
                    #
                    # Do NOT add Authorization or
                    # Content-Type here because this
                    # is a presigned upload URL.
                    headers={
                        "Content-Length": str(
                            file_size
                        )
                    },

                    timeout=(
                        UPLOAD_CONNECT_TIMEOUT,
                        UPLOAD_READ_TIMEOUT,
                    ),
                )

            response.raise_for_status()

            print(
                "Upload complete."
            )

            return

        except requests.exceptions.RequestException as exc:

            print(
                f"Upload attempt {attempt} failed: "
                f"{exc}"
            )

            if attempt == MAX_UPLOAD_RETRIES:
                raise RuntimeError(
                    "MinerU PDF upload failed after "
                    f"{MAX_UPLOAD_RETRIES} attempts."
                ) from exc

            wait_time = (
                RETRY_DELAY * attempt
            )

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )


# ============================================================
# Step 3: Wait for MinerU processing
# ============================================================

def wait_for_result(batch_id: str):
    """
    Poll MinerU until the parsing task finishes.
    """

    url = (
        f"{BASE_URL}/api/v4/extract-results/"
        f"batch/{batch_id}"
    )

    while True:

        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {API_TOKEN}"
            },
            timeout=API_TIMEOUT,
        )

        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            raise RuntimeError(
                "MinerU error while checking result: "
                f"{result}"
            )

        results = result["data"].get(
            "extract_result",
            []
        )

        if not results:
            print(
                "MinerU has not returned a result yet..."
            )

            time.sleep(5)

            continue

        file_result = results[0]

        state = file_result.get(
            "state"
        )

        print(
            f"MinerU status: {state}"
        )

        if state == "done":
            return file_result

        if state == "failed":
            raise RuntimeError(
                "MinerU parsing failed: "
                f"{file_result.get('err_msg')}"
            )

        time.sleep(5)


# ============================================================
# Step 4: Download and extract result
# ============================================================

def download_and_extract(
    result,
    output_dir: Path
):
    """
    Download the MinerU result ZIP and extract it.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_url = result[
        "full_zip_url"
    ]

    zip_path = (
        output_dir /
        "mineru_result.zip"
    )

    print(
        "Downloading MinerU result..."
    )

    response = requests.get(
        zip_url,
        timeout=DOWNLOAD_TIMEOUT,
    )

    response.raise_for_status()

    zip_path.write_bytes(
        response.content
    )

    print(
        "Extracting result..."
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        archive.extractall(
            output_dir
        )

    print(
        f"Result saved to: {output_dir}"
    )


# ============================================================
# Main pipeline
# ============================================================

def main():

    pdf_path = Path(
        "data/input/"
        "COP26-ClimateBrief-AdaptationResilience-Final-2610.pdf"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # --------------------------------------------------------
    # 1. Create MinerU upload task
    # --------------------------------------------------------

    batch_id, upload_url = (
        create_upload_task(
            pdf_path
        )
    )

    print(
        f"Batch ID: {batch_id}"
    )

    # --------------------------------------------------------
    # 2. Upload PDF to MinerU
    # --------------------------------------------------------

    upload_pdf(
        pdf_path,
        upload_url
    )

    # --------------------------------------------------------
    # 3. Wait for MinerU processing
    # --------------------------------------------------------

    print(
        "Waiting for MinerU..."
    )

    result = wait_for_result(
        batch_id
    )

    # --------------------------------------------------------
    # 4. Download and extract results
    # --------------------------------------------------------

    output_dir = Path(
        "data/output"
    )

    download_and_extract(
        result,
        output_dir
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print(
        "\nPhase 1 completed successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()