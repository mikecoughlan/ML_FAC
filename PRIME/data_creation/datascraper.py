import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Global configuration
TIMEOUT_CONFIG = (10, 30)  # (Connect timeout, Read timeout)
MAX_RETRIES = 3  # Number of times to retry a failed download
TARGET_EXTENSIONS = ".cdf"


def get_links(url):
    response = requests.get(url, timeout=TIMEOUT_CONFIG)
    soup = BeautifulSoup(response.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]


def is_folder(link):
    return link.endswith("/")


def download_file_with_retry(file_url, local_path):
    """
    Attempts to download a file, retrying up to MAX_RETRIES if a timeout occurs.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            head = requests.head(file_url, allow_redirects=True, timeout=TIMEOUT_CONFIG)

            if head.status_code not in (200, 206):
                print(f"[Attempt {attempt}/{MAX_RETRIES}] HEAD request failed (HTTP {head.status_code}). Trying GET headers...")
                head = requests.get(file_url, stream=True, timeout=TIMEOUT_CONFIG)

            remote_size = int(head.headers.get("Content-Length", 0))

            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                if remote_size > 0 and local_size == remote_size:
                    print(f"** File skipped (already complete): {file_url}")
                    return (0, file_url, None)
                else:
                    print(f"Size mismatch/incomplete file - re-downloading: {file_url} (Local: {local_size}, Remote: {remote_size})")
            else:
                print(f"Downloading new file: {file_url}")

            # Stream and write the file
            with requests.get(file_url, stream=True, timeout=TIMEOUT_CONFIG) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            print(f"✔ Downloaded: {file_url}")
            return (0, file_url, None)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"⚠ [Attempt {attempt}/{MAX_RETRIES}] Network timeout/error for {file_url}: {e}")
            if attempt < MAX_RETRIES:
                wait_time = attempt * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {MAX_RETRIES} attempts.")
                return (1, file_url, f"Timed out after {MAX_RETRIES} retries")

        except Exception as e:
            # Treat structural errors (like Permission Denied or HTTP 404) as fatal without retrying
            print(f"❌ Fatal error downloading {file_url}: {e}")
            return (1, file_url, str(e))


def crawl_and_download(base_url, current_dir):
    links = get_links(base_url)
    error_num = 0
    errors = []

    for link in links:
        full_url = urljoin(base_url, link)
        if is_folder(link) and link not in ("../", "./") and not link.startswith("/pub/data/"):
            sub_dir = os.path.join(current_dir, link.strip("/"))
            os.makedirs(sub_dir, exist_ok=True)
            crawl_and_download(full_url, sub_dir)

        elif link.endswith(TARGET_EXTENSIONS):
            local_file_path = os.path.join(current_dir, link)
            error = download_file_with_retry(full_url, local_file_path)
            if error[0] == 1:
                errors.append(error)
                error_num += 1

    if error_num > 0:
        print(f"\nFinished directory with {error_num} errors.")
        print("-" * 40)
        for error in errors:
            print(f"Error downloading {error[1]}: {error[2]}")
    else:
        print(f"\nFinished directory with no errors: {current_dir}")


if __name__ == "__main__":
    BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spe/l3/spe_sf0_pad/"
    DOWNLOAD_DIR = os.path.expanduser("data/prime/")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    crawl_and_download(BASE_URL, DOWNLOAD_DIR)
