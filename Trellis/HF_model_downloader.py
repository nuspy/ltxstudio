from huggingface_hub import HfApi, list_repo_files, hf_hub_url, get_hf_file_metadata
import os
import argparse
import requests
import threading
import time
import sys
import hashlib
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
import shutil
import json

# Configuration - Download TRELLIS.2 models to Trellis_2_3D_Generator/models
# This matches TRELLIS.2 runtime local-loading logic:
#   TRELLIS_MODELS_DIR/<org>--<repo>/...
TARGET_DIR = os.path.join("Trellis_2_3D_Generator", "models")

DOWNLOAD_STATUS_DOWNLOADED = "downloaded"
DOWNLOAD_STATUS_SKIPPED = "skipped"
DOWNLOAD_STATUS_FAILED = "failed"

# Model configurations for TRELLIS.2 runtime.
# NOTE: Some dependency repos (e.g. DINOv3 model) are discovered automatically
#       by parsing downloaded pipeline config JSONs.
MODEL_CONFIGS = {
    # Bundled/mirrored model pack hosted under a single HF repo subfolder.
    # This downloads everything under:
    #   https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main/Trellv2
    # into:
    #   Trellis_2_3D_Generator/models/
    # (i.e. strips the leading "Trellv2/" prefix so folders land directly under models)
    "trellv2_bundle": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "subdir": "Trellv2",
        "files": None,  # None = download all files under subdir
        "name": "Trellv2 bundle (MonsterMMORPG/Wan_GGUF)",
        "description": "All required TRELLIS.2 offline model folders mirrored under a single Hugging Face repo subfolder"
    },
    # Additional bundle for UltraShape models.
    # This downloads everything under:
    #   https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main/UltraShape_For_Trellis2
    # into:
    #   Trellis_2_3D_Generator/models/UltraShape/
    "ultrashape_bundle": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "subdir": "UltraShape_For_Trellis2",
        "local_target_subdir": "UltraShape",
        "files": None,  # None = download all files under subdir
        "name": "UltraShape bundle (MonsterMMORPG/Wan_GGUF)",
        "description": "UltraShape model files mirrored under a single Hugging Face repo subfolder"
    },
    # DINOv2 large image encoder bundle.
    # This downloads everything under:
    #   https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main/dinov2-large
    # into:
    #   Trellis_2_3D_Generator/models/dinov2-large/
    "dinov2_large_bundle": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "subdir": "dinov2-large",
        "local_target_subdir": "dinov2-large",
        "files": None,  # None = download all files under subdir
        "name": "DINOv2-large bundle (MonsterMMORPG/Wan_GGUF)",
        "description": "DINOv2-large model files mirrored under a single Hugging Face repo subfolder"
    },
    "trellis2_4b": {
        "repo_id": "microsoft/TRELLIS.2-4B",
        "files": None,  # None = download all repo files
        "name": "TRELLIS.2-4B",
        "description": "Main TRELLIS.2 pipeline + checkpoints (image-to-3D and texturing)"
    },
    "birefnet": {
        "repo_id": "ZhengPeng7/BiRefNet",
        "files": None,  # download all repo files (trust_remote_code model)
        "name": "BiRefNet",
        "description": "Background removal model used by TRELLIS.2 preprocessing"
    },
}

DOWNLOAD_CONFIG = {
    "num_connections": 16,      # Fixed 16 connections
    "chunk_size": 10485760,     # 10MB buffer for streaming
    "max_retries": 5,           # Reasonable retry count
    "retry_delay": 2,           # Base delay between retries
    "max_retry_delay": 30,      # Cap exponential backoff
    "timeout": 300,             # 5 minute timeout per request
}

class RobustDownloader:
    def __init__(self, config: Dict):
        self.config = config
        # Create session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            # Avoid HTTP-level compression which can hide Content-Length and break Range math.
            # We want raw bytes on disk to match HF hashes (for LFS) and to keep resume reliable.
            'Accept-Encoding': 'identity',
        })
        # Optional Hugging Face auth token (for gated/private repos)
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        self.hf_token = hf_token
        if hf_token:
            self.session.headers.update({"Authorization": f"Bearer {hf_token}"})
        # Cache files in the same directory where script runs
        script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
        
        # SHA256 cache file
        self.sha_cache_file = os.path.join(script_dir, "sha256_cache.json")
        self.sha_cache = self.load_sha_cache()
        
        # Verified files cache to avoid re-verification
        self.verified_cache_file = os.path.join(script_dir, "verified_files_cache.json")
        self.verified_cache = self.load_verified_cache()
        
        # Debug: Print cache file locations
        print(f"[DEBUG] Cache files will be saved in: {script_dir}")
        print(f"[DEBUG] SHA256 cache: {self.sha_cache_file}")
        print(f"[DEBUG] Verified cache: {self.verified_cache_file}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")

        # Console/progress management (single-line, cross-platform)
        self._progress_lock = threading.Lock()
        self._active_progress = False
        self._last_progress_len = 0

        # HF metadata caches (per-run, in-memory)
        # repo_id -> { rfilename -> {"size": Optional[int], "sha256": Optional[str]} }
        self._repo_files_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Cache Range support per URL (avoid repeated probes on large files)
        self._range_support_cache: Dict[str, bool] = {}

    # ------------------------ Hugging Face metadata helpers ------------------------

    def _hf_api(self) -> HfApi:
        """Create an HfApi client using the same token strategy as requests session."""
        if not self.hf_token:
            return HfApi()
        try:
            return HfApi(token=self.hf_token)
        except TypeError:
            # Back-compat for older huggingface_hub versions
            return HfApi()

    def _hf_file_metadata(self, url: str):
        """Call get_hf_file_metadata with optional token (backward compatible)."""
        if not self.hf_token:
            return get_hf_file_metadata(url)
        try:
            return get_hf_file_metadata(url, token=self.hf_token)
        except TypeError:
            # Back-compat for older huggingface_hub versions
            return get_hf_file_metadata(url)

    def prefetch_repo_metadata(self, repo_id: str):
        """
        Prefetch repo file metadata (size + LFS sha256 when available) via HF API.
        This is much more reliable than inferring size from HTTP headers, and avoids
        per-file API calls.
        """
        if repo_id in self._repo_files_meta:
            return

        try:
            api = self._hf_api()
            info = api.model_info(repo_id, files_metadata=True)
            meta: Dict[str, Dict[str, Any]] = {}
            for file_info in getattr(info, "siblings", []) or []:
                rfilename = getattr(file_info, "rfilename", None)
                if not rfilename:
                    continue

                size = getattr(file_info, "size", None)
                sha256: Optional[str] = None

                lfs = getattr(file_info, "lfs", None)
                if lfs:
                    if isinstance(lfs, dict):
                        sha256 = lfs.get("sha256") or lfs.get("oid")
                        if size is None:
                            size = lfs.get("size")
                    else:
                        sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
                        if size is None:
                            size = getattr(lfs, "size", None)

                # Normalize size to int where possible
                if isinstance(size, str) and size.isdigit():
                    size = int(size)
                elif isinstance(size, float):
                    size = int(size)

                meta[rfilename] = {"size": size, "sha256": sha256}

            self._repo_files_meta[repo_id] = meta
        except Exception as e:
            # Don't fail downloads if metadata can't be prefetched; fall back to HTTP/streaming.
            self._repo_files_meta[repo_id] = {}
            self.log(f"Warning: Could not prefetch metadata for {repo_id}: {e}")

    def _get_prefetched_meta(self, repo_id: str, filename: str) -> Dict[str, Any]:
        if repo_id not in self._repo_files_meta:
            self.prefetch_repo_metadata(repo_id)
        return self._repo_files_meta.get(repo_id, {}).get(filename, {}) or {}

    def supports_range(self, url: str) -> bool:
        """Return True if the remote URL honors Range requests (needed for parallel download)."""
        if url in self._range_support_cache:
            return self._range_support_cache[url]
        ok = False
        try:
            headers = {"Range": "bytes=0-0", "Accept-Encoding": "identity"}
            resp = self.session.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
            try:
                ok = resp.status_code == 206 and bool(resp.headers.get("content-range"))
            finally:
                resp.close()
        except Exception:
            ok = False
        self._range_support_cache[url] = ok
        return ok

    # ------------- Console helpers to ensure single-line progress -------------

    def _get_terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except Exception:
            return 100

    def _shorten_middle(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        left = (max_len - 3) // 2
        right = (max_len - 3) - left
        return text[:left] + "..." + text[-right:]

    def _clear_progress_line_locked(self):
        # Clear current line safely
        width = self._get_terminal_width()
        clear_len = max(self._last_progress_len, width)
        sys.stdout.write("\r" + " " * clear_len + "\r")
        sys.stdout.flush()
        self._last_progress_len = 0
        self._active_progress = False

    def clear_progress_line(self):
        with self._progress_lock:
            if self._active_progress:
                self._clear_progress_line_locked()

    def show_progress_line(self, text: str):
        # Ensure we never wrap: truncate to terminal width - 1
        with self._progress_lock:
            width = self._get_terminal_width()
            max_len = max(1, width - 1)
            if len(text) > max_len:
                text = text[:max_len]

            # Write text and erase any remainder from previous longer line
            sys.stdout.write("\r" + text)
            extra_spaces = max(0, max(self._last_progress_len - len(text), 0))
            if extra_spaces:
                sys.stdout.write(" " * extra_spaces)
                sys.stdout.write("\r" + text)
            sys.stdout.flush()
            self._last_progress_len = len(text)
            self._active_progress = True

    def finalize_progress_line(self, final_text: Optional[str] = None):
        with self._progress_lock:
            if final_text is not None:
                width = self._get_terminal_width()
                max_len = max(1, width - 1)
                if len(final_text) > max_len:
                    final_text = final_text[:max_len]
                # Write final text, clear remainder, then newline
                sys.stdout.write("\r" + final_text)
                extra_spaces = max(0, max(self._last_progress_len - len(final_text), 0))
                if extra_spaces:
                    sys.stdout.write(" " * extra_spaces)
                    sys.stdout.write("\r" + final_text)
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                if self._active_progress:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            self._last_progress_len = 0
            self._active_progress = False

    def log(self, msg: str):
        # Print a normal log line, ensuring it doesn't collide with progress line
        with self._progress_lock:
            if self._active_progress:
                self._clear_progress_line_locked()
            print(msg, flush=True)

    # --------------------------- SHA256 cache utils ---------------------------

    def load_sha_cache(self) -> Dict[str, str]:
        """Load SHA256 cache from file"""
        if os.path.exists(self.sha_cache_file):
            try:
                with open(self.sha_cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_sha_cache(self):
        """Save SHA256 cache to file"""
        try:
            with open(self.sha_cache_file, 'w') as f:
                json.dump(self.sha_cache, f, indent=2)
        except Exception as e:
            self.log(f"Warning: Could not save SHA cache: {e}")

    def load_verified_cache(self) -> Dict[str, Dict]:
        """Load verified files cache from file"""
        if os.path.exists(self.verified_cache_file):
            try:
                with open(self.verified_cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_verified_cache(self):
        """Save verified files cache to file"""
        try:
            with open(self.verified_cache_file, 'w') as f:
                json.dump(self.verified_cache, f, indent=2)
            print(f"[DEBUG] Verified cache saved to: {self.verified_cache_file}")
            print(f"[DEBUG] Cache contains {len(self.verified_cache)} entries")
        except Exception as e:
            self.log(f"Warning: Could not save verified cache: {e}")

    def is_file_verified(self, repo_id: str, filename: str, filepath: str, expected_sha: str) -> bool:
        """Check if file has already been verified and hasn't changed"""
        if not expected_sha:
            return False
            
        cache_key = f"{repo_id}/{filename}"
        
        if cache_key not in self.verified_cache:
            return False
            
        cached_info = self.verified_cache[cache_key]
        
        # Check if file exists and has same size and modification time
        if not os.path.exists(filepath):
            return False
            
        current_size = os.path.getsize(filepath)
        current_mtime = os.path.getmtime(filepath)
        
        return (cached_info.get('sha256') == expected_sha and
                cached_info.get('size') == current_size and
                abs(cached_info.get('mtime', 0) - current_mtime) < 1.0)  # Allow 1 second tolerance

    def mark_file_verified(self, repo_id: str, filename: str, filepath: str, sha256: str):
        """Mark file as verified in cache"""
        cache_key = f"{repo_id}/{filename}"
        
        print(f"[DEBUG] Marking file as verified: {filename}")
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            file_mtime = os.path.getmtime(filepath)
            
            self.verified_cache[cache_key] = {
                'sha256': sha256,
                'size': file_size,
                'mtime': file_mtime,
                'verified_at': time.time()
            }
            
            print(f"[DEBUG] Added to cache: {cache_key}")
            self.save_verified_cache()
        else:
            print(f"[DEBUG] File does not exist, cannot mark as verified: {filepath}")

    def get_file_sha256(self, repo_id: str, filename: str) -> Optional[str]:
        """Get SHA256 hash for a file from Hugging Face"""
        cache_key = f"{repo_id}/{filename}"

        # Check cache first
        if cache_key in self.sha_cache:
            return self.sha_cache[cache_key]

        try:
            # Method 0: Prefer prefetched repo metadata (fast + reliable)
            meta = self._get_prefetched_meta(repo_id, filename)
            sha256 = meta.get("sha256")
            if isinstance(sha256, str) and len(sha256) == 64:
                self.sha_cache[cache_key] = sha256
                self.save_sha_cache()
                return sha256

            # Method 1: Try to get from file metadata
            url = hf_hub_url(repo_id, filename)
            metadata = self._hf_file_metadata(url)

            # The etag contains the SHA256 hash
            if hasattr(metadata, 'etag'):
                # Remove quotes and 'W/' prefix if present
                sha256_from_etag = metadata.etag.replace('"', '').replace('W/', '')
                if len(sha256_from_etag) == 64:  # Valid SHA256 length
                    self.sha_cache[cache_key] = sha256_from_etag
                    self.save_sha_cache()
                    return sha256_from_etag

            # Method 2: Try API approach
            # (Kept as fallback for environments where prefetch didn't run)
            api = self._hf_api()
            model_info = api.model_info(repo_id, files_metadata=True)
            for file_info in getattr(model_info, "siblings", []) or []:
                if getattr(file_info, "rfilename", None) != filename:
                    continue
                lfs = getattr(file_info, "lfs", None)
                if not lfs:
                    break
                if isinstance(lfs, dict):
                    sha256 = lfs.get("sha256") or lfs.get("oid")
                else:
                    sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
                if isinstance(sha256, str) and len(sha256) == 64:
                    self.sha_cache[cache_key] = sha256
                    self.save_sha_cache()
                    return sha256

            return None

        except Exception as e:
            self.log(f"Warning: Could not get SHA256 for {filename}: {e}")
            return None

    # ----------------------- Formatting helpers (unchanged) -------------------

    def format_bytes(self, bytes_val):
        """Format bytes to human readable format"""
        if bytes_val < 0:
            return "0 B"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"

    def format_time(self, seconds):
        """Format seconds to human readable format"""
        if seconds < 0:
            return "0s"

        seconds = int(seconds)  # Convert to integer to avoid floating point issues

        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            mins = seconds // 60
            secs = seconds % 60
            if secs == 0:
                return f"{mins}m"
            else:
                return f"{mins}m {secs}s"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            if mins == 0:
                return f"{hours}h"
            else:
                return f"{hours}h {mins}m"

    # ------------------------- Progress line rendering ------------------------

    def print_progress(self, current, total, start_time, filename, speed_bytes_per_sec=None):
        """Render a single-line, non-wrapping progress line for the active file"""
        if total <= 0:
            return

        elapsed = max(0.001, time.time() - start_time)
        percent = min(100.0, (current / total * 100))

        # Calculate speed if not provided
        if speed_bytes_per_sec is None:
            speed_bytes_per_sec = current / elapsed if elapsed > 0 else 0.0

        # Calculate ETA
        if speed_bytes_per_sec > 0 and total > current:
            eta = (total - current) / speed_bytes_per_sec
            eta_str = self.format_time(eta)
        else:
            eta_str = "Complete" if current >= total else "Unknown"

        # Build the line
        # We will ensure the final line does not wrap by truncating to terminal width.
        # Progress bar target length (will be truncated by terminal width logic if needed)
        bar_length = 40
        filled = int(percent * bar_length / 100)
        bar = "=" * max(0, min(filled, bar_length)) + "-" * max(0, bar_length - filled)

        speed_str = self.format_bytes(speed_bytes_per_sec) + "/s" if speed_bytes_per_sec else "0 B/s"
        line = (
            f"{filename}: [{bar}] {percent:.1f}% "
            f"({self.format_bytes(current)}/{self.format_bytes(total)}) "
            f"{speed_str} ETA: {eta_str}"
        )

        # Show single-line progress (truncated to fit, padded/cleared safely)
        self.show_progress_line(line)

    # ------------------------------ Networking --------------------------------

    def get_file_url(self, repo_id: str, filename: str) -> str:
        """Get direct download URL for HuggingFace file"""
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    def get_file_size(self, url: str, repo_id: Optional[str] = None, filename: Optional[str] = None) -> Optional[int]:
        """Get file size from remote server"""
        try:
            # Method 0: Prefer prefetched repo metadata (HF API lists sizes for all files)
            if repo_id and filename:
                meta = self._get_prefetched_meta(repo_id, filename)
                size = meta.get("size")
                if isinstance(size, int) and size >= 0:
                    return size
                if isinstance(size, str) and size.isdigit():
                    return int(size)

            # Method 1: HF file metadata (often reliable even when HTTP headers are tricky)
            if repo_id and filename:
                try:
                    hf_url = hf_hub_url(repo_id, filename)
                    metadata = self._hf_file_metadata(hf_url)
                    meta_size = getattr(metadata, "size", None)
                    if isinstance(meta_size, int) and meta_size >= 0:
                        return meta_size
                    if isinstance(meta_size, str) and meta_size.isdigit():
                        return int(meta_size)
                except Exception:
                    pass

            response = self.session.head(url, timeout=30, allow_redirects=True)
            if response.status_code in (200, 206):
                content_length = response.headers.get('content-length')
                if content_length and str(content_length).isdigit():
                    return int(content_length)
                # Sometimes Content-Range is included even on HEAD
                content_range = response.headers.get('content-range')
                if content_range and '/' in content_range:
                    total_size = content_range.split('/')[-1]
                    if total_size.isdigit():
                        return int(total_size)

            # Fallback to range request
            headers = {'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'}
            response = self.session.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
            try:
                if response.status_code == 206:
                    content_range = response.headers.get('content-range')
                    if content_range and '/' in content_range:
                        total_size = content_range.split('/')[-1]
                        if total_size.isdigit():
                            return int(total_size)
                elif response.status_code == 200:
                    # Server ignored Range but still may provide Content-Length for the full file
                    content_length = response.headers.get('content-length')
                    if content_length and str(content_length).isdigit():
                        return int(content_length)
            finally:
                response.close()

            # Final fallback: check if file is likely small and compressed
            # Look for gzip encoding or chunked transfer which indicates small files
            # Note: many servers use brotli (br) or other encodings; any content-encoding
            # without a usable Content-Length means we must stream the file to determine size.
            content_encoding = (response.headers.get('content-encoding') or '').lower()
            transfer_encoding = (response.headers.get('transfer-encoding') or '').lower()
            if (content_encoding not in ("", "identity") or transfer_encoding == 'chunked'):
                # For compressed/chunked files, we'll return a special marker
                # to indicate we need to download the full file to get size
                return -1

            return None
        except Exception as e:
            self.log(f"Warning: Could not get file size: {e}")
            return None

    # ----------------------- Chunk I/O and verification -----------------------

    def verify_file_sha256(self, filepath: str, expected_sha: str, filename: str = "") -> bool:
        """Verify file SHA256 hash with single-line progress"""
        if not expected_sha:
            self.log(f"[WARNING] No SHA256 available for verification")
            return True  # Can't verify, assume OK

        display_name = filename or os.path.basename(filepath) or filepath
        self.log(f"[VERIFYING] Computing SHA256 for {display_name}...")

        try:
            sha256_hash = hashlib.sha256()
            file_size = os.path.getsize(filepath)
            bytes_read = 0
            start_time = time.time()
            last_update = 0.0

            # Use 8MB chunks for better performance
            chunk_size = 8 * 1024 * 1024

            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256_hash.update(chunk)
                    bytes_read += len(chunk)

                    # Update progress once per ~0.2s to stay responsive
                    now = time.time()
                    if now - last_update >= 0.2:
                        percent = (bytes_read / file_size) * 100 if file_size > 0 else 0.0
                        speed = bytes_read / max(0.001, (now - start_time))
                        line = (
                            f"[VERIFYING] {display_name}: {percent:.1f}% "
                            f"({self.format_bytes(bytes_read)}/{self.format_bytes(file_size)}) "
                            f"{self.format_bytes(speed)}/s"
                        )
                        self.show_progress_line(line)
                        last_update = now

            # Finalize
            computed_sha = sha256_hash.hexdigest()

            if computed_sha == expected_sha:
                self.finalize_progress_line(f"[VERIFIED] SHA256 match: {computed_sha[:16]}...")
                return True
            else:
                self.finalize_progress_line(f"[ERROR] SHA256 mismatch!")
                self.log(f"  Expected: {expected_sha}")
                self.log(f"  Got:      {computed_sha}")
                return False

        except Exception as e:
            self.finalize_progress_line()
            self.log(f"[ERROR] Failed to verify SHA256: {e}")
            return False

    def download_chunk(self, url: str, start: int, end: int,
                      filepath: str, chunk_id: int,
                      progress_callback=None) -> bool:
        """Download a specific chunk with resume support"""
        chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
        chunk_size_expected = end - start + 1

        # Check if chunk already complete
        if os.path.exists(chunk_file):
            existing_size = os.path.getsize(chunk_file)
            if existing_size == chunk_size_expected:
                if progress_callback:
                    progress_callback(chunk_id, chunk_size_expected)
                return True
            elif existing_size > chunk_size_expected:
                # Corrupted chunk, delete and restart
                os.remove(chunk_file)
                resume_pos = 0
            else:
                # Valid partial chunk, resume
                resume_pos = existing_size
        else:
            resume_pos = 0

        max_retries = self.config["max_retries"]

        for attempt in range(max_retries):
            try:
                # Download from resume position
                actual_start = start + resume_pos
                headers = {'Range': f'bytes={actual_start}-{end}', 'Accept-Encoding': 'identity'}

                response = self.session.get(url, headers=headers,
                                          timeout=self.config["timeout"],
                                          stream=True,
                                          allow_redirects=True)

                # Parallel chunking requires Range support. If server ignores Range it will reply 200,
                # which would corrupt chunk math (each chunk would download the full file).
                if response.status_code == 200:
                    raise Exception("Server did not honor Range request (got 200)")
                if response.status_code != 206:
                    raise Exception(f"Bad status code: {response.status_code}")

                # Download chunk
                mode = 'ab' if resume_pos > 0 else 'wb'
                downloaded = resume_pos

                with open(chunk_file, mode) as f:
                    for data in response.iter_content(chunk_size=self.config["chunk_size"]):
                        if data:
                            f.write(data)
                            downloaded += len(data)
                            if progress_callback:
                                progress_callback(chunk_id, downloaded)

                # Verify chunk is complete
                final_size = os.path.getsize(chunk_file)
                if final_size == chunk_size_expected:
                    if progress_callback:
                        progress_callback(chunk_id, chunk_size_expected)
                    return True
                elif final_size < chunk_size_expected:
                    # Incomplete, will retry
                    resume_pos = final_size
                    raise Exception(f"Chunk incomplete: {final_size}/{chunk_size_expected}")
                else:
                    # Too large, corrupted
                    os.remove(chunk_file)
                    resume_pos = 0
                    raise Exception(f"Chunk too large: {final_size}/{chunk_size_expected}")

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = min(self.config["retry_delay"] * (2 ** attempt),
                              self.config["max_retry_delay"])
                    time.sleep(delay)
                else:
                    self.log(f"Chunk {chunk_id} failed after {max_retries} attempts: {e}")
                    return False

        return False

    # ------------------------------ Merge chunks ------------------------------

    def merge_chunks(self, filepath: str, num_chunks: int) -> bool:
        """Merge downloaded chunks into final file with optimized I/O"""
        temp_file = os.path.normpath(f"{filepath}.tmp")

        try:
            # Check all chunks exist and calculate total size
            total_size = 0
            chunk_files = []
            for i in range(num_chunks):
                chunk_file = os.path.normpath(f"{filepath}.part{i}")
                if not os.path.exists(chunk_file):
                    self.log(f"Error: Missing chunk {i}")
                    return False
                size = os.path.getsize(chunk_file)
                chunk_files.append((chunk_file, size))
                total_size += size

            # For very large files (>1GB), use optimized OS-level copy
            if total_size > 1024 * 1024 * 1024:  # 1GB
                return self.merge_chunks_optimized(filepath, chunk_files, total_size, temp_file)

            # For smaller files, use buffered I/O
            buffer_size = 64 * 1024 * 1024  # 64MB buffer

            merged_size = 0
            start_time = time.time()
            last_update = 0.0

            with open(temp_file, 'wb') as outfile:
                for chunk_file, chunk_size in chunk_files:
                    with open(chunk_file, 'rb') as infile:
                        bytes_copied = 0
                        while bytes_copied < chunk_size:
                            to_read = min(buffer_size, chunk_size - bytes_copied)
                            data = infile.read(to_read)
                            if not data:
                                break
                            outfile.write(data)
                            bytes_copied += len(data)
                            merged_size += len(data)

                            # Update progress once per ~0.5s
                            now = time.time()
                            if now - last_update >= 0.5:
                                percent = (merged_size / total_size) * 100 if total_size > 0 else 0.0
                                speed = merged_size / max(0.001, (now - start_time))
                                line = (
                                    f"[MERGING] {percent:.1f}% "
                                    f"({self.format_bytes(merged_size)}/{self.format_bytes(total_size)}) "
                                    f"Speed: {self.format_bytes(speed)}/s"
                                )
                                self.show_progress_line(line)
                                last_update = now

            # Final progress line
            elapsed = max(0.001, time.time() - start_time)
            avg_speed = total_size / elapsed
            self.finalize_progress_line(
                f"[MERGING] 100.0% ({self.format_bytes(total_size)}/{self.format_bytes(total_size)}) "
                f"Completed in {self.format_time(elapsed)} - Avg: {self.format_bytes(avg_speed)}/s"
            )

            # Replace target file
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_file, filepath)

            # Clean up chunks after successful merge
            for chunk_file, _ in chunk_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass

            return True

        except Exception as e:
            self.finalize_progress_line()
            self.log(f"Error merging: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    def merge_chunks_optimized(self, filepath: str, chunk_files: List[Tuple[str, int]],
                               total_size: int, temp_file: str) -> bool:
        """Optimized merge for very large files using OS-level operations"""
        try:
            self.log(f"[MERGING] Using optimized merge for large file ({self.format_bytes(total_size)})")

            start_time = time.time()
            merged_size = 0
            last_update = 0.0

            # Use buffered concat for portability
            with open(temp_file, 'wb') as outfile:
                for i, (chunk_file, chunk_size) in enumerate(chunk_files):
                    # Update line per chunk and overall bytes to avoid extra prints
                    now = time.time()
                    if now - last_update >= 0.5:
                        line = f"[MERGING] Chunk {i+1}/{len(chunk_files)} " \
                               f"({self.format_bytes(merged_size)}/{self.format_bytes(total_size)})"
                        self.show_progress_line(line)
                        last_update = now

                    with open(chunk_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile, length=128 * 1024 * 1024)  # 128MB buffer
                    merged_size += chunk_size

            elapsed = max(0.001, time.time() - start_time)
            avg_speed = total_size / elapsed
            self.finalize_progress_line(
                f"[MERGING] 100.0% - Completed in {self.format_time(elapsed)} - Avg: {self.format_bytes(avg_speed)}/s"
            )

            # Replace target file
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_file, filepath)

            # Clean up chunks
            for chunk_file, _ in chunk_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass

            return True

        except Exception as e:
            self.finalize_progress_line()
            self.log(f"Error in optimized merge: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    # ------------------------------ Download API ------------------------------

    def download_file(self, repo_id: str, filename: str, local_dir: str, local_relpath: Optional[str] = None) -> str:
        """Main download function with integrity checks.

        Returns one of: DOWNLOAD_STATUS_DOWNLOADED / DOWNLOAD_STATUS_SKIPPED / DOWNLOAD_STATUS_FAILED
        """
        url = self.get_file_url(repo_id, filename)
        # Allow mapping a remote filename to a different local relative path (used for bundle/subdir mirroring).
        local_path = local_relpath if local_relpath else filename
        filepath = os.path.normpath(os.path.join(local_dir, local_path))
        display_name = local_path

        # Ensure the directory for the file exists (handle subdirectories)
        file_dir = os.path.dirname(filepath)
        os.makedirs(file_dir, exist_ok=True)

        # Get expected SHA256
        expected_sha = self.get_file_sha256(repo_id, filename)
        if expected_sha:
            self.log(f"[INFO] Expected SHA256: {expected_sha[:16]}...")

        # Get file size
        file_size = self.get_file_size(url, repo_id=repo_id, filename=filename)

        # Check existing file (skip only when we can validate completeness)
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath)
            if isinstance(file_size, int) and file_size >= 0:
                if actual_size == file_size:
                    # Size matches, optionally verify sha
                    if expected_sha and self.is_file_verified(repo_id, filename, filepath, expected_sha):
                        self.log(f"[SKIP] {display_name} already complete and verified (cached) ({self.format_bytes(file_size)})")
                        return DOWNLOAD_STATUS_SKIPPED
                    if expected_sha:
                        if self.verify_file_sha256(filepath, expected_sha, display_name):
                            self.mark_file_verified(repo_id, filename, filepath, expected_sha)
                            self.log(f"[SKIP] {display_name} already complete and verified ({self.format_bytes(file_size)})")
                            return DOWNLOAD_STATUS_SKIPPED
                        self.log(f"[WARNING] {display_name} failed verification, re-downloading")
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
                    else:
                        self.log(f"[SKIP] {display_name} already complete ({self.format_bytes(file_size)})")
                        return DOWNLOAD_STATUS_SKIPPED
                elif actual_size > file_size:
                    self.log(f"[WARNING] {display_name} corrupted (larger than expected), re-downloading")
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                else:
                    # Partial file: proceed to resume download (known size)
                    self.log(f"[RESUMING] {display_name} from {self.format_bytes(actual_size)}/{self.format_bytes(file_size)}")
            else:
                # Unknown remote size: if we have an expected SHA, we can still validate + skip.
                if expected_sha and actual_size > 0:
                    if self.is_file_verified(repo_id, filename, filepath, expected_sha):
                        self.log(f"[SKIP] {display_name} exists and verified (cached) ({self.format_bytes(actual_size)})")
                        return DOWNLOAD_STATUS_SKIPPED
                    if self.verify_file_sha256(filepath, expected_sha, display_name):
                        self.mark_file_verified(repo_id, filename, filepath, expected_sha)
                        self.log(f"[SKIP] {display_name} exists and verified ({self.format_bytes(actual_size)})")
                        return DOWNLOAD_STATUS_SKIPPED
                    self.log(f"[WARNING] {display_name} failed verification, re-downloading")
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass

        # If we can't reliably determine size, we still download via streaming mode (resume-capable).
        if file_size == -1:
            # Handle compressed/chunked files - download without size info
            self.log(f"[INFO] {display_name} is compressed, downloading without size info")
            return self.download_unknown_size(url, filepath, display_name, expected_sha, repo_id)
        if file_size is None:
            self.log(f"[INFO] Could not determine size for {display_name}; downloading without size info")
            return self.download_unknown_size(url, filepath, display_name, expected_sha, repo_id)

        # Use parallel download for files > 10MB
        if file_size > 10 * 1024 * 1024:
            if self.supports_range(url):
                success = self.download_parallel(url, filepath, display_name, file_size)
            else:
                self.log(f"[INFO] Range requests not supported for {display_name}; using single connection")
                success = self.download_single(url, filepath, display_name, file_size)
        else:
            success = self.download_single(url, filepath, display_name, file_size)

        # Verify SHA256 after successful download
        if success and expected_sha:
            if not self.verify_file_sha256(filepath, expected_sha, display_name):
                self.log(f"[ERROR] {display_name} downloaded but failed SHA256 verification")
                try:
                    os.remove(filepath)
                except:
                    pass
                return DOWNLOAD_STATUS_FAILED
            else:
                # Mark file as verified after successful verification
                self.mark_file_verified(repo_id, filename, filepath, expected_sha)

        return DOWNLOAD_STATUS_DOWNLOADED if success else DOWNLOAD_STATUS_FAILED

    def download_unknown_size(self, url: str, filepath: str, filename: str, expected_sha: str, repo_id: str = "") -> str:
        """Download file when size cannot be determined (resume-capable)."""
        max_retries = self.config["max_retries"]

        for attempt in range(max_retries):
            try:
                resume_pos = os.path.getsize(filepath) if os.path.exists(filepath) else 0

                if resume_pos > 0:
                    self.log(f"[RESUMING] {filename} (unknown size) from {self.format_bytes(resume_pos)}")
                    headers = {"Range": f"bytes={resume_pos}-", "Accept-Encoding": "identity"}
                else:
                    self.log(f"[DOWNLOADING] {filename} (unknown size)")
                    headers = {"Accept-Encoding": "identity"}

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.config["timeout"],
                    stream=True,
                    allow_redirects=True,
                )

                # If we tried to resume but the server says "range not satisfiable", the file is likely complete.
                if resume_pos > 0 and response.status_code == 416:
                    response.close()
                    if expected_sha:
                        if self.is_file_verified(repo_id, filename, filepath, expected_sha):
                            self.log(f"[SKIP] {filename} already complete and verified (cached)")
                            return DOWNLOAD_STATUS_SKIPPED
                        if self.verify_file_sha256(filepath, expected_sha, filename):
                            self.mark_file_verified(repo_id, filename, filepath, expected_sha)
                            self.log(f"[SKIP] {filename} already complete and verified")
                            return DOWNLOAD_STATUS_SKIPPED
                        self.log(f"[WARNING] {filename} failed verification, re-downloading")
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
                        resume_pos = 0
                    else:
                        self.log(f"[SKIP] {filename} already complete (range not satisfiable)")
                        return DOWNLOAD_STATUS_SKIPPED

                # If we attempted to resume but the server didn't honor it, restart from scratch.
                if resume_pos > 0 and response.status_code != 206:
                    self.log(f"[WARNING] Resume not supported for {filename} (status {response.status_code}), restarting")
                    response.close()
                    resume_pos = 0
                    response = self.session.get(
                        url,
                        headers={"Accept-Encoding": "identity"},
                        timeout=self.config["timeout"],
                        stream=True,
                        allow_redirects=True,
                    )

                response.raise_for_status()

                # Download the file
                downloaded = 0
                start_time = time.time()
                last_update = 0.0
                total_size: Optional[int] = None

                # If server provides total size (e.g. Content-Range), we can show percent progress.
                if response.status_code == 206:
                    content_range = response.headers.get("content-range")
                    if content_range and "/" in content_range:
                        total_str = content_range.split("/")[-1]
                        if total_str.isdigit():
                            total_size = int(total_str)
                elif response.status_code == 200:
                    content_length = response.headers.get("content-length")
                    if content_length and str(content_length).isdigit():
                        total_size = int(content_length)

                mode = "ab" if resume_pos > 0 else "wb"
                with open(filepath, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.config["chunk_size"]):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Show progress (with total if available)
                            now = time.time()
                            if now - last_update >= 0.5:
                                if total_size and total_size > 0:
                                    self.print_progress(resume_pos + downloaded, total_size, start_time, filename)
                                else:
                                    elapsed = max(0.001, now - start_time)
                                    speed = downloaded / elapsed
                                    line = (f"[DOWNLOADING] {filename}: {self.format_bytes(resume_pos + downloaded)} "
                                           f"@ {self.format_bytes(speed)}/s")
                                    self.show_progress_line(line)
                                last_update = now
                response.close()

                # Finalize download
                final_size = os.path.getsize(filepath)
                elapsed = max(0.001, time.time() - start_time)
                avg_speed = final_size / elapsed
                
                self.finalize_progress_line(
                    f"[OK] {filename} completed ({self.format_bytes(final_size)}) "
                    f"in {self.format_time(elapsed)} - Avg: {self.format_bytes(avg_speed)}/s"
                )

                # Verify SHA256 if available
                if expected_sha:
                    if not self.verify_file_sha256(filepath, expected_sha, filename):
                        self.log(f"[ERROR] {filename} downloaded but failed SHA256 verification")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return DOWNLOAD_STATUS_FAILED
                    else:
                        # Mark file as verified after successful verification
                        if repo_id:
                            self.mark_file_verified(repo_id, filename, filepath, expected_sha)

                return DOWNLOAD_STATUS_DOWNLOADED

            except Exception as e:
                self.finalize_progress_line()
                self.log(f"[ERROR] Attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = min(self.config["retry_delay"] * (2 ** attempt),
                              self.config["max_retry_delay"])
                    time.sleep(delay)
                else:
                    return DOWNLOAD_STATUS_FAILED

        return DOWNLOAD_STATUS_FAILED

    def download_parallel(self, url: str, filepath: str, filename: str,
                         file_size: int) -> bool:
        """Download using 16 parallel connections"""
        num_chunks = self.config["num_connections"]

        # Calculate chunk boundaries
        base_chunk_size = file_size // num_chunks
        chunks = []

        for i in range(num_chunks):
            start = i * base_chunk_size
            if i == num_chunks - 1:
                # Last chunk gets all remaining bytes
                end = file_size - 1
            else:
                end = (i + 1) * base_chunk_size - 1
            chunks.append((i, start, end))

        self.log(f"[DOWNLOADING] {filename} ({self.format_bytes(file_size)}) using {num_chunks} connections")

        # Progress tracking
        chunk_progress = {}
        progress_lock = threading.Lock()

        def update_progress(chunk_id: int, bytes_downloaded: int):
            with progress_lock:
                chunk_progress[chunk_id] = bytes_downloaded

        # Check existing progress
        for chunk_id, start, end in chunks:
            chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
            if os.path.exists(chunk_file):
                size = os.path.getsize(chunk_file)
                chunk_progress[chunk_id] = size
            else:
                chunk_progress[chunk_id] = 0

        initial_bytes = sum(chunk_progress.values())
        if initial_bytes > 0:
            self.log(f"[RESUMING] Already downloaded: {self.format_bytes(initial_bytes)}")

        # Download chunks
        start_time = time.time()
        failed_chunks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
            # Submit all chunks
            futures = {}
            for chunk_id, start, end in chunks:
                # Only download if not complete
                chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
                expected_size = end - start + 1

                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) == expected_size:
                    continue  # Skip completed chunks

                future = executor.submit(
                    self.download_chunk,
                    url, start, end, filepath, chunk_id,
                    update_progress
                )
                futures[future] = chunk_id

            # If all chunks already complete
            if not futures:
                self.log("[MERGING] All chunks already complete")
                if self.merge_chunks(filepath, num_chunks):
                    # Verify final file
                    if os.path.getsize(filepath) == file_size:
                        self.log(f"[OK] {filename} completed")
                        return True
                    else:
                        self.log(f"[ERROR] Size mismatch after merge")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return False
                return False

            # Monitor progress
            last_update = 0.0
            last_bytes = initial_bytes

            while futures:
                # Wait for any future to complete
                done, pending = concurrent.futures.wait(
                    list(futures.keys()),
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                # Process completed futures
                for future in done:
                    chunk_id = futures.pop(future)
                    try:
                        success = future.result()
                        if not success:
                            failed_chunks.append(chunk_id)
                    except Exception as e:
                        self.log(f"Chunk {chunk_id} exception: {e}")
                        failed_chunks.append(chunk_id)

                # Update progress (once per ~1s)
                current_time = time.time()
                if current_time - last_update >= 1.0 or not futures:
                    with progress_lock:
                        current_bytes = sum(chunk_progress.values())

                    # Calculate recent speed
                    time_delta = current_time - last_update if last_update > 0 else (current_time - start_time)
                    bytes_delta = current_bytes - last_bytes if last_update > 0 else (current_bytes - initial_bytes)
                    speed = bytes_delta / max(0.001, time_delta)

                    self.print_progress(current_bytes, file_size, start_time, filename, speed)

                    last_update = current_time
                    last_bytes = current_bytes

        # Remove the active progress line so the next messages appear cleanly
        self.clear_progress_line()

        # Check results
        if failed_chunks:
            self.log(f"[ERROR] Failed chunks: {failed_chunks}")
            return False

        # Verify all chunks are complete before merging
        for chunk_id, start, end in chunks:
            chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
            expected_size = end - start + 1
            if not os.path.exists(chunk_file):
                self.log(f"[ERROR] Missing chunk {chunk_id}")
                return False
            actual_size = os.path.getsize(chunk_file)
            if actual_size != expected_size:
                self.log(f"[ERROR] Chunk {chunk_id} incomplete: {actual_size}/{expected_size}")
                return False

        # Merge chunks
        self.log(f"[MERGING] Merging {num_chunks} chunks...")
        if self.merge_chunks(filepath, num_chunks):
            # Verify final file size
            final_size = os.path.getsize(filepath)
            if final_size == file_size:
                elapsed = max(0.001, time.time() - start_time)
                avg_speed = (file_size - initial_bytes) / elapsed if elapsed > 0 else 0
                self.log(f"[OK] {filename} completed in {self.format_time(elapsed)} "
                         f"- Average: {self.format_bytes(avg_speed)}/s")
                return True
            else:
                self.log(f"[ERROR] Final size mismatch: {final_size} != {file_size}")
                try:
                    os.remove(filepath)
                except:
                    pass
                return False
        else:
            self.log(f"[ERROR] Merge failed")
            return False

    def download_single(self, url: str, filepath: str, filename: str,
                       file_size: int) -> bool:
        """Single connection download for small files"""
        max_retries = self.config["max_retries"]

        for attempt in range(max_retries):
            try:
                # Check existing file
                resume_pos = 0
                if os.path.exists(filepath):
                    resume_pos = os.path.getsize(filepath)
                    if resume_pos >= file_size:
                        self.log(f"[OK] {filename} already complete")
                        return True

                # Download
                headers = {'Range': f'bytes={resume_pos}-'} if resume_pos > 0 else {}
                response = self.session.get(url, headers=headers,
                                          timeout=self.config["timeout"],
                                          stream=True)

                if resume_pos > 0 and response.status_code != 206:
                    self.log(f"[WARNING] Resume not supported, restarting")
                    resume_pos = 0
                    response = self.session.get(url, timeout=self.config["timeout"],
                                              stream=True)

                response.raise_for_status()

                # Write file
                mode = 'ab' if resume_pos > 0 else 'wb'
                downloaded = 0
                start_time = time.time()
                last_update = 0.0

                if resume_pos > 0:
                    self.log(f"[RESUMING] {filename} from {self.format_bytes(resume_pos)}")
                else:
                    self.log(f"[DOWNLOADING] {filename} ({self.format_bytes(file_size)})")

                with open(filepath, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.config["chunk_size"]):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Progress update once per ~0.5s
                            now = time.time()
                            if now - last_update >= 0.5:
                                total = resume_pos + downloaded
                                self.print_progress(total, file_size, start_time, filename)
                                last_update = now

                # Final progress update
                total = resume_pos + downloaded
                self.print_progress(total, file_size, start_time, filename)
                # Move to next line cleanly
                self.clear_progress_line()

                # Verify size
                if os.path.getsize(filepath) == file_size:
                    self.log(f"[OK] {filename} completed")
                    return True
                else:
                    self.log(f"[ERROR] Size mismatch")
                    continue

            except Exception as e:
                self.finalize_progress_line()
                self.log(f"[ERROR] Attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = min(self.config["retry_delay"] * (2 ** attempt),
                              self.config["max_retry_delay"])
                    time.sleep(delay)
                else:
                    return False

        return False

# ------------------------------- Repo scanning -------------------------------

def scan_repo_files(repo_id: str, specific_files: List[str] = None) -> List[str]:
    """Get list of files to download"""
    try:
        if specific_files:
            print(f"Using predefined file list")
            return specific_files

        api = HfApi()
        all_files = list_repo_files(repo_id=repo_id, repo_type="model")
        return all_files

    except Exception as e:
        print(f"Error scanning repository: {e}")
        return specific_files if specific_files else []

def download_file_with_rename(downloader: RobustDownloader, repo_id: str, remote_filename: str,
                              local_dir: str, local_filename: str) -> bool:
    """Download a file from HuggingFace repo but save it with a different local name"""
    url = downloader.get_file_url(repo_id, remote_filename)
    filepath = os.path.normpath(os.path.join(local_dir, local_filename))

    # Ensure the directory for the file exists (handle subdirectories)
    file_dir = os.path.dirname(filepath)
    os.makedirs(file_dir, exist_ok=True)

    # Get expected SHA256 - using the original remote filename
    expected_sha = downloader.get_file_sha256(repo_id, remote_filename)
    if expected_sha:
        downloader.log(f"[INFO] Expected SHA256: {expected_sha[:16]}...")

    # Get file size
    file_size = downloader.get_file_size(url, repo_id=repo_id, filename=remote_filename)

    # Check if already complete
    if os.path.exists(filepath):
        actual_size = os.path.getsize(filepath)
        if file_size:
            if actual_size == file_size:
                # Check if already verified to skip SHA256 computation
                if expected_sha and downloader.is_file_verified(repo_id, remote_filename, filepath, expected_sha):
                    downloader.log(f"[SKIP] {local_filename} already complete and verified (cached) ({downloader.format_bytes(file_size)})")
                    return True
                # Verify SHA256 if available and not cached
                elif expected_sha:
                    if downloader.verify_file_sha256(filepath, expected_sha, local_filename):
                        downloader.mark_file_verified(repo_id, remote_filename, filepath, expected_sha)
                        downloader.log(f"[SKIP] {local_filename} already complete and verified ({downloader.format_bytes(file_size)})")
                        return True
                    else:
                        downloader.log(f"[WARNING] {local_filename} failed verification, re-downloading")
                        os.remove(filepath)
                else:
                    downloader.log(f"[SKIP] {local_filename} already complete ({downloader.format_bytes(file_size)})")
                    return True
            elif actual_size > file_size:
                downloader.log(f"[WARNING] {local_filename} corrupted, re-downloading")
                os.remove(filepath)
        else:
            # Can't verify size, try SHA256 verification
            if actual_size > 1024 and expected_sha:  # > 1KB
                # Check if already verified to skip SHA256 computation
                if downloader.is_file_verified(repo_id, remote_filename, filepath, expected_sha):
                    downloader.log(f"[SKIP] {local_filename} exists and verified (cached) ({downloader.format_bytes(actual_size)})")
                    return True
                elif downloader.verify_file_sha256(filepath, expected_sha, local_filename):
                    downloader.mark_file_verified(repo_id, remote_filename, filepath, expected_sha)
                    downloader.log(f"[SKIP] {local_filename} exists and verified ({downloader.format_bytes(actual_size)})")
                    return True
                else:
                    downloader.log(f"[WARNING] {local_filename} failed verification, re-downloading")
                    os.remove(filepath)
            # If we don't have a SHA and we don't know the expected size, we can't safely skip:
            # keep going and let the download logic attempt a resume/streaming download instead.

    if file_size == -1:
        # Handle compressed/chunked files - download without size info
        downloader.log(f"[INFO] {local_filename} is compressed, downloading without size info")
        status = downloader.download_unknown_size(url, filepath, local_filename, expected_sha, repo_id)
        return status != DOWNLOAD_STATUS_FAILED
    if file_size is None:
        downloader.log(f"[INFO] Could not determine size for {remote_filename}; downloading without size info")
        status = downloader.download_unknown_size(url, filepath, local_filename, expected_sha, repo_id)
        return status != DOWNLOAD_STATUS_FAILED

    # Use parallel download for files > 10MB
    if file_size > 10 * 1024 * 1024:
        if downloader.supports_range(url):
            success = downloader.download_parallel(url, filepath, local_filename, file_size)
        else:
            downloader.log(f"[INFO] Range requests not supported for {local_filename}; using single connection")
            success = downloader.download_single(url, filepath, local_filename, file_size)
    else:
        success = downloader.download_single(url, filepath, local_filename, file_size)

    # Verify SHA256 after successful download
    if success and expected_sha:
        if not downloader.verify_file_sha256(filepath, expected_sha, local_filename):
            downloader.log(f"[ERROR] {local_filename} downloaded but failed SHA256 verification")
            try:
                os.remove(filepath)
            except:
                pass
            return False
        else:
            # Mark file as verified after successful verification
            downloader.mark_file_verified(repo_id, remote_filename, filepath, expected_sha)

    return success

def _repo_id_to_local_dir(models_root: str, repo_id: str) -> str:
    return os.path.join(models_root, repo_id.replace("/", "--"))


def _normalize_subdir_prefix(subdir: str) -> str:
    """Normalize a subdir like 'Trellv2' to a prefix like 'Trellv2/' suitable for startswith()."""
    if not subdir:
        return ""
    subdir = subdir.replace("\\", "/").strip("/")
    return subdir + "/"


def _try_extract_dependency_repo_ids_from_pipeline_config(config_path: str) -> List[str]:
    """
    Extract HF repo IDs for dependency models from a TRELLIS pipeline config JSON.
    We intentionally only pull known fields to avoid false-positives.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    args = data.get("args") or {}
    out: List[str] = []

    def add_repo_id(value: Optional[str]):
        if not value or not isinstance(value, str):
            return
        # Expect "org/repo" (no subpaths here for dependency models).
        parts = value.split("/")
        if len(parts) == 2 and all(parts):
            out.append(value)

    image_cond = args.get("image_cond_model") or {}
    if isinstance(image_cond, dict):
        add_repo_id(((image_cond.get("args") or {}).get("model_name")))

    rembg = args.get("rembg_model") or {}
    if isinstance(rembg, dict):
        add_repo_id(((rembg.get("args") or {}).get("model_name")))

    return out


def _discover_dependency_repos_from_downloaded_trellis_repo(repo_dir: str) -> List[str]:
    """
    Discover additional HF repos that TRELLIS pipeline configs reference (e.g. DINOv3, BiRefNet).
    """
    candidates: List[str] = []
    try:
        for name in os.listdir(repo_dir):
            if not name.lower().endswith(".json"):
                continue
            if "pipeline" not in name.lower():
                continue
            candidates.append(os.path.join(repo_dir, name))
    except Exception:
        return []

    deps: List[str] = []
    for config_path in candidates:
        deps.extend(_try_extract_dependency_repo_ids_from_pipeline_config(config_path))

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for r in deps:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def ensure_ultrashape_source_checkout() -> bool:
    """Install the small UltraShape source checkout required by runtime refinement."""
    script_dir = Path(__file__).resolve().parent if __file__ else Path.cwd()
    app_dir = script_dir / "Trellis_2_3D_Generator"
    source_module = app_dir / "ultrashape_source.py"
    if not source_module.is_file():
        print(f"[WARNING] UltraShape source helper not found: {source_module}")
        return False

    app_dir_str = str(app_dir)
    if app_dir_str not in sys.path:
        sys.path.insert(0, app_dir_str)

    try:
        from ultrashape_source import ensure_ultrashape_source

        root = ensure_ultrashape_source(
            app_dir,
            allow_download=True,
            log_fn=lambda msg: print(f"[UltraShape Source] {msg}"),
        )
        print(f"[UltraShape Source] OK: {root}")
        return True
    except Exception as e:
        print(f"[ERROR] UltraShape source checkout failed: {type(e).__name__}: {e}")
        print(
            "[ERROR] UltraShape refinement needs both models/UltraShape weights and the "
            "UltraShape-1.0 source checkout. You can retry with: "
            "python Trellis_2_3D_Generator/ensure_ultrashape_source.py"
        )
        return False


def download_models(
    download_dir: Optional[str] = None,
    model_types: Optional[List[str]] = None,
    with_deps: bool = True,
):
    """Main download function for TRELLIS.2 model repos."""
    target_dir = download_dir if download_dir else TARGET_DIR

    # Default to bundle (if present) so a single run pulls everything needed for offline use.
    # Users can still explicitly request the original upstream repos via --models.
    if model_types is None:
        default_bundle_models = [m for m in ("trellv2_bundle", "ultrashape_bundle") if m in MODEL_CONFIGS]
        if default_bundle_models:
            model_types = default_bundle_models
        else:
            model_types = list(MODEL_CONFIGS.keys())

    # Validate model types
    invalid_types = [mt for mt in model_types if mt not in MODEL_CONFIGS]
    if invalid_types:
        print(f"Error: Invalid model types: {', '.join(invalid_types)}")
        print(f"Valid options: {', '.join(MODEL_CONFIGS.keys())}")
        return

    downloader = RobustDownloader(DOWNLOAD_CONFIG)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Target directory: {os.path.abspath(target_dir)}")
    print(f"Downloading model packs: {', '.join(model_types)}")
    print(f"Download dependencies: {with_deps}")

    # De-duplicate exact repo downloads while still allowing multiple subdir bundles from
    # the same repo (e.g. Trellv2 + UltraShape in MonsterMMORPG/Wan_GGUF).
    downloaded_repos: set[Tuple[str, str, str]] = set()

    total_successful = 0
    total_failed = 0
    total_skipped = 0

    def download_repo(
        repo_id: str,
        name: str = "",
        description: str = "",
        files: Optional[List[str]] = None,
        subdir: Optional[str] = None,
        local_target_subdir: Optional[str] = None,
    ):
        nonlocal total_successful, total_failed, total_skipped

        subdir_prefix = _normalize_subdir_prefix(subdir) if subdir else ""
        local_target_subdir_norm = (local_target_subdir or "").replace("\\", "/").strip("/")
        repo_download_key = (repo_id, subdir_prefix, local_target_subdir_norm)
        if repo_download_key in downloaded_repos:
            return
        downloaded_repos.add(repo_download_key)

        # Resolve file list
        files_to_download = scan_repo_files(repo_id, files)
        if not files_to_download:
            print(f"[ERROR] No files found for {repo_id}. Skipping.")
            total_failed += 1
            return

        # Optional: filter to a specific subdir within the repo (bundle mirror use-case)
        if subdir_prefix:
            files_to_download = [f for f in files_to_download if isinstance(f, str) and f.replace("\\", "/").startswith(subdir_prefix)]
            if not files_to_download:
                print(f"[ERROR] No files found under subdir '{subdir_prefix}' in {repo_id}. Skipping.")
                total_failed += 1
                return

        print(f"\n{'='*60}")
        title = name or repo_id
        print(f"Downloading {title}")
        if description:
            print(f"Description: {description}")
        print(f"Repository: {repo_id}")
        print(f"Files: {len(files_to_download)} files")
        if subdir_prefix:
            print(f"Subdir: {subdir_prefix}")
        if local_target_subdir_norm:
            print(f"Local target subdir: {local_target_subdir_norm}")

        # Prefetch metadata once per repo (sizes + LFS sha256). This avoids many HTTP header edge-cases
        # and prevents per-file API calls.
        downloader.prefetch_repo_metadata(repo_id)

        # Where to put downloaded files on disk:
        # - Normal mode: models_root/<org>--<repo>/...
        # - Bundle/subdir mode: models_root[/<local_target_subdir>]/<contents_of_subdir>/... (prefix stripped)
        if subdir_prefix:
            model_dir = os.path.join(target_dir, local_target_subdir_norm) if local_target_subdir_norm else target_dir
            os.makedirs(model_dir, exist_ok=True)
        else:
            model_dir = _repo_id_to_local_dir(target_dir, repo_id)
            os.makedirs(model_dir, exist_ok=True)

        successful = 0
        failed = 0
        skipped = 0

        for filename in files_to_download:
            # In subdir mode, strip the prefix so folders land directly in models root.
            local_relpath = None
            if subdir_prefix:
                rel = filename.replace("\\", "/")
                if rel.startswith(subdir_prefix):
                    local_relpath = rel[len(subdir_prefix):]
            status = downloader.download_file(repo_id, filename, model_dir, local_relpath=local_relpath)
            if status == DOWNLOAD_STATUS_DOWNLOADED:
                successful += 1
            elif status == DOWNLOAD_STATUS_SKIPPED:
                skipped += 1
            else:
                failed += 1

        print(f"\nRepo {repo_id} summary:")
        print(f"  Downloaded: {successful}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")

        total_successful += successful
        total_failed += failed
        total_skipped += skipped

        # If this is the main TRELLIS model repo (upstream mode), optionally discover and download deps.
        # For bundle/subdir mode, deps are expected to already be present inside the bundle.
        if (not subdir_prefix) and with_deps and repo_id == "microsoft/TRELLIS.2-4B":
            deps = _discover_dependency_repos_from_downloaded_trellis_repo(model_dir)
            deps = [d for d in deps if d != repo_id]
            if deps:
                print(f"\n[INFO] Discovered {len(deps)} dependency repo(s) from pipeline configs:")
                for d in deps:
                    print(f"  - {d}")
                for d in deps:
                    download_repo(
                        d,
                        name=d,
                        description="Auto-discovered TRELLIS.2 dependency",
                        files=None,
                    )

    # Download requested packs
    for model_type in model_types:
        cfg = MODEL_CONFIGS[model_type]
        download_repo(
            cfg["repo_id"],
            name=cfg.get("name") or cfg["repo_id"],
            description=cfg.get("description") or "",
            files=cfg.get("files"),
            subdir=cfg.get("subdir"),
            local_target_subdir=cfg.get("local_target_subdir"),
        )

    ultrashape_source_ok: Optional[bool] = None
    if "ultrashape_bundle" in model_types:
        print(f"\n{'='*60}")
        print("Checking UltraShape source checkout")
        ultrashape_source_ok = ensure_ultrashape_source_checkout()

    # Final summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY:")
    print(f"  Total downloaded: {total_successful}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Total failed: {total_failed}")
    if ultrashape_source_ok is not None:
        print(f"  UltraShape source: {'ready' if ultrashape_source_ok else 'missing'}")
    print(f"  Models location: {os.path.abspath(target_dir)}")

    if total_failed == 0 and ultrashape_source_ok is not False:
        print("\n[SUCCESS] All downloads completed successfully!")
    else:
        if ultrashape_source_ok is False and total_failed == 0:
            print("\n[ERROR] Model downloads completed, but UltraShape source setup failed.")
        else:
            print(f"\n[ERROR] {total_failed} downloads failed. Please check your internet connection and try again.")

    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TRELLIS.2 model repos (offline-ready)")
    default_bundle_models = [m for m in ("trellv2_bundle", "ultrashape_bundle") if m in MODEL_CONFIGS]
    default_models_hint = ", ".join(default_bundle_models) if default_bundle_models else "all"
    parser.add_argument(
        "--dir",
        type=str,
        help=f"Download directory (default: {TARGET_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model packs to download (default: {default_models_hint}). Choices: {', '.join(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Do not auto-download dependency repos referenced by TRELLIS pipeline configs.",
    )

    args = parser.parse_args()
    download_models(args.dir, args.models, with_deps=not args.no_deps)
