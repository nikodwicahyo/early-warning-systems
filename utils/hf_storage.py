"""
Hugging Face Storage Handler
Manages persistent storage for uploaded datasets and trained models
"""

import os
import gc
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import json
import tempfile
import psutil
from huggingface_hub import HfApi, login, hf_hub_download
import streamlit as st

class HFStorageManager:
    """
    Manages data and model storage on Hugging Face Hub
    """

    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        Initialize HF Storage Manager

        Args:
            repo_id: HuggingFace repository ID (format: username/repo-name)
            token: HF token (if None, will use st.secrets or env variable)
        """
        self.repo_id = repo_id

        # Get token from multiple sources
        if token is None:
            token = self._get_token()

        self.token = token
        self._api = None

    @property
    def api(self):
        """Get or create HfApi client"""
        if self._api is None:
            self._api = HfApi(token=self.token)
        return self._api

        # Login to HF
        try:
            login(token=self.token, add_to_git_credential=False)
        except Exception as e:
            raise

    def _get_token(self) -> str:
        """
        Retrieve HF token from secrets or environment

        Priority:
        1. Streamlit secrets
        2. Environment variable
        3. Raise error if not found
        """
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
            return st.secrets['HF_TOKEN']

        # Try environment variable
        token = os.getenv('HF_TOKEN')
        if token:
            return token

        raise ValueError(
            "HF_TOKEN not found. Please set it in:\n"
            "1. Streamlit secrets (.streamlit/secrets.toml), or\n"
            "2. Environment variable (HF_TOKEN=...)"
        )

    def upload_dataset(
        self,
        file_path: str,
        destination_path: str,
        commit_message: Optional[str] = None
    ) -> str:
        """
        Upload dataset file to HF repository

        Args:
            file_path: Local file path
            destination_path: Path in HF repo (e.g., 'data/bank_data.csv')
            commit_message: Git commit message

        Returns:
            URL of uploaded file
        """
        try:
            if commit_message is None:
                commit_message = f"Upload dataset: {Path(file_path).name}"

            # Upload file
            url = self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=destination_path,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=commit_message,
                repo_type="space"
            )

            return url

        except Exception as e:
            raise

    def upload_file(
        self,
        file_path: str,
        destination_path: str,
        commit_message: Optional[str] = None
    ) -> str:
        """Alias for upload_dataset - upload any file to HF repository"""
        return self.upload_dataset(file_path, destination_path, commit_message)

    def upload_model(
        self,
        model_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload trained model to HF repository

        Args:
            model_path: Local model file path
            model_name: Name for model in repo (e.g., 'model_tft_v1.pth')
            metadata: Model metadata (version, metrics, etc.)

        Returns:
            URL of uploaded model
        """
        try:
            destination = f"models/{model_name}"

            # Upload model file
            url = self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=destination,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Update model: {model_name}",
                repo_type="space"
            )

            # Upload metadata if provided
            if metadata:
                import json
                metadata_path = f"models/{Path(model_name).stem}_metadata.json"

                # Save metadata locally first
                temp_dir = tempfile.gettempdir()
                temp_meta = os.path.join(temp_dir, f"{Path(model_name).stem}_metadata.json")
                with open(temp_meta, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Upload metadata
                self.api.upload_file(
                    path_or_fileobj=temp_meta,
                    path_in_repo=metadata_path,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"Update metadata: {model_name}",
                    repo_type="space"
                )

                # Cleanup
                os.remove(temp_meta)

            return url

        except Exception as e:
            raise

    def download_file(
        self,
        repo_file_path: str,
        local_path: Optional[str] = None
    ) -> str:
        """
        Download file from HF repository

        Args:
            repo_file_path: Path in HF repo
            local_path: Local destination path (if None, downloads to cache)

        Returns:
            Local file path
        """
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_file_path,
                token=self.token,
                repo_type="space",
                local_dir=local_path
            )

            return downloaded_path

        except Exception as e:
            return None

    def download_to_path(
        self,
        repo_file_path: str,
        local_file_path: str
    ) -> bool:
        """
        Download file from HF repository to exact destination path

        Args:
            repo_file_path: Path in HF repo (e.g., 'data/processed/master_panel.csv')
            local_file_path: Exact local file path to write to

        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            
            # Download to temp location first
            temp_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_file_path,
                token=self.token,
                repo_type="space"
            )
            
            # Ensure parent directory exists
            Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Copy to exact destination
            shutil.copy2(temp_path, local_file_path)
            
            return True

        except Exception as e:
            return False

    def list_files(self, path_prefix: str = "") -> list:
        """
        List files in HF repository

        Args:
            path_prefix: Filter by path prefix (e.g., 'models/')

        Returns:
            List of file paths
        """
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                token=self.token,
                repo_type="space"
            )

            if path_prefix:
                files = [f for f in files if f.startswith(path_prefix)]

            return files

        except Exception as e:
            return []

    def delete_file(
        self,
        repo_file_path: str,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Delete a file from HF repository

        Args:
            repo_file_path: Path in HF repo
            commit_message: Git commit message

        Returns:
            True if successful, False otherwise
        """
        try:
            if commit_message is None:
                commit_message = f"Delete file: {repo_file_path}"

            self.api.delete_file(
                path_in_repo=repo_file_path,
                repo_id=self.repo_id,
                token=self.token,
                repo_type="space",
                commit_message=commit_message
            )
            return True
        except Exception as e:
            return False

    def delete_folder(
        self,
        folder_path: str,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Delete a folder and all its contents from HF repository

        Args:
            folder_path: Folder path in HF repo
            commit_message: Git commit message

        Returns:
            True if successful, False otherwise
        """
        try:
            if commit_message is None:
                commit_message = f"Delete folder: {folder_path}"

            self.api.delete_folder(
                path_in_repo=folder_path,
                repo_id=self.repo_id,
                token=self.token,
                repo_type="space",
                commit_message=commit_message
            )
            return True
        except Exception as e:
            # Fallback: if delete_folder fails (some API versions/repo types), 
            # try deleting files individually
            try:
                files = self.list_files(folder_path)
                if not files:
                    return True
                
                for f in files:
                    self.delete_file(f, commit_message=f"Cleanup: {f}")
                return True
            except:
                return False

    def cleanup_memory(self):
        """
        Force garbage collection to free memory
        """
        gc.collect()
        if hasattr(gc, 'collect'):
            gc.collect(generation=2)

    def check_memory_usage(self):
        """
        Check current memory usage and warn if high
        """
        import psutil

        memory = psutil.virtual_memory()

        usage_percent = memory.percent
        available_gb = memory.available / (1024**3)

        if usage_percent > 80:
            st.warning(f"""
        ⚠️ **High Memory Usage**: {usage_percent:.1f}%

        Available: {available_gb:.2f} GB

        **Recommendations**:
        - Reduce batch size
        - Close other applications
        - Consider upgrading Space to higher tier
        """)

            # Force garbage collection
            gc.collect()

        return {
            'percent': usage_percent,
            'available_gb': available_gb
        }
