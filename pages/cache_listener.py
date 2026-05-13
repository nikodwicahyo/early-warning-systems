"""
Page Cache Invalidation Listener
Helper to add cache invalidation detection to all pages
"""

import streamlit as st
import logging
from src.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def check_and_handle_cache_invalidation(data_service) -> bool:
    """
    Check if cache invalidation event occurred and handle it
    Returns True if invalidation was processed (page should refresh data)
    """
    # Initialize tracking if needed
    if 'last_cache_version' not in st.session_state:
        st.session_state.last_cache_version = -1
    
    current_version = CacheManager.get_cache_version()
    last_version = st.session_state.last_cache_version
    
    # If version changed, invalidation occurred
    if current_version > last_version:
        st.session_state.last_cache_version = current_version
        logger.info(f"Cache invalidation detected (v{last_version} → v{current_version}). Refreshing data...")
        
        # Force refresh data by clearing all caches
        data_service.loader.reload_all()
        
        return True
    
    return False
