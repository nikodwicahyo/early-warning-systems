"""
Cache Invalidation Manager
Centralized cache clearing with session-wide + event broadcasting
"""

import logging
from datetime import datetime
from typing import Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)


class CacheManager:
    """Centralized cache invalidation and event broadcasting"""
    
    @staticmethod
    def invalidate_all() -> None:
        """
        Invalidate all caches: Streamlit caches + session_state caches
        Increments cache_version to trigger fresh data loads
        Broadcasts invalidation event to all pages
        """
        try:
            # Clear Streamlit caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Increment cache version to trigger new loads
            if 'cache_version' not in st.session_state:
                st.session_state.cache_version = 0
            st.session_state.cache_version += 1
            
            # Clear all session_state cache keys (predictions_v*, master_data_v*, etc.)
            keys_to_delete = [k for k in st.session_state.keys() if '_v' in k and k not in ['cache_version']]
            for key in keys_to_delete:
                del st.session_state[key]
            
            # Broadcast invalidation event
            st.session_state['cache_invalidation_event'] = {
                'timestamp': datetime.now().isoformat(),
                'version': st.session_state.cache_version,
                'processed': False
            }
            
            logger.info(f"Cache invalidated. Version: {st.session_state.cache_version}")
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    @staticmethod
    def get_cache_version() -> int:
        """Get current cache version"""
        if 'cache_version' not in st.session_state:
            st.session_state.cache_version = 0
        return st.session_state.cache_version
    
    @staticmethod
    def has_invalidation_event() -> bool:
        """Check if invalidation event exists and not yet processed"""
        if 'cache_invalidation_event' not in st.session_state:
            return False
        event = st.session_state['cache_invalidation_event']
        return not event.get('processed', False)
    
    @staticmethod
    def mark_event_processed() -> None:
        """Mark invalidation event as processed"""
        if 'cache_invalidation_event' in st.session_state:
            st.session_state['cache_invalidation_event']['processed'] = True
    
    @staticmethod
    def get_invalidation_event() -> Dict[str, Any]:
        """Get current invalidation event"""
        return st.session_state.get('cache_invalidation_event', {})
