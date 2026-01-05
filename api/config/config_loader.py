"""
Configuration Loader for NPSketch AI Training

Loads and validates configuration from YAML file.
Provides singleton access to configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Singleton configuration loader."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        # Find config file
        config_path = Path(__file__).parent / "training_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        # Example: NPSKETCH_TRAINING_DEFAULTS_BATCH_SIZE=16
        prefix = "NPSKETCH_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Parse path by matching against actual YAML structure
                # This handles underscores in key names correctly
                parts = self._parse_config_path(config_key)
                
                if parts:
                    # Try to find and set in config
                    self._set_nested_value(self._config, parts, value)
    
    def _parse_config_path(self, env_key: str) -> list:
        """
        Parse environment variable key into config path parts.
        
        Handles underscores in key names by matching against actual YAML structure.
        
        Args:
            env_key: Environment variable key without prefix (e.g., "training_defaults_batch_size")
        
        Returns:
            List of path parts (e.g., ["training", "defaults", "batch_size"])
        """
        # Split by underscores
        all_parts = env_key.split('_')
        
        # Try to match against actual config structure
        # Start from root and try to match longest possible keys
        current = self._config
        result = []
        i = 0
        
        while i < len(all_parts):
            # Try to match progressively longer key names
            matched = False
            for length in range(len(all_parts) - i, 0, -1):
                # Try key made from next 'length' parts
                candidate_key = '_'.join(all_parts[i:i+length])
                
                if isinstance(current, dict) and candidate_key in current:
                    result.append(candidate_key)
                    current = current[candidate_key]
                    i += length
                    matched = True
                    break
            
            if not matched:
                # If no match found, try single part (for backward compatibility)
                if isinstance(current, dict) and all_parts[i] in current:
                    result.append(all_parts[i])
                    current = current[all_parts[i]]
                    i += 1
                else:
                    # Path not found, return empty list
                    return []
        
        return result
    
    def _set_nested_value(self, config: Dict, parts: list, value: str):
        """Set nested configuration value."""
        # Navigate to nested dict
        current = config
        for part in parts[:-1]:
            if part in current and isinstance(current[part], dict):
                current = current[part]
            else:
                return  # Path not found, skip
        
        # Set value with type conversion
        key = parts[-1]
        if key in current:
            # Try to maintain type
            original_type = type(current[key])
            try:
                if original_type == bool:
                    current[key] = value.lower() in ('true', '1', 'yes')
                elif original_type == int:
                    current[key] = int(value)
                elif original_type == float:
                    current[key] = float(value)
                else:
                    current[key] = value
            except ValueError:
                # Keep as string if conversion fails
                current[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., "training.defaults.batch_size")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config = ConfigLoader()
            >>> config.get("training.defaults.batch_size")
            8
        """
        parts = key.split('.')
        current = self._config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "training", "augmentation")
        
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def reload(self):
        """Reload configuration from file."""
        self._config = None
        self._load_config()


# Singleton instance
_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """
    Get singleton configuration loader instance.
    
    Returns:
        ConfigLoader instance
    
    Example:
        >>> from config import get_config
        >>> config = get_config()
        >>> batch_size = config.get("training.defaults.batch_size")
    """
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


def reload_config():
    """
    Reload configuration from file.
    
    Use this after modifying the YAML file.
    """
    global _loader
    if _loader is not None:
        _loader.reload()
    else:
        _loader = ConfigLoader()


# Convenience function for quick access
def get_value(key: str, default: Any = None) -> Any:
    """
    Quick access to configuration value.
    
    Args:
        key: Dot-notation key
        default: Default value
    
    Returns:
        Configuration value
    
    Example:
        >>> from config.config_loader import get_value
        >>> batch_size = get_value("training.defaults.batch_size", 8)
    """
    return get_config().get(key, default)

