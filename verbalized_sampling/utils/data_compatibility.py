"""
Data access compatibility layer for method naming transition.

This module provides utilities to handle the transition from legacy method names
(structure_with_prob, chain_of_thought, combined) to new paper-aligned names
(vs_standard, vs_cot, vs_multi) while maintaining backward compatibility with
existing experimental data.
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings
from verbalized_sampling.methods.factory import Method, METHOD_MIGRATION_MAP


class DataPathResolver:
    """
    Resolves data paths for both new and legacy method naming conventions.

    This class handles path resolution for accessing experimental data that may
    use either the new paper-aligned naming (vs_standard, vs_cot, vs_multi) or
    the legacy naming (structure_with_prob, chain_of_thought, combined).

    When looking for data files, it will:
    1. First try the new naming convention
    2. Fall back to legacy naming if new paths don't exist
    3. Issue deprecation warnings when using legacy paths
    """

    def __init__(self, base_data_dir: Union[str, Path] = "generated_data"):
        """
        Initialize the data path resolver.

        Args:
            base_data_dir: Base directory containing experimental data
        """
        self.base_data_dir = Path(base_data_dir)
        self._cache = {}  # Cache for path resolution results

    def resolve_method_path(self,
                           base_path: Union[str, Path],
                           method: Union[Method, str],
                           warn_on_legacy: bool = True) -> Optional[Path]:
        """
        Resolve a path that contains a method name, checking both new and legacy conventions.

        Args:
            base_path: Base path template that contains method name
            method: Method enum or string to resolve
            warn_on_legacy: Whether to emit deprecation warnings for legacy paths

        Returns:
            Resolved path if found, None if neither new nor legacy path exists

        Examples:
            >>> resolver = DataPathResolver()
            >>> # Resolves to vs_standard path if it exists, otherwise structure_with_prob
            >>> path = resolver.resolve_method_path("experiments/model/task/evaluation/{method}", Method.VS_STANDARD)
        """
        base_path = Path(base_path)

        # Convert method to string if it's an enum
        method_str = method.value if hasattr(method, 'value') else str(method)

        # Create cache key
        cache_key = (str(base_path), method_str)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try new naming first
        new_method_name = self._get_new_method_name(method_str)
        new_path = Path(str(base_path).format(method=new_method_name))

        if new_path.exists():
            self._cache[cache_key] = new_path
            return new_path

        # Try legacy naming
        legacy_method_name = self._get_legacy_method_name(method_str)
        if legacy_method_name and legacy_method_name != new_method_name:
            legacy_path = Path(str(base_path).format(method=legacy_method_name))

            if legacy_path.exists():
                if warn_on_legacy:
                    warnings.warn(
                        f"Using legacy method name '{legacy_method_name}' in path '{legacy_path}'. "
                        f"Consider migrating to new name '{new_method_name}'.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                self._cache[cache_key] = legacy_path
                return legacy_path

        # Neither path exists
        self._cache[cache_key] = None
        return None

    def find_experimental_data(self,
                              experiment_pattern: str,
                              method: Union[Method, str],
                              file_pattern: str = "*.json*") -> List[Path]:
        """
        Find experimental data files for a given method, checking both naming conventions.

        Args:
            experiment_pattern: Pattern to match experiment directories (e.g., "*_poem*")
            method: Method to search for
            file_pattern: Pattern to match files within method directories

        Returns:
            List of matching file paths

        Examples:
            >>> resolver = DataPathResolver()
            >>> files = resolver.find_experimental_data("*_poem*", Method.VS_STANDARD, "responses.jsonl")
        """
        method_str = method.value if hasattr(method, 'value') else str(method)
        found_files = []

        # Get both possible method names
        new_name = self._get_new_method_name(method_str)
        legacy_name = self._get_legacy_method_name(method_str)

        method_names = [new_name]
        if legacy_name and legacy_name != new_name:
            method_names.append(legacy_name)

        # Search for experiment directories
        for experiment_dir in self.base_data_dir.glob(f"**/{experiment_pattern}"):
            if not experiment_dir.is_dir():
                continue

            # Look for method subdirectories
            for method_name in method_names:
                method_path = experiment_dir / "evaluation" / method_name
                if not method_path.exists():
                    continue

                # Find matching files
                for file_path in method_path.glob(file_pattern):
                    found_files.append(file_path)

                # Issue warning if using legacy naming
                if method_name == legacy_name and method_name != new_name:
                    warnings.warn(
                        f"Found data using legacy method name '{method_name}' in {method_path}. "
                        f"Consider migrating to new name '{new_name}'.",
                        DeprecationWarning,
                        stacklevel=2
                    )

        return sorted(found_files)

    def create_output_path(self,
                          base_path: Union[str, Path],
                          method: Union[Method, str],
                          ensure_parent: bool = True) -> Path:
        """
        Create an output path using the new naming convention.

        Args:
            base_path: Base path template that contains method name placeholder
            method: Method enum or string
            ensure_parent: Whether to create parent directories if they don't exist

        Returns:
            Output path using new naming convention

        Examples:
            >>> resolver = DataPathResolver()
            >>> path = resolver.create_output_path("output/{method}/results.json", Method.VS_STANDARD)
            >>> # Returns Path("output/vs_standard/results.json")
        """
        method_str = method.value if hasattr(method, 'value') else str(method)
        new_method_name = self._get_new_method_name(method_str)

        output_path = Path(str(base_path).format(method=new_method_name))

        if ensure_parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def get_method_mapping(self) -> Dict[str, str]:
        """
        Get the complete mapping from legacy to new method names.

        Returns:
            Dictionary mapping legacy names to new names
        """
        return METHOD_MIGRATION_MAP.copy()

    def _get_new_method_name(self, method_str: str) -> str:
        """Get the new method name for a given method string."""
        # If it's already a new name, return as-is
        if method_str in ["vs_standard", "vs_cot", "vs_multi"]:
            return method_str

        # If it's a legacy name, map to new name
        if method_str in METHOD_MIGRATION_MAP:
            return METHOD_MIGRATION_MAP[method_str]

        # If it's not a VS method, return as-is
        return method_str

    def _get_legacy_method_name(self, method_str: str) -> Optional[str]:
        """Get the legacy method name for a given method string."""
        # If it's already a legacy name, return as-is
        if method_str in METHOD_MIGRATION_MAP:
            return method_str

        # If it's a new name, find the corresponding legacy name
        reverse_map = {v: k for k, v in METHOD_MIGRATION_MAP.items()}
        return reverse_map.get(method_str)

    def clear_cache(self):
        """Clear the internal path resolution cache."""
        self._cache.clear()


# Convenience functions for common use cases

def resolve_data_path(path_template: str,
                     method: Union[Method, str],
                     base_data_dir: Union[str, Path] = "generated_data") -> Optional[Path]:
    """
    Convenience function to resolve a single data path.

    Args:
        path_template: Path template with {method} placeholder
        method: Method to resolve
        base_data_dir: Base data directory

    Returns:
        Resolved path or None if not found
    """
    resolver = DataPathResolver(base_data_dir)
    return resolver.resolve_method_path(path_template, method)


def find_method_data(experiment_pattern: str,
                    method: Union[Method, str],
                    file_pattern: str = "*.json*",
                    base_data_dir: Union[str, Path] = "generated_data") -> List[Path]:
    """
    Convenience function to find experimental data for a method.

    Args:
        experiment_pattern: Pattern to match experiment directories
        method: Method to search for
        file_pattern: Pattern to match files
        base_data_dir: Base data directory

    Returns:
        List of matching file paths
    """
    resolver = DataPathResolver(base_data_dir)
    return resolver.find_experimental_data(experiment_pattern, method, file_pattern)


def create_new_data_path(path_template: str,
                        method: Union[Method, str],
                        ensure_parent: bool = True) -> Path:
    """
    Convenience function to create a new data path using updated naming.

    Args:
        path_template: Path template with {method} placeholder
        method: Method to use
        ensure_parent: Whether to create parent directories

    Returns:
        Path using new naming convention
    """
    resolver = DataPathResolver()
    return resolver.create_output_path(path_template, method, ensure_parent)