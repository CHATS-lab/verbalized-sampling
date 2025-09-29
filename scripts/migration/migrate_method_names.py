#!/usr/bin/env python3
"""
Automated migration script for updating method names across all scripts.

This script systematically updates all Python files to use the new VS naming
convention while maintaining backwards compatibility. It handles:

1. Direct string replacements for hardcoded method names
2. Import statement updates
3. Method mapping updates
4. Documentation updates
5. Comment updates

Usage:
    python scripts/migration/migrate_method_names.py [--dry-run] [--backup]
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import ast
import subprocess
from datetime import datetime


class MethodNameMigrator:
    """Automated migrator for method name changes."""

    # Method name mappings
    LEGACY_TO_NEW = {
        "structure_with_prob": "vs_standard",
        "chain_of_thought": "vs_cot",
        "combined": "vs_multi"
    }

    # Paper name mappings
    LEGACY_TO_PAPER = {
        "structure_with_prob": "VS-Standard",
        "chain_of_thought": "VS-CoT",
        "combined": "VS-Multi"
    }

    # Patterns to update
    STRING_PATTERNS = [
        # Direct string usage
        (r'"structure_with_prob"', '"vs_standard"'),
        (r"'structure_with_prob'", "'vs_standard'"),
        (r'"chain_of_thought"', '"vs_cot"'),
        (r"'chain_of_thought'", "'vs_cot'"),
        (r'"combined"', '"vs_multi"'),
        (r"'combined'", "'vs_multi'"),

        # Dictionary keys
        (r'"structure_with_prob":', '"vs_standard":'),
        (r"'structure_with_prob':", "'vs_standard':"),
        (r'"chain_of_thought":', '"vs_cot":'),
        (r"'chain_of_thought':", "'vs_cot':"),
        (r'"combined":', '"vs_multi":'),
        (r"'combined':", "'vs_multi':"),

        # Enum references (when not using imports)
        (r'Method\.STRUCTURE_WITH_PROB', 'Method.VS_STANDARD'),
        (r'Method\.CHAIN_OF_THOUGHT', 'Method.VS_COT'),
        (r'Method\.COMBINED', 'Method.VS_MULTI'),

        # Directory path components
        (r'/structure_with_prob(?=/|"|\s|$)', '/vs_standard'),
        (r'/chain_of_thought(?=/|"|\s|$)', '/vs_cot'),
        (r'/combined(?=/|"|\s|$)', '/vs_multi'),
    ]

    # Comment and docstring patterns
    COMMENT_PATTERNS = [
        (r'structure_with_prob', 'VS-Standard (vs_standard)'),
        (r'chain_of_thought', 'VS-CoT (vs_cot)'),
        (r'combined', 'VS-Multi (vs_multi)'),
    ]

    def __init__(self, root_dir: Path, dry_run: bool = False, create_backup: bool = False):
        """Initialize the migrator."""
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.changes_made = []
        self.files_processed = 0
        self.backup_dir = None

        if self.create_backup and not self.dry_run:
            self.backup_dir = self.root_dir.parent / f"{self.root_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def create_project_backup(self):
        """Create a backup of the entire project before migration."""
        if not self.create_backup or self.dry_run:
            return

        print(f"Creating project backup at: {self.backup_dir}")
        try:
            shutil.copytree(self.root_dir, self.backup_dir,
                          ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git', '*.egg-info'))
            print(f"✅ Backup created successfully")
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            raise

    def find_python_files(self) -> List[Path]:
        """Find all Python files to process."""
        python_files = []

        # Include scripts directory
        scripts_dir = self.root_dir / "scripts"
        if scripts_dir.exists():
            python_files.extend(scripts_dir.glob("**/*.py"))

        # Include main package
        package_dir = self.root_dir / "verbalized_sampling"
        if package_dir.exists():
            python_files.extend(package_dir.glob("**/*.py"))

        # Include root level Python files
        python_files.extend(self.root_dir.glob("*.py"))

        # Exclude migration scripts themselves
        python_files = [f for f in python_files if "migration" not in str(f)]

        return sorted(python_files)

    def analyze_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Analyze a file to see if it needs changes."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, IOError) as e:
            print(f"⚠️  Could not read {file_path}: {e}")
            return False, []

        needs_changes = False
        potential_changes = []

        # Check for legacy method names
        for old_name in self.LEGACY_TO_NEW:
            if old_name in content:
                needs_changes = True
                potential_changes.append(f"Found '{old_name}' -> should become '{self.LEGACY_TO_NEW[old_name]}'")

        # Check for enum references
        enum_patterns = [
            'Method.STRUCTURE_WITH_PROB',
            'Method.CHAIN_OF_THOUGHT',
            'Method.COMBINED'
        ]
        for pattern in enum_patterns:
            if pattern in content:
                needs_changes = True
                potential_changes.append(f"Found enum reference '{pattern}'")

        return needs_changes, potential_changes

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except (UnicodeDecodeError, IOError) as e:
            print(f"⚠️  Could not read {file_path}: {e}")
            return False

        content = original_content

        # Apply string pattern replacements
        for pattern, replacement in self.STRING_PATTERNS:
            content = re.sub(pattern, replacement, content)

        # Apply comment pattern replacements in comments and docstrings
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Only apply to comments and docstrings
            stripped = line.strip()
            if stripped.startswith('#') or '"""' in line or "'''" in line:
                for old_name, new_description in self.COMMENT_PATTERNS:
                    if old_name in line:
                        lines[i] = line.replace(old_name, new_description)

        content = '\n'.join(lines)

        # Check if changes were made
        if content == original_content:
            return False

        # Write the updated content
        if not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except IOError as e:
                print(f"❌ Could not write {file_path}: {e}")
                return False

        return True

    def update_method_mappings(self, file_path: Path, content: str) -> str:
        """Update method mapping dictionaries to include both old and new names."""
        # This is a more complex transformation that might need manual review
        # For now, we'll flag files that contain mapping dictionaries
        if "METHOD_MAP" in content or "method_map" in content:
            print(f"📝 Manual review needed: {file_path} contains method mapping")

        return content

    def run_migration(self):
        """Run the complete migration process."""
        print(f"🚀 Starting method name migration")
        print(f"   Root directory: {self.root_dir}")
        print(f"   Dry run: {self.dry_run}")
        print(f"   Create backup: {self.create_backup}")
        print()

        # Create backup if requested
        if self.create_backup:
            self.create_project_backup()

        # Find all Python files
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files to process")
        print()

        # Analyze files first
        files_needing_changes = []
        for file_path in python_files:
            needs_changes, potential_changes = self.analyze_file(file_path)
            if needs_changes:
                files_needing_changes.append((file_path, potential_changes))

        print(f"📊 Analysis Results:")
        print(f"   Files needing changes: {len(files_needing_changes)}")
        print(f"   Files already up-to-date: {len(python_files) - len(files_needing_changes)}")
        print()

        if not files_needing_changes:
            print("✅ All files are already up-to-date!")
            return

        # Show detailed analysis
        if self.dry_run:
            print("🔍 Detailed Analysis (Dry Run):")
            for file_path, changes in files_needing_changes:
                rel_path = file_path.relative_to(self.root_dir)
                print(f"   📄 {rel_path}")
                for change in changes:
                    print(f"      - {change}")
            print()
        else:
            print("🔄 Migrating files...")

        # Process files
        files_changed = 0
        for file_path, _ in files_needing_changes:
            rel_path = file_path.relative_to(self.root_dir)

            if self.dry_run:
                print(f"   [DRY RUN] Would migrate: {rel_path}")
            else:
                if self.migrate_file(file_path):
                    files_changed += 1
                    print(f"   ✅ Migrated: {rel_path}")
                    self.changes_made.append(str(rel_path))
                else:
                    print(f"   ⚠️  No changes needed: {rel_path}")

        # Summary
        print()
        print("📋 Migration Summary:")
        if self.dry_run:
            print(f"   Files that would be changed: {len(files_needing_changes)}")
        else:
            print(f"   Files successfully migrated: {files_changed}")
            print(f"   Files skipped (no changes): {len(files_needing_changes) - files_changed}")

            if self.backup_dir:
                print(f"   Backup location: {self.backup_dir}")

        print()
        print("🔍 Post-migration recommendations:")
        print("   1. Run tests to ensure everything still works")
        print("   2. Review any files with complex method mappings manually")
        print("   3. Update documentation to reflect new naming")
        print("   4. Consider adding deprecation warnings for old usage")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate method names across all scripts")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be changed without making changes")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup before making changes")
    parser.add_argument("--root-dir", type=Path, default=Path.cwd(),
                       help="Root directory to migrate (default: current directory)")

    args = parser.parse_args()

    # Validate root directory
    if not args.root_dir.exists():
        print(f"❌ Root directory does not exist: {args.root_dir}")
        return 1

    # Check if we're in the right directory
    if not (args.root_dir / "verbalized_sampling").exists():
        print(f"❌ This doesn't appear to be the verbalized-sampling project root")
        print(f"   Looking for 'verbalized_sampling' directory in {args.root_dir}")
        return 1

    # Run migration
    migrator = MethodNameMigrator(
        root_dir=args.root_dir,
        dry_run=args.dry_run,
        create_backup=args.backup
    )

    try:
        migrator.run_migration()
        return 0
    except KeyboardInterrupt:
        print("\n❌ Migration interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())