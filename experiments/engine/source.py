"""Track local Python implementation dependencies without importing them."""

import ast
import hashlib
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PACKAGES = {'experiments', 'partmm', 'skelebone'}


def local_source_signatures(entry):
    """Hash the statically imported local sources, including relative imports."""
    entry = Path(entry).resolve()
    pending = [entry]
    seen = set()
    signatures = []
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        content = path.read_bytes()
        if path != entry:
            signatures.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()})
        try:
            relative = path.relative_to(SOURCE_ROOT)
            package = list(relative.parts[:-1])
        except ValueError:
            package = []
        modules = []
        for node in ast.walk(ast.parse(content, filename=str(path))):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    if not package or node.level > len(package):
                        continue
                    prefix = package[:len(package) - node.level + 1]
                    base = '.'.join(prefix + ([base] if base else []))
                if base:
                    modules.append(base)
                    modules.extend(f'{base}.{alias.name}' for alias in node.names if alias.name != '*')
        for module in modules:
            parts = module.split('.')
            if parts[0] not in PUBLIC_PACKAGES:
                continue
            candidate = SOURCE_ROOT.joinpath(*parts)
            source = candidate.with_suffix('.py')
            if source.is_file():
                pending.append(source)
            elif (candidate / '__init__.py').is_file():
                pending.append(candidate / '__init__.py')
            # Importing a submodule also executes its parent package initializers.
            for count in range(1, len(parts)):
                initializer = SOURCE_ROOT.joinpath(*parts[:count], '__init__.py')
                if initializer.is_file():
                    pending.append(initializer)
    return sorted(signatures, key=lambda item: item['path'])
