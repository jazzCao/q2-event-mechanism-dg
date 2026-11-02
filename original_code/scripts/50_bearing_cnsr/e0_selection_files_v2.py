"""Freeze the caller-owned registry too; preserve V1's historical receipt.

V1 copied documents but built cache indices from the caller's mutable registry.
Its tests covered record/role immutability, not that external cache-reference
alias. The production runtime always uses this corrected constructor.
"""
import copy
import e0_selection_files as v1


class Catalog(v1.Catalog):
    def __init__(self, registry, documents):
        # V1's index builder now receives recursively immutable, owned values.
        super().__init__(v1.immutable(copy.deepcopy(registry)),
                         v1.immutable(copy.deepcopy(documents)))


class CacheFiles(v1.CacheFiles):
    def __init__(self, runtime, catalog=None):
        runtime.assert_authorized()
        catalog = Catalog.from_context(runtime) if catalog is None else catalog
        if not isinstance(catalog,Catalog):raise TypeError('V2 owned immutable catalog required')
        super().__init__(runtime,catalog)
