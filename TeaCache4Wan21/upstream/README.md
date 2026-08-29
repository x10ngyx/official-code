# Upstream source

`teacache_generate.py` is the byte-exact TeaCache Wan2.1 reference recorded in
`../upstream_lock.json`. It is not the project entry point. The active entry
point is `../generate.py`, which executes original Wan2.1 unchanged for the
baseline and imports the official functions from this file only when TeaCache
is explicitly enabled.
