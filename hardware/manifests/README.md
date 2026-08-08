# Hardware evidence manifests

Each primary canary capture is a compressed NumPy file containing exactly three
one-dimensional arrays: `sample_ids`, `pair_ids`, and `predictions`. Every
`pair_id` is unique. Repeated diagnostic executions are stored separately and
cannot replace the primary independent pairing.

Create the capture first, hash its bytes, then write a manifest conforming to
`schemas/HardwareRunManifest.schema.json`. A SpiNNaker2 manifest uses a null
bitstream hash; a Virtex-7 manifest requires one. Never edit a captured artifact
in place—create a new run ID.

