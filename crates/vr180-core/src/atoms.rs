//! MP4/MOV atom writers for spatial metadata.
//!
//! Replaces the Python `spatialmedia` dependency (Google's package
//! that we've patched twice this session for the `KeyError:
//! 'ambisonic_channel_ordering'` bug). Trivial to reimplement:
//! all three atoms (`sv3d`, `st3d`, `SA3D`) are < 100 bytes each
//! and we already have hex dumps from the Python output to test
//! against.
//!
//! Phase 0.8 — `write_sv3d_st3d`, `write_sa3d` byte builders +
//! `inject_into_mov(path, atoms)`.
