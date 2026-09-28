# LyricCap Studio v0.1.7

## Sync Recovery

- Every audio track is aligned and validated independently before playlist offsets are applied.
- A hard/soft sync quality gate rejects abnormal duration, timestamp jumps, overlaps, count/order changes, and out-of-track timestamps.
- Failed primary alignment is retried only for that track, then routed to the optional WhisperX Japanese fallback.
- A failed track blocks the normal playlist SRT and is recorded in `output/_diagnostics/sync_report.json` and `.txt`.
- Explicit instrumental section tags are metadata, not sung lyric lines; their silence remains subtitle-free.
