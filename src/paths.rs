use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Result};

/// Normalize an incoming path to a path inside the container's DATA_DIR.
///
/// Accepts:
///   - Container-absolute paths like `/data/foo.mp4`
///   - Windows-style paths like `D:\whisper-data\foo.mp4` (mapped → /data/foo.mp4
///     when they live under `HOST_MOUNT_ROOT`, e.g. `D:\whisper-data`)
///   - WSL paths like `/mnt/d/whisper-data/foo.mp4` (same mapping rule)
///   - Plain relative paths or filenames → resolved under DATA_DIR
///
/// Rejects any input containing `..` segments or otherwise escaping DATA_DIR.
pub fn resolve_in_data_dir(input: &str, data_dir: &Path) -> Result<PathBuf> {
    if input.contains("..") {
        return Err(anyhow!("path traversal segments are not allowed: {input}"));
    }
    let host_mount = std::env::var("HOST_MOUNT_ROOT").ok();
    let normalized = input.replace('\\', "/");

    let candidate: PathBuf = if normalized.starts_with("/data") {
        PathBuf::from(&normalized)
    } else if let Some(root) = host_mount.as_deref() {
        let root_norm = root.replace('\\', "/").trim_end_matches('/').to_string();
        let lower_input = normalized.to_ascii_lowercase();
        let lower_root = root_norm.to_ascii_lowercase();
        if let Some(rest) = lower_input.strip_prefix(&lower_root) {
            let rest_real = &normalized[normalized.len() - rest.len()..];
            data_dir.join(rest_real.trim_start_matches('/'))
        } else {
            data_dir.join(normalized.trim_start_matches('/'))
        }
    } else {
        data_dir.join(normalized.trim_start_matches('/'))
    };

    // Belt-and-braces: no Component::ParentDir survived
    if candidate
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(anyhow!("resolved path escapes DATA_DIR: {}", candidate.display()));
    }
    if !candidate.starts_with(data_dir) {
        return Err(anyhow!(
            "resolved path {} is outside DATA_DIR {}",
            candidate.display(),
            data_dir.display()
        ));
    }
    Ok(candidate)
}
