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
///
/// The host-mount root is read from `HOST_MOUNT_ROOT` here; the pure logic lives
/// in [`resolve_in_data_dir_with_mount`] so it can be tested without the env.
pub fn resolve_in_data_dir(input: &str, data_dir: &Path) -> Result<PathBuf> {
    let host_mount = std::env::var("HOST_MOUNT_ROOT").ok();
    resolve_in_data_dir_with_mount(input, data_dir, host_mount.as_deref())
}

fn resolve_in_data_dir_with_mount(
    input: &str,
    data_dir: &Path,
    host_mount: Option<&str>,
) -> Result<PathBuf> {
    if input.contains("..") {
        return Err(anyhow!("path traversal segments are not allowed: {input}"));
    }
    let normalized = input.replace('\\', "/");

    let candidate: PathBuf = if normalized.starts_with("/data") {
        PathBuf::from(&normalized)
    } else if let Some(root) = host_mount {
        let root_norm = root.replace('\\', "/").trim_end_matches('/').to_string();
        let lower_input = normalized.to_ascii_lowercase();
        let lower_root = root_norm.to_ascii_lowercase();
        // Only treat this as a host-mount match when the root ends on a path
        // boundary — otherwise a sibling like `D:\whisper-data-evil` would share
        // the string prefix and get silently folded into /data.
        match lower_input.strip_prefix(&lower_root) {
            Some(rest) if rest.is_empty() || rest.starts_with('/') => {
                let rest_real = &normalized[normalized.len() - rest.len()..];
                data_dir.join(rest_real.trim_start_matches('/'))
            }
            _ => data_dir.join(normalized.trim_start_matches('/')),
        }
    } else {
        data_dir.join(normalized.trim_start_matches('/'))
    };

    // Belt-and-braces: no Component::ParentDir survived
    if candidate
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(anyhow!(
            "resolved path escapes DATA_DIR: {}",
            candidate.display()
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn data_dir() -> PathBuf {
        PathBuf::from("/data")
    }

    fn resolve(input: &str, host_mount: Option<&str>) -> Result<PathBuf> {
        resolve_in_data_dir_with_mount(input, &data_dir(), host_mount)
    }

    #[test]
    fn rejects_parent_dir_traversal() {
        let err = resolve("../etc/passwd", None).unwrap_err().to_string();
        assert!(err.contains("traversal"), "got: {err}");
    }

    #[test]
    fn rejects_embedded_parent_segment() {
        let err = resolve("foo/../../etc/passwd", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("traversal"), "got: {err}");
    }

    #[test]
    fn rejects_dotdot_inside_filename_conservatively() {
        // The `..` guard is intentionally blunt: even a literal ".." inside a
        // filename is refused. Safe over clever — locks the behavior in.
        let err = resolve("my..movie.mp4", None).unwrap_err().to_string();
        assert!(err.contains("traversal"), "got: {err}");
    }

    #[test]
    fn container_absolute_path_passes_through() {
        assert_eq!(
            resolve("/data/clip.mp4", None).unwrap(),
            PathBuf::from("/data/clip.mp4")
        );
    }

    #[test]
    fn plain_filename_resolved_under_data_dir() {
        assert_eq!(
            resolve("clip.mp4", None).unwrap(),
            PathBuf::from("/data/clip.mp4")
        );
    }

    #[test]
    fn subdirectory_filename_resolved_under_data_dir() {
        assert_eq!(
            resolve("lectures/day1.mp4", None).unwrap(),
            PathBuf::from("/data/lectures/day1.mp4")
        );
    }

    #[test]
    fn windows_host_path_mapped_to_data_dir() {
        assert_eq!(
            resolve("D:\\whisper-data\\clip.mp4", Some("D:\\whisper-data")).unwrap(),
            PathBuf::from("/data/clip.mp4")
        );
    }

    #[test]
    fn host_mount_match_is_case_insensitive_but_preserves_filename_case() {
        assert_eq!(
            resolve("d:/WHISPER-DATA/Clip.MP4", Some("D:/whisper-data")).unwrap(),
            PathBuf::from("/data/Clip.MP4")
        );
    }

    #[test]
    fn sibling_host_dir_is_not_folded_into_mount() {
        // `D:\whisper-data-evil` shares a string prefix with the mount root but
        // is a different directory. It must NOT be mapped as if under the mount;
        // it falls through to a (nonexistent) path under DATA_DIR instead.
        assert_eq!(
            resolve("D:\\whisper-data-evil\\x.mp4", Some("D:\\whisper-data")).unwrap(),
            PathBuf::from("/data/D:/whisper-data-evil/x.mp4")
        );
    }

    #[test]
    fn wsl_host_path_mapped_to_data_dir() {
        assert_eq!(
            resolve("/mnt/d/whisper-data/clip.mp4", Some("/mnt/d/whisper-data")).unwrap(),
            PathBuf::from("/data/clip.mp4")
        );
    }

    #[test]
    fn path_outside_mount_still_lands_under_data_dir() {
        assert_eq!(
            resolve("/some/other/clip.mp4", Some("D:/whisper-data")).unwrap(),
            PathBuf::from("/data/some/other/clip.mp4")
        );
    }

    #[test]
    fn sibling_dir_with_data_prefix_is_rejected() {
        // "/database/..." shares the "/data" string prefix but is NOT under the
        // /data directory. The component-wise check must reject it.
        let err = resolve("/database/evil.mp4", None).unwrap_err().to_string();
        assert!(err.contains("outside DATA_DIR"), "got: {err}");
    }
}
