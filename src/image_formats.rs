use std::path::Path;

pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "avif"];

#[must_use]
pub fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| SUPPORTED_IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::is_supported_image_path;

    #[test]
    fn recognizes_supported_extensions_case_insensitively() {
        assert!(is_supported_image_path(Path::new("sample.jpg")));
        assert!(is_supported_image_path(Path::new("sample.AVIF")));
        assert!(is_supported_image_path(Path::new("sample.webp")));
        assert!(!is_supported_image_path(Path::new("sample.gif")));
        assert!(!is_supported_image_path(Path::new("sample")));
    }
}
