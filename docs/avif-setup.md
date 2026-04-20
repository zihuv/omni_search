# AVIF Setup Guide

本文档说明 `omni_search` 在不同平台上启用可选 AVIF 解码支持时需要安装的系统依赖。

默认构建不会启用 AVIF。需要读取 `.avif` 图片时，请显式打开 crate feature：

```bash
cargo build --features avif
```

当前仓库的 `avif` feature 会透传 Rust `image` crate 的 `avif` 与 `avif-native` 特性。其中解码路径依赖：

- `libdav1d` / `dav1d`
- `pkg-config` 或兼容实现

如果你需要在运行时读取 `.avif` 图片，除了启用 `--features avif` 之外，还需要满足下面的系统依赖。

## Requirements

构建时至少需要：

- Rust 工具链
- `dav1d >= 1.3.0`
- `pkg-config`，或者通过 `SYSTEM_DEPS_DAV1D_*` 环境变量手动覆盖链接信息

运行时还需要：

- 动态库能被系统加载器找到

在 Linux 和 macOS 上，如果你通过系统包管理器安装，通常构建和运行都会自动满足。
在 Windows 上，除了构建期能找到 `dav1d.lib` 外，运行期还需要让 `dav1d.dll` 出现在 `PATH` 或和可执行文件在同一目录。

## macOS

推荐使用 Homebrew：

```bash
brew install dav1d pkgconf
```

通常这就够了。若 `cargo build --features avif` 仍提示找不到 `dav1d`，补上：

```bash
export PKG_CONFIG_PATH="$(brew --prefix dav1d)/lib/pkgconfig:$PKG_CONFIG_PATH"
```

验证：

```bash
pkg-config --modversion dav1d
pkg-config --cflags --libs dav1d
cargo test --features avif preprocess::image_path::tests::decodes_avif_bytes_from_png_path -- --exact
```

## Linux

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install -y libdav1d-dev pkg-config
```

### Fedora / RHEL / CentOS Stream

```bash
sudo dnf install -y libdav1d-devel pkgconf-pkg-config
```

### Arch Linux

```bash
sudo pacman -S --needed dav1d pkgconf
```

验证：

```bash
pkg-config --modversion dav1d
pkg-config --cflags --libs dav1d
cargo test --features avif preprocess::image_path::tests::decodes_avif_bytes_from_png_path -- --exact
```

## Windows

Windows 推荐两种方式。

### Option 1: `pkg-config` + `vcpkg`

这是最通用、最适合给其他用户复用的方式。

1. 安装 `pkg-config`

```powershell
winget install --id bloodrock.pkg-config-lite --accept-package-agreements --accept-source-agreements
```

2. 安装 `vcpkg`

```powershell
git clone https://github.com/microsoft/vcpkg.git $env:USERPROFILE\vcpkg
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat" -disableMetrics
```

3. 安装 `dav1d`

```powershell
& "$env:USERPROFILE\vcpkg\vcpkg.exe" install dav1d:x64-windows
```

4. 配置当前终端

```powershell
$env:PKG_CONFIG_PATH = "$env:USERPROFILE\vcpkg\installed\x64-windows\lib\pkgconfig"
$env:PATH = "$env:USERPROFILE\vcpkg\installed\x64-windows\bin;$env:PATH"
```

5. 如果希望新终端自动生效，写入用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable(
  'PKG_CONFIG_PATH',
  "$env:USERPROFILE\vcpkg\installed\x64-windows\lib\pkgconfig",
  'User'
)

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$dav1dBin = "$env:USERPROFILE\vcpkg\installed\x64-windows\bin"
if ([string]::IsNullOrWhiteSpace($userPath)) {
  $newPath = $dav1dBin
} elseif (($userPath -split ';') -notcontains $dav1dBin) {
  $newPath = "$dav1dBin;$userPath"
} else {
  $newPath = $userPath
}
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
```

### Option 2: 手动提供 `dav1d.lib` 和 `dav1d.dll`

如果你已经有可用的 `dav1d.dll`，也可以手动准备一个目录，里面至少包含：

- `dav1d.lib`
- `dav1d.dll`

然后绕过 `pkg-config`：

```powershell
$env:SYSTEM_DEPS_DAV1D_NO_PKG_CONFIG = '1'
$env:SYSTEM_DEPS_DAV1D_SEARCH_NATIVE = 'C:\path\to\dav1d'
$env:SYSTEM_DEPS_DAV1D_LIB = 'dav1d'
$env:PATH = "C:\path\to\dav1d;$env:PATH"
```

如果希望新终端自动生效：

```powershell
[Environment]::SetEnvironmentVariable('SYSTEM_DEPS_DAV1D_NO_PKG_CONFIG', '1', 'User')
[Environment]::SetEnvironmentVariable('SYSTEM_DEPS_DAV1D_SEARCH_NATIVE', 'C:\path\to\dav1d', 'User')
[Environment]::SetEnvironmentVariable('SYSTEM_DEPS_DAV1D_LIB', 'dav1d', 'User')
```

注意：

- 只有 `dav1d.dll` 还不够，构建阶段还需要 `dav1d.lib`。
- 如果你手头只有 `dav1d.dll`，需要另外生成 import library，或者改用 `vcpkg` 安装。

### Windows 验证

```powershell
pkg-config --modversion dav1d
cargo test --features avif preprocess::image_path::tests::decodes_avif_bytes_from_png_path -- --exact
```

如果你走的是手动覆盖路线，`pkg-config --modversion dav1d` 可以失败，但 `cargo test --features avif` 应该能通过。

## Project Verification

在当前仓库中，除了单元测试外，还可以直接用 CLI 验证真实图片：

```bash
cargo run --features avif --bin omni_search -- "测试文本" "/absolute/path/to/example.avif"
```

如果你想指定某个 bundle，例如 OpenCLIP / MobileCLIP2：

```bash
OMNI_BUNDLE_DIR=models/mobileclip2 cargo run --features avif --bin omni_search -- "测试文本" "/absolute/path/to/example.avif"
```

Windows PowerShell 写法：

```powershell
$env:OMNI_BUNDLE_DIR = 'models/mobileclip2'
cargo run --features avif --bin omni_search -- "测试文本" "C:\path\to\example.avif"
```

## Troubleshooting

常见报错和含义：

- `The pkg-config command could not be found`
  - 没装 `pkg-config`，或者它不在 `PATH` 中
- `The system library dav1d required by crate dav1d-sys was not found`
  - 没装 `dav1d` 开发包，或者 `PKG_CONFIG_PATH` 没指到 `dav1d.pc`
- Windows 上运行时报 `dav1d.dll was not found`
  - 运行期找不到 DLL，把 DLL 目录加到 `PATH`
- 只有 `.dll` 没有 `.lib`
  - 构建期仍然会失败，需要补 import library，或者改走 `vcpkg`

## Notes

- 当前项目已经验证 AVIF 图片可用于 `mobileclip2` 和 `chinese_clip_flat` 的图片嵌入。
- 如果未来改成纯 Rust AVIF 解码路径，这份文档也需要同步更新。
