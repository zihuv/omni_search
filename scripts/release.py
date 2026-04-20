from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
CATEGORY_NAMES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
CATEGORY_PATTERN = re.compile(
    r"^### (?P<name>Added|Changed|Deprecated|Removed|Fixed|Security)[ \t]*$\n?"
    r"(?P<body>.*?)(?=^### (?:Added|Changed|Deprecated|Removed|Fixed|Security)[ \t]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
PACKAGE_VERSION_PATTERN = re.compile(
    r'(^\[package\]\s*$.*?^version\s*=\s*")([^"]+)(".*$)',
    re.MULTILINE | re.DOTALL,
)


class ReleaseError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release omni_search from Python.")
    parser.add_argument("version", help="Release version, for example 0.0.1")
    parser.add_argument("--remote", default="origin", help="Git remote name")
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Skip cargo publish; requires --skip-push",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing the branch and tag",
    )
    args = parser.parse_args()

    if not VERSION_PATTERN.fullmatch(args.version):
        parser.error("version must match X.Y.Z or X.Y.Z-prerelease")

    if args.skip_publish and not args.skip_push:
        raise ReleaseError(
            "--skip-publish requires --skip-push. Pushing a release tag without "
            "publishing the crate is not supported."
        )

    return args


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def manifest_path() -> Path:
    return repo_root() / "Cargo.toml"


def changelog_path() -> Path:
    return repo_root() / "docs" / "CHANGELOG.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def format_command(args: list[str]) -> str:
    return " ".join(args)


def invoke_external(args: list[str], cwd: Path) -> None:
    print(f">> {format_command(args)}")
    completed = subprocess.run(args, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise ReleaseError(f"Command failed: {format_command(args)}")


def get_external_output(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        if output:
            raise ReleaseError(f"Command failed: {format_command(args)}\n{output}")
        raise ReleaseError(f"Command failed: {format_command(args)}")
    return output


def get_package_version(manifest: Path) -> str:
    content = read_text(manifest)
    match = re.search(
        r'^\[package\]\s*$.*?^version\s*=\s*"([^"]+)"\s*$',
        content,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ReleaseError("Could not find [package].version in Cargo.toml.")
    return match.group(1)


def set_package_version(manifest: Path, new_version: str) -> None:
    content = read_text(manifest)
    updated, count = PACKAGE_VERSION_PATTERN.subn(rf"\g<1>{new_version}\g<3>", content, count=1)
    if count == 0 or updated == content:
        raise ReleaseError("Failed to update [package].version in Cargo.toml.")
    write_text(manifest, updated)


def update_changelog_for_release(changelog: Path, version: str) -> None:
    if not changelog.exists():
        raise ReleaseError(f"CHANGELOG.md not found at {changelog}.")

    content = read_text(changelog)
    newline = "\r\n" if "\r\n" in content else "\n"
    normalized = content.replace("\r\n", "\n")

    unreleased_match = re.search(r"^## \[Unreleased\]\s*$", normalized, re.MULTILINE)
    if not unreleased_match:
        raise ReleaseError("docs/CHANGELOG.md is missing the ## [Unreleased] section.")

    unreleased_body_start = unreleased_match.end()
    after_unreleased = normalized[unreleased_body_start:]
    next_section_match = re.search(r"^## \[", after_unreleased, re.MULTILINE)

    if next_section_match:
        unreleased_body = after_unreleased[: next_section_match.start()]
        remaining_sections = after_unreleased[next_section_match.start() :]
    else:
        unreleased_body = after_unreleased
        remaining_sections = ""

    category_bodies = {category: "" for category in CATEGORY_NAMES}
    trimmed_unreleased_body = unreleased_body.strip("\n")
    for match in CATEGORY_PATTERN.finditer(trimmed_unreleased_body):
        category_bodies[match.group("name")] = match.group("body").strip()

    unsupported_content = CATEGORY_PATTERN.sub("", trimmed_unreleased_body).strip()
    if unsupported_content:
        raise ReleaseError(
            "docs/CHANGELOG.md contains unsupported content under Unreleased. "
            "Only standard category sections are supported."
        )

    release_blocks = []
    for category in CATEGORY_NAMES:
        body = category_bodies[category]
        if body.strip():
            release_blocks.append(f"### {category}\n\n{body}")

    if not release_blocks:
        raise ReleaseError("docs/CHANGELOG.md does not contain any Unreleased entries to release.")

    release_date = date.today().strftime("%Y-%m-%d")
    empty_unreleased = "\n".join(
        [
            "## [Unreleased]",
            "",
            "### Added",
            "",
            "### Changed",
            "",
            "### Deprecated",
            "",
            "### Removed",
            "",
            "### Fixed",
            "",
            "### Security",
        ]
    )
    release_section = "\n".join(
        [
            f"## [{version}] - {release_date}",
            "",
            "\n\n".join(release_blocks),
        ]
    )

    prefix = normalized[: unreleased_match.start()]
    updated = "\n".join(
        [
            prefix.rstrip("\n"),
            "",
            empty_unreleased,
            "",
            release_section,
        ]
    )

    if remaining_sections.strip():
        updated = "\n".join([updated.rstrip("\n"), "", remaining_sections.lstrip("\n")])

    if content.endswith("\r\n") or content.endswith("\n"):
        updated = updated.rstrip("\n") + "\n"

    write_text(changelog, updated.replace("\n", newline))


def assert_clean_worktree(cwd: Path) -> None:
    status = get_external_output(["git", "status", "--short"], cwd)
    if status:
        raise ReleaseError(
            "Git worktree is not clean. Commit or stash changes before running the release script."
        )


def assert_remote_exists(remote_name: str, cwd: Path) -> None:
    get_external_output(["git", "remote", "get-url", remote_name], cwd)


def assert_tag_does_not_exist(remote_name: str, tag_name: str, cwd: Path) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag_name}"],
        cwd=cwd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode == 0:
        raise ReleaseError(f"Tag {tag_name} already exists locally.")

    remote_tag = get_external_output(["git", "ls-remote", "--tags", remote_name, f"refs/tags/{tag_name}"], cwd)
    if remote_tag:
        raise ReleaseError(f"Tag {tag_name} already exists on remote {remote_name}.")


def get_current_branch(cwd: Path) -> str:
    branch = get_external_output(["git", "branch", "--show-current"], cwd)
    if not branch:
        raise ReleaseError("Detached HEAD is not supported for releases.")
    return branch


def main() -> int:
    try:
        args = parse_args()
        root = repo_root()
        manifest = manifest_path()
        changelog = changelog_path()

        assert_clean_worktree(root)
        assert_remote_exists(args.remote, root)

        current_version = get_package_version(manifest)
        if current_version == args.version:
            raise ReleaseError(f"Cargo.toml is already at version {args.version}.")

        tag_name = f"v{args.version}"
        assert_tag_does_not_exist(args.remote, tag_name, root)

        branch = get_current_branch(root)

        print(f"Updating Cargo.toml version: {current_version} -> {args.version}")
        set_package_version(manifest, args.version)
        print(f"Updating docs/CHANGELOG.md for release {args.version}")
        update_changelog_for_release(changelog, args.version)

        invoke_external(["cargo", "test"], root)
        invoke_external(["cargo", "publish", "--dry-run", "--locked", "--allow-dirty"], root)

        invoke_external(["git", "add", "Cargo.toml", "Cargo.lock", "docs/CHANGELOG.md"], root)
        invoke_external(["git", "commit", "-m", args.version], root)

        if not args.skip_publish:
            invoke_external(["cargo", "publish", "--locked"], root)

        invoke_external(["git", "tag", tag_name], root)

        if not args.skip_push:
            invoke_external(["git", "push", args.remote, branch], root)
            invoke_external(["git", "push", args.remote, tag_name], root)

        print(f"Release {args.version} completed.")
        return 0
    except ReleaseError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
