#[cfg(target_os = "windows")]
fn main() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let rc_file = manifest_dir.join("build/windows/cambia-cli.rc");
    let res_file = out_dir.join("cambia.res");

    println!("cargo:rerun-if-changed={}", rc_file.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir
            .join("build/windows/cambia-cli.exe.manifest")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir
            .join("build/windows/assets/cambia.ico")
            .display()
    );

    let rc = find_tool("rc.exe").unwrap_or_else(|| {
        panic!(
            "rc.exe was not found. Install the Windows SDK or run from a Developer Command Prompt."
        )
    });

    let status = Command::new(&rc)
        .current_dir(manifest_dir.join("build/windows"))
        .arg("/nologo")
        .arg(format!("/fo{}", res_file.display()))
        .arg(&rc_file)
        .status()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", rc.display()));

    if !status.success() {
        panic!("rc.exe failed while compiling {}", rc_file.display());
    }

    println!("cargo:rustc-link-arg-bin=cambia-cli={}", res_file.display());
    println!("cargo:rustc-link-arg-bin=cambia-gui={}", res_file.display());
}

#[cfg(not(target_os = "windows"))]
fn main() {}

#[cfg(target_os = "windows")]
fn find_tool(name: &str) -> Option<std::path::PathBuf> {
    use std::env;
    use std::path::PathBuf;

    if let Some(path) = env::var_os("PATH") {
        for dir in env::split_paths(&path) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let sdk_root = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let versions = ["10.0.26100.0", "10.0.22621.0"];
    let arches = ["x64", "x86"];

    for version in versions {
        for arch in arches {
            let candidate = sdk_root.join(version).join(arch).join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}
