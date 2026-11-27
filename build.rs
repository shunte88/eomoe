// Compiles kernels/eomoe.cu to a fatbin (SASS, so no driver PTX JIT is needed) that
// src/gpu.rs embeds and loads by kernel name.
// NVCC overrides the compiler path, EOMOE_CUDA_ARCH the SMs, comma-separated (default 120 = Blackwell
// consumer; releases build 80,86,89,90,100,120: the mma kernels need sm_80+).
// Also stamps the build time for `eomoe --version` (EOMOE_BUILT), so it reruns on src/ changes
// and skips nvcc when the fatbin is current.
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let src = "kernels/eomoe.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-changed=src");
    let built = Command::new("date").arg("+%Y-%m-%d %H:%M:%S").output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    println!("cargo:rustc-env=EOMOE_BUILT={}", built.trim());
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=EOMOE_CUDA_ARCH");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let arch = env::var("EOMOE_CUDA_ARCH").unwrap_or_else(|_| "120".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("eomoe.fatbin");
    let (key, key_file) = (format!("{nvcc} {arch}"), out.with_extension("key"));
    let mtime = |p: &str| fs::metadata(p).and_then(|m| m.modified()).ok();
    if mtime(out.to_str().unwrap()) > mtime(src) && fs::read_to_string(&key_file).ok().as_deref() == Some(&key) {
        return;
    }
    let st = Command::new(&nvcc)
        .args(["-fatbin", "-O3", "-std=c++17"])
        .args(arch.split(',').map(|a| format!("-gencode=arch=compute_{a},code=sm_{a}")))
        // A newer MSVC STL refuses an older nvcc; the kernels use no STL.
        .args((env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")).then_some("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"))
        .arg("-o")
        .arg(&out)
        .arg(src)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc} (set NVCC=/path/to/nvcc): {e}"));
    assert!(st.success(), "nvcc failed on {src}");
    fs::write(&key_file, key).unwrap();
}
