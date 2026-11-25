// Compiles kernels/eomoe.cu to a fatbin (SASS, so no driver PTX JIT is needed) that
// src/gpu.rs embeds and loads by kernel name.
// NVCC overrides the compiler path, EOMOE_CUDA_ARCH the SM (default 120 = Blackwell consumer).
use std::{env, path::PathBuf, process::Command};

fn main() {
    let src = "kernels/eomoe.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=EOMOE_CUDA_ARCH");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let arch = env::var("EOMOE_CUDA_ARCH").unwrap_or_else(|_| "120".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("eomoe.fatbin");
    let st = Command::new(&nvcc)
        .args(["-fatbin", "-O3", "-std=c++17"])
        .arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .arg("-o")
        .arg(&out)
        .arg(src)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc} (set NVCC=/path/to/nvcc): {e}"));
    assert!(st.success(), "nvcc failed on {src}");
}
