// Embed an FNV-1a 64 hash of src/lib.rs, so a test can tell whether an
// installed (e.g. committed) wheel was built from the source in this repo.
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    let src = fs::read("src/lib.rs").expect("src/lib.rs");
    let mut h: u64 = 0xcbf29ce484222325;
    for b in src {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    println!("cargo:rustc-env=WPFAST_SRC_FNV={:016x}", h);
}
