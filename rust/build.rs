fn main() {
    let hteb_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ must be inside hteb/")
        .to_path_buf();

    cc::Build::new()
        .include(&hteb_dir)
        .file(hteb_dir.join("trine_encode.c"))
        .file(hteb_dir.join("trine_stage1.c"))
        .file(hteb_dir.join("trine_route.c"))
        .file(hteb_dir.join("trine_canon.c"))
        .file(hteb_dir.join("trine_csidf.c"))
        .file(hteb_dir.join("trine_field.c"))
        .opt_level(2)
        .warnings(false) // C lib compiled with its own warnings via its Makefile
        .compile("trine");

    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_encode.c").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_stage1.c").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_route.c").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_canon.c").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_encode.h").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_stage1.h").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_route.h").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_canon.h").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_csidf.c").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_csidf.h").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_field.c").display());
    println!("cargo:rerun-if-changed={}", hteb_dir.join("trine_field.h").display());
}
