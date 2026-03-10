fn main() {
    let project_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ must be inside bindings/")
        .parent()
        .expect("bindings/ must be inside project root")
        .to_path_buf();

    let src = project_root.join("src");

    cc::Build::new()
        // Stage-1 include directories
        .include(src.join("encode"))
        .include(src.join("compare"))
        .include(src.join("index"))
        .include(src.join("canon"))
        .include(src.join("algebra"))
        .include(src.join("model"))
        // Stage-2 include directories
        .include(src.join("stage2/projection"))
        .include(src.join("stage2/cascade"))
        .include(src.join("stage2/inference"))
        .include(src.join("stage2/hebbian"))
        .include(src.join("stage2/persist"))
        // Stage-1 source files
        .file(src.join("encode/trine_encode.c"))
        .file(src.join("compare/trine_stage1.c"))
        .file(src.join("index/trine_route.c"))
        .file(src.join("canon/trine_canon.c"))
        .file(src.join("compare/trine_csidf.c"))
        .file(src.join("index/trine_field.c"))
        // Stage-2 source files
        .file(src.join("stage2/projection/trine_project.c"))
        .file(src.join("stage2/projection/trine_majority.c"))
        .file(src.join("stage2/cascade/trine_learned_cascade.c"))
        .file(src.join("stage2/cascade/trine_topology_gen.c"))
        .file(src.join("stage2/inference/trine_stage2.c"))
        .file(src.join("stage2/hebbian/trine_hebbian.c"))
        .file(src.join("stage2/hebbian/trine_accumulator.c"))
        .file(src.join("stage2/hebbian/trine_freeze.c"))
        .file(src.join("stage2/hebbian/trine_self_deepen.c"))
        .file(src.join("stage2/persist/trine_s2_persist.c"))
        .file(src.join("stage2/persist/trine_accumulator_persist.c"))
        .opt_level(2)
        .warnings(false) // C lib compiled with its own warnings via its Makefile
        .compile("trine");

    println!("cargo:rustc-link-lib=m");

    // Stage-1 rerun-if-changed
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", src.join("encode/trine_encode.c").display());
    println!("cargo:rerun-if-changed={}", src.join("compare/trine_stage1.c").display());
    println!("cargo:rerun-if-changed={}", src.join("index/trine_route.c").display());
    println!("cargo:rerun-if-changed={}", src.join("canon/trine_canon.c").display());
    println!("cargo:rerun-if-changed={}", src.join("encode/trine_encode.h").display());
    println!("cargo:rerun-if-changed={}", src.join("compare/trine_stage1.h").display());
    println!("cargo:rerun-if-changed={}", src.join("index/trine_route.h").display());
    println!("cargo:rerun-if-changed={}", src.join("canon/trine_canon.h").display());
    println!("cargo:rerun-if-changed={}", src.join("compare/trine_csidf.c").display());
    println!("cargo:rerun-if-changed={}", src.join("compare/trine_csidf.h").display());
    println!("cargo:rerun-if-changed={}", src.join("index/trine_field.c").display());
    println!("cargo:rerun-if-changed={}", src.join("index/trine_field.h").display());

    // Stage-2 rerun-if-changed — source files
    println!("cargo:rerun-if-changed={}", src.join("stage2/projection/trine_project.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/projection/trine_majority.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/cascade/trine_learned_cascade.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/cascade/trine_topology_gen.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/inference/trine_stage2.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_hebbian.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_accumulator.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_freeze.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_self_deepen.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/persist/trine_s2_persist.c").display());

    // Stage-2 rerun-if-changed — header files
    println!("cargo:rerun-if-changed={}", src.join("stage2/projection/trine_project.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/cascade/trine_learned_cascade.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/inference/trine_stage2.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_hebbian.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_accumulator.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/hebbian/trine_freeze.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/persist/trine_s2_persist.h").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/persist/trine_accumulator_persist.c").display());
    println!("cargo:rerun-if-changed={}", src.join("stage2/persist/trine_accumulator_persist.h").display());
}
