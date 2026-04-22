use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&["../../../proto/engine.proto"], &["../../../proto"])?;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let engine_path = out_dir.join("engine.rs");
    let generated = fs::read_to_string(&engine_path)?;
    let patched = generated.replace(
        "_cx: &mut Context<'_>",
        "_cx: &mut tonic::codegen::Context<'_>",
    );
    if patched != generated {
        fs::write(engine_path, patched)?;
    }

    Ok(())
}
