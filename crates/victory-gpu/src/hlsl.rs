//! Our own shaders (the indirect argument clamp, the rectangle expander,
//! the pass that draws a depth plane from its twin), compiled at run time
//! from HLSL with their root signature inside: by the system's compiler
//! (`D3DCompile`, `cs_5_1`, `ps_5_1`, `vs_5_1`) and, when it refuses them,
//! by DXC (the `_6_0` profile of the same stage). Wine's `d3dcompiler_47`
//! (vkd3d-shader) rejects `RWByteAddressBuffer` loads, which the clamp uses;
//! DXC compiles the same source. On Windows `D3DCompile` is tried first, so
//! the picture there is what the system compiler makes of the source.
//!
//! A shader's root signature comes from that attribute, and Wine's compiler
//! is the one place that compiles the source and leaves the signature out
//! (no `RDAT` part, no `RTS0`): `CreateRootSignature(0, blob)` then has
//! nothing to read. Use the matching build-time DXC shader when available,
//! then try the run-time DXC.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::core::{PCSTR, s};

/// Compiles compute shader `source` (entry `main`) named `name` in
/// messages: the system's compiler first, then DXC.
pub(crate) fn compile_compute(name: &str, source: &str) -> windows::core::Result<Vec<u8>> {
    compile_stage(name, source, "main", "cs_5_1")
}

/// Compiles `source`, entry `entry`, as the shader model 5.1 stage
/// `target` (`cs_5_1`, `vs_5_1`, `ps_5_1`): the system's compiler first,
/// then the bundled or run-time DXC shader for that stage.
pub(crate) fn compile_stage(
    name: &str,
    source: &str,
    entry: &str,
    target: &str,
) -> windows::core::Result<Vec<u8>> {
    match system_compile_stage(name, source, entry, target) {
        Ok(code) if !signature_left_out(source, &code) => return Ok(code),
        Ok(_) => tracing::info!("{name}: the system compiler left out the root signature"),
        Err((e, msg)) => {
            tracing::info!(error = %e, "{name}: the system compiler refused it ({})", msg.trim())
        }
    }
    #[cfg(victory_bundled_hlsl)]
    if let Some(bytes) = bundled_shader(source, entry, target) {
        tracing::info!("{name}: using build-time DXC shader");
        return Ok(bytes.to_vec());
    }
    let profile = format!("{}_6_0", target.split('_').next().unwrap_or("cs"));
    dxc_compile_stage(name, source, entry, &profile)
        .inspect_err(|e| tracing::warn!(error = %e, "{name}: DXC failed too"))
}

/// Whether `source` declares a root signature and the compiled `code` does
/// not carry one: DXC writes it as `RTS0`, the system's compiler as `RDAT`.
/// Windows' compiler writes `RDAT`; Wine's writes neither.
fn signature_left_out(source: &str, code: &[u8]) -> bool {
    if !source.contains("[RootSignature(") {
        return false;
    }
    let Some(container) = victory_shader::Container::parse(code).ok() else {
        return false;
    };
    let has = |fourcc| container.part(victory_shader::FourCC(fourcc)).is_some();
    !has(*b"RTS0") && !has(*b"RDAT")
}

// Match the source so a changed shader can never pick up another helper's bytecode.
#[cfg(victory_bundled_hlsl)]
fn bundled_shader(source: &str, entry: &str, target: &str) -> Option<&'static [u8]> {
    macro_rules! bytecode {
        ($name:literal) => {
            include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".dxil"))
        };
    }
    match (source, entry, target) {
        (include_str!("../shaders/argument_clamp.hlsl"), "main", "cs_5_1") => {
            Some(bytecode!("argument_clamp"))
        }
        (include_str!("../shaders/rect_rewrite.hlsl"), "main", "cs_5_1") => {
            Some(bytecode!("rect_rewrite"))
        }
        (include_str!("../shaders/rect_expand.hlsl"), "main", "cs_5_1") => {
            Some(bytecode!("rect_expand"))
        }
        (include_str!("../shaders/plane_corners.hlsl"), "corner", "vs_5_1") => {
            Some(bytecode!("plane_corners"))
        }
        (include_str!("../shaders/plane_steps.hlsl"), "plane", "ps_5_1") => {
            Some(bytecode!("plane_steps"))
        }
        (include_str!("../shaders/plane_floats.hlsl"), "plane", "ps_5_1") => {
            Some(bytecode!("plane_floats"))
        }
        _ => None,
    }
}

/// Compiles `source` with the system's `D3DCompile` as `cs_5_1`; on failure
/// returns the error and the compiler's messages (the tests' probe).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn system_compile(
    name: &str,
    source: &str,
) -> Result<Vec<u8>, (windows::core::Error, String)> {
    system_compile_stage(name, source, "main", "cs_5_1")
}

/// Compiles `source` with the system's `D3DCompile` as `target`, entry
/// `entry`; on failure returns the error and the compiler's messages.
pub(crate) fn system_compile_stage(
    name: &str,
    source: &str,
    entry: &str,
    target: &str,
) -> Result<Vec<u8>, (windows::core::Error, String)> {
    let cname = std::ffi::CString::new(name).unwrap_or_default();
    let centry = std::ffi::CString::new(entry).unwrap_or_default();
    let ctarget = std::ffi::CString::new(target).unwrap_or_default();
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR(cname.as_ptr().cast()),
            None,
            None,
            PCSTR(centry.as_ptr().cast()),
            PCSTR(ctarget.as_ptr().cast()),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        return Err((
            e,
            errors
                .as_ref()
                .map(blob_bytes)
                .map(String::from_utf8_lossy)
                .map(Into::into)
                .unwrap_or_default(),
        ));
    }
    code.as_ref()
        .map(|c| blob_bytes(c).to_vec())
        .ok_or_else(|| (windows::core::Error::empty(), String::new()))
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, borrowed for the blob's lifetime.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

/// Where `dxcompiler.dll` is looked for: `VICTORY_DXC` (the DLL or its
/// folder), beside the `dxil.dll` shader validation uses (the SDK ships them
/// together), then the executable's folder and System32.
fn dxc_library() -> windows::core::Result<windows::Win32::Foundation::HMODULE> {
    use windows::Win32::System::LibraryLoader::{
        LOAD_LIBRARY_SEARCH_APPLICATION_DIR, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };
    use windows::core::{HSTRING, w};
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("VICTORY_DXC").map(std::path::PathBuf::from) {
        paths.push(if p.is_dir() {
            p.join("dxcompiler.dll")
        } else {
            p
        });
    }
    if let Some(dir) = victory_shader::emit::validator::Validator::default_path()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        paths.push(dir.join("dxcompiler.dll"));
    }
    for path in paths.iter().filter(|p| p.is_file()) {
        // SAFETY: a NUL-terminated path; its folder (dxil.dll) is searched first.
        if let Ok(m) = unsafe {
            LoadLibraryExW(
                &HSTRING::from(path.as_path()),
                None,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        } {
            return Ok(m);
        }
    }
    // SAFETY: a bare name, searched only in the executable's folder and System32.
    unsafe {
        LoadLibraryExW(
            w!("dxcompiler.dll"),
            None,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    }
}

/// Compiles compute shader `source` with DXC as `cs_6_0` (see
/// [`dxc_library`]); `name` labels its messages.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dxc_compile(name: &str, source: &str) -> windows::core::Result<Vec<u8>> {
    dxc_compile_stage(name, source, "main", "cs_6_0")
}

/// Compiles `source` with DXC as `profile` (`cs_6_0`, `ps_6_0`), entry
/// `entry` (see [`dxc_library`]); `name` labels its messages.
pub(crate) fn dxc_compile_stage(
    name: &str,
    source: &str,
    entry: &str,
    profile: &str,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Dxc::*;
    use windows::Win32::System::LibraryLoader::GetProcAddress;
    use windows::core::{HSTRING, Interface, PCWSTR, w};
    type Create = unsafe extern "system" fn(
        *const windows::core::GUID,
        *const windows::core::GUID,
        *mut *mut std::ffi::c_void,
    ) -> windows::core::HRESULT;
    // SAFETY: loading the compiler and calling its documented entry point with
    // out pointers of ours; the buffer and arguments outlive the calls.
    unsafe {
        let library = dxc_library()?;
        let create: Create = std::mem::transmute(
            GetProcAddress(library, s!("DxcCreateInstance"))
                .ok_or_else(windows::core::Error::from_thread)?,
        );
        let mut raw = std::ptr::null_mut();
        create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw).ok()?;
        let compiler = IDxcCompiler3::from_raw(raw);
        let buffer = DxcBuffer {
            Ptr: source.as_ptr().cast(),
            Size: source.len(),
            Encoding: DXC_CP_UTF8.0,
        };
        let target = HSTRING::from(profile);
        let entry = HSTRING::from(entry);
        let arguments = [
            w!("-T"),
            PCWSTR(target.as_ptr()),
            w!("-E"),
            PCWSTR(entry.as_ptr()),
            w!("-O3"),
        ];
        let result: IDxcResult = compiler.Compile(&buffer, Some(&arguments), None)?;
        if let Err(e) = result.GetStatus()?.ok() {
            if let Ok(errors) = result.GetErrorBuffer() {
                let text = std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                );
                tracing::warn!("{name}: DXC: {}", String::from_utf8_lossy(text).trim());
            }
            return Err(e);
        }
        let blob = result.GetResult()?;
        Ok(
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROBE: &str = r#"
        RWByteAddressBuffer b : register(u0);
        [RootSignature("UAV(u0)")]
        [numthreads(1, 1, 1)]
        void main() { b.Store(0, b.Load(0)); }
    "#;

    #[cfg(victory_bundled_hlsl)]
    #[test]
    fn bundled_helpers_have_shader_code_and_root_signatures() {
        use victory_shader::{Container, FourCC};

        for source in [
            include_str!("../shaders/argument_clamp.hlsl"),
            include_str!("../shaders/rect_rewrite.hlsl"),
            include_str!("../shaders/rect_expand.hlsl"),
        ] {
            let expected = bundled_shader(source, "main", "cs_5_1").unwrap();
            let bytes = compile_compute("bundled helper", source).unwrap();
            if system_compile("bundled helper", source).is_err() {
                assert_eq!(bytes, expected);
            }
            let container = Container::parse(expected).unwrap();
            assert!(container.part(FourCC::DXIL).is_some());
            assert!(container.part(FourCC::RTS0).is_some());
        }
        assert!(bundled_shader(PROBE, "main", "cs_5_1").is_none());
    }

    /// A raw-buffer compute shader compiles with one of the two compilers
    /// whenever either is present, and both results are DXBC containers.
    #[test]
    fn raw_buffer_shader_compiles_with_a_fallback() {
        let system = system_compile("probe", PROBE);
        let dxc = dxc_compile("probe", PROBE);
        for code in [system.as_ref().ok(), dxc.as_ref().ok()]
            .into_iter()
            .flatten()
        {
            assert_eq!(&code[..4], b"DXBC");
        }
        if system.is_ok() || dxc.is_ok() {
            assert_eq!(&compile_compute("probe", PROBE).unwrap()[..4], b"DXBC");
        } else {
            assert!(compile_compute("probe", PROBE).is_err());
        }
    }
}
