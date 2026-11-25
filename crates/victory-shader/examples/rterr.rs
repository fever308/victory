//! Local tool (not upstream): translate one dumped container and print the
//! refusal exactly (with its code offset). usage: rterr <file.dxbc>
fn main() {
    let path = &std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(path).unwrap();
    match std::panic::catch_unwind(|| victory_shader::emit::translate(&bytes)) {
        Ok(Ok(h)) => println!("ok dxil={}", h.dxil.len()),
        Ok(Err(e)) => println!("err {e}"),
        Err(_) => println!("panic"),
    }
}
