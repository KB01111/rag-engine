fn main() -> anyhow::Result<()> {
    apply_cli_overrides()?;
    let addr = std::env::var("CONTEXT_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8090".to_string())
        .parse()?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let engine = context_server::engine_from_env().await?;
        context_server::serve(addr, engine).await
    })
}

/// Process supported CLI flags and set corresponding `CONTEXT_*` environment variables.
///
/// Recognizes the following flags (each consumes a single value): `--host`, `--port`,
/// `--bind`, `--data-dir`, `--managed-root` (can be repeated), `--openviking-url`,
/// and `--openviking-api-key`. When provided, values are written to these environment
/// variables: `CONTEXT_BIND`, `CONTEXT_DATA_DIR`, `CONTEXT_ROOTS` (semicolon-separated),
/// `CONTEXT_OPENVIKING_URL`, and `CONTEXT_OPENVIKING_API_KEY`. If `--bind` is not given
/// but `--host` and/or `--port` are, `CONTEXT_BIND` is constructed as `host:port` using
/// defaults `127.0.0.1` and `8090` for missing parts.
///
/// # Returns
///
/// `Ok(())` on success; an `Err` if an unknown flag is encountered or a flag is missing its value.
///
/// # Examples
///
/// ```
/// // With no additional command-line arguments this should succeed.
/// assert!(crate::apply_cli_overrides().is_ok());
/// ```
fn apply_cli_overrides() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut host: Option<String> = None;
    let mut port: Option<String> = None;
    let mut bind: Option<String> = None;
    let mut data_dir: Option<String> = None;
    let mut managed_roots: Vec<String> = Vec::new();
    let mut openviking_url: Option<String> = None;
    let mut openviking_api_key: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = Some(expect_value(&mut args, "--host")?),
            "--port" => port = Some(expect_value(&mut args, "--port")?),
            "--bind" => bind = Some(expect_value(&mut args, "--bind")?),
            "--data-dir" => data_dir = Some(expect_value(&mut args, "--data-dir")?),
            "--managed-root" => managed_roots.push(expect_value(&mut args, "--managed-root")?),
            "--openviking-url" => {
                openviking_url = Some(expect_value(&mut args, "--openviking-url")?)
            }
            "--openviking-api-key" => {
                openviking_api_key = Some(expect_value(&mut args, "--openviking-api-key")?)
            }
            _ => {
                return Err(anyhow::anyhow!("unknown flag: {}", arg));
            }
        }
    }

    if let Some(bind) = bind.or_else(|| match (host, port) {
        (None, None) => None,
        (host, port) => Some(format!(
            "{}:{}",
            host.unwrap_or_else(|| "127.0.0.1".to_string()),
            port.unwrap_or_else(|| "8090".to_string())
        )),
    }) {
        std::env::set_var("CONTEXT_BIND", bind);
    }
    if let Some(data_dir) = data_dir {
        std::env::set_var("CONTEXT_DATA_DIR", data_dir);
    }
    if !managed_roots.is_empty() {
        std::env::set_var("CONTEXT_ROOTS", managed_roots.join(";"));
    }
    if let Some(openviking_url) = openviking_url {
        std::env::set_var("CONTEXT_OPENVIKING_URL", openviking_url);
    }
    if let Some(openviking_api_key) = openviking_api_key {
        std::env::set_var("CONTEXT_OPENVIKING_API_KEY", openviking_api_key);
    }

    Ok(())
}

/// Retrieves the next value from an argument iterator for the specified flag, returning an error if no value is available.
///
/// # Parameters
///
/// - `args`: iterator over command-line arguments positioned after the flag.
/// - `flag`: flag name used in the error message when a value is missing.
///
/// # Returns
///
/// `Ok(String)` containing the next argument value, or `Err` with the message `missing value for {flag}` if the iterator is exhausted.
///
/// # Examples
///
/// ```
/// let mut it = vec!["bar".to_string()].into_iter();
/// assert_eq!(expect_value(&mut it, "--foo").unwrap(), "bar");
/// ```
fn expect_value(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}
