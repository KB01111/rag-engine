# AMD Acceleration Sidecars

`mistralrs` does not expose a ROCm/HIP backend. AMD and NPU acceleration should run as an OpenAI-compatible sidecar and be configured as a runtime provider.

## Lemonade NPU

```yaml
runtime:
  providers:
    - name: "lemonade"
      preset: "lemonade"
      # Defaults to http://127.0.0.1:8000/api/v1
      api_key: ""
```

## llama.cpp ROCm

```yaml
runtime:
  providers:
    - name: "llama-rocm"
      preset: "llama-cpp"
      # Defaults to http://127.0.0.1:8080/v1
      api_key: ""
```

## Ollama ROCm

```yaml
runtime:
  providers:
    - name: "ollama-rocm"
      preset: "ollama"
      # Defaults to http://127.0.0.1:11434/v1
      api_key: ""
```
