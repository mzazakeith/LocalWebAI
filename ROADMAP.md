# Project Roadmap: Local-First AI Model Runner

This document outlines the planned features and development trajectory for the Local-First AI Model Runner. Our vision is to create a comprehensive TypeScript library that enables developers to run AI models directly in browsers and Node.js environments without requiring external API calls, prioritizing privacy, offline usage, and low latency.

## Core Pillars

1.  **Cross-Platform Runtime & Model Support**: Seamlessly run models on various JavaScript environments.
2.  **Performance & Hardware Acceleration**: Optimize inference speed and resource usage.
3.  **Developer Experience**: Provide intuitive APIs, comprehensive tooling, and framework integrations.
4.  **Ecosystem & Community**: Foster a vibrant community and a rich model/plugin ecosystem.

## Phases & Features

### Phase 0: Proof of Concept (POC) - Completed

*   **Goal**: Validate core feasibility of running a GGUF model (TinyLLaMA, Qwen) in the browser using WebAssembly.
*   **Key Achievements**:
    *   Compiled `llama.cpp` to WebAssembly (leveraging `llama-cpp-wasm` builds).
    *   Created a TypeScript wrapper (`ts-wrapper`) for the Wasm module.
    *   Implemented model loading from URL or File, with progress reporting.
    *   Enabled token-by-token streaming for text generation.
    *   Ran inference in a Web Worker to keep the UI responsive.
    *   Basic IndexedDB caching for models (with current size limitations noted).
    *   Functional browser-based demo (`ts-wrapper/index.html`).
    *   HTTP server setup with COOP/COEP headers for `SharedArrayBuffer`.

### Phase 1: Foundation & Enhancement (Largely Completed)

*   **Goal**: Solidify the core library, expand model support, and improve basic performance.
*   **Runtime & API**: 
    *   Refine `LlamaRunner` (browser) and implement `NodeJsLlamaCppRunner` (Node.js) APIs.
    *   **[COMPLETED]** Implemented comprehensive error handling and reporting.
    *   **[COMPLETED]** **Node.js Runtime**: Successfully implemented a Node.js runtime environment by integrating a local clone of `node-llama-cpp` ([https://github.com/withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)). This provides access to mature native bindings for `llama.cpp`, including potential GPU support.
        *   **Metal GPU Troubleshooting (macOS)**: During Node.js runner development, Metal shader compilation errors were encountered with the default `node-llama-cpp` build. This was resolved by rebuilding `node-llama-cpp` with the `NODE_LLAMA_CPP_GPU=false` environment variable and ensuring the `NodeJsLlamaCppRunner` was instantiated with `gpu: false` to force CPU execution.
    *   **[NEXT FOCUS]** Unified API surface for both browser and Node.js environments (ongoing refinement).
*   **Model Management**:
    *   **[COMPLETED]** **Robust Model Loading**: Implemented chunking for IndexedDB caching (supporting large models), granular progress reporting for all stages (download, VFS write, metadata parsing, initialization), cancellation support, and detailed metadata/provenance handling (parsing, validation, storage, display).
*   **Performance**:
    *   **[NEXT FOCUS]** **WASM SIMD**: Ensure SIMD optimizations are effectively utilized (present in current `llama-cpp-wasm` builds, verify and document).
    *   **[NEXT FOCUS]** **Basic WebGL Acceleration**: Investigate and implement WebGL-based acceleration for matrix operations as an enhancement layer (as per POC spec).
    *   **[NEXT FOCUS]** Performance benchmarking tools and documented metrics.
*   **Developer Experience**:
    *   **[COMPLETED]** Basic test scripts for `NodeJsLlamaCppRunner` (`nodellamacpp-load-test.ts`, `nodellamacpp-inference-test.ts`).
    *   **[NEXT FOCUS]** Comprehensive unit and integration tests for `ts-wrapper` (both browser and Node.js runners).
    *   **[NEXT FOCUS]** Clearer documentation for setup, usage, and available parameters for both runners.

### Phase 2: Multi-Format Support & Task Abstractions (Node.js Runner Enhancements here too)

*   **Goal**: Broaden model compatibility and provide higher-level APIs for common tasks.
*   **Node.js Runner Enhancements (Leveraging `node-llama-cpp` capabilities)**:
    *   **Expand `GenerateTextParams`**: Augment parameters for `NodeJsLlamaCppRunner` to include more options supported by `node-llama-cpp` (e.g., `seed`, `repeatPenalty` structure, `grammar` string, `stopSequences`, `logitBias`).
    *   **Configurable Options**: Allow configuration of system prompt, context size, and other `LlamaContextOptions`/`LlamaChatSessionOptions` via `NodeJsLlamaCppRunner`.
    *   **Expose More APIs**: Investigate exposing other `node-llama-cpp` features like direct tokenization, embedding generation, and advanced context management through `NodeJsLlamaCppRunner`.
    *   **Advanced Testing**: Implement more complex tests for the Node.js runner, covering context management, chat history, varied sampling parameters, GBNF grammar usage, and potentially concurrent operations.
*   **Model Formats**:
    *   **ONNX Runtime**: Integrate ONNX model support, likely using `onnxruntime-web` for browsers and `onnxruntime-node` for Node.js.
        *   Adapter pattern for adding new model formats with minimal core changes.
    *   **SafeTensors**: Support for loading models and weights in SafeTensors format.
*   **Task-Specific Abstractions**: 
    *   High-level APIs for common AI tasks (initially for browser, then for Node.js where applicable):
        *   `generateText()` (refine for both)
        *   `chat()` (for conversational AI, handling chat history and templates)
        *   `embed()` (for generating text embeddings)
        *   Potentially: `summarize()`, `classify()`.
    *   Input/output formatting utilities.
    *   Prompt templating and management features.
*   **Hardware Acceleration**:
    *   **WebGPU Acceleration**: Implement WebGPU support for cutting-edge performance in supporting browsers.
    *   **Node.js GPU Support**: Further investigate and document leveraging `node-llama-cpp`'s existing GPU support (CUDA, Vulkan, Metal where stable/desired) for the Node.js runner, beyond the initial CPU-focused bring-up.
    *   Automatic capability detection and fallback between WebGPU, WebGL, SIMD, and basic WASM (browser) and CPU/GPU (Node.js).
*   **Browser Runner Strategy**:
    *   **Investigate `node-llama-cpp` for Browser**: Explore the feasibility and benefits of adapting `node-llama-cpp`'s architecture or its direct `llama.cpp` bindings/approach for the browser Wasm runner. This could potentially unify the core runner logic further, provide more comprehensive `llama.cpp` feature parity, and simplify long-term maintenance.

### Phase 3: Ecosystem & Framework Integrations

*   **Goal**: Make the library easily adoptable in popular frameworks and foster a community.
*   **Framework-Specific Integrations**:
    *   **React**: Hooks (e.g., `useLlamaCompletion`, `useLlamaChat`, `useLlamaEmbedding`).
    *   **Vue**: Composables with similar functionality.
    *   **Svelte**: Stores and actions.
    *   **Next.js/Nuxt.js**: Examples and guidance for both client-side and server-side (Node.js runtime) usage.
    *   **Node.js Frameworks (NestJS, Express, etc.)**: Clear integration patterns for backend AI tasks.
*   **Developer Tooling**: 
    *   Model conversion and quantization utility integration or recommendations.
    *   Interactive playground for testing prompts and parameters.
*   **Community & Model Hub (Longer Term Vision)**:
    *   Versioned model registry concepts (license compliance, integrity checks).
    *   Plugin architecture for community extensions (e.g., new model backends, custom task APIs).
    *   Showcase of community projects and use cases.

## Cross-Cutting Concerns

*   **Documentation**: Continuously improve and expand documentation with interactive examples, tutorials, and API references.
*   **Performance Telemetry**: Tools and methods for developers to understand performance characteristics on different devices/environments.
*   **Security**: Model integrity checks, sandboxed execution considerations, privacy by design.
*   **Licensing**: Clear guidance on model licenses and library usage.

## Acknowledgements

This project builds upon the fantastic open-source work of others. We are deeply grateful to the developers and communities behind these projects:

*   **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: For the core C/C++ inference engine.
*   **[llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)**: For providing the WebAssembly build and JavaScript bindings that enabled our initial browser POC.
*   **[node-llama-cpp](https://github.com/withcatai/node-llama-cpp)**: For the comprehensive Node.js bindings for `llama.cpp`, which significantly accelerated the development of our Node.js runtime and provides a rich set of features. Thank you for open-sourcing your work!

This roadmap is a living document and will be updated as the project evolves and based on community feedback. 