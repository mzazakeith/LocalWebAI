# Local-First AI Model Runner

This project aims to create a powerful TypeScript library that enables developers to run Large Language Models (LLMs) directly in web browsers and Node.js environments. The core value is to provide a privacy-preserving, offline-capable, and low-latency solution for AI inference without relying on external API calls.

This project is currently in **Phase 1: Foundation & Enhancement (Node.js integration largely complete, Browser work ongoing)**.

 <!-- [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mzazakeith/LocalWebAI) -->

## Current Status

We have successfully demonstrated the following capabilities:

*   **In-Browser Inference**: Running `llama.cpp` compiled to WebAssembly via a `LlamaRunner` in the `ts-wrapper`, now powered by the `wllama` library ([https://github.com/ngxson/wllama](https://github.com/ngxson/wllama)) for up-to-date `llama.cpp` features and active maintenance.
*   **Node.js Inference**: Successfully integrated `node-llama-cpp` ([https://github.com/withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)) to create a `NodeJsLlamaCppRunner`, enabling robust LLM execution in Node.js environments. This involved troubleshooting and resolving Metal GPU build issues on macOS to ensure stable CPU-based execution for initial testing.
*   **TypeScript Wrappers**: The `ts-wrapper` provides developer-friendly APIs for both browser (`LlamaRunner`) and Node.js (`NodeJsLlamaCppRunner`) environments.
*   **Model Loading**: Support for loading GGUF models from various sources.
*   **Robust Caching (Browser)**: Models are cached in IndexedDB using chunking, supporting large models and providing faster subsequent loads.
*   **Granular Progress**: Detailed progress reporting for all loading stages (download, VFS write, metadata parsing, initialization).
*   **Metadata & Provenance**: Parsing, validation, and display of GGUF model metadata, including provenance information (source URL, download date, file details).
*   **Error Handling**: Specific and user-friendly error reporting for various issues (network, file, format, compatibility, Wasm).
*   **Cancellation**: Support for cancelling in-progress model downloads/loads.
*   **Streaming Output**: Token-by-token text generation streamed to the UI.
*   **Web Worker**: Inference runs in a separate Web Worker to maintain UI responsiveness.
*   **Demo**: A functional `index.html` within the `ts-wrapper` directory showcases these features.
*   **Server Setup**: Includes an Express.js server (`server.js`) at the project root, configured with necessary COOP/COEP headers for `SharedArrayBuffer` support (relevant for multi-threaded browser Wasm).

## Project Structure

*   `ts-wrapper/`: Contains the core TypeScript library (`LlamaRunner` for browser, `NodeJsLlamaCppRunner` for Node.js, `ModelCache`), a demo `index.html`, and its `package.json` for building the wrapper.
*   `wllama/`: A local clone of the `wllama` project ([https://github.com/ngxson/wllama](https://github.com/ngxson/wllama)), which provides the WebAssembly bindings and JavaScript interface to `llama.cpp` for the browser-based `LlamaRunner`.
*   `llama-cpp-wasm/`: A git submodule or separate checkout of the `llama-cpp-wasm` project, which was used for the initial Proof of Concept browser Wasm build. It has now been superseded by `wllama` for the main browser runner.
*   `node-llama-cpp/`: A local clone of the `node-llama-cpp` project, used for its native Node.js bindings.
*   `emsdk/`: (If used directly) The Emscripten SDK, potentially used by `llama-cpp-wasm` for its builds.
*   `models/`: A suggested directory for storing downloaded GGUF model files (not version-controlled by default).
*   `server.js`: Node.js Express server at the root to serve the project with appropriate headers.
*   `package.json`: Root `package.json` for managing server dependencies (like Express) and root-level scripts.
*   `README.md`: This file.
*   `ROADMAP.md`: Outlines the future development plans and features.
*   `poc.md`: The initial Proof of Concept strategy document.
*   `full.txt`: The comprehensive long-term strategy document.

## How to Run the POC

1.  **Clone the Repository** (if you haven't already).
    ```bash
     git clone ...
     cd LocalWebAI
    ```

2.  **Ensure `wllama` Artifacts are Present and Configured**:
    *   The browser demo relies on build artifacts (specifically `.wasm` files and JavaScript modules) from the `wllama` project (see `wllama/` directory, which should be a local clone).
    *   Ensure the `wllama` project is built (e.g., by running `npm install && npm run build` within the `wllama` directory).
    *   The `LlamaRunner` in `ts-wrapper` will need to be configured with the correct paths to these `wllama` artifacts (e.g., `wllama/esm/single-thread/wllama.wasm`, `wllama/esm/multi-thread/wllama.wasm`). This is typically handled during `LlamaRunner` instantiation in the demo HTML/JS.

3.  **Install Root Dependencies** (for the server):
    *   Navigate to the project root (`LocalWebAI/`).
    *   Run: `npm install`

4.  **Install `ts-wrapper` Dependencies & Build**: 
    *   Navigate to the `ts-wrapper/` directory: `cd ts-wrapper`
    *   Run: `npm install` (to install TypeScript and any other wrapper-specific dev dependencies).
    *   Run: `npm run build` (to compile the TypeScript wrapper to JavaScript in `ts-wrapper/dist/`).

5.  **Start the Server**:
    *   Navigate back to the project root: `cd ..`
    *   Run: `npm start`
    *   This will start an HTTP server (usually on `http://localhost:8080`) with the necessary COOP/COEP headers.

6.  **Open in Browser**:
    *   Open `http://localhost:8080/ts-wrapper/index.html` in your web browser.

7.  **Test**: 
    *   Use the UI to load a GGUF model via URL or file upload.
        *   Example Small Model (previously mentioned as >1GiB with caching issues, now cacheable): `https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_k_m.gguf`
    *   Enter a prompt and click "Generate Text".
    *   Check the browser's developer console for logs and any errors.

8.  **Test Node.js Runner (from `ts-wrapper` directory)**:
    *   Ensure `node-llama-cpp` has been built for CPU (e.g., `cd ../node-llama-cpp && NODE_LLAMA_CPP_GPU=false npm run dev:build && cd ../ts-wrapper`).
    *   Run the load test: `npm run test:node-load`
    *   Run the inference test: `npm run test:node-inference` (uses `/tmp/phi-2.Q4_K_M.gguf` by default, adjust path or copy model if needed).

## Next Steps & Future Vision

The successful integration of `node-llama-cpp` for the Node.js runtime and `wllama` for the browser runtime marks significant milestones. Key next steps include:

*   Refining the unified API surface for both browser and Node.js runners.
*   Verifying and optimizing **WASM SIMD** performance for the browser runner (leveraging `wllama` capabilities).
*   Investigating basic **WebGL acceleration** for the browser.
*   Expanding `NodeJsLlamaCppRunner` capabilities by leveraging more features from `node-llama-cpp` (e.g., advanced generation parameters, embeddings).
*   Adding comprehensive **unit/integration tests** and improving **documentation** for both environments.

Future phases will focus on expanding model format support (ONNX, SafeTensors), adding higher-level task APIs (chat, embeddings, summarize), and creating integrations for popular JavaScript frameworks (React, Vue, Next.js, etc.).

For a detailed plan, please see the [**Project Roadmap (ROADMAP.md)**](./ROADMAP.md).

## Acknowledgements

This project builds upon the fantastic open-source work of others. We are deeply grateful to the developers and communities behind these projects:

*   **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: For the core C/C++ inference engine that makes high-performance LLM execution possible on a wide range of hardware. Their work is foundational to this project.
*   **[wllama](https://github.com/ngxson/wllama)**: For their excellent WebAssembly bindings for `llama.cpp`. `wllama` now powers our browser-based `LlamaRunner`, providing a modern, actively maintained, and feature-rich interface to `llama.cpp` directly in the browser.
*   **[node-llama-cpp](https://github.com/withcatai/node-llama-cpp)**: For their excellent Node.js bindings for `llama.cpp`. Their comprehensive library, active development, and clear documentation have been invaluable for implementing our Node.js runtime.
*   **[llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)**: For providing the WebAssembly build and JavaScript bindings for `llama.cpp` that enabled the initial browser-based proof of concept for this library.

Thank you for open-sourcing your work and enabling projects like this one!

## Contributing

We welcome contributions to the Local-First AI Model Runner! Whether it's reporting a bug, suggesting a new feature, or submitting a pull request, your help is valued.

Please see our [**Contribution Guidelines (CONTRIBUTING.md)**](./CONTRIBUTING.md) for more details on how to get started.

We encourage you to:
*   Open an issue for any bugs you find or features you'd like to see.
*   Fork the repository and submit pull requests with your improvements.
