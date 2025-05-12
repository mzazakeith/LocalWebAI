# Local-First AI Model Runner

This project aims to create a powerful TypeScript library that enables developers to run Large Language Models (LLMs) directly in web browsers and Node.js environments. The core value is to provide a privacy-preserving, offline-capable, and low-latency solution for AI inference without relying on external API calls.

This project is currently in **Phase 1: Foundation & Enhancement** after POC was achieved .

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mzazakeith/LocalWebAI)

## Current Status: Phase 1 In Progress

We have successfully demonstrated the following capabilities:

*   **In-Browser Inference**: Running `llama.cpp` compiled to WebAssembly (leveraging `llama-cpp-wasm` for the Wasm build).
*   **TypeScript Wrapper**: A `ts-wrapper` provides a developer-friendly API (`LlamaRunner`) to interact with the Wasm module.
*   **Model Loading**: Support for loading GGUF models from a URL or a user-provided File object.
*   **Robust Caching**: Models are cached in IndexedDB using chunking, supporting large models and providing faster subsequent loads.
*   **Granular Progress**: Detailed progress reporting for all loading stages (download, VFS write, metadata parsing, initialization).
*   **Metadata & Provenance**: Parsing, validation, and display of GGUF model metadata, including provenance information (source URL, download date, file details).
*   **Error Handling**: Specific and user-friendly error reporting for various issues (network, file, format, compatibility, Wasm).
*   **Cancellation**: Support for cancelling in-progress model downloads/loads.
*   **Streaming Output**: Token-by-token text generation streamed to the UI.
*   **Web Worker**: Inference runs in a separate Web Worker to maintain UI responsiveness.
*   **Demo**: A functional `index.html` within the `ts-wrapper` directory showcases these features.
*   **Server Setup**: Includes an Express.js server (`server.js`) at the project root, configured with necessary COOP/COEP headers for `SharedArrayBuffer` support, enabling multi-threaded Wasm execution.

## Project Structure

*   `ts-wrapper/`: Contains the core TypeScript library (`LlamaRunner`, `ModelCache`, `worker.ts`), a demo `index.html`, and its `package.json` for building the wrapper.
*   `llama-cpp-wasm/`: A git submodule or separate checkout of the `llama-cpp-wasm` project, used for its WebAssembly build artifacts (`main.js`, `main.wasm`).
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

2.  **Ensure `llama-cpp-wasm` Artifacts are Present**:
    *   The POC relies on build artifacts (specifically `main.js` and `main.wasm`) from the `llama-cpp-wasm` project. Ensure the `llama-cpp-wasm/dist/llama-mt/` directory (for multi-threaded) or `llama-cpp-wasm/dist/llama-st/` (for single-threaded, though `mt` is currently configured in the demo) contains these files. If not, you may need to build `llama-cpp-wasm` first by following its own `build.sh` or `build-multi-thread.sh` scripts.

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

## Next Steps & Future Vision

The recently completed enhancements to model loading, caching, and metadata handling have solidified the foundation. Key next steps from Phase 1 of our roadmap include:

*   Developing a **Node.js runtime** for the library.
*   Verifying and optimizing **WASM SIMD** performance.
*   Investigating basic **WebGL acceleration**.
*   Adding comprehensive **unit/integration tests** and improving **documentation**.
*   Creating a unified API surface for both browser and Node.js environments.

Future phases will focus on expanding model format support (ONNX, SafeTensors), adding higher-level task APIs (chat, embeddings, summarize), and creating integrations for popular JavaScript frameworks (React, Vue, Next.js, etc.).

For a detailed plan, please see the [**Project Roadmap (ROADMAP.md)**](./ROADMAP.md).

## Acknowledgements

This project builds upon the fantastic open-source work of others. We are deeply grateful to the developers and communities behind these projects:

*   **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: For the core C/C++ inference engine that makes high-performance LLM execution possible on a wide range of hardware. Their work is foundational to this project.
*   **[llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)**: For providing the WebAssembly build and JavaScript bindings for `llama.cpp`, which enabled the initial browser-based proof of concept for this library.

Thank you for open-sourcing your work and enabling projects like this one!

## Contributing

We welcome contributions to the Local-First AI Model Runner! Whether it's reporting a bug, suggesting a new feature, or submitting a pull request, your help is valued.

Please see our [**Contribution Guidelines (CONTRIBUTING.md)**](./CONTRIBUTING.md) for more details on how to get started.

We encourage you to:
*   Open an issue for any bugs you find or features you'd like to see.
*   Fork the repository and submit pull requests with your improvements.
