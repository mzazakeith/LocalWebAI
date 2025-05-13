# Node.js Integration Strategy using Cloned `node-llama-cpp`

This document outlines the plan to integrate a locally cloned `node-llama-cpp` repository into the `ts-wrapper` to provide Node.js runtime capabilities for the Local-First AI Model Runner. This approach supersedes the previous custom Node.js worker and Wasm runner.

## Rationale

Leveraging `node-llama-cpp` (even as a local clone rather than an NPM dependency) offers several advantages:
- Access to mature native bindings for `llama.cpp`, likely improving performance and stability over a pure Emscripten Node.js Wasm build.
- GPU support (Metal, CUDA, Vulkan) handled by `node-llama-cpp`.
- A higher-level JavaScript/TypeScript API for model interaction.
- Potential resolution of issues like `munmap` warnings and non-zero exit codes experienced with the direct Emscripten Wasm approach.
- More direct control over the `node-llama-cpp` codebase for modifications or deep debugging.

## Current `llama-cpp-wasm` Usage (Browser POC)

- **Build Artifacts:** Relies on pre-compiled Wasm artifacts (`main.js`, `main.wasm`) from `llama-cpp-wasm/dist/llama-mt/`.
- **`LlamaRunner` (Browser):**
    - Instantiates a Web Worker (`worker.ts`).
    - Worker imports `main.js` and locates `main.wasm`.
    - Communication via `postMessage` for commands and results.
    - Manages caching (`ModelCache`) and GGUF parsing (`gguf-parser.ts`).

## Plan Overview

The integration will proceed in the following phases:

**Phase A: Project Restructuring and Integration of `node-llama-cpp` Source**
**Phase B: Implement New Node.js Runner using `node-llama-cpp` API**
**Phase C: Testing and Refinement**
**Phase D: Documentation and Future Considerations**

---

## Phase A: Project Restructuring and Integration of `node-llama-cpp` Source

### 1. Integrate `node-llama-cpp` Source Code
   - **Status:** The `node-llama-cpp` repository is assumed to be already cloned locally (e.g., at `/Users/mzaza/Documents/Practice Projects/LocalWebAIV2/node-llama-cpp`).
   - **Action:** Configure `ts-wrapper/tsconfig.json` to use TypeScript Path Mapping to directly reference modules from the local `node-llama-cpp/src` directory.
     ```json
     // ts-wrapper/tsconfig.json
     {
       "compilerOptions": {
         // ... other options
         "baseUrl": ".", // This should be the directory of tsconfig.json
         "paths": {
           // Adjust the path based on the relative location of node-llama-cpp from ts-wrapper/src
           "@node-llama-cpp/*": ["../../node-llama-cpp/src/*"] 
         }
       }
     }
     ```
   - **Rationale:** Path mapping simplifies local development and avoids complex `npm link` or packaging steps for now.

### 2. Clean Up Obsolete `ts-wrapper` Code (Previous Node.js POC)
   - **Action:** Delete `ts-wrapper/src/node-llama-runner.ts`.
   - **Action:** Delete `ts-wrapper/src/node-worker.ts`.
   - **Action:** Delete `ts-wrapper/src/poc-node-test.ts`.
   - **Action:** Delete `ts-wrapper/src/poc-node-inference-test.ts`.
   - **Rationale:** These files were part of the custom Node.js Wasm runner which is being replaced.

### 3. Modify `ts-wrapper/package.json`
   - **Action:** Ensure `"type": "module"` is set in `ts-wrapper/package.json`.
   - **Action:** Review and update build scripts (`npm run build`) if necessary to ensure `tsc` correctly resolves path mappings.
   - **Action:** Remove any direct dependencies related to the old Node.js Wasm approach if they are no longer needed.

### 4. Build `node-llama-cpp` Native Addons
   - **Action:** Navigate to the local `node-llama-cpp` directory.
   - **Action:** Consult its `package.json` for build commands (e.g., `build`, `rebuild`, `compile`). Typically, running `npm install` within this directory should trigger the `cmake-js` build process for its native addons.
   - **Goal:** Successfully generate the `.node` addon files (e.g., in `node-llama-cpp/build/Release/`).
   - **Dependency Check:** Ensure all build dependencies for `node-llama-cpp` (CMake, C++ compiler etc.) are installed on the development machine.

---

## Phase B: Implement New Node.js Runner using `node-llama-cpp` API

### 1. Create `NodeJsLlamaCppRunner.ts`
   - **Action:** Create a new file: `ts-wrapper/src/node-llama-cpp-runner.ts`.
   - **Purpose:** This class will serve as the primary interface for using `node-llama-cpp`'s functionalities within `ts-wrapper` for Node.js environments. It will *not* use Web Workers for its core operations, as `node-llama-cpp` uses native bindings.

### 2. Implement `NodeJsLlamaCppRunner` Functionality
   - **Reference:** Utilize the official `node-llama-cpp` documentation ([https://node-llama-cpp.withcat.ai/guide/](https://node-llama-cpp.withcat.ai/guide/)).
   - **Imports:** Use path-mapped imports to access `node-llama-cpp` modules.
     ```typescript
     // Example:
     import { getLlama, LlamaModel, LlamaContext, LlamaChatSession /* ...etc */ } from '@node-llama-cpp/index';
     import * as path from 'path'; 
     ```
   - **Constructor:**
     - Design to accept necessary configurations.
   - **`loadModel(modelPath: string, params?: ModelLoadParams)`:**
     - Internal logic: `const llama = await getLlama();`, then `llama.loadModel({ modelPath: absoluteModelPath, ... })`.
     - Map `ts-wrapper`'s `ModelLoadParams` to `node-llama-cpp` parameters.
     - Adapt `node-llama-cpp` progress events to `ts-wrapper`'s `ProgressCallback` and `LoadingStage`.
     - Implement robust error handling, converting errors to `ts-wrapper`'s custom error types.
   - **`generateText(prompt: string, params: GenerateTextParams, tokenCallback: TokenCallback, completionCallback: CompletionCallback)`:**
     - Manage `LlamaModel`, `LlamaContext`, and potentially `LlamaChatSession` or `LlamaCompletion`.
     - Investigate `node-llama-cpp`'s API for token-by-token streaming (e.g., `sequence.evaluate(tokens)`) and adapt to `TokenCallback` and `CompletionCallback`.
   - **`getModelMetadata(): ModelSpecification | null`:**
     - Fetch metadata via `node-llama-cpp` and map to `ts-wrapper`'s `ModelSpecification` type.
   - **`terminate(): void`:**
     - Implement resource cleanup for `node-llama-cpp` (e.g., unload models, dispose contexts).
   - **Error Handling:** Consistently use custom error classes from `ts-wrapper/src/errors.ts`.

---

## Phase C: Testing and Refinement

### 1. Create New POC Test Files
   - **Action:** Develop `ts-wrapper/src/nodellamacpp-load-test.ts` for model loading tests.
   - **Action:** Develop `ts-wrapper/src/nodellamacpp-inference-test.ts` for text generation tests.
   - **Test Scope:**
     - Instantiate and use `NodeJsLlamaCppRunner`.
     - Load local models (e.g., `phi-2.Q4_K_M.gguf`).
     - Monitor progress via callbacks.
     - Perform inference and validate output.
     - Log metadata and errors.
     - Verify clean termination and absence of `exit(1)` issues.

### 2. Update Build and Run Scripts
   - **Action:** Modify `ts-wrapper/package.json` to include scripts for running the new Node.js tests (e.g., `npm run test:node-new`).

### 3. Iterative Debugging and Refinement
   - **Focus Areas:**
     - Correct model path resolutions.
     - Proper usage of the `node-llama-cpp` API.
     - Accurate adaptation of progress, token, and error reporting.
     - Successful build and location of `node-llama-cpp` native addons.
     - Confirmation that `munmap` warnings and `exit(1)` issues are resolved.

---

## Phase D: Documentation and Future Considerations

### 1. Update Project Documentation
   - **Action:** Revise `README.md` and `ROADMAP.md` to reflect the new Node.js strategy.
   - **Action:** Document the build process for `node-llama-cpp`'s native addons if it requires manual steps.
   - **Action:** Document the API of the new `NodeJsLlamaCppRunner`.

### 2. Review `llama-cpp-wasm` Fork Usage
   - **Decision Point:** Once `node-llama-cpp` successfully fulfills Node.js requirements, re-evaluate the role of our `llama-cpp-wasm` fork. It remains essential for the browser Wasm runner.
   - **Cleanup:** Our custom Node.js build scripts within `llama-cpp-wasm` (e.g., `build-node-single-thread.sh`) will become obsolete.

---
This plan provides a structured approach to integrating `node-llama-cpp` as a local clone, aiming for a robust and feature-rich Node.js runtime for the `ts-wrapper`. 