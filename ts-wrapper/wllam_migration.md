# Plan: Integrating Local wllama into ts-wrapper/LlamaRunner.ts

**Objective:** Replace the current WebAssembly backend in `ts-wrapper/src/LlamaRunner.ts` with the `wllama` library, using a local clone of `wllama` for imports. This aims to leverage `wllama`'s active maintenance, modern features, and potentially simplified WASM management.

**Assumptions:**
*   A local clone of the `wllama` repository is available and located in a known path relative to the `ts-wrapper` project (e.g., `../wllama` if `ts-wrapper` and `wllama` are sibling directories).
*   The local `wllama` clone can be built to produce its JavaScript modules (e.g., in an `esm/` or `dist/` subdirectory within `wllama`).
*   The necessary server headers (`Cross-Origin-Embedder-Policy: require-corp` and `Cross-Origin-Opener-Policy: same-origin`) can be configured for the development and deployment environment if multi-threading from `wllama` is to be used.

---

## Phase A: Setup and Preparation

1.  **Verify `wllama` Local Build:**
    *   Ensure the local `wllama` clone is up-to-date and can be successfully built (e.g., using `npm install && npm run build` within the `wllama` directory).
    *   Identify the path to the main `wllama` ES module entry point (e.g., `wllama/esm/index.js` or `wllama/dist/index.js`) and the paths to its WASM artifacts (`wllama.wasm`, `wllama-mt.wasm`, `wllama.worker.js`).

2.  **Establish Import Strategy for Local `wllama`:**
    *   Determine the correct relative import path for `wllama`'s main module from within `LlamaRunner.ts`.
        *   Example: `import { Wllama, /* other types */ } from '../../wllama/esm/index.js';`
    *   Confirm paths for `wllama`'s WASM artifacts (`.wasm` files, `worker.js`) that will be passed to the `Wllama` constructor.

3.  **Backup Existing `LlamaRunner.ts`:**
    *   Create a backup copy of `ts-wrapper/src/LlamaRunner.ts` before making significant modifications.

4.  **Review `ts-wrapper` Interfaces:**
    *   Re-familiarize with `ModelSpecification` (`ts-wrapper/src/model-spec.ts`), `GenerateTextParams`, `TokenCallback`, `CompletionCallback`, `ProgressCallback`, and `LoadingStage` (`ts-wrapper/src/llama-runner.ts`, `ts-wrapper/src/loading-progress.ts`).
    *   Re-familiarize with `LocalWebAIError` and its subtypes (`ts-wrapper/src/errors.ts`).

---

## Phase B: Core `wllama` Integration & Model Loading in `LlamaRunner.ts`

1.  **Modify `LlamaRunner` Constructor and Initialization:**
    *   Update the `LlamaRunner` constructor:
        *   Change parameters from `(workerPath: string, wasmModulePath: string, wasmPath: string)` to accept a configuration object suitable for `wllama`, e.g., `wllamaArtifactPaths: { singleThreadWasm: string; multiThreadWasm: string; workerJs?: string; }` or individual paths. This configuration will provide the paths to the local `wllama` `.wasm` and `worker.js` files.
    *   Remove the existing `this.worker: Worker | null` property related to `ts-wrapper/src/worker.ts`.
    *   Add a new property: `private wllamaInstance: Wllama | null = null;` (Import `Wllama` type from local `wllama`).
    *   Rename/refactor `initWorker()` to `initWllama()` (or similar):
        *   Inside `initWllama()`, instantiate `this.wllamaInstance = new Wllama(wllamaConfigPaths, wllamaOptions);`.
            *   `wllamaConfigPaths` will be constructed from the constructor arguments, mapping to `wllama`'s expected format (e.g., `{'single-thread/wllama.wasm': pathToSingleThreadWasm, ...}`).
            *   `wllamaOptions` can include `logger` (e.g., map `ts-wrapper`'s logging if any, or use `wllama`'s defaults/`LoggerWithoutDebug`), and potentially `n_threads` or rely on `wllama`'s auto-detection.
        *   Remove the `postMessage` call that was used to initialize the old worker.
        *   The `isInitialized` flag can be set after `Wllama` instantiation (or based on a successful first interaction if needed, though `Wllama` constructor is synchronous).

2.  **Refactor `loadModel` Method:**
    *   This method will now use `this.wllamaInstance.loadModel(...)`.
    *   **Input Handling:**
        *   If `source` is a `string` (URL), pass it directly.
        *   If `source` is a `File`, read its content as an `ArrayBuffer` and pass the buffer to `wllama.loadModel()`.
    *   **Progress Callback:**
        *   Adapt `ts-wrapper`'s `ProgressCallback` to `wllama`'s `progress_callback` format. `wllama` provides `{ loaded, total }`. Map this to `ProgressInfo` structure, including appropriate `LoadingStage` updates (e.g., `DOWNLOADING_MODEL_DATA`, `VERIFYING_MODEL_DATA`).
    *   **Abort Signal:** Pass the `AbortSignal` directly to `wllama.loadModel()`.
    *   **Model Metadata:**
        *   On successful model load, `wllama.loadModel()` resolves with an object containing `model_meta`.
        *   Implement a private helper method, e.g., `private mapWllamaMetaToModelSpec(wllamaMeta: any, sourceInfo: { url?: string, fileName?: string, fileSize?: number }): ModelSpecification`.
        *   Populate `this.currentModelMetadata` using this mapping function. Include fields like `architecture`, `quantization`, `layerCount`, `embeddingLength`, `contextLength`, `vocabSize`, `headCount`, `headCountKv`, `ropeFrequencyBase`, `ropeFrequencyScale`, `modelName`, `ggufVersion`.
        *   Populate provenance fields in `ModelSpecification` (`sourceURL`, `fileName`, `fileSize`) based on the input source and `wllama`'s progress info. `downloadDate` should be set by `LlamaRunner`.
    *   **GGUF Validation:** After obtaining `ggufVersion` from `wllama.model_meta.version`, perform the GGUF version compatibility check (using `MIN_SUPPORTED_GGUF_VERSION`, `MAX_SUPPORTED_GGUF_VERSION` from `LlamaRunner`). Throw `ModelCompatibilityError` if needed.
    *   **State Management:** Update `this.isLoadingModel`, `this.onModelLoadedCallback`, `this.onModelLoadErrorCallback` using `async/await` and Promise resolution/rejection from `wllama.loadModel()`.

3.  **Update `getModelMetadata` Method:**
    *   This method will simply return `this.currentModelMetadata`.

---

## Phase C: Text Generation and Streaming

1.  **Refactor `generateText` Method:**
    *   This method will primarily use `this.wllamaInstance.runCompletion(...)` (or an equivalent streaming method from `wllama`).
    *   **Parameter Mapping (`GenerateTextParams` to `wllama`'s `CompletionTaskConfig`):**
        *   `prompt`: Pass directly.
        *   `n_predict`: Map to `n_predict`.
        *   `ctx_size`: Map to `context_params.ctx_len` (or similar in `wllama`). Note: `wllama` might also take `context_size` during `loadModel`. Clarify precedence if both are set.
        *   `batch_size`: Map to `context_params.batch_size` (or similar).
        *   `temp`, `top_k`, `top_p`: Map to `sampling_params` in `wllama`.
        *   `n_gpu_layers`: This parameter is likely not supported by `wllama`'s browser WASM. Log a warning if set, or ignore.
        *   `no_display_prompt`: Investigate if `wllama` or `llama.cpp` via `wllama` has a direct setting. If not, `LlamaRunner` might have to filter the prompt from the initial tokens if this flag is true (less ideal).
        *   `chatml`: If this implies a specific prompt template, `LlamaRunner` should apply the template to the `prompt` string before sending it to `wllama`. If `llama.cpp` has a specific mode for ChatML that `wllama` exposes, use that.
    *   **Callbacks:**
        *   Map `tokenCallback: TokenCallback` to `wllama`'s `on_token` callback.
        *   Map `completionCallback: CompletionCallback` to `wllama`'s `on_complete` callback or the Promise resolution of `runCompletion`.
    *   **Abort Signal:** Ensure an `AbortController`'s signal can be passed to `wllama`'s generation method to allow cancellation.
    *   Ensure the method correctly handles the asynchronous nature and Promise returned by `wllama`'s generation function.

---

## Phase D: Ancillary Functionality and Cleanup

1.  **Implement `cancelLoading` Method:**
    *   This should call `.abort()` on the `AbortController` whose signal was passed to `this.wllamaInstance.loadModel()`.
    *   Ensure state (`isLoadingModel`, etc.) is reset correctly.

2.  **Implement `terminate` Method:**
    *   Call `this.wllamaInstance.exit()` if `wllama` provides such a method for explicit resource cleanup. (Verify `wllama`'s API for this).
    *   Set `this.wllamaInstance = null;`.
    *   Reset any other relevant state in `LlamaRunner` (e.g., `currentModelMetadata`, `isInitialized`).

3.  **Robust Error Handling:**
    *   Wrap all calls to `this.wllamaInstance` methods (`loadModel`, `runCompletion`, etc.) in `try...catch` blocks.
    *   In the `catch` block, map errors thrown by `wllama` (e.g., `WllamaError`, `WllamaLoadModelError`, or generic `Error`s) to the appropriate `LocalWebAIError` subtypes defined in `ts-wrapper/src/errors.ts`.
        *   For example, a model loading failure in `wllama` could be mapped to `ModelInitializationError` or a more specific error.
        *   Use the existing error reporting mechanisms in `LlamaRunner` (e.g., rejecting the main Promise, calling `currentProgressCallback` with an error stage).

4.  **Remove Redundant Code from `LlamaRunner.ts`:**
    *   Remove all code related to managing the old `ts-wrapper/src/worker.ts` (e.g., `this.worker.postMessage`, `this.worker.onmessage`, `this.worker.onerror` handlers for LLM operations).
    *   Remove any direct GGUF parsing logic if `wllama.model_meta` provides all necessary information. The `validateModelMetadata` method might still be useful for checking fields in the mapped `ModelSpecification`.
    *   The `ModelCache` logic should still be relevant for caching model files/ArrayBuffers *before* they are passed to `wllama.loadModel()`.

---

## Phase E: Testing and Refinement

1.  **Develop Testing Strategy:**
    *   **Unit Tests:** Test individual methods of `LlamaRunner` with a mocked `Wllama` instance to verify parameter mapping, callback handling, and state changes.
    *   **Integration Tests:**
        *   Test `LlamaRunner.loadModel` with a small, valid GGUF model file, using the actual local `wllama` build. Verify metadata extraction and progress reporting.
        *   Test `LlamaRunner.generateText` with simple prompts. Verify token streaming and completion.
        *   Test cancellation of model loading and text generation.
        *   Test error handling for various scenarios (invalid model, network error if loading from URL via `wllama`, etc.).
    *   Ensure tests cover both single-threaded and multi-threaded scenarios if `wllama` is configured to support both.

2.  **Iterative Refinement:**
    *   During implementation and testing, address any discrepancies found in API mapping or feature parity between `ts-wrapper`'s requirements and `wllama`'s capabilities.
    *   Refine error mapping and progress reporting to be as informative as possible.
    *   Update documentation within `LlamaRunner.ts` to reflect the new `wllama`-based implementation.
    *   Ensure the public API of `LlamaRunner` remains consistent to minimize impact on other parts of `ts-wrapper` unless a change is explicitly desired.

---

This plan provides a structured approach to the integration. Each phase and step should be carefully executed and tested.