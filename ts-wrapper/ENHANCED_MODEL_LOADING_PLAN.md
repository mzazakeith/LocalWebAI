# Plan: Enhanced Model Loading & Metadata Handling

## 1. Introduction & Goal

This document outlines the plan to significantly enhance the model loading process for the Local-First AI Model Runner. The primary goals are:

*   **Granular Progress:** Provide detailed progress tracking beyond simple download/file read, including VFS writing, metadata parsing, and model initialization steps.
*   **Rich Metadata Handling:** Implement robust parsing, validation, storage, and utilization of model-specific metadata (e.g., from GGUF headers), including provenance.
*   **Robustness & Reliability:** Introduce comprehensive error handling, cancellation support for model loading, schema versioning for model formats, and integrity checks.
*   **Clean Architecture & Maintainability:** Apply SOLID principles, ensure clear separation of concerns, and provide thorough documentation for all new components and changes.
*   **Performance:** Ensure that new features introduce minimal overhead during the loading process.

The project will follow an iterative, phased approach. **Crucially, after the completion of each phase, the system will be tested to ensure it remains fully functional and that the changes integrate correctly.**

## 2. Phased Implementation

### Phase A: Project Structure Refactor & Core Metadata Definitions

*   **Goal:** Establish a cleaner project structure more conducive to new modules and define the core data structures for enhanced metadata.
*   **Tasks:**
    1.  **Create `src` directory:** Move `llama-runner.ts`, `model-cache.ts`, and `worker.ts` into a new `ts-wrapper/src/` directory.
    2.  **Update `tsconfig.json`:**
        *   Modify `rootDir` to `"./src"`.
        *   Update `include` to `["./src/**/*.ts"]`.
        *   Ensure `outDir` remains `"./dist"`.
    3.  **Update `package.json`:**
        *   Adjust `main`, `types`, and `files` paths to reflect the new `src` subdirectory structure (e.g., `dist/src/llama-runner.js`).
    4.  **Define `ModelSpecification.ts`:** Create `ts-wrapper/src/model-spec.ts`.
        *   Define an interface `ModelSpecification` to hold detailed, parsed metadata from the model file (e.g., architecture, layer count, context size, quantization, creator, license, GGUF version) and provenance data (original source URL, download date).
    5.  **Refine `ModelCache` Metadata:**
        *   In `model-cache.ts`, rename the existing `ModelMetadata` (used for cache entry management) to `CacheEntryMetadata`.
        *   Modify `ModelCache` to store and retrieve the new `ModelSpecification` object associated with a `modelId`, alongside `CacheEntryMetadata` and model chunks. The IndexedDB metadata store will likely need to accommodate this combined structure.
*   **Post-Phase Validation:** Verify project builds correctly, existing demo functionality in `index.html` (model loading from URL/file, text generation) works as before with the new structure.

### Phase B: Metadata Parsing in Worker & Initial Validation

*   **Goal:** Implement the logic to parse metadata from the model file (GGUF header) within the Web Worker and communicate it back to the main thread for initial validation and storage.
*   **Tasks:**
    1.  **Create `gguf-parser.ts`:** Create `ts-wrapper/src/gguf-parser.ts`.
        *   Research GGUF file format structure (header details).
        *   Implement functions to parse GGUF headers from an `ArrayBuffer` (or `DataView`) and populate a `ModelSpecification` object.
        *   Initially, focus on extracting key fields and the GGUF version.
    2.  **Enhance `worker.ts` (`loadModelData` or new function):**
        *   After `FS_createDataFile`, the worker will read the initial part (header) of the model file from the Emscripten VFS.
        *   Use `GGUFParser` to parse this header data.
        *   `postMessage` the populated `ModelSpecification` object (or an error object if parsing fails) back to `LlamaRunner`.
    3.  **Enhance `LlamaRunner.ts`:**
        *   Handle the new message from the worker containing `ModelSpecification`.
        *   Perform basic validation (e.g., critical fields present, GGUF version within a simple supported range).
        *   If valid, store the `ModelSpecification` using `ModelCache`.
        *   Propagate errors appropriately if parsing/validation fails.
*   **Post-Phase Validation:** Test loading a GGUF model. Verify that metadata is parsed, sent to `LlamaRunner`, and stored in `ModelCache` (can be checked via IndexedDB inspection). Ensure model still loads and generates text.

### Phase C: Granular Progress Reporting

*   **Goal:** Provide more detailed feedback to the user about the various stages of model loading, including metadata parsing.
*   **Tasks:**
    1.  **Define Progress Stages:** Create an enum or constant object in a shared location (e.g., `ts-wrapper/src/types.ts` or within `llama-runner.ts`) for distinct loading stages:
        *   `DOWNLOADING_FROM_SOURCE` / `READING_FROM_FILE`
        *   `VFS_WRITE_START` / `VFS_WRITE_PROGRESS` / `VFS_WRITE_COMPLETE`
        *   `METADATA_PARSE_START` / `METADATA_PARSE_COMPLETE`
        *   `MODEL_INITIALIZATION_START` (Worker preparing Wasm module for the specific model)
        *   `MODEL_READY`
    2.  **Update `ProgressCallback` Type:** Modify `ProgressCallback` in `llama-runner.ts` to accept an object like: `{ stage: string; loaded?: number; total?: number; message?: string; metadata?: ModelSpecification }`.
    3.  **Implement Stage Reporting:**
        *   `LlamaRunner`: Emit progress for download/file reading.
        *   `worker.ts`: `postMessage` updates for VFS writing (potentially with loaded/total if feasible for large VFS writes, otherwise start/complete), metadata parsing, and model initialization stages.
        *   `LlamaRunner`: Forward these structured progress updates to the user's `ProgressCallback`.
    4.  **Update `index.html`:**
        *   Modify the UI to clearly display the current loading stage.
        *   Once `METADATA_PARSE_COMPLETE` is received and `ModelSpecification` is available in the progress object, display some key metadata fields (e.g., model architecture, GGUF version).
*   **Post-Phase Validation:** Test model loading. Verify that the UI in `index.html` shows the new granular progress stages and displays basic metadata after it's parsed. Model should still load and generate text.

### Phase D: Cancellation Support for Model Loading

*   **Goal:** Allow the user to cancel an in-progress model load operation cleanly.
*   **Tasks:**
    1.  **Introduce `AbortSignal`:** Modify `LlamaRunner.loadModel` to accept an optional `AbortSignal` as a parameter.
    2.  **Propagate Signal:**
        *   Pass the `AbortSignal` to `fetch` calls in `LlamaRunner`.
        *   When posting messages to `worker.ts` that initiate long-running operations (like VFS writing or model processing), include information if cancellation is requested. This might involve sending a specific `CANCEL_LOAD` message to the worker if the main thread's `AbortSignal` is aborted.
    3.  **Worker Cancellation Logic:**
        *   `worker.ts` should check for a cancellation request/flag:
            *   Before major synchronous operations.
            *   Periodically during very long operations if they can be chunked internally (less likely for current VFS ops).
        *   If cancelled, the worker should clean up (e.g., remove any partial VFS files for the current model) and `postMessage` a cancellation confirmation/error.
    4.  **`LlamaRunner` Cleanup:**
        *   If loading is aborted (either via its own `AbortSignal` or a message from the worker), reject the `loadModel` promise with an `AbortError`.
        *   Ensure any partially downloaded data is discarded.
        *   Instruct `ModelCache` to remove any incomplete cache entries for the aborted model load.
    5.  **Update `index.html`:** Add a "Cancel Load" button that becomes active during loading and uses `AbortController` to signal cancellation to `LlamaRunner`.
*   **Post-Phase Validation:** Test model loading. Verify that clicking the "Cancel Load" button stops the loading process at various stages (during download, during VFS write in worker if possible to test granularity). Ensure resources are cleaned up.

### Phase E: Comprehensive Error Handling & Final Validation

*   **Goal:** Enhance error reporting with specific, actionable messages and ensure model integrity/compatibility before use.
*   **Tasks:**
    1.  **Define Error Types/Codes:** Create specific error classes or an enum of error codes for failures at different stages (download, VFS write, metadata parsing, metadata validation, model compatibility, Wasm initialization).
    2.  **Propagate Specific Errors:**
        *   Ensure errors from `GGUFParser` (e.g., "Invalid GGUF Magic Number", "Unsupported GGUF Version"), VFS operations (e.g., "VFS Write Failed"), and Wasm module initialization are caught in `worker.ts` and `postMessage`'d back with specific details/codes.
        *   `LlamaRunner` should convert these into user-friendly, actionable errors.
    3.  **Metadata Integrity & Compatibility (in `LlamaRunner`):**
        *   After receiving `ModelSpecification`, perform thorough validation.
        *   Check GGUF version against a supported range (e.g., `if (metadata.ggufVersion < MIN_SUPPORTED_GGUF_VERSION || metadata.ggufVersion > MAX_SUPPORTED_GGUF_VERSION) throw new ModelCompatibilityError(...)`).
        *   Consider adding checks for essential metadata fields required by `llama.cpp`.
    4.  **Update `index.html`:** Display detailed error messages clearly to the user.
*   **Post-Phase Validation:** Test various error scenarios: invalid model URL, corrupted GGUF file (if possible to simulate for header parsing), model with unsupported GGUF version. Verify that specific and helpful error messages are shown in `index.html`.

### Phase F: Documentation, SOLID Review, and Cleanup

*   **Goal:** Ensure the new code is well-documented, adheres to good design principles, manages memory effectively, and integrates provenance tracking.
*   **Tasks:**
    1.  **TSDoc Comments:** Add/update comprehensive TSDoc comments for all new and modified public APIs, interfaces, and complex internal logic in all affected files (`llama-runner.ts`, `worker.ts`, `model-cache.ts`, `gguf-parser.ts`, `model-spec.ts`).
    2.  **SOLID/Clean Architecture Review:**
        *   Review the changes against SOLID principles.
        *   Ensure clear separation of concerns (e.g., `GGUFParser` for parsing, `ModelCache` for storage, `LlamaRunner` for orchestration).
    3.  **Memory Management Review:**
        *   Double-check that large `ArrayBuffer`s are not held in memory longer than necessary, particularly on the main thread (`LlamaRunner`) once data is passed to the worker or cache.
    4.  **Provenance Tracking:**
        *   Confirm that `ModelSpecification` includes fields for original source URL and download timestamp.
        *   Ensure `ModelCache` correctly stores and can retrieve this provenance information.
        *   Consider displaying this provenance information in `index.html` when a model is loaded.
*   **Post-Phase Validation:** Review code for documentation and architectural consistency. Verify provenance data is stored and can be displayed. Perform a final round of testing on all model loading features.

## 3. Clarifications & Scope Notes

*   **GGUF Schema Versioning:** For the initial implementation, parsing the GGUF version from the header and checking against a known "supported GGUF version range" is sufficient.
*   **Model Provenance & Configuration Snapshots:** Storing the source URL, download date, and the parsed GGUF metadata (which acts as a configuration snapshot) is adequate for this iteration.
*   **GPU Context Management:** Not in scope for these specific enhancements.
*   **Advanced Thread-Safe Resource Management:** Focus will be on robust cancellation and clean data handling between the main thread and the worker.

This plan provides a structured approach to delivering these enhancements. Each phase builds upon the last, with validation steps to ensure continuous stability. 