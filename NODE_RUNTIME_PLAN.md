# Node.js Runtime Implementation Plan

This document outlines the plan to implement the Node.js runtime environment for `llama-cpp-wasm-ts-wrapper`, addressing the goal set in Phase 1 of the project roadmap:

> **Node.js Runtime**: Create a parallel runtime environment for Node.js using `worker_threads` and native Wasm bindings (if beneficial over Emscripten's Node output).

Based on initial research, we will proceed using Emscripten's Node.js output (`-sENVIRONMENT=node -pthread`) with `worker_threads`.

## Phased Implementation

The implementation will proceed in the following phases:

### Phase A: Build System Setup (`llama-cpp-wasm`)

**Goal:** Configure the `llama-cpp-wasm` build system to produce Wasm artifacts suitable for Node.js using `worker_threads`.

**Steps:**

1.  **CMake/Emscripten Configuration:**
    *   Verify or add CMake build options to target Node.js.
    *   Ensure the Emscripten flags `-sENVIRONMENT=node` and `-pthread` are used for the Node.js build configuration. This enables `worker_threads` support and `SharedArrayBuffer` usage.
    *   Consider separate build outputs for single-threaded (`st`) and multi-threaded (`mt`) Node.js versions if needed, mirroring the browser builds.
2.  **Build Script Updates:** Modify existing build scripts (e.g., in `package.json` or standalone scripts) to trigger the Node.js Wasm build.
3.  **Output Directory:** Define a clear output directory for the Node.js artifacts (e.g., `llama-cpp-wasm/dist/node/`).
4.  **Initial Build:** Perform an initial build to confirm the Node.js artifacts (`.wasm`, `.js` glue code) are generated without errors.

**Verification:** *After this phase, confirm that the necessary Wasm and JS files for the Node.js environment are successfully generated in the designated output directory.*

### Phase B: Core Node.js Worker Logic (`ts-wrapper`)

**Goal:** Create the basic structure of the Node.js worker script, adapting the existing browser worker logic for the Node.js environment and setting up communication.

**Steps:**

1.  **Create New Worker File:** Create `ts-wrapper/src/node-worker.ts`.
2.  **Adapt Core Structure:** Copy the basic structure, imports, and type definitions from `ts-wrapper/src/worker.ts`.
3.  **Import `worker_threads`:** Import necessary components (`parentPort`, `isMainThread`, `workerData`, `Worker`) from the `worker_threads` module.
4.  **Replace Communication API:**
    *   Replace `self.postMessage` with `parentPort.postMessage`.
    *   Replace `self.onmessage` handler with `parentPort.on('message', ...)`.
5.  **Environment Check:** Adapt any environment checks (like `isMainThread`) to use the `worker_threads` equivalents.
6.  **Stub Implementation:** Initially stub out the complex parts like Wasm module initialization, model loading (`fs` interaction), and `callMain` to focus on the worker structure and communication plumbing. Define the `workerActions` and basic message handling.

**Verification:** *After this phase, ensure the `node-worker.ts` file compiles and can be theoretically started (though it won't do much yet). Basic message passing structure should be in place.*

### Phase C: Node.js Model Loading (`ts-wrapper`)

**Goal:** Implement the model loading logic within the Node.js worker using the Node.js filesystem (`fs`) module.

**Steps:**

1.  **Implement `LOAD` Handler:**
    *   Modify the `LOAD` action handler in `node-worker.ts` to expect a local file path (`modelPath`) in the message data instead of `modelData` or `modelUrl`.
    *   Use the Node.js `fs` module (e.g., `fs.promises.readFile` for smaller headers/metadata, consider `fs.createReadStream` for writing large models to VFS if memory becomes an issue) to interact with the filesystem.
2.  **Header Validation:** Read an initial chunk of the model file from the provided path using `fs` to perform the `validateGGUFHeader` check *before* attempting to load the full model into VFS.
3.  **VFS Integration:**
    *   Use `wasmModuleInstance.FS_createDataFile` to write the model content (read from the filesystem) into the Emscripten VFS (e.g., at `/models/model.bin`).
    *   Handle potential errors during file reading (`fs`) and VFS writing.
4.  **Metadata Parsing:** After successfully writing to VFS, use `wasmModuleInstance.FS.readFile` or `wasmModuleInstance.FS.open/read/close` to read the header *from VFS* for metadata parsing (`parseGGUFHeader`).
5.  **Progress & Error Reporting:** Integrate progress reporting (`reportProgress`) for VFS writing stages and robust error handling (`reportError`) for `fs` and VFS errors.
6.  **Cancellation:** Ensure cancellation checks (`checkCancellation`) are integrated appropriately during file reading and VFS writing.

**Verification:** *After this phase, the Node.js worker should be able to receive a file path, validate the model header, write the model to the VFS, parse metadata from VFS, and report progress/errors related to these steps. Wasm module initialization might still be basic.*

### Phase D: Main Thread Integration (`ts-wrapper`)

**Goal:** Create the main thread Node.js wrapper class to manage the worker and expose a user-friendly API.

**Steps:**

1.  **Create Wrapper Class File:** Create `ts-wrapper/src/node-llama-runner.ts`.
2.  **Define Class Structure:** Define a `NodeLlamaRunner` class, mirroring the API design of the browser's `LlamaRunner` (e.g., `loadModel`, `runInference`, `cancel` methods, event handling).
3.  **Worker Management:**
    *   Use `new Worker('./path/to/node-worker.js', { workerData: ... })` from `worker_threads` to spawn the worker instance within the `loadModel` method (or similar).
    *   Pass necessary initialization data (like the path to the Node.js-specific Wasm JS file) via `workerData`.
4.  **Implement Communication:**
    *   Implement the main thread side of the message handling (`worker.on('message', ...)`). Process events like `INITIALIZED`, `PROGRESS_UPDATE`, `WRITE_RESULT`, `MODEL_METADATA`, `RUN_COMPLETED`, and `ERROR`.
    *   Implement methods to send commands to the worker (`worker.postMessage(...)`), such as `LOAD` (sending the *model file path*), `RUN_MAIN` (sending prompt and parameters), and `CANCEL_LOAD`.
5.  **API Mapping:** Map the public API methods (`loadModel`, `runInference`) to the appropriate worker communication sequences. Manage internal state (e.g., `isLoading`, `isReady`).

**Verification:** *After this phase, you should be able to instantiate `NodeLlamaRunner`, call `loadModel` with a local file path, and potentially initiate `runInference`. The full Wasm execution might not be complete yet, but the main thread <-> worker communication loop should be functional.*

### Phase E: Wasm Integration & Build (`ts-wrapper` / `llama-cpp-wasm`)

**Goal:** Fully integrate the Node.js Wasm module loading and execution within the worker, and update the `ts-wrapper` build process.

**Steps:**

1.  **Wasm Module Initialization (`node-worker.ts`):**
    *   Finalize the `initWasmModule` function in `node-worker.ts`.
    *   Ensure the dynamic `import()` correctly loads the Node.js-specific Emscripten JS glue code.
    *   Verify the Emscripten `Module` factory instantiation works correctly in the Node.js environment.
    *   Ensure `locateFile` correctly points to the `.wasm` file path relative to the worker script.
2.  **`callMain` Execution (`node-worker.ts`):**
    *   Ensure the `runMain` function correctly constructs arguments for `wasmModuleInstance.callMain` based on the parameters received from the main thread.
    *   Adapt environment-specific details (e.g., use `os.cpus().length` instead of `navigator.hardwareConcurrency`).
    *   Verify TTY output capture and streaming (`stdout`/`stderr` handlers) work as expected.
3.  **TypeScript Configuration (`ts-wrapper`):** Update `tsconfig.json` to include necessary Node.js types (`@types/node`) if not already present. Ensure build targets are compatible.
4.  **Build Script Updates (`ts-wrapper`):** Modify `package.json` scripts or bundler configurations (if used) to:
    *   Compile the new Node.js files (`node-worker.ts`, `node-llama-runner.ts`).
    *   Produce distinct outputs for the browser and Node.js runtimes.
5.  **Package Exports (`ts-wrapper`):** Update the `package.json` `exports` field to provide clear entry points for consumers wanting to use either the browser or the Node.js runtime.

**Verification:** *After this phase, the end-to-end flow should work. You should be able to use the `ts-wrapper` package in a Node.js project, load a model from a file, and run inference. Basic functionality should be demonstrable.*

### Phase F: Comprehensive Testing

**Goal:** Ensure the Node.js runtime is robust, correct, and performs as expected through dedicated testing.

**Steps:**

1.  **Develop Test Suite:** Create a new test suite specifically for the Node.js runtime (e.g., using Jest, Mocha, or Node's built-in test runner).
2.  **Test Cases:** Implement test cases covering:
    *   Successful model loading from various valid file paths.
    *   Failure modes for model loading (invalid paths, non-existent files, permission errors, corrupted models, incompatible GGUF versions).
    *   Successful inference runs with different prompts and parameters.
    *   Correct streaming output during inference.
    *   Accurate progress reporting during model loading.
    *   Effective cancellation of model loading.
    *   Handling of runtime errors during inference.
    *   Memory usage and potential leaks (more advanced).
    *   Concurrency testing (running multiple inferences).
3.  **Run Tests:** Execute the test suite and debug any failures.
4.  **Refinement:** Refine the implementation based on test results.

**Verification:** *After this phase, a comprehensive suite of automated tests should pass, demonstrating the correctness and robustness of the Node.js runtime implementation.* 