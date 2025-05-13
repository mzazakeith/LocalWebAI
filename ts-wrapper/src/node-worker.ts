// ts-wrapper/node-worker.ts

// Node.js worker threads imports
import { parentPort, isMainThread, workerData, TransferListItem } from 'worker_threads';
import os from 'os'; // Needed for potential hardware concurrency later
import fs from 'fs/promises'; // <-- Import fs.promises
import { constants as fsConstants } from 'fs'; // <-- Import fs constants

// Type definition for Emscripten module (adjust as needed)
interface EmscriptenModule {
  [key: string]: any; // Allow any properties
  noInitialRun: boolean;
  preInit: any[];
  TTY: {
    register: (dev: any, ops: any) => void;
  };
  FS_createPath: (path: string, name: string, canRead?: boolean, canWrite?: boolean) => void;
  FS_createDataFile: (parent: string, name: string, data: Uint8Array, canRead?: boolean, canWrite?: boolean, canOwn?: boolean) => any;
  callMain: (args: string[]) => any;
  FS: any;
}

// Import shared types and utilities
import { parseGGUFHeader } from './gguf-parser.js'; // Keep for stubbed validation
import { ModelSpecification } from './model-spec.js';
import { LoadingStage } from './loading-progress.js';
import { 
  GGUFParsingError, 
  ModelCompatibilityError, 
  VFSError, 
  WasmError,
  OperationCancelledError,
  ModelInitializationError,
  FileError,
  LocalWebAIError // Ensure LocalWebAIError is imported if used in stubs
} from './errors.js';

// Define the expected structure of Module factory (keep for stub)
declare function Module(settings?: Partial<EmscriptenModule>): Promise<EmscriptenModule>;

// Worker actions (shared between main thread and worker)
const workerActions = {
  LOAD: 'LOAD',
  INITIALIZED: 'INITIALIZED',
  RUN_MAIN: 'RUN_MAIN',
  WRITE_RESULT: 'WRITE_RESULT',
  RUN_COMPLETED: 'RUN_COMPLETED',
  LOAD_MODEL_DATA: 'LOAD_MODEL_DATA', // Changed for Node.js: Expect file path
  MODEL_METADATA: 'MODEL_METADATA',
  PROGRESS_UPDATE: 'PROGRESS_UPDATE',
  CANCEL_LOAD: 'CANCEL_LOAD',
  LOAD_NODE: 'LOAD_NODE' // New action specific to Node.js init
};

let wasmModuleInstance: EmscriptenModule | null = null; // Initialize as null
const modelPath = "/models/model.bin"; // VFS path remains the same for now
const headerReadSize = 1024 * 1024; // 1MB header read size

const decoder = new TextDecoder('utf-8');
const stdoutBuffer: number[] = [];
const stderrBuffer: number[] = []; // Buffer for stderr

// Helper function to safely post messages to the parent port
function postMessageToParent(message: any, transferList?: ReadonlyArray<TransferListItem>) {
  if (!parentPort) {
    console.error('[NodeWorker] Error: parentPort is not available.');
    return;
  }
  try {
    parentPort.postMessage(message, transferList);
  } catch (error) {
    console.error('[NodeWorker] Error posting message:', error, 'Message:', message);
    // Optionally, post a simplified error back if the original message failed
    if (message.event !== 'ERROR') { // Avoid infinite error loops
        try {
            parentPort.postMessage({ event: 'ERROR', error: 'Failed to serialize message', errorDetails: { name: 'SerializationError' } });
        } catch (nestedError) {
            console.error('[NodeWorker] Error posting serialization error message:', nestedError);
        }
    }
  }
}

// Stdout/Stderr handling adapted for Node worker communication
const stdin = () => { /* no-op */ };

const stdout = (c: number) => {
  stdoutBuffer.push(c);
  // Logic remains similar, but use postMessageToParent
  const punctuationBytes = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 58, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126];
  const whitespaceBytes = [32, 9, 10, 13, 11, 12];
  const splitBytes = [...punctuationBytes, ...whitespaceBytes];
  if (splitBytes.includes(c) || stdoutBuffer.length > 20) {
    const text = decoder.decode(new Uint8Array(stdoutBuffer));
    stdoutBuffer.length = 0; // Clear buffer
    postMessageToParent({
      event: workerActions.WRITE_RESULT,
      text: text,
    });
  }
};

const stderr = (c: number) => {
  stderrBuffer.push(c);
  // Optionally, still print to worker's stderr for live debugging if desired
  // process.stderr.write(String.fromCharCode(c));
};

let isCancellationRequested = false;

function checkCancellation(): boolean {
  return isCancellationRequested;
}

function resetCancellationState() {
  isCancellationRequested = false;
}

function cleanupAfterCancellation() {
  // Basic VFS cleanup (if needed and instance exists)
  try {
    if (wasmModuleInstance && wasmModuleInstance.FS) {
      try {
        wasmModuleInstance.FS.unlink(modelPath); 
      } catch (e) { /* Ignore if file doesn't exist */ }
    }
  } catch (e) {
    console.error('[NodeWorker] Error during VFS cleanup after cancellation:', e);
  }

  reportProgress(LoadingStage.CANCELLED, { message: 'Model loading cancelled' });
  reportError(new OperationCancelledError('Model loading cancelled by user'));
  resetCancellationState();
}

function reportError(error: Error | string, stage: LoadingStage = LoadingStage.ERROR) {
  let errorMsg: string;
  let errorDetails: any = {};
  
  if (error instanceof Error) {
    errorMsg = error.message;
    errorDetails.name = error.constructor.name;
    errorDetails.message = error.message;
    // Copy relevant properties for specific error types
    if (error instanceof ModelCompatibilityError) {
      errorDetails.actualVersion = error.actualVersion;
      errorDetails.minSupported = error.minSupported;
      errorDetails.maxSupported = error.maxSupported;
    } else if (error instanceof GGUFParsingError) {
      errorDetails.details = error.details;
    } else if (error instanceof VFSError) {
      errorDetails.path = error.path;
    }
    // Include stack for debugging if available
    if (error.stack) {
        errorDetails.stack = error.stack;
    }
  } else {
    errorMsg = String(error);
    errorDetails.message = errorMsg;
    errorDetails.name = 'GenericError';
  }

  // Report progress first
  reportProgress(stage, { message: errorMsg, error: errorMsg });

  // Send structured error event
  postMessageToParent({
    event: 'ERROR',
    error: errorMsg,
    errorDetails: errorDetails
  });
}

function reportProgress(stage: LoadingStage, details: any = {}) {
  postMessageToParent({
    event: workerActions.PROGRESS_UPDATE,
    stage,
    ...details
  });
}

// --- Actual Implementations for Phase E ---

async function initWasmModule(wasmNodeModulePath: string, wasmNodePath: string) {
  console.log(`[NodeWorker] Initializing Wasm module from: ${wasmNodeModulePath}, Wasm file: ${wasmNodePath}`);
  reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
    message: 'Initializing WebAssembly module for Node.js'
  });

  // Polyfill for Emscripten code that might expect web worker globals
  if (typeof globalThis.onmessage === 'undefined') {
    (globalThis as any).onmessage = null; // Or a no-op function: () => {};
    console.log('[NodeWorker] Polyfilled globalThis.onmessage');
  }
  if (typeof globalThis.postMessage === 'undefined') {
    // Emscripten might also try to use self.postMessage for its own workers if pthreads are involved.
    // We don't want it to conflict with parentPort, so a simple log or no-op might be best.
    (globalThis as any).postMessage = (message: any) => {
      console.warn('[NodeWorker] globalThis.postMessage polyfill called. This might indicate an unexpected Wasm behavior. Message:', message);
    };
    console.log('[NodeWorker] Polyfilled globalThis.postMessage');
  }

  const emscriptenModuleConfig: Partial<EmscriptenModule> = {
    noInitialRun: true,
    preInit: [() => {
      if (!emscriptenModuleConfig.FS || !emscriptenModuleConfig.TTY) {
        console.error('[NodeWorker] Emscripten FS or TTY not available on config during preInit.');
        return;
      }
      emscriptenModuleConfig.TTY.register(emscriptenModuleConfig.FS.makedev(5, 0), {
        get_char: (tty: any) => stdin(),
        put_char: (tty: any, val: number) => { tty.output.push(val); stdout(val); },
        flush: (tty: any) => tty.output = [],
      });
      emscriptenModuleConfig.TTY.register(emscriptenModuleConfig.FS.makedev(6, 0), {
        get_char: (tty: any) => stdin(),
        put_char: (tty: any, val: number) => { tty.output.push(val); stderr(val); },
        flush: (tty: any) => tty.output = [],
      });
    }],
    locateFile: (path: string) => {
      if (path.endsWith('.wasm')) {
        // wasmNodePath should be an absolute path or resolvable relative to the JS glue file.
        // For Node.js, directly using the provided wasmNodePath is usually correct if it's absolute
        // or relative to where the worker process is started from, assuming the glue file can find it.
        // Emscripten's default locateFile often expects the .wasm file to be sibling to the main .js glue file.
        // If wasmNodePath is absolute, it should just work.
        // If relative, it needs to be relative to the wasmNodeModulePath's directory.
        // For now, we assume wasmNodePath is correctly provided to be resolvable.
        return wasmNodePath;
      }
      return path;
    },
    // Other configurations like print, printErr could be routed here if needed
    // print: console.log,
    // printErr: console.error,
  };

  try {
    // Dynamically import the Emscripten-generated JS glue file
    // The import path should be resolvable by Node.js (e.g., absolute path or correct relative path)
    const ModuleFactory = (await import(wasmNodeModulePath)).default;
    if (typeof ModuleFactory !== 'function') {
        throw new WasmError('Wasm module factory not found or not a function. Check Emscripten build (MODULARIZE, EXPORT_ES6).');
    }

    wasmModuleInstance = await ModuleFactory(emscriptenModuleConfig);
    
    if (!wasmModuleInstance || typeof wasmModuleInstance.callMain !== 'function') {
        throw new WasmError('Failed to instantiate Wasm module or callMain is not available.');
    }

    console.log('[NodeWorker] Wasm module instance created successfully.');
    reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
      message: 'WebAssembly module initialized (No model loaded yet)'
    });
    postMessageToParent({ event: workerActions.INITIALIZED }); // Signal Wasm module is ready

  } catch (error) {
    console.error('[NodeWorker] Error during Wasm module initialization:', error);
    const errInstance = error instanceof Error ? error : new WasmError(String(error));
    reportError(errInstance);
    // Do not proceed if Wasm initialization fails critically
  }
}

async function loadModelData(modelPathFromArgs: string) {
  console.log(`[NodeWorker] Loading model from path: ${modelPathFromArgs}`);
  if (!wasmModuleInstance) {
    reportError(new WasmError('Wasm module not initialized before loading model data.'));
    return;
  }
  
  let fileHandle: fs.FileHandle | null = null;
  let modelDataBuffer: Buffer | null = null; // Use Node.js Buffer initially
  
  try {
    // Check for cancellation before opening file
    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    reportProgress(LoadingStage.PREPARING_MODEL_DATA, { message: 'Opening model file' });
    
    // 1. Open the file
    try {
      fileHandle = await fs.open(modelPathFromArgs, fsConstants.O_RDONLY);
    } catch (err: any) {
      throw new FileError(
        `Failed to open model file: ${err.message || 'Unknown error'}`,
        modelPathFromArgs
      );
    }

    // Check for cancellation after opening file
    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // 2. Get file stats for size
    const stats = await fileHandle.stat();
    const totalSize = stats.size;
    reportProgress(LoadingStage.PREPARING_MODEL_DATA, { 
      message: 'Reading model file header',
      total: totalSize
    });

    // 3. Read header for validation
    const headerBuffer = Buffer.alloc(headerReadSize);
    const { bytesRead } = await fileHandle.read(headerBuffer, 0, headerReadSize, 0);

    if (bytesRead < 8) { // Basic check for enough data
        throw new GGUFParsingError(
          `Failed to read sufficient header data: only read ${bytesRead} bytes`,
          { bytesRead, minimumRequired: 8 }
        );
    }
    
    try {
        const headerArrayBuffer = headerBuffer.buffer.slice(headerBuffer.byteOffset, headerBuffer.byteOffset + bytesRead);
        validateGGUFHeader(headerArrayBuffer);
    } catch (error) {
      reportError(error instanceof Error ? error : new GGUFParsingError(String(error)));
      return; 
    }
    
    console.log('[NodeWorker] GGUF header validated successfully.');
    reportProgress(LoadingStage.PREPARING_MODEL_DATA, { 
      message: 'Validated model header. Reading full model file...',
      loaded: bytesRead, 
      total: totalSize
    });

    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    console.log(`[NodeWorker] Reading full file (${(totalSize / (1024*1024)).toFixed(2)} MB) into buffer...`);
    modelDataBuffer = await fileHandle.readFile();
    console.log(`[NodeWorker] File read complete. Buffer size: ${modelDataBuffer.byteLength}`);

    if (modelDataBuffer.byteLength !== totalSize) {
        console.warn(`[NodeWorker] Warning: Read buffer size (${modelDataBuffer.byteLength}) does not match file stat size (${totalSize}). Proceeding anyway.`);
    }

    reportProgress(LoadingStage.PREPARING_MODEL_DATA, { 
      message: 'Model file read into memory',
      loaded: totalSize, 
      total: totalSize
    });

    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    reportProgress(LoadingStage.VFS_WRITE_START, { 
        message: 'Writing model to virtual filesystem',
        total: totalSize
    });
    
    try {
      wasmModuleInstance.FS_createPath("/", "models", true, true);
      const modelDataView = new Uint8Array(modelDataBuffer.buffer, modelDataBuffer.byteOffset, modelDataBuffer.byteLength);
      wasmModuleInstance.FS_createDataFile('/models', 'model.bin', modelDataView, true, true, true);
      console.log('[NodeWorker] Successfully wrote model to VFS path:', modelPath);
    } catch (error) {
      throw new VFSError(
        `Failed to write model data to VFS: ${error instanceof Error ? error.message : String(error)}`,
        modelPath
      );
    }
    
    modelDataBuffer = null; 

    reportProgress(LoadingStage.VFS_WRITE_COMPLETE, { 
        message: 'Model written to virtual filesystem',
        loaded: totalSize,
        total: totalSize
    });

    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    let metadata: ModelSpecification | null = null;
    try {
      metadata = await parseModelMetadata(); 
    } catch (error) {
      console.warn("[NodeWorker] Metadata parsing failed, continuing without full metadata:", error);
      reportError(error instanceof Error ? error : new GGUFParsingError(String(error)), LoadingStage.METADATA_PARSE_COMPLETE); 
    }
    
    if (checkCancellation()) {
        cleanupAfterCancellation();
        return;
    }

    reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
      message: 'Model ready for inference',
      metadata: metadata || undefined
    });

    if (metadata) {
      postMessageToParent({ event: workerActions.MODEL_METADATA, metadata });
    }

    postMessageToParent({ event: workerActions.INITIALIZED }); // Signal model is loaded and VFS ready

    reportProgress(LoadingStage.MODEL_READY, {
      message: 'Model loaded and ready',
      loaded: totalSize,
      total: totalSize,
      metadata: metadata || undefined
    });

  } catch (error) {
    console.error('[NodeWorker] Error during model loading:', error);
    reportError(error instanceof Error ? error : new Error(String(error)));
  } finally {
    if (fileHandle) {
      try {
        await fileHandle.close();
        console.log(`[NodeWorker] Closed file handle for: ${modelPathFromArgs}`);
      } catch (closeError) {
        console.error(`[NodeWorker] Error closing file handle: ${closeError}`);
      }
    }
    modelDataBuffer = null;
    resetCancellationState(); 
  }
}

async function parseModelMetadata(): Promise<ModelSpecification | null> {
  console.log("[NodeWorker] Parsing model metadata from VFS...");
  if (!wasmModuleInstance) {
    const error = new WasmError('Wasm module not initialized when trying to parse model metadata');
    reportError(error);
    return null;
  }

  try {
    reportProgress(LoadingStage.METADATA_PARSE_START, {
      message: 'Reading model header from VFS for metadata extraction'
    });

    let stream;
    try {
      stream = wasmModuleInstance.FS.open(modelPath, 'r');
    } catch (err) {
      throw new FileError(`Failed to open model file in VFS at ${modelPath}: ${err instanceof Error ? err.message : String(err)}`, modelPath);
    }

    const headerBuffer = new Uint8Array(headerReadSize);
    let bytesRead: number;
    try {
      bytesRead = wasmModuleInstance.FS.read(stream, headerBuffer, 0, headerReadSize, 0);
    } catch (err) {
      throw new FileError(`Failed to read model file header from VFS: ${err instanceof Error ? err.message : String(err)}`, modelPath);
    } finally {
      try {
        wasmModuleInstance.FS.close(stream);
      } catch (closeErr) {
        console.error('[NodeWorker] Error closing VFS file stream after read attempt:', closeErr);
      }
    }

    if (bytesRead < 8) {
      throw new GGUFParsingError(
        `Failed to read sufficient header data from VFS: only read ${bytesRead} bytes`,
        { bytesRead, minimumRequired: 8 }
      );
    }

    const headerData = headerBuffer.slice(0, bytesRead).buffer; // Get ArrayBuffer view

    validateGGUFHeader(headerData);
    const metadata = parseGGUFHeader(headerData);

    reportProgress(LoadingStage.METADATA_PARSE_COMPLETE, {
      message: 'Model metadata extracted successfully from VFS',
      metadata: metadata
    });

    return metadata;
  } catch (error) {
    console.error("[NodeWorker] Error parsing metadata from VFS:", error);
    if (error instanceof Error) {
        throw error; 
    } else {
        throw new GGUFParsingError(String(error)); 
    }
  }
}

function runMain(prompt: string, params: Record<string, any>) {
  console.log(`[NodeWorker] Running main with prompt: "${prompt}", params:`, params);
  if (!wasmModuleInstance || typeof wasmModuleInstance.callMain !== 'function') {
    reportError(new WasmError('Wasm module not ready or callMain not available.'));
    return;
  }

  // Clear previous stderr buffer
  stderrBuffer.length = 0;

  try {
    const args: string[] = [
      // Default llama.cpp args for inference
      "--model", modelPath, // Model path in VFS
      "--prompt", prompt,
      // Basic sensible defaults, can be overridden by params
      "--n-predict", (params.n_predict || 256).toString(), // Max tokens to predict
      "--ctx-size", (params.ctx_size || 512).toString(),    // Context size
      "--batch-size", (params.batch_size || 512).toString(), // Batch size for prompt processing
      "--temp", (params.temp || 0.8).toString(),             // Temperature
      "--top-k", (params.top_k || 40).toString(),
      "--top-p", (params.top_p || 0.9).toString(),
      "--simple-io", // Ensures output is plain text tokens
    ];

    if (params.chatml) {
      args.push("--chatml");
    }

    // By default, llama.cpp main example displays the prompt. 
    // The --no-display-prompt flag in the original worker seems to be a custom patch or older version behavior.
    // Standard llama.cpp main usually includes prompt unless specific flags are used to suppress parts of output.
    // If `no_display_prompt` is true, we aim to suppress it. Llama.cpp doesn't have a direct flag for this for simple-io.
    // This might need more complex output filtering if the Wasm doesn't support suppressing it directly.
    // For now, we assume the main Wasm build behaves as expected or we filter on JS side.
    // The browser worker used --no-display-prompt. If this is from a patch, the Node build needs it too.
    // Assuming standard llama.cpp args for now.
    if (params.no_display_prompt === true) {
        // This flag might not exist in standard llama.cpp `main` example.
        // If it's custom, it needs to be supported by the Wasm build.
        // args.push("--no-display-prompt"); 
        console.warn('[NodeWorker] no_display_prompt: true is set, but standard llama.cpp main might still display prompt with simple-io.');
    }

    // The Emscripten flags -s USE_PTHREADS=1 and -s PTHREAD_POOL_SIZE must be active in the build.
    // The llama.cpp CMakeLists.txt usually handles -pthread for native builds.
    // For Emscripten, this enables pthreads. Llama.cpp itself will use these if compiled with thread support.
    // const threadCount = (os.cpus() || []).length > 1 ? (os.cpus() || []).length : 2; // Default to 2 if os.cpus() is weird
    // args.push("--threads", threadCount.toString());
    // console.log(`[NodeWorker] Using --threads ${threadCount}`);
    console.log('[NodeWorker] Explicitly NOT passing --threads argument for single-threaded Node.js build.');

    // Add other parameters as needed from GenerateTextParams
    if (params.n_gpu_layers !== undefined && params.n_gpu_layers > 0) {
        // GPU layers are typically not supported in browser/Node Wasm builds of llama.cpp
        // but we include the arg if passed, in case a special build supports it.
        // args.push("-ngl", params.n_gpu_layers.toString());
        console.warn('[NodeWorker] n_gpu_layers specified, but typically not effective in Node.js Wasm builds.');
    }

    console.log('[NodeWorker] Calling wasmModuleInstance.callMain with args:', args);
    wasmModuleInstance.callMain(args);

    // Ensure any remaining buffered output is sent after callMain completes or errors
    if (stdoutBuffer.length > 0) {
      const text = decoder.decode(new Uint8Array(stdoutBuffer));
      stdoutBuffer.length = 0;
      postMessageToParent({ event: workerActions.WRITE_RESULT, text });
    }
    const finalStderr = decoder.decode(new Uint8Array(stderrBuffer));
    stderrBuffer.length = 0;
    postMessageToParent({ event: workerActions.RUN_COMPLETED, stderr: finalStderr });

  } catch (e) {
    const error = e instanceof Error ? e : new ModelInitializationError('Error running model inference');
    console.error('[NodeWorker] Error executing callMain:', error);
    reportError(error);
  }
}

// Main message handler for the worker
if (!parentPort) {
  throw new Error('[NodeWorker] Error: This script must be run as a worker thread.');
}

parentPort.on('message', async (data: any) => {
  const { event: action, wasmNodeModulePath, wasmNodePath, modelPath: modelPathArg, params, prompt } = data;
  console.log(`[NodeWorker] Received action: ${action}`, data); // Log received messages

  switch (action) {
    case workerActions.LOAD_NODE: // Use a specific Node action to avoid conflict with browser LOAD
      // Initialize the Wasm module (stubbed for now)
      await initWasmModule(wasmNodeModulePath, wasmNodePath);
      break;
      
    case workerActions.LOAD_MODEL_DATA: // Re-use this action, but expect path
      // Load model data from the provided path (stubbed for now)
      await loadModelData(modelPathArg); // Pass the path to the stub
      break;
      
    case workerActions.RUN_MAIN:
      // Generate text using the model (stubbed for now)
      runMain(prompt, params);
      break;
      
    case workerActions.CANCEL_LOAD:
      console.log('[NodeWorker] Cancellation requested via message');
      isCancellationRequested = true;
      // Cleanup might happen within the operation being cancelled
      break;
      
    default:
      console.warn(`[NodeWorker] Unknown action received: ${action}`);
      reportError(`Unknown worker action: ${action}`);
  }
});

// Signal worker is ready to receive messages (optional)
console.log('[NodeWorker] Worker started and listening for messages.');

// Handle worker exit gracefully
process.on('exit', (code) => {
  console.log(`[NodeWorker] Exiting with code: ${code}`);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.error('[NodeWorker] Uncaught Exception:', err);
  reportError(err instanceof Error ? err : new Error('Uncaught exception in worker'));
  // Optional: exit the worker process
  // process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('[NodeWorker] Unhandled Rejection at:', promise, 'reason:', reason);
  reportError(reason instanceof Error ? reason : new Error('Unhandled promise rejection in worker'));
  // Optional: exit the worker process
  // process.exit(1);
});

// --- Helper: Validate GGUF Header (Copied from worker.ts for standalone use if needed, or ensure import works) ---
// Ensure this function is available
function validateGGUFHeader(buffer: ArrayBuffer): void {
  if (!buffer || buffer.byteLength < 8) {
    throw new GGUFParsingError(
      "Insufficient data to validate file format (minimum 8 bytes required)",
      { byteLength: buffer?.byteLength ?? 0 }
    );
  }

  const dataView = new DataView(buffer);
  
  // Check magic number "GGUF" (0x46554747 in little-endian)
  const magic = dataView.getUint32(0, true);
  if (magic !== 0x46554747) {
    throw new GGUFParsingError(
      `Invalid file format: Not a GGUF file (magic number mismatch)`,
      { expected: 0x46554747, actual: magic, hexActual: `0x${magic.toString(16).toUpperCase()}` }
    );
  }

  // Check version
  const version = dataView.getUint32(4, true);
  const minSupported = 2;
  const maxSupported = 3; // Align with gguf-parser.ts if changed there
  
  if (version < minSupported || version > maxSupported) {
    throw new ModelCompatibilityError(
      `Unsupported GGUF version`,
      version,
      minSupported,
      maxSupported
    );
  }
}

// --- End Helper --- 