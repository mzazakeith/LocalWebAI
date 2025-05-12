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
  // For now, just log stderr to console
  // Consider sending critical stderr messages to parent
  process.stderr.write(String.fromCharCode(c));
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

// --- Stubbed Implementations for Phase B ---

async function initWasmModule(wasmModulePath: string, wasmPath: string) {
  console.log(`[NodeWorker] Stub: initWasmModule called with modulePath: ${wasmModulePath}, wasmPath: ${wasmPath}`);
  reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
    message: 'Stub: Initializing WebAssembly module'
  });
  
  // Simulate success for Phase B structure testing
  // In later phases, this will perform the actual import and instantiation
  wasmModuleInstance = { 
      callMain: (args) => console.log(`[NodeWorker] Stub: callMain(${JSON.stringify(args)})`), 
      FS_createPath: (p, n) => console.log(`[NodeWorker] Stub: FS_createPath(${p}, ${n})`), 
      FS_createDataFile: (p, n) => console.log(`[NodeWorker] Stub: FS_createDataFile(${p}, ${n})`), 
      FS: { 
          open: () => ({ /* stub stream */ }), 
          read: () => 0, 
          close: () => {}, 
          stat: () => ({ /* stub stat */ }),
          unlink: () => console.log("[NodeWorker] Stub: FS.unlink()")
      },
      TTY: { register: () => {} }, // Add stubs for expected properties
      noInitialRun: true,
      preInit: []
  }; 
  console.log("[NodeWorker] Stub: Wasm module instance created.");
  
  // No model is loaded yet in this stub
  reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
    message: 'Stub: Wasm Module Initialized (No Model Loaded)'
  });
  
  // Send INITIALIZED immediately after stubbing for Phase B testing
  postMessageToParent({ event: workerActions.INITIALIZED });
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
    
    // Validate GGUF Header (using the Buffer directly, ArrayBuffer view is compatible)
    // Note: validateGGUFHeader itself might need adjustment if it strictly requires ArrayBuffer
    // For now, assume it works or adapt it later. We use slice().buffer for safety.
    try {
        // Create an ArrayBuffer view from the relevant part of the Buffer
        const headerArrayBuffer = headerBuffer.buffer.slice(headerBuffer.byteOffset, headerBuffer.byteOffset + bytesRead);
        validateGGUFHeader(headerArrayBuffer);
    } catch (error) {
      // If validation fails, report the specific error and stop
      reportError(error instanceof Error ? error : new GGUFParsingError(String(error)));
      return; // Stop processing this model
    }
    
    console.log('[NodeWorker] GGUF header validated successfully.');
    reportProgress(LoadingStage.PREPARING_MODEL_DATA, { 
      message: 'Validated model header. Reading full model file...',
      loaded: bytesRead, // Report header bytes as initial loaded count
      total: totalSize
    });

    // Check for cancellation before reading the full file
    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // 4. Read the entire file into a buffer
    // TODO: Implement streaming for large files to avoid memory issues
    console.log(`[NodeWorker] Reading full file (${(totalSize / (1024*1024)).toFixed(2)} MB) into buffer...`);
    modelDataBuffer = await fileHandle.readFile();
    console.log(`[NodeWorker] File read complete. Buffer size: ${modelDataBuffer.byteLength}`);

    // Check buffer size matches stats
    if (modelDataBuffer.byteLength !== totalSize) {
        console.warn(`[NodeWorker] Warning: Read buffer size (${modelDataBuffer.byteLength}) does not match file stat size (${totalSize}). Proceeding anyway.`);
    }

    // Report completion of file reading phase (approximated)
    reportProgress(LoadingStage.PREPARING_MODEL_DATA, { 
      message: 'Model file read into memory',
      loaded: totalSize, // Mark as fully loaded into buffer
      total: totalSize
    });

    // Check for cancellation before writing to VFS
    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // 5. Write the model data to VFS
    reportProgress(LoadingStage.VFS_WRITE_START, { 
        message: 'Writing model to virtual filesystem',
        total: totalSize
    });
    
    try {
      // Ensure VFS directory exists
      wasmModuleInstance.FS_createPath("/", "models", true, true);
      
      // Write the buffer (must be Uint8Array for FS_createDataFile)
      // Create a Uint8Array view of the Node.js Buffer without copying memory
      const modelDataView = new Uint8Array(modelDataBuffer.buffer, modelDataBuffer.byteOffset, modelDataBuffer.byteLength);
      wasmModuleInstance.FS_createDataFile('/models', 'model.bin', modelDataView, true, true, true);
      
      console.log('[NodeWorker] Successfully wrote model to VFS path:', modelPath);
      
    } catch (error) {
      throw new VFSError(
        `Failed to write model data to VFS: ${error instanceof Error ? error.message : String(error)}`,
        modelPath
      );
    }
    
    // Release reference to the large buffer now that it's in VFS
    modelDataBuffer = null; 

    reportProgress(LoadingStage.VFS_WRITE_COMPLETE, { 
        message: 'Model written to virtual filesystem',
        loaded: totalSize,
        total: totalSize
    });

    // Check for cancellation after writing to VFS
    if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // 6. Parse Metadata (from VFS)
    let metadata: ModelSpecification | null = null;
    try {
      metadata = await parseModelMetadata(); // Call the actual implementation now
    } catch (error) {
      console.warn("[NodeWorker] Metadata parsing failed, continuing without full metadata:", error);
      // Report the error but allow initialization to continue if possible
      reportError(error instanceof Error ? error : new GGUFParsingError(String(error)), LoadingStage.METADATA_PARSE_COMPLETE); 
    }
    
    // Check for cancellation after metadata parsing
    if (checkCancellation()) {
        cleanupAfterCancellation();
        return;
    }

    // 7. Report completion
    reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
      message: 'Model ready for inference',
      metadata: metadata || undefined
    });

    if (metadata) {
      postMessageToParent({ event: workerActions.MODEL_METADATA, metadata });
    }

    // Signal that the model is loaded and ready
    postMessageToParent({ event: workerActions.INITIALIZED });

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
    // Ensure the file handle is closed
    if (fileHandle) {
      try {
        await fileHandle.close();
        console.log(`[NodeWorker] Closed file handle for: ${modelPathFromArgs}`);
      } catch (closeError) {
        console.error(`[NodeWorker] Error closing file handle: ${closeError}`);
      }
    }
    // Release buffer reference if it wasn't already
    modelDataBuffer = null;
    // Reset cancellation state for the next operation
    resetCancellationState(); 
  }
}

// Un-stub: Keep the actual implementation that reads from VFS
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

    // Validate first (reuse existing validation)
    validateGGUFHeader(headerData);

    // Parse
    const metadata = parseGGUFHeader(headerData);

    reportProgress(LoadingStage.METADATA_PARSE_COMPLETE, {
      message: 'Model metadata extracted successfully from VFS',
      metadata: metadata
    });

    return metadata;
  } catch (error) {
    // Throw the error to be caught by the caller (loadModelData)
    console.error("[NodeWorker] Error parsing metadata from VFS:", error);
    if (error instanceof Error) {
        throw error; // Re-throw specific errors
    } else {
        throw new GGUFParsingError(String(error)); // Wrap unknown errors
    }
  }
}

// Keep runMain stubbed for Phase C
function runMain(prompt: string, params: Record<string, any>) {
  console.log(`[NodeWorker] Stub: runMain called with prompt: "${prompt}", params:`, params);
  if (!wasmModuleInstance) {
    reportError(new WasmError('Wasm module not ready (stub)'));
    return;
  }

  // Simulate generation
  const stubOutput = `\nStub response to prompt: "${prompt}"`;
  let i = 0;
  const interval = setInterval(() => {
      if (i < stubOutput.length) {
          stdout(stubOutput.charCodeAt(i));
          i++;
      } else {
          clearInterval(interval);
          // Ensure buffer is flushed
          if (stdoutBuffer.length > 0) {
            const text = decoder.decode(new Uint8Array(stdoutBuffer));
            stdoutBuffer.length = 0;
            postMessageToParent({ event: workerActions.WRITE_RESULT, text });
          }
          postMessageToParent({ event: workerActions.RUN_COMPLETED });
      }
  }, 20); // Simulate token streaming speed
}

// --- End Stubbed Implementations ---

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