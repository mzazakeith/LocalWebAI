// ts-wrapper/node-worker.ts

// Node.js worker threads imports
import { parentPort, isMainThread, workerData, TransferListItem } from 'worker_threads';
import os from 'os'; // Needed for potential hardware concurrency later

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
  console.log(`[NodeWorker] Stub: loadModelData called with path: ${modelPathFromArgs}`);
  if (!wasmModuleInstance) {
    reportError(new WasmError('Wasm module not initialized (stub)'));
    return;
  }
  
  if (checkCancellation()) {
      cleanupAfterCancellation();
      return;
  }

  reportProgress(LoadingStage.VFS_WRITE_START, { message: 'Stub: Reading model file header' });
  
  // Simulate reading header and validation success
  await new Promise(resolve => setTimeout(resolve, 50)); // Short delay
  
  if (checkCancellation()) { cleanupAfterCancellation(); return; }

  reportProgress(LoadingStage.VFS_WRITE_PROGRESS, { message: 'Stub: Writing model to VFS' });
  await new Promise(resolve => setTimeout(resolve, 100)); // Simulate write time

  if (checkCancellation()) { cleanupAfterCancellation(); return; }
  
  reportProgress(LoadingStage.VFS_WRITE_COMPLETE, { message: 'Stub: Model written to VFS' });
  
  // Simulate metadata parsing
  const metadata: ModelSpecification = await parseModelMetadata(); // Call the stubbed version
  
  if (checkCancellation()) { cleanupAfterCancellation(); return; }

  reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
    message: 'Stub: Model Ready for Inference',
    metadata: metadata
  });

  if (metadata) {
      postMessageToParent({ event: workerActions.MODEL_METADATA, metadata });
  }
  
  // Post INITIALIZED again to signify model readiness in this stub context
  postMessageToParent({ event: workerActions.INITIALIZED });

  reportProgress(LoadingStage.MODEL_READY, {
    message: 'Stub: Model Ready',
    metadata: metadata
  });
}

async function parseModelMetadata(): Promise<ModelSpecification> {
  console.log("[NodeWorker] Stub: parseModelMetadata called");
  reportProgress(LoadingStage.METADATA_PARSE_START, { message: 'Stub: Parsing model metadata' });
  
  // Simulate parsing success
  await new Promise(resolve => setTimeout(resolve, 30));
  
  const stubMetadata: ModelSpecification = {
    ggufVersion: 3, // Example value
    architecture: 'stub-arch',
    contextLength: 2048,
    embeddingLength: 4096,
    modelName: 'Stub Model',
    // Add other fields as needed for basic testing
  };
  
  reportProgress(LoadingStage.METADATA_PARSE_COMPLETE, { 
    message: 'Stub: Metadata parsed', 
    metadata: stubMetadata 
  });
  return stubMetadata;
}

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