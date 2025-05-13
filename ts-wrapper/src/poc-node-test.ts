import { NodeLlamaRunner } from './node-llama-runner.js';
import { LoadingStage } from './loading-progress.js';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs'; // To check if files exist

// ESM equivalent of __dirname
// When running the compiled .js file from dist/src, __dirname will be /path/to/LocalWebAIV2/ts-wrapper/dist/src
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Construct absolute paths

// Path to the compiled worker script.
// If poc-node-test.js is in dist/src (after compilation), node-worker.js (also compiled from src) will be in dist/src as well.
const workerPath = path.resolve(__dirname, 'node-worker.js');

// Path to the project root (LocalWebAIV2)
// From ts-wrapper/dist/src/, navigate up two levels.
const projectRoot = path.resolve(__dirname, '../../');

// Paths to Wasm artifacts (single-thread Node.js build, using .mjs)
const wasmNodeModulePath = path.resolve(projectRoot, 'llama-cpp-wasm/dist/node/st/main.mjs');
const wasmNodePath = path.resolve(projectRoot, 'llama-cpp-wasm/dist/node/st/main.wasm');

// Path to the model file (ensure this model exists)
const modelToLoadPath = path.resolve(projectRoot, 'models/phi-2.Q4_K_M.gguf');

async function runPocTest() {
  console.log('--- Starting Node.js LlamaRunner POC Test ---');

  console.log(`Derived paths for the test (these are resolved based on the compiled script's location):
  Project Root: ${projectRoot}
  Compiled Worker JS Path (for NodeLlamaRunner): ${workerPath}
  Wasm Module (JS glue) Path: ${wasmNodeModulePath}
  Wasm File Path: ${wasmNodePath}
  Model File Path: ${modelToLoadPath}
  `);

  // Verify paths before attempting to use them
  const pathsToVerify: { name: string, pathValue: string, type: 'file' | 'dir' }[] = [
    { name: "Compiled Worker JS (relative to this script's compiled location)", pathValue: workerPath, type: 'file' },
    { name: "Wasm Module (JS glue - .mjs)", pathValue: wasmNodeModulePath, type: 'file' },
    { name: "Wasm File (.wasm)", pathValue: wasmNodePath, type: 'file' },
    { name: "Model File", pathValue: modelToLoadPath, type: 'file' },
    { name: "Parent directory of Wasm Module", pathValue: path.dirname(wasmNodeModulePath), type: 'dir' },
    { name: "Parent directory of Model File", pathValue: path.dirname(modelToLoadPath), type: 'dir' },
  ];

  let pathsValid = true;
  for (const item of pathsToVerify) {
    if (!fs.existsSync(item.pathValue)) {
      console.error(`Error: Path for "${item.name}" does not exist: ${item.pathValue}`);
      pathsValid = false;
    } else {
      try {
        const stats = fs.statSync(item.pathValue);
        if (item.type === 'file' && !stats.isFile()) {
          console.error(`Error: Path for "${item.name}" is not a file: ${item.pathValue}`);
          pathsValid = false;
        } else if (item.type === 'dir' && !stats.isDirectory()) {
           console.error(`Error: Path for "${item.name}" is not a directory: ${item.pathValue}`);
          pathsValid = false;
        }
      } catch (e: any) {
        console.error(`Error stating path for "${item.name}" (${item.pathValue}): ${e.message}`);
        pathsValid = false;
      }
    }
  }

  if (!pathsValid) {
    console.error('Path verification failed. Please check the paths and ensure all required files/directories exist.');
    console.log('Ensure you have run:');
    console.log('1. The appropriate build script in `llama-cpp-wasm` (e.g., `build-node-single-thread.sh`) to generate Node.js artifacts including `main.mjs`.');
    console.log('2. `npm run build` (or `tsc`) in `ts-wrapper` to compile the TypeScript files.');
    console.log('3. The model file `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` exists in the `models` directory.');
    return;
  }
  console.log('All specified paths exist and seem correct. Proceeding with the test.');

  let runner: NodeLlamaRunner | null = null;

  try {
    console.log('Instantiating NodeLlamaRunner...');
    runner = new NodeLlamaRunner(
      workerPath,          // Absolute path to the compiled node-worker.js
      wasmNodeModulePath,  // Absolute path to the Wasm JS glue file (main.mjs)
      wasmNodePath         // Absolute path to the .wasm file
    );
    console.log('NodeLlamaRunner instantiated successfully.');

    console.log(`Attempting to load model: ${modelToLoadPath}`);
    await runner.loadModel(modelToLoadPath, (progress) => {
      console.log(`[Progress] Stage: ${progress.stage}, Message: ${progress.message || 'No message'}`);
      if (progress.loaded !== undefined && progress.total !== undefined && progress.total > 0) {
        const percent = Math.round((progress.loaded / progress.total) * 100);
        console.log(`           ${percent}% (${progress.loaded} / ${progress.total} bytes)`);
      } else if (progress.loaded !== undefined) {
        console.log(`           ${progress.loaded} bytes loaded`);
      }
      if (progress.stage === LoadingStage.ERROR) {
        console.error(`           Error during loading: ${progress.error}`);
      }
      if (progress.metadata) {
        // Truncate metadata for console logging if it's too long
        const metadataString = JSON.stringify(progress.metadata);
        const shortMetadata = metadataString.length > 200 ? metadataString.substring(0, 197) + "..." : metadataString;
        console.log('           Metadata received (partial):', shortMetadata);
      }
    });

    console.log('------------------------------------------');
    console.log('SUCCESS: Model loaded successfully!');
    console.log('------------------------------------------');
    const metadata = runner.getModelMetadata();
    if (metadata) {
        console.log('Final Model Metadata:', JSON.stringify(metadata, null, 2));
    } else {
        console.log('No metadata available after successful load.');
    }

  } catch (error) {
    console.error('--- POC Test FAILED ---');
    if (error instanceof Error) {
        console.error(`Error Type: ${error.constructor.name}`);
        console.error(`Error Message: ${error.message}`);
        if (error.stack) {
            console.error(`Stack Trace: ${error.stack}`);
        }
        // Log additional properties if they exist (like in custom errors)
        Object.entries(error).forEach(([key, value]) => {
            if (key !== 'message' && key !== 'name' && key !== 'stack') {
                console.error(`    ${key}: ${value}`);
            }
        });
    } else {
        console.error('An unknown error occurred:', error);
    }
    console.error('------------------------------------------');
  } finally {
    if (runner) {
      console.log('Terminating NodeLlamaRunner...');
      try {
        runner.terminate();
        console.log('NodeLlamaRunner terminated.');
      } catch (termError) {
        console.error('Error during NodeLlamaRunner termination:', termError);
      }
    }
  }
  console.log('--- Node.js LlamaRunner POC Test Finished ---');
}

runPocTest().catch(err => {
    console.error("Unhandled critical error in POC test execution:", err);
}); 