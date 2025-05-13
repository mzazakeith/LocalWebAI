import { NodeLlamaRunner, GenerateTextParams } from './node-llama-runner.js';
import { LoadingStage } from './loading-progress.js';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs'; // To check if files exist

// ESM equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Configuration ---
const modelFileName = 'phi-2.Q4_K_M.gguf';
const prompt = 'What is the capital of Italy?';
const generationParams: GenerateTextParams = {
  n_predict: 50,
  temp: 0.7,
  top_k: 40,
  top_p: 0.9,
  no_display_prompt: true, // Optional: set to true if you don't want the prompt in the output stream
};
// ---------------------

// Construct absolute paths
const workerPath = path.resolve(__dirname, 'node-worker.js');
const projectRoot = path.resolve(__dirname, '../../');
const wasmNodeModulePath = path.resolve(projectRoot, 'llama-cpp-wasm/dist/node/st/main.mjs');
const wasmNodePath = path.resolve(projectRoot, 'llama-cpp-wasm/dist/node/st/main.wasm');
const modelToLoadPath = path.resolve(projectRoot, 'models', modelFileName);

async function runPocInferenceTest() {
  console.log(`--- Starting Node.js LlamaRunner Inference POC Test (Model: ${modelFileName}) ---`);

  console.log(`Derived paths for the test:
  Project Root: ${projectRoot}
  Compiled Worker JS Path: ${workerPath}
  Wasm Module (JS glue): ${wasmNodeModulePath}
  Wasm File: ${wasmNodePath}
  Model File: ${modelToLoadPath}
  `);

  const pathsToVerify: { name: string, pathValue: string, type: 'file' | 'dir' }[] = [
    { name: "Compiled Worker JS", pathValue: workerPath, type: 'file' },
    { name: "Wasm Module (.mjs)", pathValue: wasmNodeModulePath, type: 'file' },
    { name: "Wasm File (.wasm)", pathValue: wasmNodePath, type: 'file' },
    { name: "Model File", pathValue: modelToLoadPath, type: 'file' },
  ];

  let pathsValid = true;
  for (const item of pathsToVerify) {
    if (!fs.existsSync(item.pathValue)) {
      console.error(`Error: Path for "${item.name}" does not exist: ${item.pathValue}`);
      pathsValid = false;
    }
  }

  if (!pathsValid) {
    console.error('Path verification failed. Please ensure all required files exist and build steps are complete.');
    return;
  }
  console.log('All specified paths exist. Proceeding...');

  let runner: NodeLlamaRunner | null = null;

  try {
    console.log('Instantiating NodeLlamaRunner...');
    runner = new NodeLlamaRunner(workerPath, wasmNodeModulePath, wasmNodePath);
    console.log('NodeLlamaRunner instantiated.');

    console.log(`Loading model: ${modelToLoadPath}`);
    await runner.loadModel(modelToLoadPath, (progress) => {
      console.log(`[Load Progress] Stage: ${progress.stage}, Msg: ${progress.message || ''}`);
      if (progress.loaded !== undefined && progress.total !== undefined && progress.total > 0) {
        console.log(`                 ${Math.round((progress.loaded / progress.total) * 100)}% (${progress.loaded}/${progress.total})`);
      }
      if (progress.stage === LoadingStage.ERROR) console.error(`                 ERROR: ${progress.error}`);
    });
    console.log('------------------------------------------');
    console.log('SUCCESS: Model loaded!');
    console.log('------------------------------------------');

    const metadata = runner.getModelMetadata();
    if (metadata) {
        console.log('Model Metadata:', JSON.stringify(metadata, null, 2));
    }

    console.log(`
--- Starting Inference ---`);
    console.log(`Prompt: "${prompt}"`);
    console.log('Generated Text:');
    
    let fullResponse = "";
    process.stdout.write('[STREAM] '); // For visual separation of streamed output

    await new Promise<void>((resolve, reject) => {
      runner!.generateText(
        prompt,
        generationParams,
        (token) => {
          process.stdout.write(token);
          fullResponse += token;
        },
        () => {
          process.stdout.write('\n[STREAM END]\n');
          console.log('------------------------------------------');
          console.log('SUCCESS: Inference completed!');
          console.log('------------------------------------------');
          console.log('Full Response Received:');
          console.log(fullResponse);
          resolve();
        }
      );
    });

  } catch (error) {
    console.error('--- POC Inference Test FAILED ---');
    if (error instanceof Error) {
        console.error(`Error Type: ${error.constructor.name}`);
        console.error(`Message: ${error.message}`);
        if (error.stack) console.error(`Stack: ${error.stack}`);
    } else {
        console.error('Unknown error:', error);
    }
    console.error('------------------------------------------');
  } finally {
    if (runner) {
      console.log('Terminating NodeLlamaRunner...');
      runner.terminate();
      console.log('NodeLlamaRunner terminated.');
    }
  }
  console.log(`--- Node.js LlamaRunner Inference POC Test Finished (Model: ${modelFileName}) ---`);
}

runPocInferenceTest().catch(err => {
    console.error("Unhandled critical error in POC inference test execution:", err);
}); 