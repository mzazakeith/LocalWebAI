import * as path from 'path';
import { fileURLToPath } from 'url';
import { NodeJsLlamaCppRunner, NodeModelLoadParams } from './node-llama-cpp-runner.js';
import { LoadingStage, ProgressCallback } from './loading-progress.js';
import { TokenCallback, CompletionCallback, GenerateTextParams } from './llama-runner.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// const projectRoot = path.resolve(__dirname, '../../..');
// const modelPath = path.join(projectRoot, 'models', 'phi-2.Q4_K_M.gguf');
const modelPath = '/tmp/phi-2.Q4_K_M.gguf'; // Using the simplified path for now
const samplePrompt = 'The capital of Kenya is';

const progressCallback: ProgressCallback = (progress) => {
    console.log(`[INFERENCE TEST] Progress: Stage: ${progress.stage}, Message: ${progress.message || ''}`);
    if (progress.stage === LoadingStage.ERROR) {
        console.error(`[INFERENCE TEST] Error during loading: ${progress.error}`);
    }
};

const tokenCallback: TokenCallback = (textChunk) => {
    process.stdout.write(textChunk); // Stream token to console
};

const completionCallback: CompletionCallback = () => {
    console.log('\n[INFERENCE TEST] Text generation stream finished.'); // Indicates streaming is done
};

async function testModelInference() {
    console.log('[INFERENCE TEST] Starting model inference test...');
    const runner = new NodeJsLlamaCppRunner({ gpu: false }); // Force CPU

    try {
        console.log(`[INFERENCE TEST] Attempting to load model from: ${modelPath}`);
        const loadParams: Partial<NodeModelLoadParams & { progressCallback?: ProgressCallback; signal?: AbortSignal }> = {
            progressCallback: progressCallback,
            gpuLayers: 0 // Force CPU layers
        };
        await runner.loadModel(modelPath, loadParams);
        console.log('[INFERENCE TEST] Model loaded successfully.');

        const generateParams: GenerateTextParams = {
            n_predict: 50,
            temp: 0.7,
        };

        console.log(`[INFERENCE TEST] Generating text for prompt: "${samplePrompt}"`);
        // NodeJsLlamaCppRunner's generateText expects a CompletionCallback that matches () => void
        await runner.generateText(samplePrompt, generateParams, tokenCallback, completionCallback);
        // The actual completion message or error handling specific to generateText outcome
        // should ideally be handled within NodeJsLlamaCppRunner or its internal callbacks if needed.
        // For this test, completionCallback just signals the end of the stream.

    } catch (error: any) {
        console.error('[INFERENCE TEST] Model inference test failed:');
        if (error instanceof Error) {
            console.error(`Error Type: ${error.constructor.name}`);
            console.error(`Error Message: ${error.message}`);
            if (error.stack) {
                console.error(`Stack Trace: ${error.stack}`);
            }
        } else {
            console.error(error);
        }
    } finally {
        console.log('[INFERENCE TEST] Terminating runner...');
        await runner.terminate();
        console.log('[INFERENCE TEST] Runner terminated.');
    }
}

testModelInference(); 