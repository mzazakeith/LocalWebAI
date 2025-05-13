import * as path from 'path';
import { fileURLToPath } from 'url';
import { NodeJsLlamaCppRunner, NodeModelLoadParams } from './node-llama-cpp-runner.js';
import { LoadingStage, ProgressCallback } from './loading-progress.js';
import { ModelSpecification } from './model-spec.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// const projectRoot = path.resolve(__dirname, '../../..');
// const modelPath = path.join(projectRoot, 'models', 'phi-2.Q4_K_M.gguf'); // Adjust path as needed
const modelPath = '/tmp/phi-2.Q4_K_M.gguf'; // Using the simplified path

const progressCallback: ProgressCallback = (progress) => {
    console.log(`[LOAD TEST] Progress: Stage: ${progress.stage}, Message: ${progress.message || ''}`);
    if (progress.loaded !== undefined && progress.total !== undefined) {
        const percentage = progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : 0;
        console.log(`[LOAD TEST] ${percentage}% (${progress.loaded}/${progress.total})`);
    }
    if (progress.stage === LoadingStage.ERROR) {
        console.error(`[LOAD TEST] Error during loading: ${progress.error}`);
    }
    if (progress.stage === LoadingStage.METADATA_PARSE_COMPLETE && progress.metadata) {
        console.log('[LOAD TEST] Model metadata loaded:', JSON.stringify(progress.metadata, null, 2));
    }
};

async function testModelLoad() {
    console.log('[LOAD TEST] Starting model load test...');
    const runner = new NodeJsLlamaCppRunner({ gpu: false });

    try {
        console.log(`[LOAD TEST] Attempting to load model from: ${modelPath}`);
        
        const loadParams: Partial<NodeModelLoadParams & { progressCallback?: ProgressCallback; signal?: AbortSignal }> = {
            progressCallback: progressCallback,
            gpuLayers: 0 // Force CPU execution by disabling GPU layers
        };

        await runner.loadModel(modelPath, loadParams);
        console.log('[LOAD TEST] Model loaded successfully.');

        const metadata: ModelSpecification | null = runner.getModelMetadata();
        if (metadata) {
            console.log('[LOAD TEST] Retrieved metadata after load:', JSON.stringify(metadata, null, 2));
        } else {
            console.warn('[LOAD TEST] Could not retrieve metadata after successful load.');
        }

        console.log('[LOAD TEST] Terminating runner...');
        await runner.terminate();
        console.log('[LOAD TEST] Runner terminated.');
        console.log('[LOAD TEST] Model load test completed successfully.');

    } catch (error: any) {
        console.error('[LOAD TEST] Model load test failed:');
        if (error.name && error.message) {
            console.error(`Error Type: ${error.name}`);
            console.error(`Error Message: ${error.message}`);
            if (error.stack) {
                 console.error(`Stack Trace: ${error.stack}`);
            }
        } else {
            console.error(error);
        }
        // Ensure runner is terminated even on error
        try {
            await runner.terminate();
            console.log('[LOAD TEST] Runner terminated after error.');
        } catch (terminateError) {
            console.error('[LOAD TEST] Error terminating runner after test failure:', terminateError);
        }
    }
}

testModelLoad().catch(e => {
    console.error("[LOAD TEST] Unhandled error in testModelLoad:", e);
}); 