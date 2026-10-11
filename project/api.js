/**
 * NOWCAST AI - Convective Scale Weather Intelligence Platform
 * Frontend API Service (api.js) - Part 4 Architecture
 * Connects the web GIS mission control directly to the FastAPI PyTorch ConvLSTM backend.
 */

const API_BASE_URL = window.location.origin;

/**
 * Checks operational status of the ConvLSTM model and dataset provider.
 */
async function getModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model-status`);
        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch model status:", err);
        throw err;
    }
}

/**
 * Triggers actual PyTorch ConvLSTM inference on the backend.
 * @param {number} sequenceIndex - Sample index in dataset [0..79]
 */
async function runForecast(sequenceIndex = 0) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/run-forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sequence_index: sequenceIndex,
                demo_mode: true
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server responded with ${response.status}`);
        }

        return await response.json();
    } catch (err) {
        console.error("Failed to run ConvLSTM forecast:", err);
        throw err;
    }
}

/**
 * Retrieves the currently cached latest forecast result.
 */
async function getForecast() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/forecast`);
        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}`);
        }
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch cached forecast:", err);
        throw err;
    }
}

/**
 * Reloads the PyTorch ConvLSTM weights into memory.
 */
async function reloadModel() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/reload-model`, {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}`);
        }
        return await response.json();
    } catch (err) {
        console.error("Failed to reload model:", err);
        throw err;
    }
}

/**
 * Fetches data source telemetry.
 */
async function getDataSources() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data-sources`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch data sources:", err);
        throw err;
    }
}

/**
 * Part 4: Fetches authentic provider live status, latency, and QC flags.
 */
async function getProvidersStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/providers/status`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch providers status:", err);
        throw err;
    }
}

/**
 * Part 4: Sends natural language queries to context-aware AI chatbot assistant.
 */
async function sendChatMessage(message, contextOverride = null) {
    try {
        const payload = { message };
        if (contextOverride) {
            payload.context_override = contextOverride;
        }
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to send chat message:", err);
        throw err;
    }
}

/**
 * Part 4: Generates automated 6-hour meteorological nowcasting bulletin.
 */
async function getOperationalReport() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/reports/generate`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to generate operational report:", err);
        throw err;
    }
}

/**
 * Part 4: Fetches structured backend pipeline logs.
 */
async function getPipelineLogs(limit = 30) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/logs?limit=${limit}`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch pipeline logs:", err);
        throw err;
    }
}

/**
 * Fetches impact calculation for specified timestep.
 */
async function getImpact(stepIndex = 0) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/impact?step_index=${stepIndex}`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch impact:", err);
        throw err;
    }
}

/**
 * Fetches top 5 risk locations for specified timestep.
 */
async function getRiskLocations(stepIndex = 0) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/risk-locations?step_index=${stepIndex}`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch risk locations:", err);
        throw err;
    }
}

// ==============================================================================
// PART 7: VALIDATION & EXPLAINABILITY API CLIENT METHODS
// ==============================================================================

async function getEvaluationSummary() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/summary`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch evaluation summary:", err);
        throw err;
    }
}

async function getEvaluationLeadTime() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/lead-time`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch lead-time metrics:", err);
        throw err;
    }
}

async function getEvaluationBaselines() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/baselines`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch baseline comparisons:", err);
        throw err;
    }
}

async function getEvaluationAblation() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/ablation`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch ablation results:", err);
        throw err;
    }
}

async function getWhyForecast() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/explainability/why-forecast`);
        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        return await response.json();
    } catch (err) {
        console.error("Failed to fetch why-forecast explanation:", err);
        throw err;
    }
}

// Export for window
if (typeof window !== 'undefined') {
    window.NowcastAPI = {
        getModelStatus,
        runForecast,
        getForecast,
        reloadModel,
        getDataSources,
        getProvidersStatus,
        sendChatMessage,
        getOperationalReport,
        getPipelineLogs,
        getImpact,
        getRiskLocations,
        getEvaluationSummary,
        getEvaluationLeadTime,
        getEvaluationBaselines,
        getEvaluationAblation,
        getWhyForecast
    };
}
