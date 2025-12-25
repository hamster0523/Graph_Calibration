/**
 * API客户端
 * 处理与后端API的所有HTTP通信
 */

class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.timeout = 120000; // 增加到120秒超时，适应Flow初始化
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    /**
     * 通用请求方法
     */
    async request(method, url, options = {}) {
        const fullURL = this.baseURL + url;
        const config = {
            method: method.toUpperCase(),
            headers: { ...this.defaultHeaders, ...options.headers },
            signal: this.createTimeoutSignal(options.timeout || this.timeout),
            ...options
        };

        // 处理请求体
        if (config.body && typeof config.body === 'object' && !(config.body instanceof FormData)) {
            config.body = JSON.stringify(config.body);
        }

        try {
            console.log(`🌐 ${method.toUpperCase()} ${fullURL}`);

            const response = await fetch(fullURL, config);

            // 检查HTTP错误
            if (!response.ok) {
                const errorData = await this.parseResponse(response);
                throw new APIError(
                    errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
                    response.status,
                    errorData
                );
            }

            return await this.parseResponse(response);

        } catch (error) {
            if (error.name === 'AbortError') {
                throw new APIError('Request timeout', 408);
            }

            if (error instanceof APIError) {
                throw error;
            }

            // 网络错误等
            throw new APIError(
                error.message || 'Network error',
                0,
                { originalError: error }
            );
        }
    }

    /**
     * 创建超时信号
     */
    createTimeoutSignal(timeout) {
        if (typeof AbortController === 'undefined') {
            return undefined;
        }

        const controller = new AbortController();
        setTimeout(() => controller.abort(), timeout);
        return controller.signal;
    }

    /**
     * 解析响应
     */
    async parseResponse(response) {
        const contentType = response.headers.get('content-type');

        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }

        if (contentType && contentType.includes('text/')) {
            return await response.text();
        }

        return await response.blob();
    }

    // === GET 请求 ===
    async get(url, params = {}, options = {}) {
        const urlWithParams = this.buildURL(url, params);
        return this.request('GET', urlWithParams, options);
    }

    // === POST 请求 ===
    async post(url, data = null, options = {}) {
        return this.request('POST', url, {
            body: data,
            ...options
        });
    }

    // === PUT 请求 ===
    async put(url, data = null, options = {}) {
        return this.request('PUT', url, {
            body: data,
            ...options
        });
    }

    // === DELETE 请求 ===
    async delete(url, options = {}) {
        return this.request('DELETE', url, options);
    }

    // === 文件上传 ===
    async upload(url, formData, options = {}) {
        const uploadOptions = {
            ...options,
            headers: {
                // 不设置Content-Type，让浏览器自动设置
                ...options.headers
            },
            body: formData
        };

        // 移除Content-Type以支持multipart/form-data
        delete uploadOptions.headers['Content-Type'];

        return this.request('POST', url, uploadOptions);
    }

    /**
     * 构建URL参数
     */
    buildURL(url, params = {}) {
        if (Utils.isEmpty(params)) {
            return url;
        }

        const urlObj = new URL(url, window.location.origin);
        Object.entries(params).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                urlObj.searchParams.append(key, value);
            }
        });

        return urlObj.pathname + urlObj.search;
    }

    // === 聊天相关API ===

    /**
     * 发送聊天消息
     */
    async sendMessage(message, timestamp = null) {
        return this.post('/api/chat', {
            message,
            timestamp: timestamp || new Date().toISOString()
        });
    }

    /**
     * 获取Agent状态
     */
    async getAgentStatus() {
        return this.get('/api/status');
    }

    /**
     * 重置Agent
     */
    async resetAgent() {
        return this.post('/api/agent/reset');
    }

    /**
     * 初始化Agent
     */
    async initializeAgent() {
        return this.post('/api/agent/initialize');
    }

    /**
     * 获取Agent信息
     */
    async getAgentInfo() {
        return this.get('/api/agent/info');
    }

    // === 工作空间相关API ===

    /**
     * 获取工作空间文件列表
     */
    async getWorkspaceFiles() {
        return this.get('/api/workspace/files');
    }

    /**
     * 上传文件
     */
    async uploadFiles(files) {
        const results = [];

        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const result = await this.upload('/api/workspace/upload', formData);
                results.push({ file: file.name, result, success: true });
            } catch (error) {
                results.push({ file: file.name, error, success: false });
            }
        }

        return results;
    }

    /**
     * 下载文件
     */
    async downloadFile(filePath) {
        const response = await fetch(`/api/workspace/download/${filePath}`);

        if (!response.ok) {
            throw new APIError(`Failed to download file: ${response.statusText}`, response.status);
        }

        return response.blob();
    }

    /**
     * 删除文件
     */
    async deleteFile(filePath) {
        return this.delete(`/api/workspace/delete/${filePath}`);
    }

    /**
     * 获取工作空间统计
     */
    async getWorkspaceStats() {
        return this.get('/api/workspace/stats');
    }

    /**
     * 创建文件夹
     */
    async createFolder(folderName) {
        return this.post('/api/workspace/create-folder', null, {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({ folder_name: folderName })
        });
    }

    /**
     * 搜索文件
     */
    async searchFiles(query) {
        return this.get('/api/workspace/search', { query });
    }

    // === 配置相关API ===

    /**
     * 获取配置
     */
    async getConfig() {
        return this.get('/api/config');
    }

    /**
     * 更新配置
     */
    async updateConfig(configData) {
        return this.post('/api/config', configData);
    }

    /**
     * 测试配置
     */
    async testConfig(testData) {
        return this.post('/api/config/test', testData);
    }

    /**
     * 获取可用模型
     */
    async getAvailableModels() {
        return this.get('/api/config/models');
    }

    /**
     * 获取搜索引擎列表
     */
    async getSearchEngines() {
        return this.get('/api/config/search-engines');
    }

    /**
     * 备份配置
     */
    async backupConfig() {
        return this.post('/api/config/backup');
    }

    /**
     * 恢复默认配置
     */
    async restoreDefaultConfig() {
        return this.post('/api/config/restore-default');
    }

    /**
     * 验证配置
     */
    async validateConfig() {
        return this.get('/api/config/validate');
    }

    /**
     * 获取配置状态
     */
    async getConfigStatus() {
        return this.get('/api/config/status');
    }

    // === 系统相关API ===

    /**
     * 健康检查
     */
    async healthCheck() {
        return this.get('/health');
    }
}

/**
 * API错误类
 */
class APIError extends Error {
    constructor(message, status = 0, data = null) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }

    /**
     * 检查是否为特定状态码的错误
     */
    isStatus(status) {
        return this.status === status;
    }

    /**
     * 检查是否为网络错误
     */
    isNetworkError() {
        return this.status === 0;
    }

    /**
     * 检查是否为客户端错误 (4xx)
     */
    isClientError() {
        return this.status >= 400 && this.status < 500;
    }

    /**
     * 检查是否为服务器错误 (5xx)
     */
    isServerError() {
        return this.status >= 500 && this.status < 600;
    }

    /**
     * 获取错误的详细信息
     */
    getDetails() {
        return this.data;
    }

    /**
     * 获取用户友好的错误消息
     */
    getUserMessage() {
        if (this.isNetworkError()) {
            return 'Network connection failed. Please check your internet connection.';
        }

        if (this.isStatus(408)) {
            return 'Request timed out. Please try again.';
        }

        if (this.isStatus(401)) {
            return 'Authentication required. Please check your credentials.';
        }

        if (this.isStatus(403)) {
            return 'Access denied. You don\'t have permission to perform this action.';
        }

        if (this.isStatus(404)) {
            return 'The requested resource was not found.';
        }

        if (this.isStatus(429)) {
            return 'Too many requests. Please wait a moment and try again.';
        }

        if (this.isServerError()) {
            return 'Server error occurred. Please try again later.';
        }

        return this.message;
    }
}

// 导出到全局
window.APIClient = APIClient;
window.APIError = APIError;

// 创建全局API客户端实例
window.apiClient = {
    chat: {
        sendMessage: async (data) => {
            const client = new APIClient();
            return client.request('POST', '/api/chat', { body: data });
        },
        getAgentStatus: async () => {
            const client = new APIClient();
            return client.request('GET', '/api/status');
        }
    },
    agent: {
        reset: async () => {
            const client = new APIClient();
            return client.request('POST', '/api/agent/reset');
        }
    },
    workspace: {
        listFiles: async () => {
            const client = new APIClient();
            return client.request('GET', '/api/workspace/files');
        },
        uploadFile: async (file, onProgress) => {
            // 验证文件对象
            if (!file || !file.name) {
                throw new Error('Invalid file object');
            }

            // 检查文件大小 (50MB限制)
            const maxSize = 50 * 1024 * 1024;
            if (file.size > maxSize) {
                throw new Error('File size exceeds 50MB limit');
            }

            const formData = new FormData();
            formData.append('file', file);

            const client = new APIClient();

            // 使用XMLHttpRequest来支持进度回调
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();

                // 进度监听
                if (onProgress) {
                    xhr.upload.addEventListener('progress', (e) => {
                        if (e.lengthComputable) {
                            const percentComplete = (e.loaded / e.total) * 100;
                            onProgress(percentComplete);
                        }
                    });
                }

                // 完成监听
                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            resolve(response);
                        } catch (e) {
                            resolve(xhr.responseText);
                        }
                    } else {
                        try {
                            const errorResponse = JSON.parse(xhr.responseText);
                            reject(new Error(errorResponse.detail || `HTTP ${xhr.status}`));
                        } catch (e) {
                            reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                        }
                    }
                });

                // 错误监听
                xhr.addEventListener('error', () => {
                    reject(new Error('Network error during upload'));
                });

                // 发送请求
                xhr.open('POST', '/api/workspace/upload');
                xhr.send(formData);
            });
        },
        deleteFile: async (filePath) => {
            const client = new APIClient();
            return client.request('DELETE', `/api/workspace/delete/${encodeURIComponent(filePath)}`);
        }
    },
    config: {
        getConfiguration: async () => {
            const client = new APIClient();
            return client.request('GET', '/api/config');
        },
        updateConfiguration: async (config) => {
            const client = new APIClient();
            return client.request('POST', '/api/config', { body: config });
        },
        testConfiguration: async (config) => {
            const client = new APIClient();
            return client.request('POST', '/api/config/test', { body: config });
        }
    }
};
