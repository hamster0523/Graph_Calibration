/**
 * 工具函数库
 * 提供通用的工具方法和辅助函数
 */

window.Utils = {
    /**
     * 转义HTML字符串
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * 反转义HTML字符串
     */
    unescapeHtml(html) {
        const div = document.createElement('div');
        div.innerHTML = html;
        return div.textContent || div.innerText || '';
    },

    /**
     * 格式化文件大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * 格式化时间
     */
    formatTime(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        
        if (!(date instanceof Date) || isNaN(date)) {
            return 'Invalid Date';
        }
        
        const now = new Date();
        const diff = now - date;
        
        // 小于1分钟
        if (diff < 60000) {
            return 'Just now';
        }
        
        // 小于1小时
        if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        }
        
        // 小于24小时
        if (diff < 86400000) {
            const hours = Math.floor(diff / 3600000);
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        }
        
        // 小于7天
        if (diff < 604800000) {
            const days = Math.floor(diff / 86400000);
            return `${days} day${days > 1 ? 's' : ''} ago`;
        }
        
        // 超过7天显示具体日期
        return date.toLocaleDateString();
    },

    /**
     * 格式化相对时间
     */
    formatRelativeTime(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        
        return new Intl.RelativeTimeFormat('en', { numeric: 'auto' }).format(
            Math.round((date - new Date()) / (1000 * 60 * 60 * 24)),
            'day'
        );
    },

    /**
     * 深拷贝对象
     */
    deepClone(obj) {
        if (obj === null || typeof obj !== 'object') {
            return obj;
        }
        
        if (obj instanceof Date) {
            return new Date(obj);
        }
        
        if (Array.isArray(obj)) {
            return obj.map(item => this.deepClone(item));
        }
        
        const cloned = {};
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                cloned[key] = this.deepClone(obj[key]);
            }
        }
        
        return cloned;
    },

    /**
     * 防抖函数
     */
    debounce(func, wait, immediate = false) {
        let timeout;
        
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func.apply(this, args);
            };
            
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            
            if (callNow) func.apply(this, args);
        };
    },

    /**
     * 节流函数
     */
    throttle(func, limit) {
        let inThrottle;
        
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * 生成唯一ID
     */
    generateId(prefix = '') {
        const timestamp = Date.now().toString(36);
        const randomPart = Math.random().toString(36).substr(2);
        return prefix + timestamp + randomPart;
    },

    /**
     * 验证邮箱格式
     */
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    },

    /**
     * 验证URL格式
     */
    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    },

    /**
     * 获取文件扩展名
     */
    getFileExtension(filename) {
        return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
    },

    /**
     * 获取文件类型图标
     */
    getFileTypeIcon(filename) {
        const extension = this.getFileExtension(filename).toLowerCase();
        const iconMap = {
            // 图片
            'jpg': 'icon-image',
            'jpeg': 'icon-image', 
            'png': 'icon-image',
            'gif': 'icon-image',
            'svg': 'icon-image',
            'webp': 'icon-image',
            
            // 文档
            'pdf': 'icon-file-pdf',
            'doc': 'icon-file-word',
            'docx': 'icon-file-word',
            'xls': 'icon-file-excel',
            'xlsx': 'icon-file-excel',
            'ppt': 'icon-file-powerpoint',
            'pptx': 'icon-file-powerpoint',
            
            // 代码
            'js': 'icon-file-code',
            'ts': 'icon-file-code',
            'py': 'icon-file-code',
            'java': 'icon-file-code',
            'cpp': 'icon-file-code',
            'c': 'icon-file-code',
            'php': 'icon-file-code',
            'html': 'icon-file-code',
            'css': 'icon-file-code',
            'scss': 'icon-file-code',
            'json': 'icon-file-code',
            'xml': 'icon-file-code',
            'yml': 'icon-file-code',
            'yaml': 'icon-file-code',
            
            // 压缩包
            'zip': 'icon-file-archive',
            'rar': 'icon-file-archive',
            '7z': 'icon-file-archive',
            'tar': 'icon-file-archive',
            'gz': 'icon-file-archive',
            
            // 音视频
            'mp3': 'icon-file-audio',
            'wav': 'icon-file-audio',
            'flac': 'icon-file-audio',
            'mp4': 'icon-file-video',
            'avi': 'icon-file-video',
            'mov': 'icon-file-video',
            'wmv': 'icon-file-video',
            
            // 文本
            'txt': 'icon-file-text',
            'md': 'icon-file-text',
            'csv': 'icon-file-text'
        };
        
        return iconMap[extension] || 'icon-file';
    },

    /**
     * 复制文本到剪贴板
     */
    async copyToClipboard(text) {
        if (navigator.clipboard) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch (err) {
                console.error('Failed to copy to clipboard:', err);
            }
        }
        
        // 降级方案
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            document.body.removeChild(textArea);
            return true;
        } catch (err) {
            console.error('Fallback copy failed:', err);
            document.body.removeChild(textArea);
            return false;
        }
    },

    /**
     * 下载文件
     */
    downloadFile(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    },

    /**
     * 检查是否为移动设备
     */
    isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    },

    /**
     * 获取设备类型
     */
    getDeviceType() {
        if (this.isMobile()) {
            return 'mobile';
        }
        
        if (window.innerWidth <= 768) {
            return 'tablet';
        }
        
        return 'desktop';
    },

    /**
     * 滚动到元素
     */
    scrollToElement(element, options = {}) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (!element) return;
        
        const defaultOptions = {
            behavior: 'smooth',
            block: 'start',
            inline: 'nearest'
        };
        
        element.scrollIntoView({ ...defaultOptions, ...options });
    },

    /**
     * 创建DOM元素
     */
    createElement(tag, attributes = {}, children = []) {
        const element = document.createElement(tag);
        
        // 设置属性
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'textContent') {
                element.textContent = value;
            } else if (key === 'innerHTML') {
                element.innerHTML = value;
            } else if (key === 'className' || key === 'class') {
                element.className = value;
            } else {
                element.setAttribute(key, value);
            }
        });
        
        // 添加子元素
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else {
                element.appendChild(child);
            }
        });
        
        return element;
    },

    /**
     * 等待指定时间
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    /**
     * 重试函数
     */
    async retry(fn, maxRetries = 3, delay = 1000) {
        let lastError;
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                if (i < maxRetries - 1) {
                    await this.sleep(delay * (i + 1));
                }
            }
        }
        
        throw lastError;
    },

    /**
     * 格式化JSON
     */
    formatJson(obj, indent = 2) {
        return JSON.stringify(obj, null, indent);
    },

    /**
     * 安全的JSON解析
     */
    safeJsonParse(str, defaultValue = null) {
        try {
            return JSON.parse(str);
        } catch {
            return defaultValue;
        }
    },

    /**
     * 检查对象是否为空
     */
    isEmpty(obj) {
        if (obj == null) return true;
        if (Array.isArray(obj) || typeof obj === 'string') return obj.length === 0;
        return Object.keys(obj).length === 0;
    },

    /**
     * 合并对象
     */
    merge(...objects) {
        return Object.assign({}, ...objects);
    },

    /**
     * 省略文本
     */
    truncate(str, length = 50, suffix = '...') {
        if (str.length <= length) return str;
        return str.slice(0, length) + suffix;
    }
};