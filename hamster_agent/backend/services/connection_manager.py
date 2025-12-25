"""
WebSocket连接管理器
处理WebSocket连接的创建、删除和消息广播
"""

import json
import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket
from datetime import datetime


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 保存连接信息
        self.connection_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now().isoformat(),
            "last_ping": datetime.now().isoformat()
        }
        
        print(f"WebSocket connected: {self.connection_info[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            client_info = self.connection_info.get(websocket, {})
            print(f"WebSocket disconnected: {client_info.get('client_id', 'unknown')}")
            
            self.active_connections.remove(websocket)
            if websocket in self.connection_info:
                del self.connection_info[websocket]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def send_json_message(self, data: Dict[str, Any], websocket: WebSocket):
        """发送JSON消息"""
        try:
            message = json.dumps(data, ensure_ascii=False)
            await self.send_personal_message(message, websocket)
        except Exception as e:
            print(f"Error sending JSON message: {e}")
    
    async def broadcast(self, message: str):
        """广播消息给所有连接"""
        if not self.active_connections:
            return
        
        # 并发发送消息
        tasks = []
        for connection in self.active_connections.copy():
            tasks.append(self._safe_send(connection, message))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """广播JSON消息给所有连接"""
        try:
            message = json.dumps(data, ensure_ascii=False)
            await self.broadcast(message)
        except Exception as e:
            print(f"Error broadcasting JSON message: {e}")
    
    async def _safe_send(self, websocket: WebSocket, message: str):
        """安全发送消息（处理断开的连接）"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending message to client: {e}")
            self.disconnect(websocket)
    
    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self.active_connections)
    
    def get_connection_info(self) -> List[Dict[str, Any]]:
        """获取所有连接信息"""
        return [
            {
                "client_id": info["client_id"],
                "connected_at": info["connected_at"],
                "last_ping": info.get("last_ping")
            }
            for info in self.connection_info.values()
        ]
    
    async def update_ping(self, websocket: WebSocket):
        """更新连接的ping时间"""
        if websocket in self.connection_info:
            self.connection_info[websocket]["last_ping"] = datetime.now().isoformat()


# 全局连接管理器实例
manager = ConnectionManager()