"""
WebSocket相关的API路由
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

try:
    from ...services.connection_manager import manager
except ImportError:
    try:
        from services.connection_manager import manager
    except ImportError:
        from backend.services.connection_manager import manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket连接端点
    
    处理客户端的WebSocket连接，支持实时双向通信
    """
    client_id = None
    
    try:
        # 接受连接
        await manager.connect(websocket)
        client_id = manager.connection_info.get(websocket, {}).get("client_id", "unknown")
        print(f"🔗 WebSocket client connected: {client_id}")
        print(f"   Total active connections: {manager.get_connection_count()}")
        
        # 发送连接成功消息
        await manager.send_json_message({
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "server_time": datetime.now().isoformat(),
                "message": "WebSocket connection established successfully"
            },
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        # 主消息循环
        while True:
            try:
                # 接收客户端消息
                data = await websocket.receive_text()
                
                # 处理消息
                await _handle_websocket_message(websocket, data, client_id)
                
            except WebSocketDisconnect:
                print(f"WebSocket client {client_id} disconnected normally")
                break
            except Exception as e:
                print(f"Error handling WebSocket message from {client_id}: {e}")
                # 发送错误消息给客户端
                await manager.send_json_message({
                    "type": "error",
                    "data": {
                        "message": f"Error processing message: {str(e)}",
                        "error_code": "MESSAGE_PROCESSING_ERROR"
                    },
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        # 清理连接
        manager.disconnect(websocket)


async def _handle_websocket_message(websocket: WebSocket, data: str, client_id: str):
    """处理WebSocket消息"""
    try:
        # 解析JSON消息
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await manager.send_json_message({
                "type": "error",
                "data": {
                    "message": "Invalid JSON format",
                    "error_code": "INVALID_JSON"
                },
                "timestamp": datetime.now().isoformat()
            }, websocket)
            return
        
        message_type = message.get("type")
        message_data = message.get("data", {})
        
        # 根据消息类型处理
        if message_type == "ping":
            # 处理心跳包
            await _handle_ping(websocket, message_data)
            
        elif message_type == "subscribe":
            # 处理订阅请求
            await _handle_subscribe(websocket, message_data)
            
        elif message_type == "unsubscribe":
            # 处理取消订阅
            await _handle_unsubscribe(websocket, message_data)
            
        elif message_type == "get_status":
            # 获取服务器状态
            await _handle_get_status(websocket)
            
        elif message_type == "broadcast_test":
            # 广播测试消息（仅用于调试）
            if message_data.get("message"):
                await manager.broadcast_json({
                    "type": "broadcast_message",
                    "data": {
                        "message": message_data["message"],
                        "from_client": client_id
                    },
                    "timestamp": datetime.now().isoformat()
                })
        
        else:
            # 未知消息类型
            await manager.send_json_message({
                "type": "error",
                "data": {
                    "message": f"Unknown message type: {message_type}",
                    "error_code": "UNKNOWN_MESSAGE_TYPE"
                },
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
    except Exception as e:
        print(f"Error handling message from {client_id}: {e}")
        await manager.send_json_message({
            "type": "error", 
            "data": {
                "message": f"Internal error: {str(e)}",
                "error_code": "INTERNAL_ERROR"
            },
            "timestamp": datetime.now().isoformat()
        }, websocket)


async def _handle_ping(websocket: WebSocket, data: dict):
    """处理ping消息"""
    await manager.update_ping(websocket)
    await manager.send_json_message({
        "type": "pong",
        "data": {
            "server_time": datetime.now().isoformat(),
            "client_message": data.get("message", "")
        },
        "timestamp": datetime.now().isoformat()
    }, websocket)


async def _handle_subscribe(websocket: WebSocket, data: dict):
    """处理订阅请求"""
    channels = data.get("channels", [])
    
    # 这里可以实现频道订阅逻辑
    # 目前所有客户端都会收到所有广播消息
    
    await manager.send_json_message({
        "type": "subscription_confirmed",
        "data": {
            "channels": channels,
            "message": "Subscribed to channels"
        },
        "timestamp": datetime.now().isoformat()
    }, websocket)


async def _handle_unsubscribe(websocket: WebSocket, data: dict):
    """处理取消订阅"""
    channels = data.get("channels", [])
    
    # 这里可以实现取消订阅逻辑
    
    await manager.send_json_message({
        "type": "unsubscription_confirmed",
        "data": {
            "channels": channels,
            "message": "Unsubscribed from channels"
        },
        "timestamp": datetime.now().isoformat()
    }, websocket)


async def _handle_get_status(websocket: WebSocket):
    """处理获取状态请求"""
    await manager.send_json_message({
        "type": "server_status",
        "data": {
            "connected_clients": manager.get_connection_count(),
            "server_time": datetime.now().isoformat(),
            "connections": manager.get_connection_info()
        },
        "timestamp": datetime.now().isoformat()
    }, websocket)