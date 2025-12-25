import json
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional, Any, Union

import boto3
import requests

from .config import config

# Global variables to track the current tool use ID across function calls
# Tmp solution
CURRENT_TOOLUSE_ID = None


# Class to handle OpenAI-style response formatting
class OpenAIResponse:
    def __init__(self, data):
        # Recursively convert nested dicts and lists to OpenAIResponse objects
        for key, value in data.items():
            if isinstance(value, dict):
                value = OpenAIResponse(value)
            elif isinstance(value, list):
                value = [
                    OpenAIResponse(item) if isinstance(item, dict) else item
                    for item in value
                ]
            setattr(self, key, value)

    def model_dump(self, *args, **kwargs):
        # Convert object to dict and add timestamp
        data = self.__dict__
        data["created_at"] = datetime.now().isoformat()
        return data


# Main client class for interacting with Amazon Bedrock
class BedrockClient:
    def __init__(self):
        # Initialize Bedrock client, you need to configure AWS env first
        try:
            self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
            self.chat = Chat(self.client)
        except Exception as e:
            print(f"Error initializing Bedrock client: {e}")
            sys.exit(1)

class BedrockClient_request:
    def __init__(self, api_url: Optional[str] = None, api_token: Optional[str] = None):
        self.api_url = api_url
        self.api_token = api_token
        self.chat = Chat_request(api_url, api_token)

# Chat interface class
class Chat:
    def __init__(self, client):
        self.completions = ChatCompletions(client)

class Chat_request:
    def __init__(self, api_url: Optional[str] = None, api_token: Optional[str] = None):
        self.completions = ChatCompletions_request(api_url, api_token)

# Core class handling chat completions functionality
class ChatCompletions:
    def __init__(self, client):
        self.client = client

    def _convert_openai_tools_to_bedrock_format(self, tools):
        # Convert OpenAI function calling format to Bedrock tool format
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                bedrock_tool = {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": function.get("parameters", {}).get(
                                    "properties", {}
                                ),
                                "required": function.get("parameters", {}).get(
                                    "required", []
                                ),
                            }
                        },
                    }
                }
                bedrock_tools.append(bedrock_tool)
        return bedrock_tools

    def _convert_openai_messages_to_bedrock_format(self, messages):
        # Convert OpenAI message format to Bedrock message format
        bedrock_messages = []
        system_prompt = []
        for message in messages:
            if message.get("role") == "system":
                system_prompt = [{"text": message.get("content")}]
            elif message.get("role") == "user":
                bedrock_message = {
                    "role": message.get("role", "user"),
                    "content": [{"text": message.get("content")}],
                }
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "assistant":
                bedrock_message = {
                    "role": "assistant",
                    "content": [{"text": message.get("content")}],
                }
                openai_tool_calls = message.get("tool_calls", [])
                if openai_tool_calls:
                    bedrock_tool_use = {
                        "toolUseId": openai_tool_calls[0]["id"],
                        "name": openai_tool_calls[0]["function"]["name"],
                        "input": json.loads(
                            openai_tool_calls[0]["function"]["arguments"]
                        ),
                    }
                    bedrock_message["content"].append({"toolUse": bedrock_tool_use})
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = openai_tool_calls[0]["id"]
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "tool":
                bedrock_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": CURRENT_TOOLUSE_ID,
                                "content": [{"text": message.get("content")}],
                            }
                        }
                    ],
                }
                bedrock_messages.append(bedrock_message)
            else:
                raise ValueError(f"Invalid role: {message.get('role')}")
        return system_prompt, bedrock_messages

    def _convert_bedrock_response_to_openai_format(self, bedrock_response):
        # Convert Bedrock response format to OpenAI format
        content = ""
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            content_array = bedrock_response["output"]["message"]["content"]
            content = "".join(item.get("text", "") for item in content_array)
        if content == "":
            content = "."

        # Handle tool calls in response
        openai_tool_calls = []
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            for content_item in bedrock_response["output"]["message"]["content"]:
                if content_item.get("toolUse"):
                    bedrock_tool_use = content_item["toolUse"]
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                    openai_tool_call = {
                        "id": CURRENT_TOOLUSE_ID,
                        "type": "function",
                        "function": {
                            "name": bedrock_tool_use["name"],
                            "arguments": json.dumps(bedrock_tool_use["input"]),
                        },
                    }
                    openai_tool_calls.append(openai_tool_call)

        # Construct final OpenAI format response
        openai_format = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "created": int(time.time()),
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "finish_reason": bedrock_response.get("stopReason", "end_turn"),
                    "index": 0,
                    "message": {
                        "content": content,
                        "role": bedrock_response.get("output", {})
                        .get("message", {})
                        .get("role", "assistant"),
                        "tool_calls": (
                            openai_tool_calls if openai_tool_calls != [] else None
                        ),
                        "function_call": None,
                    },
                }
            ],
            "usage": {
                "completion_tokens": bedrock_response.get("usage", {}).get(
                    "outputTokens", 0
                ),
                "prompt_tokens": bedrock_response.get("usage", {}).get(
                    "inputTokens", 0
                ),
                "total_tokens": bedrock_response.get("usage", {}).get("totalTokens", 0),
            },
        }
        return OpenAIResponse(openai_format)

    async def _invoke_bedrock(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Non-streaming invocation of Bedrock model
        (
            system_prompt,
            bedrock_messages,
        ) = self._convert_openai_messages_to_bedrock_format(messages)
        response = self.client.converse(
            modelId=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            toolConfig={"tools": tools} if tools else None,
        )
        openai_response = self._convert_bedrock_response_to_openai_format(response)
        return openai_response

    async def _invoke_bedrock_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Streaming invocation of Bedrock model
        (
            system_prompt,
            bedrock_messages,
        ) = self._convert_openai_messages_to_bedrock_format(messages)
        response = self.client.converse_stream(
            modelId=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            toolConfig={"tools": tools} if tools else None,
        )

        # Initialize response structure
        bedrock_response = {
            "output": {"message": {"role": "", "content": []}},
            "stopReason": "",
            "usage": {},
            "metrics": {},
        }
        bedrock_response_text = ""
        bedrock_response_tool_input = ""

        # Process streaming response
        stream = response.get("stream")
        if stream:
            for event in stream:
                if event.get("messageStart", {}).get("role"):
                    bedrock_response["output"]["message"]["role"] = event[
                        "messageStart"
                    ]["role"]
                if event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                    bedrock_response_text += event["contentBlockDelta"]["delta"]["text"]
                    print(
                        event["contentBlockDelta"]["delta"]["text"], end="", flush=True
                    )
                if event.get("contentBlockStop", {}).get("contentBlockIndex") == 0:
                    bedrock_response["output"]["message"]["content"].append(
                        {"text": bedrock_response_text}
                    )
                if event.get("contentBlockStart", {}).get("start", {}).get("toolUse"):
                    bedrock_tool_use = event["contentBlockStart"]["start"]["toolUse"]
                    tool_use = {
                        "toolUseId": bedrock_tool_use["toolUseId"],
                        "name": bedrock_tool_use["name"],
                    }
                    bedrock_response["output"]["message"]["content"].append(
                        {"toolUse": tool_use}
                    )
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                if event.get("contentBlockDelta", {}).get("delta", {}).get("toolUse"):
                    bedrock_response_tool_input += event["contentBlockDelta"]["delta"][
                        "toolUse"
                    ]["input"]
                    print(
                        event["contentBlockDelta"]["delta"]["toolUse"]["input"],
                        end="",
                        flush=True,
                    )
                if event.get("contentBlockStop", {}).get("contentBlockIndex") == 1:
                    bedrock_response["output"]["message"]["content"][1]["toolUse"][
                        "input"
                    ] = json.loads(bedrock_response_tool_input)
        print()
        openai_response = self._convert_bedrock_response_to_openai_format(
            bedrock_response
        )
        return openai_response

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Optional[bool] = True,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Main entry point for chat completion
        bedrock_tools = []
        if tools is not None:
            bedrock_tools = self._convert_openai_tools_to_bedrock_format(tools)
        if stream:
            return self._invoke_bedrock_stream(
                model,
                messages,
                max_tokens,
                temperature,
                bedrock_tools,
                tool_choice,
                **kwargs,
            )
        else:
            return self._invoke_bedrock(
                model,
                messages,
                max_tokens,
                temperature,
                bedrock_tools,
                tool_choice,
                **kwargs,
            )


class ChatCompletions_request(ChatCompletions):
    def __init__(
        self, api_url: Optional[str] = None, api_token: Optional[str] = None
    ):
        #super().__init__(client)
        self.api_url = api_url
        self.api_token = api_token

    def _modify_bedrock_messages(self, message):
        new_message = []
        for m in message:
            if m.get("role") == "system":
                new_message.append(m)
                continue

            new_content = []
            for c in m.get("content", []):
                if "text" in c:
                    new_content.append({"type": "text", "text": c["text"]})
                elif "toolResult" in c:
                    # 关键：保留 toolResult 结构，不要抹平
                    new_content.append({"type": "text", "text": c["toolResult"]})
                else:
                    # 其他类型按需透传或处理
                    new_content.append(c)

            # 注意：m 可能本来就是 "assistant" 或 "user"；对于包含 toolResult 的消息，
            # Bedrock 语义要求 role="user" 回传结果
            new_role = m.get("role", "user")
            if any("toolResult" in b for b in new_content):
                new_role = "user"

            new_message.append({"role": new_role, "content": new_content})
        return new_message

    def _hamster_convert_bedrock_response_to_openai_format(self, bedrock_response: Union[str, Dict[str, Any]]):
        if isinstance(bedrock_response, str):
            bedrock_response = json.loads(bedrock_response)

        role = "assistant"
        content_text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        output = bedrock_response.get("output", {})
        message = (output or {}).get("message", {}) or {}
        role = message.get("role", role)

        content_blocks = message.get("content")
        if isinstance(content_blocks, list):
            for blk in content_blocks:
                if isinstance(blk, dict) and "text" in blk:
                    content_text_parts.append(blk.get("text", ""))
                if isinstance(blk, dict) and "toolUse" in blk:
                    tu = blk["toolUse"] or {}
                    tool_calls.append({
                        "id": tu.get("toolUseId") or f"tool_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(tu.get("input", {}), ensure_ascii=False)
                        }
                    })

        if not content_text_parts:
            result_txt = bedrock_response.get("result")
            if isinstance(result_txt, str):
                content_text_parts.append(result_txt)

        if not tool_calls:
            tu = bedrock_response.get("tool_use")
            if tu:
                tu_list = tu if isinstance(tu, list) else [tu]
                for t in tu_list:
                    if not isinstance(t, dict):
                        continue
                    tool_calls.append({
                        "id": t.get("toolUseId") or f"tool_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": t.get("name", ""),
                            "arguments": json.dumps(t.get("input", {}), ensure_ascii=False)
                        }
                    })

        content_str = "".join(content_text_parts) if content_text_parts else ""

        u = bedrock_response.get("usage") or {}
        prompt_tokens = u.get("inputTokens", u.get("input_tokens", 0)) or 0
        completion_tokens = u.get("outputTokens", u.get("output_tokens", 0)) or 0
        total_tokens = u.get("totalTokens", u.get("total_tokens")) or (prompt_tokens + completion_tokens)

        stop_reason = bedrock_response.get("stopReason") or bedrock_response.get("stop_reason")
        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = stop_map.get(stop_reason, "tool_calls" if tool_calls else "stop")

        openai_format: Dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "created": int(time.time()),
            "object": "chat.completion",
            "system_fingerprint": "",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": content_str,
                        "role": role or "assistant",
                        "tool_calls": tool_calls,
                        "function_call": None,
                    },
                }
            ],
            "usage": {
                "completion_tokens": int(completion_tokens),
                "prompt_tokens": int(prompt_tokens),
                "total_tokens": int(total_tokens),
            },
        }

        return OpenAIResponse(openai_format)

    def _get_bedrock_messages(self, messages):
        new_messages = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "tool":
                role = "user"
            new_messages.append(
                {
                    "role" : role,
                    "content" : [
                        {
                            "type" : "text",
                            "text" : m.get("content", "")
                        }
                    ]
                }
            )
        return new_messages

    async def _invoke_bedrock(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        """
        使用HTTP请求调用远程API（重写父类方法）
        """
        # 重用父类的消息转换逻辑
        (
            system_prompt,
            bedrock_messages,
        ) = self._convert_openai_messages_to_bedrock_format(messages)

        # 准备HTTP请求
        headers = {"Content-Type": "application/json", "token": self.api_token}

        bedrock_messages = self._modify_bedrock_messages(bedrock_messages)

        bedrock_messages = self._get_bedrock_messages(
            messages)

        data = {
            "model_name": model,
            "system": system_prompt,
            "message": bedrock_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            #data["tools"] = tools
            # data["tool_choice"] = tool_choice
            data['toolConfig'] = {
                "tools": tools
            }

        # 发送HTTP请求替换 client.converse 调用
        try:
            response = requests.post(
                url=self.api_url, headers=headers, json=data, timeout=600, stream=False
            )
            response.raise_for_status()
            response_data = response.json()

            # 重用父类的响应转换逻辑
            openai_response = self._hamster_convert_bedrock_response_to_openai_format(
                response_data
            )
            return openai_response

        except Exception as e:
            print(f"HTTP request error: {e}")
            raise

    async def _invoke_bedrock_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        """
        流式调用远程API（当前实现为非流式）
        """
        return await self._invoke_bedrock(
            model, messages, max_tokens, temperature, tools, tool_choice, **kwargs
        )
