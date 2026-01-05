"""
Anthropic API Adapter - 用我们的 Grok/GLM 模拟 Anthropic Claude API

提供与 Anthropic SDK 兼容的接口，底层使用我们的 LLM 配置
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm


@dataclass
class Usage:
    """模拟 Anthropic 的 usage 响应"""
    input_tokens: int
    output_tokens: int


@dataclass
class TextBlock:
    """模拟 Anthropic 的 text block"""
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    """模拟 Anthropic 的 tool_use block"""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class Message:
    """模拟 Anthropic 的 Message 响应"""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Any] = None
    model: str = ""
    stop_reason: str = "end_turn"
    usage: Usage = None

    def __post_init__(self):
        if self.content is None:
            self.content = []

    def model_dump(self) -> Dict:
        """转换为字典格式"""
        content_list = []
        for block in self.content:
            if isinstance(block, TextBlock):
                content_list.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content_list.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
            elif isinstance(block, dict):
                content_list.append(block)
        
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "content": content_list,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens
            }
        }


class BetaMessages:
    """模拟 Anthropic 的 beta.messages 接口"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def create(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 4000,
        tools: List[Dict] = None,
        thinking: Dict = None,
        **kwargs
    ) -> Message:
        """创建消息（模拟 Anthropic API）"""
        return self.parent._create_message(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            thinking=thinking,
            **kwargs
        )


class Beta:
    """模拟 Anthropic 的 beta 接口"""
    
    def __init__(self, parent):
        self.messages = BetaMessages(parent)


class AnthropicAdapter:
    """
    Anthropic SDK 兼容适配器
    
    使用我们的 Grok/GLM 模拟 Anthropic Claude API
    """
    
    def __init__(self, api_key: str = None):
        """初始化适配器（api_key 被忽略，使用我们的配置）"""
        self.beta = Beta(self)
        self._call_id = 0
    
    def _create_message(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 4000,
        tools: List[Dict] = None,
        thinking: Dict = None,
        **kwargs
    ) -> Message:
        """内部方法：创建消息响应"""
        
        # 获取 LLM 实例 (executor 需要多步执行能力)
        llm = get_llm("multi_step", temperature=0.3)
        
        # 构建提示
        system_msg = ""
        conversation = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_msg = content if isinstance(content, str) else json.dumps(content)
            elif role == "user":
                if isinstance(content, str):
                    conversation.append(f"User: {content}")
                elif isinstance(content, list):
                    # 处理 tool_result 等复杂内容
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                tool_content = block.get("content", "")
                                conversation.append(f"Tool Result ({block.get('tool_use_id', '')}): {tool_content}")
                            elif block.get("type") == "text":
                                conversation.append(f"User: {block.get('text', '')}")
            elif role == "assistant":
                if isinstance(content, str):
                    conversation.append(f"Assistant: {content}")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                conversation.append(f"Assistant: {block.get('text', '')}")
                            elif block.get("type") == "tool_use":
                                conversation.append(f"Assistant used tool: {block.get('name', '')} with {json.dumps(block.get('input', {}))}")
        
        # 添加工具说明
        tools_desc = ""
        if tools:
            tools_desc = "\n\n## Available Tools\n"
            for tool in tools:
                name = tool.get("name", "")
                desc = tool.get("description", "")
                tools_desc += f"\n### {name}\n{desc}\n"
                if tool.get("input_schema", {}).get("properties"):
                    tools_desc += "Parameters:\n"
                    for param, info in tool["input_schema"]["properties"].items():
                        tools_desc += f"  - {param}: {info.get('description', '')}\n"
            
            tools_desc += """
## Tool Usage Instructions
To use a tool, respond with JSON in this format:
```json
{"tool_use": {"name": "tool_name", "input": {"param1": "value1"}}}
```

If you need to provide a text response along with tool usage, first write your text, then add the tool_use JSON.
If no tool is needed, just respond with text.
"""
        
        # 组合完整提示
        full_prompt = system_msg + tools_desc + "\n\n" + "\n".join(conversation) + "\n\nAssistant:"
        
        try:
            # 调用 LLM
            response = llm.invoke(full_prompt)
            content_text = response.content
            
            # 估算 token 使用量
            input_tokens = len(full_prompt) // 4
            output_tokens = len(content_text) // 4
            
            # 解析响应，检查是否有工具调用
            content_blocks = []
            tool_use = self._parse_tool_use(content_text)
            
            if tool_use:
                # 提取工具调用前的文本
                text_before_tool = self._extract_text_before_tool(content_text)
                if text_before_tool:
                    content_blocks.append(TextBlock(text=text_before_tool))
                
                # 添加工具调用
                self._call_id += 1
                content_blocks.append(ToolUseBlock(
                    id=f"toolu_{self._call_id:05d}",
                    name=tool_use["name"],
                    input=tool_use["input"]
                ))
            else:
                # 纯文本响应
                content_blocks.append(TextBlock(text=content_text))
            
            return Message(
                id=f"msg_{int(time.time())}",
                content=content_blocks,
                model=model,
                stop_reason="tool_use" if tool_use else "end_turn",
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            )
            
        except Exception as e:
            # 返回错误消息
            return Message(
                id=f"msg_{int(time.time())}",
                content=[TextBlock(text=f"Error: {str(e)}")],
                model=model,
                stop_reason="error",
                usage=Usage(input_tokens=0, output_tokens=0)
            )
    
    def _parse_tool_use(self, content: str) -> Optional[Dict]:
        """从响应中解析工具调用"""
        import re
        
        # 首先尝试直接解析整个内容为 JSON（处理纯 JSON 响应）
        content_stripped = content.strip()
        try:
            data = json.loads(content_stripped)
            if "tool_use" in data:
                return data["tool_use"]
            # 如果直接是工具调用格式（没有外层 tool_use）
            if "name" in data and "input" in data:
                return data
        except json.JSONDecodeError:
            pass
        
        # 匹配代码块中的 JSON
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
        ]
        
        for pattern in code_block_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    if "tool_use" in data:
                        return data["tool_use"]
                    if "name" in data and "input" in data:
                        return data
                except json.JSONDecodeError:
                    continue
        
        # 使用括号匹配来找到完整的 JSON 对象（处理嵌套）
        # 查找 {"tool_use": ...} 模式
        tool_use_pattern = r'\{"tool_use"\s*:'
        matches = list(re.finditer(tool_use_pattern, content))
        
        for match in matches:
            start_pos = match.start()
            # 从 { 开始，找到匹配的 }
            brace_count = 0
            json_start = None
            
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start is not None:
                        try:
                            json_str = content[json_start:i+1]
                            data = json.loads(json_str)
                            if "tool_use" in data:
                                return data["tool_use"]
                        except json.JSONDecodeError:
                            pass
                        break
        
        # 也检查直接的工具调用格式 {"name": ..., "input": ...}
        direct_pattern = r'\{"name"\s*:\s*"[^"]+"\s*,\s*"input"'
        matches = list(re.finditer(direct_pattern, content))
        
        for match in matches:
            start_pos = match.start()
            brace_count = 0
            json_start = None
            
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start is not None:
                        try:
                            json_str = content[json_start:i+1]
                            data = json.loads(json_str)
                            if "name" in data and "input" in data:
                                return data
                        except json.JSONDecodeError:
                            pass
                        break
        
        return None
    
    def _extract_text_before_tool(self, content: str) -> str:
        """提取工具调用之前的文本"""
        import re
        
        # 找到 JSON 块或 tool_use 开始的位置
        patterns = [
            r'```json\s*\{.*?"tool_use"',
            r'\{"tool_use"',
            r'```json\s*\{"name"',
            r'\{"name":\s*"[^"]+",\s*"input"',
        ]
        
        earliest_pos = len(content)
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
        
        if earliest_pos < len(content):
            text = content[:earliest_pos].strip()
            return text if text else ""
        
        return ""


# 提供与 Anthropic SDK 相同的导入方式
Anthropic = AnthropicAdapter
