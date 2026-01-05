"""
LLM Adapter - 将我们的 config.py 适配到 deep_research_agent

提供 OpenAI 兼容的 chat completion 接口，底层使用 Grok/GLM
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm, get_default_model, create_grok_llm, create_local_llm

from common import TokenTracker


@dataclass
class FunctionCall:
    """模拟 OpenAI 的 function_call 响应"""
    name: str
    arguments: str


@dataclass
class Message:
    """模拟 OpenAI 的 message 响应"""
    content: Optional[str]
    function_call: Optional[FunctionCall] = None
    role: str = "assistant"


@dataclass
class Choice:
    """模拟 OpenAI 的 choice 响应"""
    message: Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class Usage:
    """模拟 OpenAI 的 usage 响应"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Any = None


@dataclass
class ChatCompletionResponse:
    """模拟 OpenAI 的 chat completion 响应"""
    choices: List[Choice]
    usage: Usage
    model: str
    id: str = "chatcmpl-adapter"


class LLMAdapter:
    """
    LLM 适配器 - 提供 OpenAI 兼容的 chat_completion 接口
    
    将 deep_research_agent 的 OpenAI/Claude 调用统一为我们的 Grok/GLM
    """
    
    # 角色映射: 原版用什么模型 -> 我们用什么
    ROLE_MODEL_MAP = {
        "planner": "complex_reasoning",  # Planner 需要复杂推理 -> Grok
        "executor": "multi_step",         # Executor 执行任务 -> Grok
        "default": "simple_extraction",   # 默认 -> GLM
    }
    
    def __init__(self):
        self.token_tracker = TokenTracker()
        self._last_usage = None
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        reasoning_effort: str = 'high',
        role: str = "default",  # 新增: 指定角色以选择合适的模型
    ) -> ChatCompletionResponse:
        """
        OpenAI 兼容的 chat completion 接口
        
        Args:
            messages: 对话消息列表
            model: 原始模型名 (会被忽略，使用我们的配置)
            temperature: 生成温度
            functions: 函数定义列表
            function_call: 函数调用模式
            reasoning_effort: 推理强度 (用于选择模型)
            role: 角色 ("planner", "executor", "default")
        
        Returns:
            OpenAI 兼容的响应对象
        """
        start_time = time.time()
        
        # 根据角色选择任务类型
        task_type = self.ROLE_MODEL_MAP.get(role, "simple_extraction")
        
        # 获取 LLM 实例
        llm = get_llm(task_type, temperature=temperature)
        
        # 构建完整的对话历史提示
        # 重要: 需要保留完整的对话历史，否则模型不知道之前做了什么
        system_msg = ""
        conversation_parts = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                conversation_parts.append(f"[User]: {msg['content']}")
            elif msg["role"] == "assistant":
                if msg.get("content"):
                    conversation_parts.append(f"[Assistant]: {msg['content']}")
                elif msg.get("function_call"):
                    fc = msg["function_call"]
                    fc_name = fc.get("name", fc) if isinstance(fc, dict) else fc
                    conversation_parts.append(f"[Assistant]: Called function {fc_name}")

        # 如果有 functions，添加函数说明到系统提示
        if functions:
            func_descriptions = self._format_functions(functions)
            system_msg += f"\n\n## Available Functions\n{func_descriptions}\n"
            system_msg += "\nTo call a function, respond with JSON in this exact format:\n"
            system_msg += '```json\n{"function_call": {"name": "function_name", "arguments": {"arg1": "value1"}}}\n```'

        # 组合完整提示 - 包含完整对话历史
        conversation_history = "\n\n".join(conversation_parts)

        # 检查是否刚刚成功写入了 scratchpad.md，如果是，强调下一步行动
        if conversation_history and "Successfully created/updated file: scratchpad.md" in conversation_history:
            system_msg += "\n\n## CRITICAL WORKFLOW INSTRUCTION"
            system_msg += "\nYou have just written to scratchpad.md. Now you MUST follow this exact workflow:"
            system_msg += "\n1. If the scratchpad contains instructions in 'Next Steps and Action Items' for the Executor, respond with ONLY the text: INVOKE_EXECUTOR"
            system_msg += "\n2. Do NOT call create_file again unless you need to revise the scratchpad."
            system_msg += "\n3. Do NOT output TASK_COMPLETE - the research has not started yet!"
            system_msg += "\n4. The Executor will perform web searches and data collection based on your scratchpad instructions."
            system_msg += "\n\nYour next response should be exactly: INVOKE_EXECUTOR"

        full_prompt = f"{system_msg}\n\n## Conversation History\n{conversation_history}" if system_msg else conversation_history
        
        try:
            # 调用 LLM
            response = llm.invoke(full_prompt)
            content = response.content
            
            # 计算 token 使用量 (估算)
            prompt_tokens = len(full_prompt) // 4  # 粗略估算
            completion_tokens = len(content) // 4
            
            # 尝试解析函数调用
            function_call_obj = None
            if functions:
                function_call_obj = self._parse_function_call(content)
                if function_call_obj:
                    content = None  # 如果有函数调用，content 设为 None
            
            # 构建响应
            message = Message(
                content=content,
                function_call=function_call_obj,
                role="assistant"
            )
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            self._last_usage = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'cached_prompt_tokens': 0,
                'total_cost': 0.0  # 我们的模型成本可以忽略
            }
            
            thinking_time = time.time() - start_time
            self._last_usage['thinking_time'] = thinking_time
            
            return ChatCompletionResponse(
                choices=[Choice(message=message)],
                usage=usage,
                model=get_default_model()
            )
            
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}")
    
    def _format_functions(self, functions: List[Dict]) -> str:
        """格式化函数定义为文本"""
        lines = []
        for func in functions:
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            
            lines.append(f"### {name}")
            lines.append(f"Description: {desc}")
            
            if params.get("properties"):
                lines.append("Parameters:")
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required = param_name in params.get("required", [])
                    lines.append(f"  - {param_name} ({param_type}{'*' if required else ''}): {param_desc}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_function_call(self, content: str) -> Optional[FunctionCall]:
        """从响应内容中解析函数调用"""
        if not content:
            return None

        import re

        # 方法1: 提取 ```json ... ``` 代码块中的内容
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', content)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                data = json.loads(json_str)
                if "function_call" in data:
                    fc = data["function_call"]
                    return FunctionCall(
                        name=fc.get("name", ""),
                        arguments=json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                    )
            except json.JSONDecodeError:
                pass

        # 方法2: 查找 {"function_call": ...} 格式的 JSON
        # 使用平衡括号匹配
        start_idx = content.find('{"function_call"')
        if start_idx != -1:
            # 找到匹配的结束括号
            depth = 0
            end_idx = start_idx
            for i, c in enumerate(content[start_idx:], start_idx):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            if end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                try:
                    data = json.loads(json_str)
                    if "function_call" in data:
                        fc = data["function_call"]
                        return FunctionCall(
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                        )
                except json.JSONDecodeError:
                    pass

        return None
    
    def get_token_usage(self) -> Dict[str, Union[int, float]]:
        """获取上次调用的 token 使用量"""
        return self._last_usage or {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cached_prompt_tokens': 0,
            'total_cost': 0.0
        }


# 全局实例
_adapter = None


def get_adapter() -> LLMAdapter:
    """获取全局适配器实例"""
    global _adapter
    if _adapter is None:
        _adapter = LLMAdapter()
    return _adapter


def chat_completion_with_adapter(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    role: str = "default",
    **kwargs
) -> ChatCompletionResponse:
    """便捷函数: 使用适配器进行 chat completion"""
    adapter = get_adapter()
    return adapter.chat_completion(messages, model, role=role, **kwargs)
