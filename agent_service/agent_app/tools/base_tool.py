"""
Base Tool Class
Utilities for creating LangChain tools
"""

from typing import Any, Callable, Optional, Dict, List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
import logging

from agent_app.core import progress, tool_cache

logger = logging.getLogger(__name__)


class ToolMetadata(BaseModel):
    """Metadata for tool registration"""
    name: str = Field(description="Tool name (unique identifier)")
    description: str = Field(description="What this tool does")
    category: str = Field(description="Tool category (service, camera, recipe, etc.)")
    requires_approval: bool = Field(
        default=False,
        description="Whether tool requires human approval before execution"
    )


class BaseTool:
    """
    Utility class for creating LangChain tools

    Simplifies tool creation with consistent interface
    """

    @staticmethod
    def create_tool(
        func: Callable,
        metadata: ToolMetadata,
        args_schema: Optional[BaseModel] = None
    ) -> StructuredTool:
        """
        Create a LangChain StructuredTool from a function

        Args:
            func: Python function to wrap
            metadata: Tool metadata
            args_schema: Pydantic model for function arguments

        Returns:
            StructuredTool ready to use with LangChain

        Example:
            def my_func(param: str) -> str:
                return f"Result: {param}"

            class MyFuncArgs(BaseModel):
                param: str = Field(description="Input parameter")

            tool = BaseTool.create_tool(
                func=my_func,
                metadata=ToolMetadata(
                    name="my_tool",
                    description="Does something useful",
                    category="utility"
                ),
                args_schema=MyFuncArgs
            )
        """
        logger.debug(f"Creating tool: {metadata.name}")

        # Cache đặt ở ĐÂY vì đây là chỗ duy nhất mọi tool đi qua — bọc ở từng
        # tool thì chắc chắn bỏ sót cái này cái kia. `should_cache` liệt kê CÓ
        # thay vì loại trừ, nên category mới mặc định không cache: bỏ sót một
        # tool đọc thì chỉ chậm, còn cache lỡ một tool ghi thì trả kết quả sai.
        run = func
        if tool_cache.should_cache(metadata.name, metadata.category,
                                   metadata.requires_approval):
            run = tool_cache.wrap(func, metadata.name)

        # Báo tiến trình cũng đặt ở đây, cùng lý do với cache: một chỗ duy nhất mọi
        # tool đi qua. Bọc NGOÀI cache để lần cache hit cũng được báo — người dùng
        # thấy "đang tính pass/fail" rồi xong ngay, đó là thông tin đúng.
        #
        # Bỏ qua tool `agent` (bốn agent con): chúng đã tự báo bằng
        # `progress.agent_started`, báo thêm ở đây thành hai dòng cho một việc.
        if metadata.category != "agent":
            run = progress.timed(run, metadata.name)

        return StructuredTool(
            name=metadata.name,
            description=metadata.description,
            func=run,
            args_schema=args_schema,
            metadata={
                "category": metadata.category,
                "requires_approval": metadata.requires_approval
            }
        )


class ToolRegistry:
    """
    Global registry for all tools

    Allows tools to be shared across multiple agents
    """

    _tools: Dict[str, StructuredTool] = {}

    @classmethod
    def register(cls, tool: StructuredTool):
        """Register a tool globally"""
        if tool.name in cls._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        cls._tools[tool.name] = tool
        logger.info(f"✅ Registered tool: {tool.name}")

    @classmethod
    def get_tool(cls, name: str) -> Optional[StructuredTool]:
        """Get a tool by name"""
        return cls._tools.get(name)

    @classmethod
    def get_tools_by_category(cls, category: str) -> List[StructuredTool]:
        """Get all tools in a category"""
        return [
            tool for tool in cls._tools.values()
            if tool.metadata.get("category") == category
        ]

    @classmethod
    def get_all_tools(cls) -> List[StructuredTool]:
        """Get all registered tools"""
        return list(cls._tools.values())

    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]:
        """List all tools with metadata"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.metadata.get("category"),
                "requires_approval": tool.metadata.get("requires_approval", False)
            }
            for tool in cls._tools.values()
        ]
